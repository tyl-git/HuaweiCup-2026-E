"""Train a deterministic availability-aware Q2 fusion model.

The model is trained only on the official train split.  Each sample/epoch gets
a fixed, seed-derived contiguous mask or complete input, generated in
content-token coordinates before mask-aware pooling.  Validation is evaluated
on complete input and on the same fixed single-modality missing plans used by
the baseline robustness audit.  The shared loader validates all native split
structures; test is then discarded before preprocessing.  Test and attachments
are never evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
import os
import random
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "03_Results" / "e" / "question-two"
OUT = RESULTS / "q2-robust-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
TEXT_CACHE = RESULTS / "bert-cache-v1.0"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
DIMENSIONS = {"text": 768, "audio": 74, "vision": 35}
FRACTIONS = (0.2, 0.4)
POSITIONS = ("start", "middle", "end", "random")
PLAN_SEED = 20260930


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class RobustConfig:
    max_epochs: int = 40
    patience: int = 8
    batch_size: int = 64
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    dropout: float = 0.2
    hidden_dim: int = 64
    fusion_dim: int = 128
    classification_loss_weight: float = 1.0
    regression_loss_weight: float = 1.0
    augmentation_probability: float = 0.5
    augmentation_fractions: tuple[float, ...] = FRACTIONS
    augmentation_positions: tuple[str, ...] = POSITIONS
    checkpoint_selection: str = "minimum_valid_cross_entropy_plus_l1_on_complete_input"


def set_seed(seed: int, torch) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def build_model(priors, config: RobustConfig, torch):
    nn = torch.nn

    class AvailabilityAwareFusion(nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = MODALITIES
            self.encoders = nn.ModuleDict({
                m: nn.Sequential(nn.LayerNorm(DIMENSIONS[m]),
                                 nn.Linear(DIMENSIONS[m], config.hidden_dim), nn.ReLU())
                for m in MODALITIES
            })
            self.gates = nn.ModuleDict({
                m: nn.Sequential(nn.Linear(config.hidden_dim + 1, config.hidden_dim // 2),
                                 nn.ReLU(), nn.Linear(config.hidden_dim // 2, 1))
                for m in MODALITIES
            })
            self.availability = nn.Linear(len(MODALITIES), config.hidden_dim)
            self.fusion = nn.Sequential(nn.Linear(config.hidden_dim, config.fusion_dim),
                                        nn.ReLU(), nn.Dropout(config.dropout))
            self.classifier = nn.Linear(config.fusion_dim, 3)
            self.regressor = nn.Linear(config.fusion_dim, 1)
            self.register_buffer("prior_logits", torch.tensor(
                np.log(priors["class_probabilities"]), dtype=torch.float32))
            self.register_buffer("prior_intensity", torch.tensor(
                priors["intensity_mean"], dtype=torch.float32))

        def forward(self, features, observed):
            encoded, scores, flags = [], [], []
            for modality in self.modalities:
                x = features[modality]
                flag = observed[modality].reshape(-1, 1).to(dtype=x.dtype)
                hidden = self.encoders[modality](x) * flag
                encoded.append(hidden)
                flags.append(flag)
                scores.append(self.gates[modality](torch.cat((hidden, flag), dim=1)))
            flag_matrix = torch.cat(flags, dim=1)
            score_matrix = torch.cat(scores, dim=1)
            score_matrix = score_matrix.masked_fill(~flag_matrix.bool(), -1e9)
            weights = torch.softmax(score_matrix, dim=1)
            weights = weights * flag_matrix
            fused = sum(weights[:, i:i + 1] * encoded[i] for i in range(len(encoded)))
            fused = fused + self.availability(flag_matrix)
            hidden = self.fusion(fused)
            logits = self.classifier(hidden)
            intensity = self.regressor(hidden).squeeze(1)
            any_available = flag_matrix.bool().any(dim=1)
            # An all-empty row gets the train prior; no invented vector is used.
            logits = torch.where(any_available[:, None], logits, self.prior_logits[None, :])
            intensity = torch.where(any_available, intensity, self.prior_intensity)
            return logits, intensity

    return AvailabilityAwareFusion()


def pool_normalized(batch, normalized_features, normalized_masks):
    pooled, available = {}, {}
    for modality, values in normalized_features.items():
        mask = np.asarray(normalized_masks[modality], dtype=bool)
        counts = mask.sum(axis=1, dtype=np.int32)
        numerators = np.einsum("nt,ntd->nd", mask, values, dtype=np.float64)
        result = (numerators / np.maximum(counts[:, None], 1)).astype(np.float32)
        result[counts == 0] = 0
        if not np.isfinite(result).all():
            raise RuntimeError(f"Nonfinite pooled features for {modality}")
        pooled[modality] = result
        available[modality] = counts > 0
    return pooled, available


def make_pooled(batch, normalized_features, masks, normalizer=None):
    return pool_normalized(batch, normalized_features, masks)


def normalize_once(batch, raw, normalizer, chunk_size=64):
    """Bound temporary memory; preserve effective masks independently of values."""
    output = {m: np.empty((len(batch), 50, d), dtype=np.float32) for m, d in DIMENSIONS.items()}
    masks = {m: np.empty((len(batch), 50), dtype=bool) for m in MODALITIES}
    for start in range(0, len(batch), chunk_size):
        stop = min(start + chunk_size, len(batch))
        transformed = normalizer.transform({m: raw[m][start:stop] for m in MODALITIES},
                                            batch.content_mask[start:stop],
                                            {m: batch.observation_masks[m][start:stop] for m in MODALITIES})
        for modality in MODALITIES:
            output[modality][start:stop] = transformed.features[modality]
            masks[modality][start:stop] = transformed.observation_masks[modality]
    return output, masks


def tensors_for(split, pooled, available, torch, device):
    return (
        {m: torch.as_tensor(pooled[m], dtype=torch.float32, device=device) for m in MODALITIES},
        {m: torch.as_tensor(available[m], dtype=torch.bool, device=device) for m in MODALITIES},
        torch.as_tensor(np.asarray(split.classification_labels), dtype=torch.long, device=device),
        torch.as_tensor(np.asarray(split.regression_labels), dtype=torch.float32, device=device),
    )


def evaluate(model, tensors, config, torch, metrics, *, return_predictions=False):
    model.eval()
    n = tensors[2].shape[0]
    logits_all, score_all = [], []
    total_ce = total_l1 = 0.0
    with torch.inference_mode():
        for start in range(0, n, config.batch_size):
            index = slice(start, min(n, start + config.batch_size))
            features, observed, classes, target = tensors
            logits, scores = model({m: x[index] for m, x in features.items()},
                                   {m: x[index] for m, x in observed.items()})
            total_ce += torch.nn.functional.cross_entropy(logits, classes[index], reduction="sum").item()
            total_l1 += torch.nn.functional.l1_loss(scores, target[index], reduction="sum").item()
            logits_all.append(logits.cpu().numpy())
            score_all.append(scores.cpu().numpy())
    logits = np.concatenate(logits_all)
    scores = np.concatenate(score_all)
    classes = tensors[2].cpu().numpy()
    target = tensors[3].cpu().numpy()
    report = {
        "cross_entropy": total_ce / n,
        "l1": total_l1 / n,
        "selection_loss": config.classification_loss_weight * total_ce / n
        + config.regression_loss_weight * total_l1 / n,
        "classification": metrics.classification_metrics(classes, logits.argmax(axis=1)),
        "regression": metrics.regression_metrics(target, scores),
    }
    if return_predictions:
        return report, {"logits": logits, "intensity": scores,
                        "true_class": classes, "true_intensity": target}
    return report


def interval_digest(plan_records):
    canonical = json.dumps(plan_records, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def train_epoch_masks(batch, original_masks, missing, seed, epoch, config):
    """Deterministically assign complete/missing conditions per sample."""
    conditions = {}
    for row, sample_id in enumerate(batch.ids):
        key = f"{PLAN_SEED}\0{seed}\0{epoch}\0{sample_id}".encode("utf-8")
        value = int.from_bytes(hashlib.sha256(key).digest()[:8], "big")
        if value % 10000 >= round(10000 * config.augmentation_probability):
            continue
        choice = value // 10000
        modality = MODALITIES[choice % len(MODALITIES)]
        choice //= len(MODALITIES)
        fraction = config.augmentation_fractions[choice % len(config.augmentation_fractions)]
        choice //= len(config.augmentation_fractions)
        position = config.augmentation_positions[choice % len(config.augmentation_positions)]
        conditions.setdefault((modality, fraction, position), []).append(row)
    effective = {m: np.asarray(original_masks[m]).copy() for m in MODALITIES}
    records = []
    for (modality, fraction, position), rows in sorted(conditions.items()):
        index = np.asarray(rows, dtype=np.int64)
        subset_masks = {m: original_masks[m][index] for m in MODALITIES}
        plan = missing.make_interval_plan(batch.content_mask[index], subset_masks,
                                          [batch.ids[i] for i in rows],
                                          modalities=(modality,), fraction=fraction,
                                          position=position, seed=PLAN_SEED + seed + epoch)
        effective[modality][index] &= ~plan.removed_masks[modality]
        records.append({"modality": modality, "fraction": fraction,
                        "position": position, "samples": len(rows),
                        "summary": plan.summary(),
                        "sample_indices_sha256": hashlib.sha256(index.tobytes()).hexdigest(),
                        "removed_masks_sha256": hashlib.sha256(
                            plan.removed_masks[modality].tobytes()).hexdigest()})
    return effective, {"epoch": epoch, "complete_samples": len(batch.ids) - sum(map(len, conditions.values())),
                       "conditions": records}


def train_one(variant, seed, train, valid, raw_train, raw_valid, masks_train, masks_valid,
              normalizer, priors, config, torch, metrics, missing, baseline, device):
    set_seed(seed, torch)
    folder = OUT / variant / f"seed_{seed}"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {folder}")
    folder.mkdir(parents=True)
    complete_train, train_available = make_pooled(train, raw_train, masks_train, normalizer)
    complete_valid, valid_available = make_pooled(valid, raw_valid, masks_valid, normalizer)
    valid_tensors = tensors_for(valid, complete_valid, valid_available, torch, device)
    model = (baseline.build_model(MODALITIES, priors, config, torch) if variant == "fusion_aug"
             else build_model(priors, config, torch)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    plan_records = []
    best_loss, best_epoch, stale = float("inf"), 0, 0
    expected_best_predictions = None
    history = []
    for epoch in range(1, config.max_epochs + 1):
        if variant == "gated_clean":
            pooled, available = complete_train, train_available
            epoch_masks = masks_train
            record = {"epoch": epoch, "complete_samples": len(train), "conditions": []}
        else:
            epoch_masks, record = train_epoch_masks(train, masks_train, missing, seed, epoch, config)
            pooled, available = make_pooled(train, raw_train, epoch_masks, normalizer)
        plan_records.append(record)
        np.savez_compressed(folder / f"train_epoch_{epoch:02d}_masks.npz", ids=np.asarray(train.ids),
                            **{f"observed_{m}": epoch_masks[m] for m in MODALITIES})
        train_tensors = tensors_for(train, pooled, available, torch, device)
        model.train()
        order = torch.randperm(len(train), generator=generator).to(device)
        for index in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            features, observed, classes, target = train_tensors
            logits, scores = model({m: x[index] for m, x in features.items()},
                                   {m: x[index] for m, x in observed.items()})
            loss = (config.classification_loss_weight * torch.nn.functional.cross_entropy(logits, classes[index])
                    + config.regression_loss_weight * torch.nn.functional.l1_loss(scores, target[index]))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite robust loss at seed={seed}, epoch={epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        report, predictions = evaluate(model, valid_tensors, config, torch, metrics,
                                       return_predictions=True)
        history.append({"epoch": epoch, "augmentation": plan_records[epoch - 1],
                        "valid_complete_selection_loss": report["selection_loss"]})
        if report["selection_loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = report["selection_loss"], epoch, 0
            expected_best_predictions = {key: value.copy() for key, value in predictions.items()}
            torch.save({"model_state": model.state_dict(), "model": variant,
                        "modalities": MODALITIES, "seed": seed, "epoch": epoch,
                        "config": asdict(config), "augmentation_enabled": variant != "gated_clean",
                        "train_priors": priors,
                        "normalization_sha256": sha256(STATS),
                        "augmentation_plan_seed": PLAN_SEED}, folder / "best.pt")
        else:
            stale += 1
        print(f"{variant} seed={seed} epoch={epoch} valid_complete_loss={report['selection_loss']:.6f}", flush=True)
        if stale >= config.patience:
            break

    checkpoint = folder / "best.pt"
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    complete_report, complete_predictions = evaluate(model, valid_tensors, config, torch, metrics,
                                                     return_predictions=True)
    for field, expected in expected_best_predictions.items():
        np.testing.assert_array_equal(complete_predictions[field], expected,
                                      err_msg=f"Reloaded checkpoint changes {field}")
    train_complete_tensors = tensors_for(train, complete_train, train_available, torch, device)
    train_report = evaluate(model, train_complete_tensors, config, torch, metrics)
    np.savez_compressed(folder / "valid_complete_predictions.npz", ids=np.asarray(valid.ids),
                        available=np.column_stack([valid_available[m] for m in MODALITIES]),
                        **complete_predictions)
    # Paired validation robustness evaluation uses one immutable plan per
    # condition, shared by every seed and with the baseline audit.
    missing_reports = {}
    missing_plan_records = []
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            for position in ("start", "middle", "end"):
                key = f"{modality}_missing_{int(round(fraction * 100)):02d}pct_{position}"
                plan = missing.make_interval_plan(valid.content_mask, masks_valid, valid.ids,
                                                  modalities=(modality,), fraction=fraction,
                                                  position=position, seed=PLAN_SEED)
                effective_masks = {m: np.asarray(masks_valid[m]).copy() for m in MODALITIES}
                effective_masks[modality] &= ~plan.removed_masks[modality]
                pooled, available = make_pooled(valid, raw_valid, effective_masks)
                tensors = tensors_for(valid, pooled, available, torch, device)
                missing_reports[key], predictions = evaluate(model, tensors, config, torch, metrics,
                                                              return_predictions=True)
                np.savez_compressed(folder / f"valid_{key}_predictions.npz", ids=np.asarray(valid.ids),
                                    available=np.column_stack([available[m] for m in MODALITIES]),
                                    **predictions)
                missing_plan_records.append({"condition": key, "modality": modality,
                                             "fraction": fraction, "position": position,
                                             "seed": PLAN_SEED, "summary": plan.summary(),
                                             "intervals": plan.intervals[modality].tolist()})

    report = {"model": variant, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "config": asdict(config), "train_priors": priors,
              "augmentation_enabled": variant != "gated_clean",
              "normalization_sha256": saved["normalization_sha256"],
              "checkpoint_sha256": sha256(checkpoint), "train": train_report,
              "valid_complete": complete_report, "augmentation_plan_seed": PLAN_SEED,
              "augmentation_plan_digest": interval_digest(plan_records[:len(history)]),
              "augmentation_epochs": plan_records[:len(history)],
              "valid_missing_plan_seed": PLAN_SEED,
              "valid_missing_plan_digest": interval_digest(missing_plan_records),
              "valid_missing": missing_reports,
              "prediction_files_sha256": {path.name: sha256(path)
                                            for path in sorted(folder.glob("*_predictions.npz"))},
              "training_mask_files_sha256": {path.name: sha256(path)
                                             for path in sorted(folder.glob("train_epoch_*_masks.npz"))},
              "test_evaluated": False, "specialists_evaluated": False,
              "checkpoint_reload_verified": True}
    (folder / "valid_missing_plan.json").write_text(json.dumps(
        {"sample_ids": list(valid.ids), "seed": PLAN_SEED, "conditions": missing_plan_records},
        ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2,
                                                     allow_nan=False) + "\n", encoding="utf-8")
    return report, model, saved


def self_test():
    import torch
    priors = {"class_probabilities": [0.2, 0.3, 0.5], "intensity_mean": 0.1}
    model = build_model(priors, RobustConfig(), torch).eval()
    rng = np.random.default_rng(7)
    features = {m: torch.tensor(rng.normal(size=(4, DIMENSIONS[m])).astype(np.float32))
                for m in MODALITIES}
    observed = {m: torch.tensor([True, False, True, False]) for m in MODALITIES}
    observed["audio"][1] = True
    observed["vision"][1] = False
    observed["text"][3] = False
    observed["audio"][3] = False
    observed["vision"][3] = False
    logits, scores = model(features, observed)
    assert logits.shape == (4, 3) and scores.shape == (4,)
    assert torch.isfinite(logits).all() and torch.isfinite(scores).all()
    np.testing.assert_allclose(logits[3].detach().numpy(), np.log(priors["class_probabilities"]), atol=1e-7)
    np.testing.assert_allclose(scores[3].item(), priors["intensity_mean"], atol=1e-7)
    model.zero_grad(set_to_none=True)
    (logits[3].sum() + scores[3]).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.parameters())
    before = model(features, observed)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(priors, RobustConfig(), torch).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for expected, actual in zip(before, restored(features, observed)):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    missing = load_module("q2_robust_missing_test", "2026-09-24_q2-missing-intervals_v1.0.py")
    content = np.zeros((120, 50), dtype=bool)
    content[:, 1:21] = True
    original = {m: content.copy() for m in MODALITIES}
    original["vision"][1] = False
    original["audio"][::3, 4:7] = False
    class Fixture(SimpleNamespace):
        def __len__(self):
            return len(self.ids)
    batch = Fixture(ids=tuple(f"fixture_{i}" for i in range(120)), content_mask=content)
    first, record = train_epoch_masks(batch, original, missing, 20260924, 1, RobustConfig())
    repeated, repeated_record = train_epoch_masks(batch, original, missing, 20260924, 1, RobustConfig())
    assert record == repeated_record and 0 < record["complete_samples"] < len(batch)
    assert {row["modality"] for row in record["conditions"]} == set(MODALITIES)
    for modality in MODALITIES:
        np.testing.assert_array_equal(first[modality], repeated[modality])
        assert not (first[modality] & ~original[modality]).any()
    raw = {m: rng.normal(size=(120, 50, d)).astype(np.float32) for m, d in DIMENSIONS.items()}
    pooled, flags = pool_normalized(batch, raw, first)
    perturbed = {m: value.copy() for m, value in raw.items()}
    for modality in MODALITIES:
        perturbed[modality][~first[modality]] = 10000
    after_pool, _ = pool_normalized(batch, perturbed, first)
    for modality in MODALITIES:
        np.testing.assert_array_equal(pooled[modality], after_pool[modality])
    assert not flags["vision"][1] and not pooled["vision"][1].any()
    print("Robust self-test OK: empty priors/gradients, checkpoint roundtrip, reproducible diverse masks, no observation resurrection, masked pooling")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--preflight", action="store_true", help="Verify real train/valid inputs and inference without training or output")
    group.add_argument("--run", action="store_true", help="Train three ablation variants at three seeds")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", choices=("all", "fusion_aug", "gated_clean", "gated_aug"),
                        default="all", help="Run all variants or one ablation at three fixed seeds")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    loader = load_module("q2_robust_loader", "2026-09-24_q2_data_v1.0.py")
    norm = load_module("q2_robust_norm", "2026-09-24_q2-normalization_v1.0.py")
    cache = load_module("q2_robust_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    metrics = load_module("q2_robust_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    missing = load_module("q2_robust_missing", "2026-09-24_q2-missing-intervals_v1.0.py")
    baseline = load_module("q2_robust_baseline", "2026-09-24_train-q2-baselines_v1.0.py")
    batches = loader.load_official()
    del batches["test"]
    for split in ("train", "valid"):
        batches[split] = replace(batches[split], reference_text=None)
    standardizer = norm.MultimodalStandardizer.load(STATS)
    raw = {}
    normalized_masks = {}
    cache_hashes = {}
    model_info = cache.model_fingerprint(cache.MODEL)
    for split in ("train", "valid"):
        batch = batches[split]
        meta = cache.verify_cache(batch, model_info)
        source = {"text": np.load(TEXT_CACHE / f"{split}.npy", mmap_mode="r", allow_pickle=False),
                  "audio": batch.audio, "vision": batch.vision}
        raw[split], normalized_masks[split] = normalize_once(batch, source, standardizer)
        del source
        print(f"{split}: standardized {len(batch)} verified samples once", flush=True)
        cache_hashes[split] = meta["feature_sha256"]
        if split == "train" and standardizer.metadata["source_metadata"]["text_cache_contract"] != meta["contract"]:
            raise RuntimeError("Normalizer was fitted with a different BERT source/model/token contract")
    if standardizer.metadata["source_metadata"]["source_sha256"] != dict(batches["train"].source_sha256):
        raise RuntimeError("Normalizer was fitted on a different official source")
    if standardizer.metadata["source_metadata"]["text_feature_sha256"] != cache_hashes["train"]:
        raise RuntimeError("Normalizer was fitted with a different train BERT cache")
    priors = {"class_counts": np.bincount(batches["train"].classification_labels, minlength=3).tolist()}
    priors["class_probabilities"] = (np.asarray(priors["class_counts"]) / len(batches["train"])).tolist()
    priors["intensity_mean"] = float(np.mean(batches["train"].regression_labels, dtype=np.float64))
    config = RobustConfig()
    baseline_provenance_path = RESULTS / "q2-baselines-v1.0" / "provenance.json"
    baseline_provenance = json.loads(baseline_provenance_path.read_text(encoding="utf-8"))
    baseline_filename = "2026-09-24_train-q2-baselines_v1.0.py"
    if baseline_provenance["dependency_sha256"][baseline_filename] != sha256(SRC / baseline_filename):
        raise RuntimeError("Baseline trainer changed since the baseline run; use a reviewed experiment version")
    if (baseline_provenance["normalization_sha256"] != sha256(STATS)
            or baseline_provenance["bert_cache_sha256"] != cache_hashes):
        raise RuntimeError("Baseline inputs differ from the robust experiment inputs")
    if baseline_provenance["official_source_sha256"] != next(iter(batches["train"].source_sha256.values())):
        raise RuntimeError("Baseline official source differs from the robust experiment")
    for field, value in asdict(baseline.TrainingConfig()).items():
        if field != "checkpoint_selection" and asdict(config).get(field) != value:
            raise RuntimeError(f"Base training hyperparameter differs: {field}")
    provenance = {"schema": "q2_robust_training_v1.0", "split_fit": "train",
                  "selection_split": "valid", "official_source_sha256": dict(batches["train"].source_sha256),
                  "bert_cache_sha256": cache_hashes, "normalization_sha256": sha256(STATS),
                  "plan_seed": PLAN_SEED, "seeds": list(SEEDS), "config": asdict(config),
                  "variants": ["fusion_aug", "gated_clean", "gated_aug"],
                  "baseline_provenance_sha256": sha256(baseline_provenance_path),
                  "torch": torch.__version__, "numpy": np.__version__,
                  "deterministic_algorithms": True, "allow_tf32": False,
                  "text_missing_scope": "post_BERT_feature_rows; retained contextual embeddings may encode removed token information",
                  "device": args.device, "test_evaluated": False, "specialists_evaluated": False,
                  "dependency_sha256": {name: sha256(SRC / name) for name in (
                      Path(__file__).name, "2026-09-24_q2_data_v1.0.py",
                      "2026-09-24_prepare-q2-text_v1.0.py", "2026-09-24_q2-normalization_v1.0.py",
                      "2026-09-24_q2-missing-intervals_v1.0.py", "2026-09-24_q2-evaluation_v1.0.py",
                      "2026-09-24_train-q2-baselines_v1.0.py")}}
    provenance_path = OUT / "provenance.json"
    # JSON converts tuples to lists; compare like representations on later
    # single-variant launches so unchanged provenance remains reusable.
    provenance = json.loads(json.dumps(provenance, ensure_ascii=False, allow_nan=False))
    variants = ("fusion_aug", "gated_clean", "gated_aug") if args.model == "all" else (args.model,)
    if OUT.exists():
        if not provenance_path.is_file() or json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
            raise RuntimeError("Existing robust output has different or incomplete provenance")
        for variant in variants:
            if (OUT / variant).exists():
                raise FileExistsError(f"Refusing to overwrite prior variant: {OUT / variant}")
    if args.preflight:
        sample_count = 16
        tiny = SimpleNamespace(ids=batches["train"].ids[:sample_count],
                               content_mask=batches["train"].content_mask[:sample_count])
        tiny_masks = {m: normalized_masks["train"][m][:sample_count] for m in MODALITIES}
        augmented, mask_record = train_epoch_masks(tiny, tiny_masks, missing, SEEDS[0], 1, config)
        full_pool, full_flags = pool_normalized(tiny,
            {m: raw["train"][m][:sample_count] for m in MODALITIES}, tiny_masks)
        masked_pool, masked_flags = pool_normalized(tiny,
            {m: raw["train"][m][:sample_count] for m in MODALITIES}, augmented)
        for variant in variants:
            set_seed(SEEDS[0], torch)
            model = (baseline.build_model(MODALITIES, priors, config, torch) if variant == "fusion_aug"
                     else build_model(priors, config, torch)).to(args.device).eval()
            with torch.inference_mode():
                for pool, flags in ((full_pool, full_flags), (masked_pool, masked_flags)):
                    logits, score = model(
                        {m: torch.as_tensor(pool[m], device=args.device) for m in MODALITIES},
                        {m: torch.as_tensor(flags[m], device=args.device) for m in MODALITIES})
                    if logits.shape != (sample_count, 3) or score.shape != (sample_count,) or not (
                            torch.isfinite(logits).all() and torch.isfinite(score).all()):
                        raise RuntimeError(f"Nonfinite or wrong-shape preflight output: {variant}")
            print(f"{variant}: real complete and masked forward OK", flush=True)
        print(f"Preflight ready: train={len(batches['train'])}, valid={len(batches['valid'])}, "
              f"masked_conditions={len(mask_record['conditions'])}; no optimizer or output written", flush=True)
        return
    if not OUT.exists():
        OUT.mkdir(parents=True)
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reports = []
    train_masks = normalized_masks["train"]
    valid_masks = normalized_masks["valid"]
    for variant in variants:
        for seed in SEEDS:
            result, _, _ = train_one(variant, seed, batches["train"], batches["valid"],
                                     raw["train"], raw["valid"], train_masks, valid_masks,
                                     standardizer, priors, config, torch, metrics,
                                     missing, baseline, args.device)
            reports.append(result)
    (OUT / f"summary_{args.model}.json").write_text(json.dumps([
        {"model": item["model"], "seed": item["seed"], "best_epoch": item["best_epoch"],
         "accuracy": item["valid_complete"]["classification"]["accuracy"],
         "macro_f1": item["valid_complete"]["classification"]["macro_f1"],
         "mae": item["valid_complete"]["regression"]["mae"],
         "pearson_r": item["valid_complete"]["regression"]["pearson_r"]}
        for item in reports], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Robust train/valid runs complete; test and specialists untouched", flush=True)


if __name__ == "__main__":
    main()
