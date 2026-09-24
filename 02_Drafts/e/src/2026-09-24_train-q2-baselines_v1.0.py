"""Train reproducible Q2 unimodal and simple-fusion baselines on official train/valid.

Only frozen-BERT cache and official audio/vision enter this pipeline. No test or
specialist sample is evaluated. Mean pooling uses the independent observation
masks after applying the train-only normalizer, so empty observations stay empty.
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
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "03_Results" / "e" / "question-two"
OUT = RESULTS / "q2-baselines-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
MODES = {"text": ("text",), "audio": ("audio",),
         "vision": ("vision",), "fusion": ("text", "audio", "vision")}
DIMENSIONS = {"text": 768, "audio": 74, "vision": 35}
SEEDS = (20260924, 20260925, 20260926)


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class TrainingConfig:
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
    checkpoint_selection: str = "minimum_valid_cross_entropy_plus_l1"


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


def pool_normalized(batch, raw_features, normalizer, chunk_size: int = 64):
    """Return mean of observed standardized content rows and availability flags."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    n = len(batch)
    if set(raw_features) != set(DIMENSIONS):
        raise ValueError("Expected exactly text/audio/vision features")
    pooled = {m: np.zeros((n, d), dtype=np.float32) for m, d in DIMENSIONS.items()}
    available = {m: np.zeros(n, dtype=bool) for m in DIMENSIONS}
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        raw = {m: np.asarray(raw_features[m][start:end]) for m in DIMENSIONS}
        masks = {m: batch.observation_masks[m][start:end] for m in DIMENSIONS}
        normalized = normalizer.transform(raw, batch.content_mask[start:end], masks)
        for modality in DIMENSIONS:
            mask = normalized.observation_masks[modality]
            counts = mask.sum(axis=1, dtype=np.int32)
            values = normalized.features[modality]
            numerators = np.einsum("nt,ntd->nd", mask, values, dtype=np.float64)
            pooled[modality][start:end] = (numerators / np.maximum(counts[:, None], 1)).astype(np.float32)
            available[modality][start:end] = counts > 0
            if not np.isfinite(pooled[modality][start:end]).all():
                raise RuntimeError(f"Nonfinite pooled {modality} features")
            if np.any(pooled[modality][start:end][counts == 0] != 0):
                raise RuntimeError(f"Empty {modality} observations must remain zero")
    return pooled, available


def train_priors(batch):
    counts = np.bincount(batch.classification_labels, minlength=3)
    if np.any(counts == 0):
        raise ValueError("All three classes are required to define a fixed prior")
    return {"class_counts": counts.tolist(),
            "class_probabilities": (counts / counts.sum()).tolist(),
            "intensity_mean": float(np.mean(batch.regression_labels, dtype=np.float64))}


def build_model(modalities, priors, config: TrainingConfig, torch):
    nn = torch.nn

    class Baseline(nn.Module):
        def __init__(self):
            super().__init__()
            self.modalities = tuple(modalities)
            self.encoders = nn.ModuleDict({
                m: nn.Sequential(nn.LayerNorm(DIMENSIONS[m]),
                                 nn.Linear(DIMENSIONS[m], config.hidden_dim), nn.ReLU())
                for m in self.modalities})
            width = len(self.modalities) * (config.hidden_dim + 1)
            self.fusion = nn.Sequential(nn.Linear(width, config.fusion_dim), nn.ReLU(),
                                        nn.Dropout(config.dropout))
            self.classifier = nn.Linear(config.fusion_dim, 3)
            self.regressor = nn.Linear(config.fusion_dim, 1)
            self.register_buffer("prior_logits", torch.tensor(
                np.log(priors["class_probabilities"]), dtype=torch.float32))
            self.register_buffer("prior_intensity", torch.tensor(
                priors["intensity_mean"], dtype=torch.float32))

        def forward(self, features, observed):
            encoded = []
            flags = []
            for modality in self.modalities:
                x = features[modality]
                flag = observed[modality].reshape(-1, 1).to(dtype=x.dtype)
                encoded.append(self.encoders[modality](x) * flag)
                flags.append(flag)
            available = torch.stack([observed[m].bool() for m in self.modalities], dim=1).any(dim=1)
            hidden = self.fusion(torch.cat(encoded + flags, dim=1))
            logits = self.classifier(hidden)
            intensity = self.regressor(hidden).squeeze(1)
            # No trainable signal is invented when every selected modality is empty.
            logits = torch.where(available[:, None], logits, self.prior_logits[None, :])
            intensity = torch.where(available, intensity, self.prior_intensity)
            return logits, intensity

    return Baseline()


def tensors_for(split, pooled, available, modalities, torch, device):
    return ({m: torch.as_tensor(pooled[m], dtype=torch.float32, device=device)
             for m in modalities},
            {m: torch.as_tensor(available[m], dtype=torch.bool, device=device)
             for m in modalities},
            torch.as_tensor(np.array(split.classification_labels), dtype=torch.long, device=device),
            torch.as_tensor(np.array(split.regression_labels), dtype=torch.float32, device=device))


def forward_indices(model, tensors, index):
    features, observed, classes, scores = tensors
    logits, intensity = model({m: x[index] for m, x in features.items()},
                              {m: x[index] for m, x in observed.items()})
    return logits, intensity, classes[index], scores[index]


def evaluate(model, tensors, config, torch, metrics, *, return_predictions=False):
    model.eval()
    count = tensors[2].shape[0]
    logits_all, scores_all = [], []
    total_ce = total_l1 = 0.0
    with torch.inference_mode():
        for start in range(0, count, config.batch_size):
            index = slice(start, min(start + config.batch_size, count))
            logits, score, classes, target = forward_indices(model, tensors, index)
            total_ce += torch.nn.functional.cross_entropy(logits, classes, reduction="sum").item()
            total_l1 += torch.nn.functional.l1_loss(score, target, reduction="sum").item()
            logits_all.append(logits.cpu().numpy())
            scores_all.append(score.cpu().numpy())
    logits = np.concatenate(logits_all)
    predictions = np.concatenate(scores_all)
    classes = tensors[2].cpu().numpy()
    target = tensors[3].cpu().numpy()
    ce, l1 = total_ce / count, total_l1 / count
    report = {"cross_entropy": ce, "l1": l1,
            "selection_loss": config.classification_loss_weight * ce + config.regression_loss_weight * l1,
            "classification": metrics.classification_metrics(classes, logits.argmax(axis=1)),
            "regression": metrics.regression_metrics(target, predictions)}
    if return_predictions:
        return report, {"logits": logits, "intensity": predictions,
                        "true_class": classes, "true_intensity": target}
    return report


def train_one(name, seed, train, valid, train_pool, train_available,
              valid_pool, valid_available, priors, config, torch, metrics, device):
    set_seed(seed, torch)
    modalities = MODES[name]
    model = build_model(modalities, priors, config, torch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_tensors = tensors_for(train, train_pool, train_available, modalities, torch, device)
    valid_tensors = tensors_for(valid, valid_pool, valid_available, modalities, torch, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    folder = OUT / name / f"seed_{seed}"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {folder}")
    folder.mkdir(parents=True)
    checkpoint = folder / "best.pt"
    best_loss = float("inf")
    best_epoch = 0
    history = []
    stale = 0
    expected_best_predictions = None
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator).to(device)
        for index in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, score, classes, target = forward_indices(model, train_tensors, index)
            loss = (config.classification_loss_weight *
                    torch.nn.functional.cross_entropy(logits, classes) +
                    config.regression_loss_weight *
                    torch.nn.functional.l1_loss(score, target))
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite training loss: {name}/{seed}/{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        result, predictions = evaluate(model, valid_tensors, config, torch, metrics,
                                        return_predictions=True)
        history.append({"epoch": epoch, "valid_cross_entropy": result["cross_entropy"],
                        "valid_l1": result["l1"], "valid_selection_loss": result["selection_loss"]})
        if result["selection_loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = result["selection_loss"], epoch, 0
            expected_best_predictions = {k: v.copy() for k, v in predictions.items()}
            torch.save({"model_state": model.state_dict(), "model": name, "modalities": modalities,
                        "seed": seed, "epoch": epoch, "config": asdict(config), "train_priors": priors,
                        "normalization_sha256": file_sha256(STATS)}, checkpoint)
        else:
            stale += 1
        print(f"{name} seed={seed} epoch={epoch} valid_loss={result['selection_loss']:.6f}", flush=True)
        if stale >= config.patience:
            break
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    valid_report, valid_predictions = evaluate(model, valid_tensors, config, torch, metrics,
                                                return_predictions=True)
    for field, expected in expected_best_predictions.items():
        np.testing.assert_array_equal(valid_predictions[field], expected,
                                      err_msg=f"Reloaded checkpoint changes {field}")
    train_report, train_predictions = evaluate(model, train_tensors, config, torch, metrics,
                                                return_predictions=True)
    for split, predictions, batch, observed in (
            ("train", train_predictions, train, train_available),
            ("valid", valid_predictions, valid, valid_available)):
        np.savez_compressed(folder / f"{split}_predictions.npz", ids=np.asarray(batch.ids),
                            available=np.column_stack([observed[m] for m in modalities]),
                            **predictions)
    report = {"model": name, "modalities": list(modalities), "seed": seed,
              "best_epoch": best_epoch, "epochs_run": len(history),
              "checkpoint_selection": config.checkpoint_selection,
              "config": asdict(config), "train_priors": priors,
              "normalization_sha256": saved["normalization_sha256"],
              "valid": valid_report, "train": train_report,
              "checkpoint_reload_predictions_identical": True,
              "checkpoint_sha256": file_sha256(checkpoint),
              "prediction_files_sha256": {split: file_sha256(folder / f"{split}_predictions.npz")
                                            for split in ("train", "valid")},
              "empty_observation_samples": {
                  "train": int((~np.column_stack([train_available[m] for m in modalities]).any(axis=1)).sum()),
                  "valid": int((~np.column_stack([valid_available[m] for m in modalities]).any(axis=1)).sum())},
              "test_evaluated": False, "specialists_evaluated": False}
    (folder / "metrics.json").write_text(json.dumps(report, indent=2, ensure_ascii=False,
                                               allow_nan=False) + "\n", encoding="utf-8")
    (folder / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n",
                                           encoding="utf-8")
    return report


def self_test():
    import torch
    norm = module("q2_baseline_norm_test", "2026-09-24_q2-normalization_v1.0.py")
    rng = np.random.default_rng(123)
    content = np.zeros((3, 50), dtype=bool)
    content[:, 1:4] = True
    raw = {m: rng.normal(size=(3, 50, d)).astype(np.float32)
           for m, d in DIMENSIONS.items()}
    masks = {m: content.copy() for m in DIMENSIONS}
    for m in DIMENSIONS:
        masks[m][0, 2] = False
        raw[m][0, 1] = 0
        masks[m][2] = False
    class TinyBatch:
        content_mask = content
        observation_masks = masks
        def __len__(self):
            return 3
    standardizer = norm.MultimodalStandardizer().fit(
        raw, content, masks, split="train", source_metadata={"fixture": "synthetic"})
    pooled, available = pool_normalized(TinyBatch(), raw, standardizer, chunk_size=2)
    transformed = standardizer.transform(raw, content, masks)
    for m in DIMENSIONS:
        expected = transformed.features[m][1, 1:4].mean(axis=0)
        np.testing.assert_allclose(pooled[m][1], expected, atol=1e-7)
        assert not available[m][2] and not pooled[m][2].any()
        assert available[m][0]  # One surviving observed content position.
    priors = {"class_probabilities": [0.2, 0.3, 0.5], "intensity_mean": 0.25}
    model = build_model(MODES["fusion"], priors, TrainingConfig(), torch)
    features = {m: torch.tensor(pooled[m]) for m in DIMENSIONS}
    flags = {m: torch.tensor(available[m]) for m in DIMENSIONS}
    logits, intensity = model(features, flags)
    np.testing.assert_allclose(logits[2].detach().numpy(), np.log(priors["class_probabilities"]), atol=1e-7)
    assert intensity[2].item() == priors["intensity_mean"]
    # Empty rows do not update any model parameters through a fabricated vector.
    model.zero_grad(set_to_none=True)
    (logits[2].sum() + intensity[2]).backward()
    assert all(p.grad is None or not p.grad.any() for p in model.parameters())
    for mode, modalities in MODES.items():
        output = build_model(modalities, priors, TrainingConfig(), torch)(features, flags)
        assert output[0].shape == (3, 3) and output[1].shape == (3,)
        assert torch.isfinite(output[0]).all() and torch.isfinite(output[1]).all()
    # Checkpoint state is sufficient for exact deterministic eval predictions.
    model.eval()
    before = model(features, flags)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(MODES["fusion"], priors, TrainingConfig(), torch).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    after = restored(features, flags)
    for expected, actual in zip(before, after):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    set_seed(42, torch)
    a = build_model(MODES["audio"], priors, TrainingConfig(), torch).eval()(features, flags)
    set_seed(42, torch)
    b = build_model(MODES["audio"], priors, TrainingConfig(), torch).eval()(features, flags)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    print("Baseline self-test OK: masked means, empty priors/gradients, four models, checkpoint roundtrip, deterministic seeds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--run", action="store_true", help="Train all four baselines at three fixed seeds")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    loader = module("q2_baseline_loader", "2026-09-24_q2_data_v1.0.py")
    cache = module("q2_baseline_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    norm = module("q2_baseline_norm", "2026-09-24_q2-normalization_v1.0.py")
    metrics = module("q2_baseline_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    batches = loader.load_official()
    # The existing loader validates all splits in the single official PKL.
    # Discard held-out objects before preprocessing or model construction.
    del batches["test"]
    model_info = cache.model_fingerprint(cache.MODEL)
    standardizer = norm.MultimodalStandardizer.load(STATS)
    train_source = next(iter(batches["train"].source_sha256.values()))
    if standardizer.metadata["source_metadata"]["source_sha256"] != dict(batches["train"].source_sha256):
        raise RuntimeError("Normalizer was fitted on a different official source")
    pooled = {}
    available = {}
    cache_hashes = {}
    for split in ("train", "valid"):
        batch = batches[split]
        meta = cache.verify_cache(batch, model_info)
        if split == "train" and standardizer.metadata["source_metadata"]["text_cache_contract"] != meta["contract"]:
            raise RuntimeError("Normalizer was fitted with a different BERT source/model/token contract")
        cache_hashes[split] = meta["feature_sha256"]
        text = np.load(cache.OUT / f"{split}.npy", mmap_mode="r", allow_pickle=False)
        pooled[split], available[split] = pool_normalized(
            batch, {"text": text, "audio": batch.audio, "vision": batch.vision}, standardizer)
        print(f"{split}: pooled {len(batch)} verified samples", flush=True)
    if standardizer.metadata["source_metadata"]["text_feature_sha256"] != cache_hashes["train"]:
        raise RuntimeError("Normalizer was fitted with a different train BERT cache")
    provenance = {"official_source_sha256": train_source,
                  "bert_cache_sha256": cache_hashes,
                  "normalization_sha256": file_sha256(STATS),
                  "torch": torch.__version__, "numpy": np.__version__,
                  "device": args.device, "seeds": list(SEEDS),
                  "model_types": list(MODES), "config": asdict(TrainingConfig()),
                  "fit_split": "train", "selection_split": "valid",
                  "test_evaluated": False, "specialists_evaluated": False,
                  "pooling": "mean_of_observed_standardized_content_rows",
                  "precision": "float32", "deterministic_algorithms": True,
                  "allow_tf32": False, "gradient_clip_norm": 1.0,
                  "dependency_sha256": {
                      filename: file_sha256(Path(__file__).with_name(filename))
                      for filename in (Path(__file__).name,
                                       "2026-09-24_q2_data_v1.0.py",
                                       "2026-09-24_prepare-q2-text_v1.0.py",
                                       "2026-09-24_q2-normalization_v1.0.py",
                                       "2026-09-24_q2-evaluation_v1.0.py")}}
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite previous experiment: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                                          encoding="utf-8")
    priors = train_priors(batches["train"])
    results = []
    for name in MODES:
        for seed in SEEDS:
            results.append(train_one(name, seed, batches["train"], batches["valid"],
                                     pooled["train"], available["train"], pooled["valid"],
                                     available["valid"], priors, TrainingConfig(),
                                     torch, metrics, args.device))
    summary = [{"model": r["model"], "seed": r["seed"], "best_epoch": r["best_epoch"],
                "accuracy": r["valid"]["classification"]["accuracy"],
                "macro_f1": r["valid"]["classification"]["macro_f1"],
                "mae": r["valid"]["regression"]["mae"],
                "pearson_r": r["valid"]["regression"]["pearson_r"]}
               for r in results]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                             allow_nan=False) + "\n", encoding="utf-8")
    print("All train/valid baselines complete; test and specialists untouched", flush=True)


if __name__ == "__main__":
    main()
