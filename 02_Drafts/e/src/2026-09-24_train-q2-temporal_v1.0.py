"""Train a small position-aware Q2 model on official aligned train/valid only.

The predictor sees content and effective observation masks at every position.
Source-zero and injected gaps remain distinct in saved audit masks, but the
predictor cannot use their provenance because attachment 3 does not supply it.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
RESULTS = ROOT / "03_Results" / "e" / "question-two"
OUT = RESULTS / "q2-temporal-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
TEXT_CACHE = RESULTS / "bert-cache-v1.0"
MODALITIES = ("text", "audio", "vision")
DIMENSIONS = {"text": 768, "audio": 74, "vision": 35}
SEEDS = (20260924, 20260925, 20260926)


def dependency(name):
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("q2_temporal_" + name, SRC / name)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class TemporalConfig:
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
    augmentation_fractions: tuple[float, ...] = (0.2, 0.4)
    augmentation_positions: tuple[str, ...] = ("start", "middle", "end", "random")
    checkpoint_selection: str = "minimum_valid_cross_entropy_plus_l1_on_complete_input"
    temporal_width: int = 32
    temporal_kernel: int = 3


def build_model(priors, config, torch, *, temporal=True, gap_signal=True):
    nn = torch.nn

    class MaskAwareTemporal(nn.Module):
        def __init__(self):
            super().__init__()
            self.project = nn.ModuleDict({m: nn.Sequential(
                nn.LayerNorm(DIMENSIONS[m]), nn.Linear(DIMENSIONS[m], config.temporal_width),
                nn.ReLU()) for m in MODALITIES})
            self.gates = nn.ModuleDict({m: nn.Linear(config.temporal_width + 1, 1)
                                        for m in MODALITIES})
            self.gap_signal = gap_signal
            self.gap_projection = nn.Linear(3, config.temporal_width) if gap_signal else None
            self.temporal = temporal
            self.local = (nn.Conv1d(config.temporal_width, config.temporal_width,
                                    config.temporal_kernel,
                                    padding=config.temporal_kernel // 2)
                          if temporal else None)
            self.time_score = nn.Linear(config.temporal_width, 1)
            self.fusion = nn.Sequential(nn.Linear(config.temporal_width, config.fusion_dim),
                                        nn.ReLU(), nn.Dropout(config.dropout))
            self.classifier = nn.Linear(config.fusion_dim, 3)
            self.regressor = nn.Linear(config.fusion_dim, 1)
            self.register_buffer("prior_logits", torch.tensor(
                np.log(priors["class_probabilities"]), dtype=torch.float32))
            self.register_buffer("prior_intensity", torch.tensor(
                priors["intensity_mean"], dtype=torch.float32))

        def forward(self, features, content, effective, *, return_evidence=False):
            if content.ndim != 2 or content.shape[1] != 50 or content.dtype != torch.bool:
                raise ValueError("content must be bool [B,50]")
            hidden, scores, flags = [], [], []
            for modality in MODALITIES:
                value, flag = features[modality], effective[modality]
                if (value.shape != (content.shape[0], 50, DIMENSIONS[modality])
                        or flag.shape != content.shape or flag.dtype != torch.bool
                        or bool((flag & ~content).any())):
                    raise ValueError(f"{modality}: feature or effective mask contract mismatch")
                # Mask before and after the affine projection: hidden values at
                # absent positions must not depend on their raw feature values.
                encoded = self.project[modality](value * flag.unsqueeze(-1))
                encoded = encoded * flag.unsqueeze(-1)
                hidden.append(encoded)
                flags.append(flag)
                scores.append(self.gates[modality](torch.cat(
                    (encoded, flag.unsqueeze(-1).to(encoded.dtype)), dim=-1)))
            flag_matrix = torch.stack(flags, dim=-1)
            score_matrix = torch.cat(scores, dim=-1).masked_fill(~flag_matrix, -1e9)
            weights = torch.softmax(score_matrix, dim=-1) * flag_matrix
            fused = sum(weights[..., i:i + 1] * hidden[i] for i in range(3))
            if self.gap_signal:
                # This is a unified within-content unavailable signal. It does
                # not reveal whether a zero came from source or augmentation.
                gaps = content.unsqueeze(-1) & ~flag_matrix
                fused = fused + self.gap_projection(gaps.to(fused.dtype))
            fused = fused * content.unsqueeze(-1)
            if self.temporal:
                local = self.local(fused.transpose(1, 2)).transpose(1, 2)
                fused = (fused + torch.relu(local)) * content.unsqueeze(-1)
            time_logits = self.time_score(fused).squeeze(-1).masked_fill(~content, -1e9)
            time_weights = torch.softmax(time_logits, dim=1) * content
            pooled = (time_weights.unsqueeze(-1) * fused).sum(dim=1)
            representation = self.fusion(pooled)
            logits = self.classifier(representation)
            intensity = self.regressor(representation).squeeze(1)
            any_observed = flag_matrix.any(dim=(1, 2))
            logits = torch.where(any_observed[:, None], logits, self.prior_logits[None, :])
            intensity = torch.where(any_observed, intensity, self.prior_intensity)
            if return_evidence:
                return logits, intensity, {"modality_gate": weights, "time_weight": time_weights}
            return logits, intensity

    return MaskAwareTemporal()


def tensors_for(batch, arrays, source_masks, torch, device):
    return (
        {m: torch.as_tensor(arrays[m], dtype=torch.float32, device=device) for m in MODALITIES},
        torch.as_tensor(batch.content_mask.copy(), dtype=torch.bool, device=device),
        {m: torch.as_tensor(source_masks[m].copy(), dtype=torch.bool, device=device)
         for m in MODALITIES},
        torch.as_tensor(batch.classification_labels.copy(), dtype=torch.long, device=device),
        torch.as_tensor(batch.regression_labels.copy(), dtype=torch.float32, device=device),
    )


def with_effective(tensors, masks, torch, device):
    return (tensors[0], tensors[1],
            {m: torch.as_tensor(masks[m].copy(), dtype=torch.bool, device=device)
             for m in MODALITIES}, tensors[3], tensors[4])


def forward_rows(model, tensors, rows):
    features, content, effective, classes, target = tensors
    logits, intensity = model({m: features[m][rows] for m in MODALITIES},
                              content[rows], {m: effective[m][rows] for m in MODALITIES})
    return logits, intensity, classes[rows], target[rows]


def evaluate(model, tensors, config, torch, metrics, *, predictions=False):
    model.eval()
    n = tensors[3].shape[0]
    logits_all, scores_all = [], []
    ce_sum = l1_sum = 0.0
    with torch.inference_mode():
        for start in range(0, n, config.batch_size):
            rows = slice(start, min(n, start + config.batch_size))
            logits, intensity, classes, target = forward_rows(model, tensors, rows)
            ce_sum += torch.nn.functional.cross_entropy(logits, classes, reduction="sum").item()
            l1_sum += torch.nn.functional.l1_loss(intensity, target, reduction="sum").item()
            logits_all.append(logits.cpu().numpy())
            scores_all.append(intensity.cpu().numpy())
    logits = np.concatenate(logits_all)
    scores = np.concatenate(scores_all)
    ce, l1 = ce_sum / n, l1_sum / n
    report = {"cross_entropy": ce, "l1": l1,
              "selection_loss": config.classification_loss_weight * ce
              + config.regression_loss_weight * l1,
              "classification": metrics.classification_metrics(
                  tensors[3].cpu().numpy(), logits.argmax(axis=1)),
              "regression": metrics.regression_metrics(tensors[4].cpu().numpy(), scores)}
    if predictions:
        return report, {"logits": logits, "intensity": scores,
                        "true_class": tensors[3].cpu().numpy(),
                        "true_intensity": tensors[4].cpu().numpy()}
    return report


def self_test():
    import torch

    torch.manual_seed(7)
    config = TemporalConfig()
    priors = {"class_probabilities": [0.2, 0.3, 0.5], "intensity_mean": 0.1}
    model = build_model(priors, config, torch).eval()
    rng = np.random.default_rng(7)
    values = {m: torch.tensor(rng.normal(size=(3, 50, d)).astype(np.float32))
              for m, d in DIMENSIONS.items()}
    content = torch.zeros((3, 50), dtype=torch.bool)
    content[:, 1:11] = True
    source = {m: content.clone() for m in MODALITIES}
    source["vision"][0, 3:5] = False
    effective = {m: mask.clone() for m, mask in source.items()}
    effective["audio"][0, 6:9] = False
    for m in MODALITIES:
        effective[m][2] = False
    before = model(values, content, effective)
    changed = {m: x.clone() for m, x in values.items()}
    for m in MODALITIES:
        changed[m][~effective[m]] = 10000
    after = model(changed, content, effective)
    for x, y in zip(before, after):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    torch.testing.assert_close(before[0][2], model.prior_logits, rtol=0, atol=1e-7)
    torch.testing.assert_close(before[1][2], model.prior_intensity, rtol=0, atol=1e-7)
    # Use identical values and observation counts so only the location changes.
    constant = {m: torch.zeros_like(value) for m, value in values.items()}
    early = {m: mask.clone() for m, mask in effective.items()}
    late = {m: mask.clone() for m, mask in effective.items()}
    for masks in (early, late):
        for m in MODALITIES:
            masks[m][0] = False
    early["audio"][0, 1:4] = True
    late["audio"][0, 7:10] = True
    with torch.no_grad():
        early_logits = model(constant, content, early)[0][0]
        late_logits = model(constant, content, late)[0][0]
        if torch.allclose(early_logits, late_logits, rtol=0, atol=1e-7):
            raise AssertionError("Temporal model ignored gap location")
        pooled = build_model(priors, config, torch, temporal=False).eval()
        pooled_early = pooled(constant, content, early)[0][0]
        pooled_late = pooled(constant, content, late)[0][0]
        torch.testing.assert_close(pooled_early, pooled_late, rtol=0, atol=1e-6)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = build_model(priors, config, torch).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    for x, y in zip(before, restored(values, content, effective)):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    print("Temporal self-test OK: mask invariance, gap position, empty prior, checkpoint")


def run_seed(variant, seed, train, valid, train_arrays, valid_arrays, train_masks,
             valid_masks, priors, config, modules, device):
    torch, robust, missing, metrics = (modules[key] for key in
                                       ("torch", "robust", "missing", "metrics"))
    robust.set_seed(seed, torch)
    folder = OUT / variant / f"seed_{seed}"
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite prior run: {folder}")
    folder.mkdir(parents=True)
    temporal = variant != "no_temporal"
    gap_signal = variant != "no_gap_signal"
    model = build_model(priors, config, torch, temporal=temporal,
                        gap_signal=gap_signal).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay)
    train_tensors = tensors_for(train, train_arrays, train_masks, torch, device)
    valid_tensors = tensors_for(valid, valid_arrays, valid_masks, torch, device)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_loss, stale, best_epoch = float("inf"), 0, 0
    best_predictions = None
    history = []
    for epoch in range(1, config.max_epochs + 1):
        effective, plan_record = robust.train_epoch_masks(
            train, train_masks, missing, seed, epoch, config)
        np.savez_compressed(folder / f"train_epoch_{epoch:02d}_masks.npz",
                            ids=np.asarray(train.ids), content=train.content_mask,
                            **{f"source_observed_{m}": train_masks[m] for m in MODALITIES},
                            **{f"source_zero_{m}": train.content_mask & ~train_masks[m]
                               for m in MODALITIES},
                            **{f"injected_missing_{m}": train_masks[m] & ~effective[m]
                               for m in MODALITIES},
                            **{f"effective_observed_{m}": effective[m] for m in MODALITIES})
        epoch_tensors = with_effective(train_tensors, effective, torch, device)
        model.train()
        for rows in torch.randperm(len(train), generator=generator).to(device).split(
                config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            logits, intensity, classes, target = forward_rows(model, epoch_tensors, rows)
            loss = (config.classification_loss_weight *
                    torch.nn.functional.cross_entropy(logits, classes)
                    + config.regression_loss_weight *
                    torch.nn.functional.l1_loss(intensity, target))
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Nonfinite loss: {variant}/{seed}/{epoch}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        report, predictions = evaluate(model, valid_tensors, config, torch, metrics,
                                       predictions=True)
        history.append({"epoch": epoch, "valid_complete_selection_loss": report["selection_loss"],
                        "augmentation": plan_record})
        if report["selection_loss"] < best_loss - 1e-8:
            best_loss, best_epoch, stale = report["selection_loss"], epoch, 0
            best_predictions = {k: v.copy() for k, v in predictions.items()}
            torch.save({"model_state": model.state_dict(), "model": variant, "seed": seed,
                        "epoch": epoch, "config": asdict(config), "train_priors": priors,
                        "normalization_sha256": sha256(STATS),
                        "augmentation_plan_seed": robust.PLAN_SEED}, folder / "best.pt")
        else:
            stale += 1
        print(f"{variant} seed={seed} epoch={epoch} valid_complete_loss="
              f"{report['selection_loss']:.6f}", flush=True)
        if stale >= config.patience:
            break
    saved = torch.load(folder / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(saved["model_state"])
    complete, checked = evaluate(model, valid_tensors, config, torch, metrics,
                                 predictions=True)
    for key in checked:
        np.testing.assert_array_equal(checked[key], best_predictions[key])
    np.savez_compressed(folder / "valid_complete_predictions.npz", ids=np.asarray(valid.ids),
                        **checked)
    train_complete = evaluate(model, train_tensors, config, torch, metrics)
    gap_reports, plans = {}, []
    for modality in MODALITIES:
        for fraction in robust.FRACTIONS:
            for position in ("start", "middle", "end"):
                name = f"{modality}_missing_{int(fraction * 100):02d}pct_{position}"
                plan = missing.make_interval_plan(
                    valid.content_mask, valid_masks, valid.ids, modalities=(modality,),
                    fraction=fraction, position=position, seed=robust.PLAN_SEED)
                effective = {m: np.asarray(valid_masks[m]).copy() for m in MODALITIES}
                effective[modality] &= ~plan.removed_masks[modality]
                masked = with_effective(valid_tensors, effective, torch, device)
                gap_reports[name], pred = evaluate(model, masked, config, torch, metrics,
                                                   predictions=True)
                np.savez_compressed(folder / f"valid_{name}_predictions.npz",
                                    ids=np.asarray(valid.ids), **pred)
                plans.append({"condition": name, "seed": robust.PLAN_SEED,
                              "summary": plan.summary(),
                              "intervals": plan.intervals[modality].tolist()})
    result = {"model": variant, "seed": seed, "best_epoch": best_epoch,
              "epochs_run": len(history), "config": asdict(config),
              "checkpoint_sha256": sha256(folder / "best.pt"),
              "normalization_sha256": saved["normalization_sha256"],
              "train_complete": train_complete, "valid_complete": complete,
              "valid_missing": gap_reports, "valid_missing_plans": plans,
              "test_evaluated": False, "specialists_evaluated": False,
              "prediction_files_sha256": {
                  p.name: sha256(p) for p in sorted(folder.glob("*_predictions.npz"))}}
    (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False,
                                                      indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    (folder / "metrics.json").write_text(json.dumps(result, ensure_ascii=False,
                                                      indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--model", choices=("temporal", "no_temporal", "no_gap_signal"),
                        default="temporal")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    loader = dependency("2026-09-24_q2_data_v1.0.py")
    cache = dependency("2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = dependency("2026-09-24_q2-normalization_v1.0.py")
    robust = dependency("2026-09-24_train-q2-robust_v1.0.py")
    baseline = dependency("2026-09-24_train-q2-baselines_v1.0.py")
    missing = dependency("2026-09-24_q2-missing-intervals_v1.0.py")
    metrics = dependency("2026-09-24_q2-evaluation_v1.0.py")
    batches = loader.load_official()
    del batches["test"]
    batches = {key: replace(batch, reference_text=None) for key, batch in batches.items()}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    model_info = cache.model_fingerprint(cache.MODEL)
    arrays, masks, cache_hashes = {}, {}, {}
    for split in ("train", "valid"):
        batch = batches[split]
        metadata = cache.verify_cache(batch, model_info)
        source = {"text": np.load(TEXT_CACHE / f"{split}.npy", mmap_mode="r",
                                  allow_pickle=False),
                  "audio": batch.audio, "vision": batch.vision}
        arrays[split], masks[split] = robust.normalize_once(batch, source, scaler)
        cache_hashes[split] = metadata["feature_sha256"]
        if split == "train" and scaler.metadata["source_metadata"]["text_cache_contract"] != metadata["contract"]:
            raise RuntimeError("Normalizer and train BERT cache contracts differ")
        print(f"{split}: verified {len(batch)} aligned samples", flush=True)
    if (scaler.metadata["source_metadata"]["source_sha256"]
            != dict(batches["train"].source_sha256)
            or scaler.metadata["source_metadata"]["text_feature_sha256"]
            != cache_hashes["train"]):
        raise RuntimeError("Normalizer was fitted on different train inputs")
    prior = baseline.train_priors(batches["train"])
    config = TemporalConfig()
    for key, value in asdict(baseline.TrainingConfig()).items():
        if key != "checkpoint_selection" and asdict(config).get(key) != value:
            raise RuntimeError(f"Baseline hyperparameter mismatch: {key}")
    for key, value in asdict(robust.RobustConfig()).items():
        if key not in ("hidden_dim", "fusion_dim") and asdict(config).get(key) != value:
            raise RuntimeError(f"Robust protocol mismatch: {key}")
    provenance = {"schema": "q2_temporal_v1.0", "aligned_source_sha256":
                  dict(batches["train"].source_sha256), "bert_cache_sha256": cache_hashes,
                  "normalization_sha256": sha256(STATS), "model": args.model,
                  "seeds": list(SEEDS), "config": asdict(config),
                  "split_fit": "train", "selection_split": "valid",
                  "plan_seed": robust.PLAN_SEED,
                  "predictor_mask_inputs": ["content", "effective_observed"],
                  "audit_only_masks": ["source_zero", "injected_missing"],
                  "text_missing_scope": "post-BERT masking; context in retained rows may leak removed words",
                  "test_evaluated": False, "specialists_evaluated": False,
                  "dependency_sha256": {p.name: sha256(p) for p in [Path(__file__),
                      SRC / "2026-09-24_q2_data_v1.0.py",
                      SRC / "2026-09-24_prepare-q2-text_v1.0.py",
                      SRC / "2026-09-24_q2-normalization_v1.0.py",
                      SRC / "2026-09-24_train-q2-robust_v1.0.py",
                      SRC / "2026-09-24_train-q2-baselines_v1.0.py",
                      SRC / "2026-09-24_q2-missing-intervals_v1.0.py",
                      SRC / "2026-09-24_q2-evaluation_v1.0.py"]}}
    folder = OUT / args.model
    if folder.exists():
        raise FileExistsError(f"Refusing to overwrite prior model: {folder}")
    reference_path = OUT / "provenance.json"
    if reference_path.exists():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        current_shared = json.loads(json.dumps(provenance))
        for record in (reference, current_shared):
            record.pop("model", None)
            record["dependency_sha256"].pop(Path(__file__).name, None)
        if current_shared != reference:
            raise RuntimeError("Existing temporal data protocol differs")
    provenance_path = (reference_path if args.model == "temporal" else
                       OUT / f"provenance_{args.model}.json")
    if provenance_path.exists():
        existing = json.loads(provenance_path.read_text(encoding="utf-8"))
        if existing != json.loads(json.dumps(provenance)):
            raise RuntimeError(f"Existing {args.model} provenance differs")
    if args.preflight:
        count = 16
        tiny = type("MaskBatch", (), {"ids": batches["train"].ids[:count],
                                      "content_mask": batches["train"].content_mask[:count]})()
        source = {m: masks["train"][m][:count] for m in MODALITIES}
        effective, plan = robust.train_epoch_masks(tiny, source, missing,
                                                    SEEDS[0], 1, config)
        model = build_model(prior, config, torch,
                            temporal=args.model != "no_temporal",
                            gap_signal=args.model != "no_gap_signal").to(args.device).eval()
        features = {m: torch.as_tensor(arrays["train"][m][:count], device=args.device)
                    for m in MODALITIES}
        content = torch.as_tensor(tiny.content_mask, device=args.device)
        with torch.inference_mode():
            for current in (source, effective):
                output = model(features, content,
                               {m: torch.as_tensor(current[m], device=args.device)
                                for m in MODALITIES})
                if output[0].shape != (count, 3) or output[1].shape != (count,) or not (
                        torch.isfinite(output[0]).all() and torch.isfinite(output[1]).all()):
                    raise RuntimeError("Preflight forward failed")
        print(f"Preflight OK: train={len(batches['train'])}, valid={len(batches['valid'])}, "
              f"masked_groups={len(plan['conditions'])}; no optimizer or output written")
        return
    OUT.mkdir(parents=True, exist_ok=True)
    if not provenance_path.exists():
        provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2)
                                   + "\n", encoding="utf-8")
    modules = {"torch": torch, "robust": robust, "missing": missing, "metrics": metrics}
    reports = [run_seed(args.model, seed, batches["train"], batches["valid"],
                        arrays["train"], arrays["valid"], masks["train"], masks["valid"],
                        prior, config, modules, args.device) for seed in SEEDS]
    (OUT / f"summary_{args.model}.json").write_text(json.dumps([
        {"model": item["model"], "seed": item["seed"], "best_epoch": item["best_epoch"],
         "valid_complete": item["valid_complete"]} for item in reports],
        ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print("Temporal train/valid complete; test and specialists untouched")


if __name__ == "__main__":
    main()
