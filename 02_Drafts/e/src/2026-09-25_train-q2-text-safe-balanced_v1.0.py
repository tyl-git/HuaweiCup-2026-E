"""Train an isolated temporal model with true input-level text-gap augmentation.

Only official train/valid data are used. Existing checkpoints/results are never
overwritten, and test/specialist splits are never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
RESULTS = ROOT / "03_Results" / "e" / "question-two"
SAFE = RESULTS / "q2-text-safe-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
TEXT_CACHE = RESULTS / "bert-cache-v1.0"
OUT = RESULTS / "q2-text-safe-balanced-sqrt-v1.0"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CONDITIONS = tuple((fraction, position) for fraction in (0.2, 0.4)
                   for position in ("start", "middle", "end"))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def train_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float64)
    if np.any(counts <= 0):
        raise RuntimeError("All classes must occur in train")
    raw = 1.0 / np.sqrt(counts / counts.sum())
    return (raw / raw.mean()).astype(np.float32)


def normalize_text(raw, mask, normalizer):
    stats = normalizer.statistics["text"]
    output = np.zeros(raw.shape, dtype=np.float32)
    flat, out, selected = raw.reshape(-1, raw.shape[-1]), output.reshape(-1, raw.shape[-1]), mask.reshape(-1)
    for start in range(0, len(flat), normalizer.chunk_rows):
        stop = min(start + normalizer.chunk_rows, len(flat))
        take = selected[start:stop]
        if np.any(take):
            values = (flat[start:stop][take].astype(np.float64) - stats.mean) / stats.scale
            out[start:stop][take] = values.astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("nonfinite normalized text gap")
    return output


def load_safe_condition(split, condition, batch, normalizer, manifest):
    fraction, position = condition
    name = f"{int(fraction * 100):02d}pct_{position}"
    path = SAFE / f"{split}_textgap_{name}.npz"
    if sha256(path) != manifest["files"][path.name]:
        raise RuntimeError(f"safe cache checksum mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as archive:
        ids = archive["ids"].astype(str)
        raw = archive["features"]
        effective = archive["effective_mask"]
        removed = archive["removed_mask"]
    if not np.array_equal(ids, np.asarray(batch.ids, dtype=str)):
        raise RuntimeError(f"ID order mismatch: {path.name}")
    if not np.array_equal(effective, batch.content_mask & ~removed):
        raise RuntimeError(f"mask mismatch: {path.name}")
    return normalize_text(raw, effective, normalizer), effective


def batch_forward(model, arrays, masks, batch, rows, torch, device):
    features = {m: torch.as_tensor(np.asarray(arrays[m][rows]), dtype=torch.float32, device=device)
                for m in MODALITIES}
    content = torch.as_tensor(np.asarray(batch.content_mask[rows]).copy(), dtype=torch.bool, device=device)
    effective = {m: torch.as_tensor(np.asarray(masks[m][rows]).copy(), dtype=torch.bool, device=device)
                 for m in MODALITIES}
    classes = torch.as_tensor(np.asarray(batch.classification_labels[rows]).copy(), dtype=torch.long, device=device)
    target = torch.as_tensor(np.asarray(batch.regression_labels[rows]).copy(), dtype=torch.float32, device=device)
    return model(features, content, effective), classes, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    data = load_module("text_safe_train_data", "2026-09-24_q2_data_v1.0.py")
    normalizer_mod = load_module("text_safe_train_norm", "2026-09-24_q2-normalization_v1.0.py")
    robust = load_module("text_safe_train_robust", "2026-09-24_train-q2-robust_v1.0.py")
    temporal = load_module("text_safe_train_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    metrics = load_module("text_safe_train_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    missing = load_module("text_safe_train_missing", "2026-09-24_q2-missing-intervals_v1.0.py")
    manifest = json.loads((SAFE / "manifest.json").read_text(encoding="utf-8"))
    batches = data.load_official()
    del batches["test"]
    train, valid = batches["train"], batches["valid"]
    scaler = normalizer_mod.MultimodalStandardizer.load(STATS)
    for split, batch in (("train", train), ("valid", valid)):
        meta = json.loads((TEXT_CACHE / f"{split}.json").read_text(encoding="utf-8"))
        path = TEXT_CACHE / f"{split}.npy"
        if sha256(path) != meta["feature_sha256"]:
            raise RuntimeError(f"BERT cache checksum mismatch: {split}")
    complete_raw = {"text": np.load(TEXT_CACHE / "train.npy", mmap_mode="r", allow_pickle=False),
                    "audio": train.audio, "vision": train.vision}
    valid_raw = {"text": np.load(TEXT_CACHE / "valid.npy", mmap_mode="r", allow_pickle=False),
                 "audio": valid.audio, "vision": valid.vision}
    complete_train, train_masks = robust.normalize_once(train, complete_raw, scaler)
    complete_valid, valid_masks = robust.normalize_once(valid, valid_raw, scaler)
    weights = train_weights(train.classification_labels)
    prior = load_module("text_safe_train_baseline", "2026-09-24_train-q2-baselines_v1.0.py").train_priors(train)
    config = temporal.TemporalConfig()
    if args.preflight:
        model = temporal.build_model(prior, config, torch).to(args.device).eval()
        rows = np.arange(min(16, len(train)))
        (logits, intensity), classes, target = batch_forward(model, complete_train, train_masks, train, rows, torch, args.device)
        if logits.shape != (len(rows), 3) or intensity.shape != (len(rows),):
            raise RuntimeError("complete forward shape mismatch")
        safe_text, safe_mask = load_safe_condition("train", CONDITIONS[0], train, scaler, manifest)
        arrays = dict(complete_train); masks = dict(train_masks)
        arrays["text"] = safe_text; masks["text"] = safe_mask
        (logits, intensity), _, _ = batch_forward(model, arrays, masks, train, rows, torch, args.device)
        if not torch.isfinite(logits).all() or not torch.isfinite(intensity).all():
            raise RuntimeError("input-level forward nonfinite")
        print(f"Preflight OK: train={len(train)}, valid={len(valid)}, text_conditions={len(CONDITIONS)}; no optimizer or output written")
        return
    # The compressed condition caches are intentionally loaded only after the
    # lightweight preflight. This keeps preflight responsive on limited RAM.
    safe_train, safe_valid = {}, {}
    for condition in CONDITIONS:
        safe_train[condition] = load_safe_condition("train", condition, train, scaler, manifest)
        safe_valid[condition] = load_safe_condition("valid", condition, valid, scaler, manifest)
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior run: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "provenance.json").write_text(json.dumps({
        "schema": "q2_text_safe_balanced_sqrt_v1.0", "train_n": len(train), "valid_n": len(valid),
        "test_evaluated": False, "specialists_evaluated": False, "selection_split": "valid_complete",
        "text_mask_scope": "input-level [MASK] before frozen BERT; removed rows unavailable after encoding",
        "conditions": [f"{int(f*100):02d}pct_{p}" for f, p in CONDITIONS],
        "classification_weights": weights.tolist(), "normalization_sha256": sha256(STATS),
        "safe_manifest_sha256": sha256(SAFE / "manifest.json")}, indent=2) + "\n", encoding="utf-8")
    results = []
    for seed in (20260924, 20260925, 20260926):
        robust.set_seed(seed, torch)
        folder = OUT / f"seed_{seed}"
        folder.mkdir()
        model = temporal.build_model(prior, config, torch).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best_loss, best_epoch, stale = float("inf"), 0, 0
        for epoch in range(1, config.max_epochs + 1):
            model.train()
            robust_masks, plan = robust.train_epoch_masks(train, train_masks, missing, seed, epoch, config)
            permutation = torch.randperm(len(train), generator=generator).numpy()
            for rows in np.array_split(permutation, max(1, int(np.ceil(len(train) / config.batch_size)))):
                use_safe = int(hashlib.sha256(f"{seed}:{epoch}:{int(rows[0])}".encode()).hexdigest()[:8], 16) % 2 == 0
                if use_safe:
                    condition = CONDITIONS[(epoch + int(rows[0])) % len(CONDITIONS)]
                    text_values, text_mask = safe_train[condition]
                    arrays = dict(complete_train); arrays["text"] = text_values
                    masks = dict(train_masks); masks["text"] = text_mask
                else:
                    arrays, masks = complete_train, robust_masks
                optimizer.zero_grad(set_to_none=True)
                (logits, intensity), classes, target = batch_forward(model, arrays, masks, train, rows, torch, args.device)
                loss = torch.nn.functional.cross_entropy(logits, classes, weight=torch.as_tensor(weights, device=args.device)) + torch.nn.functional.l1_loss(intensity, target)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"nonfinite loss seed={seed} epoch={epoch}")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            valid_tensors = temporal.tensors_for(valid, complete_valid, valid_masks, torch, args.device)
            report = temporal.evaluate(model, valid_tensors, config, torch, metrics)
            print(f"text_safe_balanced seed={seed} epoch={epoch} valid_complete_loss={report['selection_loss']:.6f}", flush=True)
            if report["selection_loss"] < best_loss - 1e-8:
                best_loss, best_epoch, stale = report["selection_loss"], epoch, 0
                torch.save({"model_state": model.state_dict(), "model": "text_safe_balanced_sqrt", "seed": seed,
                            "epoch": epoch, "config": asdict(config), "train_priors": prior,
                            "normalization_sha256": sha256(STATS)}, folder / "best.pt")
            else:
                stale += 1
            if stale >= config.patience:
                break
        saved = torch.load(folder / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(saved["model_state"]); model.eval()
        valid_tensors = temporal.tensors_for(valid, complete_valid, valid_masks, torch, args.device)
        complete_report = temporal.evaluate(model, valid_tensors, config, torch, metrics)
        gap_reports = {}
        for condition, (text_values, text_mask) in safe_valid.items():
            arrays = dict(complete_valid); arrays["text"] = text_values
            masks = dict(valid_masks); masks["text"] = text_mask
            tensors = temporal.tensors_for(valid, arrays, masks, torch, args.device)
            gap_reports[f"{int(condition[0]*100):02d}pct_{condition[1]}"] = temporal.evaluate(model, tensors, config, torch, metrics)
        result = {"seed": seed, "best_epoch": best_epoch, "valid_complete": complete_report,
                  "valid_input_text_gap": gap_reports, "test_evaluated": False,
                  "specialists_evaluated": False, "checkpoint_sha256": sha256(folder / "best.pt")}
        (folder / "metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        results.append(result)
    (OUT / "summary.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Text-safe balanced train/valid complete: {OUT}; test untouched")


if __name__ == "__main__":
    main()
