"""Train a focal-loss candidate on the locked temporal architecture.

This is an isolated train/valid-only candidate.  The architecture, aligned_50
input contract, train-only normalization, square-root class weights, and
contiguous-gap augmentation are unchanged. The sole model change is weighted
multiclass focal loss (gamma=2) replacing weighted CE. Checkpoints use the same
complete-validation CE+L1 criterion as the locked training protocol.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
RESULTS = ROOT / "03_Results" / "e" / "question-two"
OUT = RESULTS / "q2-focal-balanced-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)


def load_module(name, filename):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def weights(labels):
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=3).astype(np.float64)
    p = counts / counts.sum()
    raw = 1.0 / np.sqrt(p)
    return (raw / raw.mean()).astype(np.float32), counts.astype(int)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", action="store_true")
    actions.add_argument("--preflight", action="store_true")
    actions.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    import torch

    temporal = load_module("balanced_select_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    data = load_module("balanced_select_data", "2026-09-24_q2_data_v1.0.py")
    robust = load_module("balanced_select_robust", "2026-09-24_train-q2-robust_v1.0.py")
    baseline = load_module("balanced_select_baseline", "2026-09-24_train-q2-baselines_v1.0.py")
    missing = load_module("balanced_select_missing", "2026-09-24_q2-missing-intervals_v1.0.py")
    metrics = load_module("balanced_select_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    batches = data.load_official()
    train, valid = batches["train"], batches["valid"]
    if len(train) != 3395 or len(valid) != 728:
        raise RuntimeError("Official train/valid counts changed")
    normalizer = load_module("balanced_select_norm", "2026-09-24_q2-normalization_v1.0.py")
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    cache = RESULTS / "bert-cache-v1.0"
    train_raw = {"text": np.load(cache / "train.npy", mmap_mode="r", allow_pickle=False),
                 "audio": train.audio, "vision": train.vision}
    valid_raw = {"text": np.load(cache / "valid.npy", mmap_mode="r", allow_pickle=False),
                 "audio": valid.audio, "vision": valid.vision}
    train_arrays, train_masks = robust.normalize_once(train, train_raw, scaler)
    valid_arrays, valid_masks = robust.normalize_once(valid, valid_raw, scaler)
    prior = baseline.train_priors(train)
    config = temporal.TemporalConfig()
    class_weight, counts = weights(train.classification_labels)
    if args.self_test:
        temporal.self_test()
        print("Focal-loss candidate self-test OK")
        return
    model = temporal.build_model(prior, config, torch).to(args.device).eval()
    train_tensors = temporal.tensors_for(train, train_arrays, train_masks, torch, args.device)
    valid_tensors = temporal.tensors_for(valid, valid_arrays, valid_masks, torch, args.device)
    logits, intensity, *_ = temporal.forward_rows(model, train_tensors, np.arange(min(16, len(train))))
    if logits.shape[1] != 3 or intensity.ndim != 1:
        raise RuntimeError("Forward shape mismatch")
    if args.preflight:
        print(f"Preflight OK: train={len(train)}, valid={len(valid)}, focal-loss candidate; no optimizer or output written")
        return
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite prior run: {OUT}")
    OUT.mkdir(parents=True)
    (OUT / "provenance.json").write_text(json.dumps({
        "schema": "q2_focal_balanced_v1.0", "train_n": len(train), "valid_n": len(valid),
        "test_evaluated": False, "specialists_evaluated": False,
        "selection_split": "valid_complete", "selection_loss": "unweighted CE + L1",
        "training_classification_loss": "sqrt-balanced multiclass focal loss, gamma=2.0",
        "class_counts": counts.tolist(), "class_weights": class_weight.tolist(),
        "architecture": "locked q2 temporal mask-aware gate model",
        "normalization_sha256": sha256(STATS), "source_script": str(Path(__file__).resolve()),
        "acceptance_rule": "eligible only if valid macro-F1 improves >=0.01 and complete MAE does not worsen >0.01 versus locked balanced-sqrt ensemble",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_results = []
    for seed in SEEDS:
        robust.set_seed(seed, torch)
        folder = OUT / f"seed_{seed}"; folder.mkdir()
        model = temporal.build_model(prior, config, torch).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        best, best_epoch, stale = float("inf"), 0, 0
        history = []
        for epoch in range(1, config.max_epochs + 1):
            effective, plan = robust.train_epoch_masks(train, train_masks, missing, seed, epoch, config)
            epoch_tensors = temporal.with_effective(train_tensors, effective, torch, args.device)
            model.train()
            for rows in torch.randperm(len(train), generator=generator).to(args.device).split(config.batch_size):
                optimizer.zero_grad(set_to_none=True)
                logits, score, classes, target = temporal.forward_rows(model, epoch_tensors, rows)
                log_prob = torch.nn.functional.log_softmax(logits, dim=1)
                target_log_prob = log_prob.gather(1, classes[:, None]).squeeze(1)
                target_prob = target_log_prob.exp()
                alpha = torch.as_tensor(class_weight, device=args.device)[classes]
                focal = (-alpha * (1.0 - target_prob).square() * target_log_prob).mean()
                loss = focal + torch.nn.functional.l1_loss(score, target)
                if not torch.isfinite(loss): raise RuntimeError("Nonfinite loss")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            report, pred = temporal.evaluate(model, valid_tensors, config, torch, metrics, predictions=True)
            selection = report["selection_loss"]
            history.append({"epoch": epoch, "valid_unweighted_selection_loss": selection,
                            "augmentation": plan})
            print(f"focal_balanced seed={seed} epoch={epoch} valid_complete_loss={selection:.6f}", flush=True)
            if selection < best - 1e-8:
                best, best_epoch, stale = selection, epoch, 0
                torch.save({"model_state": model.state_dict(), "model": "focal_balanced", "seed": seed,
                            "epoch": epoch, "config": asdict(config), "train_priors": prior,
                            "class_weights": class_weight.tolist(), "normalization_sha256": sha256(STATS)}, folder / "best.pt")
            else:
                stale += 1
            if stale >= config.patience: break
        saved = torch.load(folder / "best.pt", map_location=args.device, weights_only=True)
        model.load_state_dict(saved["model_state"]); model.eval()
        complete, pred = temporal.evaluate(model, valid_tensors, config, torch, metrics, predictions=True)
        np.savez_compressed(folder / "valid_complete_predictions.npz", ids=np.asarray(valid.ids), **pred)
        result = {"seed": seed, "best_epoch": best_epoch, "valid_complete": complete,
                  "checkpoint_sha256": sha256(folder / "best.pt"), "test_evaluated": False,
                  "specialists_evaluated": False}
        (folder / "history.json").write_text(json.dumps(history, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (folder / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        all_results.append(result)
    (OUT / "summary.json").write_text(json.dumps(all_results, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Focal-loss train/valid complete: {OUT}; test and specialists untouched")


if __name__ == "__main__":
    main()
