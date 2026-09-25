"""Compare input-level and post-BERT text gaps on official validation only.

Existing temporal checkpoints are frozen. The same token intervals are used
for both evaluations, and no official test or specialist sample is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
RESULTS = ROOT / "03_Results" / "e" / "question-two"
SAFE = RESULTS / "q2-text-safe-v1.0"
CHECKPOINTS = RESULTS / "q2-temporal-v1.0" / "temporal"
OUTPUT = RESULTS / "q2-text-safe-eval-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)
FRACTIONS = (0.2, 0.4)
POSITIONS = ("start", "middle", "end")


def dependency(filename: str):
    spec = importlib.util.spec_from_file_location("text_safe_eval_" + filename, SRC / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_inputs():
    data = dependency("2026-09-24_q2_data_v1.0.py")
    normalizer = dependency("2026-09-24_q2-normalization_v1.0.py")
    robust = dependency("2026-09-24_train-q2-robust_v1.0.py")
    temporal = dependency("2026-09-24_train-q2-temporal_v1.0.py")
    metrics = dependency("2026-09-24_q2-evaluation_v1.0.py")
    missing = dependency("2026-09-24_q2-missing-intervals_v1.0.py")
    batches = data.load_official()
    valid = batches["valid"]
    del batches
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    cache_path = RESULTS / "bert-cache-v1.0" / "valid.npy"
    cache_meta = json.loads(cache_path.with_suffix(".json").read_text(encoding="utf-8"))
    if sha256(cache_path) != cache_meta["feature_sha256"]:
        raise RuntimeError("Complete validation BERT cache checksum mismatch")
    complete_raw = {"text": np.load(cache_path, mmap_mode="r", allow_pickle=False),
                    "audio": valid.audio, "vision": valid.vision}
    complete, observed = robust.normalize_once(valid, complete_raw, scaler)
    if not np.array_equal(observed["text"], valid.content_mask):
        raise RuntimeError("Unexpected complete text observation mask")
    return valid, scaler, complete, observed, temporal, metrics, missing


def safe_condition(valid, scaler, observed, missing, fraction, position, manifest):
    condition = f"{int(fraction * 100):02d}pct_{position}"
    path = SAFE / f"valid_textgap_{condition}.npz"
    if sha256(path) != manifest["files"][path.name]:
        raise RuntimeError(f"Text-safe cache checksum mismatch: {path.name}")
    with np.load(path, allow_pickle=False) as saved:
        ids = saved["ids"].astype(str)
        removed = saved["removed_mask"]
        effective = saved["effective_mask"]
        masked_input = saved["masked_input_ids"]
        features = saved["features"]
    if not np.array_equal(ids, np.asarray(valid.ids, dtype=str)):
        raise RuntimeError(f"Validation ID order mismatch: {condition}")
    plan = missing.make_interval_plan(
        valid.content_mask, valid.observation_masks, valid.ids,
        modalities=("text",), fraction=fraction, position=position, seed=20260930)
    if not np.array_equal(removed, plan.removed_masks["text"]):
        raise RuntimeError(f"Post-BERT and input-level intervals differ: {condition}")
    if not np.array_equal(effective, observed["text"] & ~removed):
        raise RuntimeError(f"Effective text mask mismatch: {condition}")
    if not np.all(masked_input[removed] == 103):
        raise RuntimeError(f"Input token mask mismatch: {condition}")
    raw = {"text": features, "audio": valid.audio, "vision": valid.vision}
    mask = {"text": effective,
            "audio": valid.observation_masks["audio"],
            "vision": valid.observation_masks["vision"]}
    transformed = scaler.transform(raw, valid.content_mask, mask)
    safe_observed = transformed.observation_masks
    for modality in ("audio", "vision"):
        if not np.array_equal(safe_observed[modality], observed[modality]):
            raise RuntimeError(f"Non-text mask changed: {condition}/{modality}")
    if not np.array_equal(safe_observed["text"], effective):
        raise RuntimeError(f"Safe normalization mask mismatch: {condition}")
    return condition, removed, transformed.features, safe_observed


def report_row(seed, condition, kind, report):
    return {"seed": seed, "condition": condition, "encoding": kind,
            "accuracy": report["classification"]["accuracy"],
            "macro_f1": report["classification"]["macro_f1"],
            "mae": report["regression"]["mae"],
            "pearson_r": report["regression"]["pearson_r"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--check", action="store_true", help="Verify existing report without writing")
    args = parser.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    manifest = json.loads((SAFE / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_encoded") is not False or manifest.get("splits", {}).keys() != {"train", "valid"}:
        raise RuntimeError("Text-safe cache provenance is not train/valid only")
    valid, scaler, complete, observed, temporal, metrics, missing = load_inputs()
    original = temporal.tensors_for(valid, complete, observed, torch, args.device)
    checkpoints = {}
    complete_reports = {}
    for seed in SEEDS:
        folder = CHECKPOINTS / f"seed_{seed}"
        record = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
        path = folder / "best.pt"
        if sha256(path) != record["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint checksum mismatch: {seed}")
        saved = torch.load(path, map_location=args.device, weights_only=True)
        config_keys = {field.name for field in fields(temporal.TemporalConfig)}
        if saved["model"] != "temporal" or saved["seed"] != seed or set(saved["config"]) != config_keys:
            raise RuntimeError(f"Checkpoint metadata mismatch: {seed}")
        config = temporal.TemporalConfig(**saved["config"])
        model = temporal.build_model(saved["train_priors"], config, torch).to(args.device).eval()
        model.load_state_dict(saved["model_state"])
        report = temporal.evaluate(model, original, config, torch, metrics)
        for key in ("accuracy", "macro_f1"):
            np.testing.assert_allclose(report["classification"][key],
                                       record["valid_complete"]["classification"][key], atol=0, rtol=0)
        complete_reports[seed] = report_row(seed, "complete", "original", report)
        checkpoints[seed] = (model, config)
    rows = list(complete_reports.values())
    matched_rows = []
    for fraction in FRACTIONS:
        for position in POSITIONS:
            condition, removed, safe_features, safe_masks = safe_condition(
                valid, scaler, observed, missing, fraction, position, manifest)
            post_masks = {key: value.copy() for key, value in observed.items()}
            post_masks["text"] &= ~removed
            post = temporal.with_effective(original, post_masks, torch, args.device)
            safe = temporal.tensors_for(valid, safe_features, safe_masks, torch, args.device)
            for seed, (model, config) in checkpoints.items():
                post_report = temporal.evaluate(model, post, config, torch, metrics)
                safe_report = temporal.evaluate(model, safe, config, torch, metrics)
                old = report_row(seed, condition, "post_bert", post_report)
                new = report_row(seed, condition, "input_level", safe_report)
                rows.extend((old, new))
                matched_rows.append({"seed": seed, "condition": condition,
                    "removed_rows": int(removed.sum()),
                    "delta_macro_f1_input_minus_post": new["macro_f1"] - old["macro_f1"],
                    "delta_accuracy_input_minus_post": new["accuracy"] - old["accuracy"],
                    "delta_mae_input_minus_post": new["mae"] - old["mae"]})
            print(f"Verified and evaluated: {condition}", flush=True)
    aggregate = {}
    for condition in sorted({row["condition"] for row in matched_rows}):
        subset = [row for row in matched_rows if row["condition"] == condition]
        aggregate[condition] = {key: float(np.mean([row[key] for row in subset]))
                                for key in ("delta_macro_f1_input_minus_post",
                                            "delta_accuracy_input_minus_post",
                                            "delta_mae_input_minus_post")}
    result = {"schema": "q2_text_safe_eval_v1.0", "split": "official_valid",
              "n": len(valid), "checkpoint_selection": "existing_valid_only",
              "test_evaluated": False, "specialists_evaluated": False,
              "comparison": "identical text intervals; post-BERT row removal vs input-level [MASK] then BERT",
              "text_safe_manifest_sha256": sha256(SAFE / "manifest.json"),
              "normalization_sha256": sha256(STATS),
              "checkpoint_sha256": {str(seed): sha256(CHECKPOINTS / f"seed_{seed}" / "best.pt")
                                    for seed in SEEDS},
              "aggregate": aggregate, "rows": rows, "matched_deltas": matched_rows}
    if args.check:
        existing = json.loads((OUTPUT / "report.json").read_text(encoding="utf-8"))
        if existing != result:
            raise RuntimeError("Recomputed validation report differs")
        print("Text-safe validation report check passed; no writing")
        return
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {OUTPUT}")
    OUTPUT.mkdir(parents=True)
    (OUTPUT / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                         encoding="utf-8")
    with (OUTPUT / "metrics.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Text-safe validation report saved: {OUTPUT}; test untouched")
    for condition, values in aggregate.items():
        print(f"{condition}: delta Macro-F1={values['delta_macro_f1_input_minus_post']:+.4f}, "
              f"delta MAE={values['delta_mae_input_minus_post']:+.4f}")


if __name__ == "__main__":
    main()
