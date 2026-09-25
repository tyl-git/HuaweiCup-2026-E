"""Fixed-ensemble validation-only simultaneous local-gap stress audit.

The 24 cases are declared before prediction: three modality pairs or all three,
each with a shared contiguous 20%/40% content-coordinate gap at start/middle/end.
No fitting, weight or checkpoint selection, official-test evaluation, or other
data is used. Frozen post-BERT text features are masked, so text conditions are
feature-availability stress tests, not raw-text deletion or re-encoding tests.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = Q2 / "q2-balanced-multigap-stress-valid-v1.0"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
GROUPS = (("text", "audio"), ("text", "vision"), ("audio", "vision"), MODALITIES)
PLAN_SEED = 20260924


def load(filename, name):
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def conditions():
    return [{"name": f"{'_'.join(group)}_missing_{percent}pct_{position}",
             "modalities": list(group), "fraction": percent / 100, "position": position}
            for group in GROUPS for percent in (20, 40) for position in ("start", "middle", "end")]


def shared_plan(batch, observed, condition, missing):
    # The established interval routine chooses one content-coordinate span.
    # All selected modalities share that span, including source-observation holes.
    coordinate_masks = {m: batch.content_mask for m in MODALITIES}
    base = missing.make_interval_plan(batch.content_mask, coordinate_masks, batch.ids,
                                       modalities=("text",), fraction=condition["fraction"],
                                       position=condition["position"], seed=PLAN_SEED)
    interval_mask = base.removed_masks["text"].copy()
    missing_masks = {m: interval_mask & observed[m] if m in condition["modalities"]
                     else np.zeros_like(interval_mask) for m in MODALITIES}
    effective = {m: batch.content_mask & observed[m] & ~missing_masks[m] for m in MODALITIES}
    return interval_mask, base.intervals["text"].copy(), missing_masks, effective


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch
    helper = load("2026-09-25_aggregate-q2-balanced-ensemble_v1.1.py", "multigap_fixed_ensemble")
    loader = load("2026-09-24_q2_data_v1.0.py", "multigap_loader")
    cache = load("2026-09-24_prepare-q2-text_v1.0.py", "multigap_cache")
    normalizer = load("2026-09-24_q2-normalization_v1.0.py", "multigap_normalizer")
    robust = load("2026-09-24_train-q2-robust_v1.0.py", "multigap_robust")
    missing = load("2026-09-24_q2-missing-intervals_v1.0.py", "multigap_masks")
    metric = load("2026-09-25_aggregate-q2-balanced-ensemble_v1.0.py", "multigap_metrics")
    # The official loader verifies its common aligned archive. Only valid is
    # transformed or predicted; no test labels/results enter this audit.
    batch = loader.load_official()["valid"]
    if len(batch) != 728:
        raise RuntimeError("Validation count changed")
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = cache.verify_cache(batch, model_info)
    provenance = helper.read_json(helper.TRAINED / "provenance.json")
    if (provenance["normalization_sha256"] != helper.sha(helper.STATS)
            or provenance["bert_cache_sha256"]["valid"] != cache_meta["feature_sha256"]
            or provenance["aligned_source_sha256"] != dict(batch.source_sha256)):
        raise RuntimeError("Frozen training inputs changed")
    scaler = normalizer.MultimodalStandardizer.load(helper.STATS)
    raw = {"text": np.load(cache.OUT / "valid.npy", allow_pickle=False, mmap_mode="r"),
           "audio": batch.audio, "vision": batch.vision}
    arrays, observed = robust.normalize_once(batch, raw, scaler)
    jobs = helper.load_frozen_models(torch)
    declared = conditions()
    plans = [shared_plan(batch, observed, condition, missing) for condition in declared]
    for interval, bounds, artificial, effective in plans:
        if np.any(interval & ~batch.content_mask):
            raise RuntimeError("Interval includes padding")
        for m in MODALITIES:
            if np.any(artificial[m] & ~observed[m]) or np.any(effective[m] & ~batch.content_mask):
                raise RuntimeError("Missing/effective mask violates source/content contract")
    print("Preflight OK: 728 valid, 24 predeclared shared local gaps, 3 frozen seed models")
    if args.check:
        print("No task inference, training, or output written")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    partial = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or partial.exists():
        raise FileExistsError("Stress-audit destination already exists")
    partial.mkdir(parents=True)
    protocol = {"schema": "q2_balanced_multigap_stress_valid_v1.0", "date": "2026-09-25",
                "evaluation_split": "valid", "n": 728, "conditions": declared,
                "conditions_frozen_before_task_predictions": True, "seeds": list(SEEDS),
                "weights": [1 / 3] * 3, "operator": helper.OPERATOR,
                "no_training_or_tuning": True, "official_test_predictions_used": False,
                "one_shot_holdout_claim": False, "plan_seed": PLAN_SEED,
                "span_rule": "one shared content-coordinate interval per sample; ceil(fraction * content_length), start/middle/end; all selected modalities use identical begin/end",
                "rounding_note": "Very short samples may lose all observed positions in a selected modality through ceiling or pre-existing holes; this is a local-interval test, not a whole-modality dropout design.",
                "mask_contract": "missing_m=shared_interval & source_observed_m for selected modalities; effective_m=content & source_observed_m & ~missing_m; source/padding masks unchanged",
                "comparison_boundary": "Older single-modality conditions target observed positions. This audit targets content positions for a common simultaneous time interval. Do not present these as identical mask interventions.",
                "text_scope": "post-BERT feature masking, not raw-text loss",
                "feature_version": "aligned_50", "feature_dimensions": [768, 74, 35],
                "sources": [helper.record(Path(__file__)), helper.record(SRC / "2026-09-25_aggregate-q2-balanced-ensemble_v1.1.py"),
                            helper.record(SRC / "2026-09-24_q2-missing-intervals_v1.0.py"),
                            helper.record(helper.STATS), helper.record(helper.TRAINED / "provenance.json")],
                "checkpoint_records": [{k: job[k] for k in ("seed", "checkpoint", "best_epoch", "config")} for job in jobs],
                "source_sha256": dict(batch.source_sha256), "valid_cache_sha256": cache_meta["feature_sha256"],
                "normalization_sha256": helper.sha(helper.STATS)}
    helper.write_json(partial / "protocol.json", protocol)
    # Preserve the exact plans independently of all predictions.
    mask_arrays = {"ids": np.asarray(batch.ids), "conditions": np.asarray([c["name"] for c in declared]),
                   "content_mask": batch.content_mask,
                   "interval_mask": np.stack([p[0] for p in plans]), "interval_bounds": np.stack([p[1] for p in plans]),
                   **{f"source_observed_{m}": observed[m] for m in MODALITIES},
                   **{f"source_unavailable_{m}": batch.content_mask & ~observed[m] for m in MODALITIES},
                   **{f"missing_{m}": np.stack([p[2][m] for p in plans]) for m in MODALITIES},
                   **{f"effective_{m}": np.stack([p[3][m] for p in plans]) for m in MODALITIES}}
    np.savez_compressed(partial / "condition_masks.npz", **mask_arrays)
    seed_logits = np.empty((24, 3, len(batch), 3), dtype=np.float32)
    seed_intensity = np.empty((24, 3, len(batch)), dtype=np.float32)
    for s, job in enumerate(jobs):
        model = job["model"].to(args.device)
        with torch.inference_mode():
            for c, (interval, bounds, artificial, effective) in enumerate(plans):
                for start in range(0, len(batch), 64):
                    stop = min(start + 64, len(batch))
                    features = {m: torch.as_tensor(np.where(effective[m][start:stop, :, None],
                                      arrays[m][start:stop], 0).copy(), device=args.device) for m in MODALITIES}
                    content = torch.as_tensor(batch.content_mask[start:stop].copy(), device=args.device)
                    masks = {m: torch.as_tensor(effective[m][start:stop].copy(), device=args.device) for m in MODALITIES}
                    logits, intensity = model(features, content, masks)
                    seed_logits[c, s, start:stop] = logits.cpu().numpy()
                    seed_intensity[c, s, start:stop] = intensity.cpu().numpy()
        model.cpu()
        print(f"Frozen seed={job['seed']}: all 24 validation local-gap conditions predicted", flush=True)
    if not np.isfinite(seed_logits).all() or not np.isfinite(seed_intensity).all():
        raise RuntimeError("Nonfinite stress prediction")
    combined_logits = seed_logits.mean(axis=1)
    combined_intensity = seed_intensity.mean(axis=1)
    np.savez_compressed(partial / "predictions.npz", ids=np.asarray(batch.ids), seeds=np.asarray(SEEDS),
                        conditions=np.asarray([c["name"] for c in declared]), seed_logits=seed_logits,
                        seed_intensity=seed_intensity, logits=combined_logits, intensity=combined_intensity,
                        true_class=batch.classification_labels, true_intensity=batch.regression_labels)
    complete_path = helper.OUT / "balanced_sqrt_ensemble_valid_predictions.npz"
    with np.load(complete_path, allow_pickle=False) as complete:
        if not np.array_equal(complete["ids"], np.asarray(batch.ids)):
            raise RuntimeError("Complete-validation reference IDs differ")
        complete_cm = metric.classification_metrics(complete["true_class"], complete["logits"].argmax(axis=1))
        complete_rm = metric.regression_metrics(complete["true_intensity"], complete["intensity"])
    rows, reports = [], {}
    for c, condition in enumerate(declared):
        cm = metric.classification_metrics(batch.classification_labels, combined_logits[c].argmax(axis=1))
        rm = metric.regression_metrics(batch.regression_labels, combined_intensity[c])
        reports[condition["name"]] = {"classification": cm, "regression": rm}
        rows.append({"condition": condition["name"], "modalities": "+".join(condition["modalities"]),
                     "fraction": condition["fraction"], "position": condition["position"],
                     "accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"], "neutral_f1": cm["neutral_f1"],
                     "mae": rm["mae"], "pearson_r": rm["pearson_r"],
                     "delta_macro_f1_from_complete": cm["macro_f1"] - complete_cm["macro_f1"],
                     "delta_mae_from_complete": rm["mae"] - complete_rm["mae"],
                     **{f"{m}_removed_observed_positions": int(plans[c][2][m].sum()) for m in MODALITIES},
                     **{f"{m}_samples_newly_fully_unavailable": int((observed[m].any(axis=1) & ~plans[c][3][m].any(axis=1)).sum()) for m in MODALITIES}})
    helper.write_csv(partial / "metrics.csv", rows)
    summary = {"schema": "q2_balanced_multigap_stress_valid_v1.0", "n": 728, "condition_count": 24,
               "complete_reference": helper.record(complete_path),
               "complete_validation": {"classification": complete_cm, "regression": complete_rm},
               "results": reports,
               "output_sha256": {path.name: helper.sha(path) for path in partial.iterdir() if path.is_file()}}
    helper.write_json(partial / "summary.json", summary)
    partial.rename(OUT)
    print(f"Validation-only fixed-ensemble stress audit saved: {OUT}")


if __name__ == "__main__":
    main()
