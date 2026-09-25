"""Correct the frozen balanced ensemble's inconsistent specialist aggregation.

Version 1.0 used mean logits on official splits but mean probabilities on
attachment 3.  Version 1.1 applies the ALREADY FIXED official mean-logits rule
to every split, without fitting, choosing weights/seeds, or changing checkpoints.
Official test results were already viewed; this is a post-test implementation
correction, not a newly untouched or one-shot holdout experiment.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
TRAINED = Q2 / "q2-temporal-balanced-sqrt-v1.0"
FINAL = Q2 / "q2-temporal-balanced-sqrt-final-v1.0"
PREVIOUS = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.0"
OUT = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.1"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CLASSES = ("Negative", "Neutral", "Positive")
OPERATOR = "arithmetic_mean_logits_and_intensity_then_softmax"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def softmax(logits):
    exp = np.exp(np.asarray(logits, dtype=np.float64)
                 - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


def ensemble_prediction(seed_logits, seed_intensity):
    """One shared operator, with float32 accumulation matching v1.0 test."""
    if seed_logits.shape[0] != 3 or seed_intensity.shape[0] != 3:
        raise ValueError("Exactly the three frozen seeds are required")
    logits = np.mean(seed_logits, axis=0)
    intensity = np.mean(seed_intensity, axis=0)
    return logits, intensity, softmax(logits)


def record(path):
    path = Path(path)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha(path)}


def load_frozen_models(torch, device="cpu"):
    """Reusable checkpoint verifier; Q3 can explain this exact predictor."""
    temporal = load("balanced_v11_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    normalization_hash = sha(STATS)
    jobs = []
    for seed in SEEDS:
        folder = TRAINED / "temporal" / f"seed_{seed}"
        checkpoint = folder / "best.pt"
        recorded = read_json(folder / "metrics.json")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (sha(checkpoint) != recorded["checkpoint_sha256"]
                or saved["model"] != "temporal" or saved["seed"] != seed
                or saved["epoch"] != recorded["best_epoch"]
                or json.loads(json.dumps(saved["config"])) != recorded["config"]
                or saved["normalization_sha256"] != normalization_hash):
            raise RuntimeError(f"Frozen checkpoint metadata differs: {seed}")
        config = temporal.TemporalConfig(**saved["config"])
        model = temporal.build_model(saved["train_priors"], config, torch)
        model.load_state_dict(saved["model_state"], strict=True)
        model.to(device).eval().requires_grad_(False)
        jobs.append({"seed": seed, "model": model, "checkpoint": record(checkpoint),
                     "best_epoch": saved["epoch"], "config": saved["config"]})
    return jobs


def frozen_text_features(batch, cache, torch, device):
    """Frozen local BERT, identical to the existing official/special interface."""
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    from transformers import BertModel
    encoder = BertModel.from_pretrained(str(cache.MODEL), local_files_only=True,
                                        attn_implementation="eager").to(device).float().eval()
    encoder.requires_grad_(False)
    features = np.empty((len(batch), 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(batch), 10):
            stop = min(start + 10, len(batch))
            inputs = {key: torch.as_tensor(np.asarray(getattr(batch, key)[start:stop]).copy(),
                                           dtype=torch.long, device=device)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            features[start:stop] = encoder(**inputs).last_hidden_state.cpu().numpy()
    if not np.isfinite(features).all():
        raise RuntimeError("Nonfinite frozen BERT output")
    del encoder
    return features


def prepare_special_tensors(batch, text_features, scaler, torch, device):
    robust = load("balanced_v11_robust", "2026-09-24_train-q2-robust_v1.0.py")
    arrays, source_observed = robust.normalize_once(
        batch, {"text": text_features, "audio": batch.audio, "vision": batch.vision}, scaler)
    features = {m: torch.as_tensor(arrays[m].copy(), device=device) for m in MODALITIES}
    content = torch.as_tensor(batch.content_mask.copy(), dtype=torch.bool, device=device)
    effective = {m: torch.as_tensor(source_observed[m].copy(), dtype=torch.bool, device=device)
                 for m in MODALITIES}
    return (features, content, effective), source_observed


def official_arrays(kind, aggregate, reference_batch):
    rows = aggregate.load_split(kind, TRAINED if kind == "valid" else FINAL)
    ids = np.asarray(reference_batch.ids)
    if not np.array_equal(ids, rows[0]["ids"]):
        raise RuntimeError(f"Official loader and frozen {kind} prediction IDs differ")
    if (not np.array_equal(reference_batch.classification_labels, rows[0]["true_class"])
            or not np.array_equal(reference_batch.regression_labels, rows[0]["true_intensity"])):
        raise RuntimeError(f"Official {kind} labels differ from frozen predictions")
    packed = {"ids": rows[0]["ids"], "seeds": np.asarray(SEEDS, dtype=np.int64),
              "logits": np.stack([row["logits"] for row in rows]),
              "intensity": np.stack([row["intensity"] for row in rows]),
              "true_class": rows[0]["true_class"], "true_intensity": rows[0]["true_intensity"]}
    for key in ("logits", "intensity"):
        if packed[key].dtype != np.float32 or not np.isfinite(packed[key]).all():
            raise RuntimeError(f"Expected finite float32 frozen {kind}/{key}")
    return packed


def frozen_validation_conditions(aggregate, reference):
    """Read all 18 existing saved gaps, without generating/selecting any mask."""
    conditions = [f"{m}_missing_{fraction}pct_{position}"
                  for m in MODALITIES for fraction in (20, 40)
                  for position in ("start", "middle", "end")]
    packed, sources = {}, []
    for condition in conditions:
        paths = [TRAINED / "temporal" / f"seed_{seed}" / f"valid_{condition}_predictions.npz"
                 for seed in SEEDS]
        rows = [aggregate.read_npz(path) for path in paths]
        for seed, row in zip(SEEDS, rows):
            for key in ("ids", "true_class", "true_intensity"):
                if not np.array_equal(row[key], reference[key]):
                    raise RuntimeError(f"Frozen validation gap IDs/labels differ: {seed}/{condition}/{key}")
        packed[condition] = {"ids": reference["ids"], "seeds": np.asarray(SEEDS),
                             "logits": np.stack([row["logits"] for row in rows]),
                             "intensity": np.stack([row["intensity"] for row in rows]),
                             "true_class": reference["true_class"], "true_intensity": reference["true_intensity"]}
        sources.extend(record(path) for path in paths)
    return packed, sources


def write_csv(path, rows):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    import torch
    aggregate = load("balanced_v10_aggregate", "2026-09-25_aggregate-q2-balanced-ensemble_v1.0.py")
    loader = load("balanced_v11_loader", "2026-09-24_q2_data_v1.0.py")
    cache = load("balanced_v11_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = load("balanced_v11_normalizer", "2026-09-24_q2-normalization_v1.0.py")
    official = loader.load_official()
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[3], 3)
    if len(official["valid"]) != 728 or len(official["test"]) != 727 or len(special) != 30:
        raise RuntimeError("Sample counts changed")
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {kind: cache.verify_cache(official[kind], model_info)
                  for kind in ("train", "valid", "test")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    training = read_json(TRAINED / "provenance.json")
    experiment = read_json(TRAINED / "balanced_experiment.json")
    final_summary = read_json(FINAL / "summary.json")
    old_provenance = read_json(PREVIOUS / "provenance.json")
    old_summary = read_json(PREVIOUS / "summary.json")
    if (training["selection_split"] != "valid"
            or training["normalization_sha256"] != sha(STATS)
            or training["aligned_source_sha256"] != dict(official["test"].source_sha256)
            or training["bert_cache_sha256"] != {
                k: cache_meta[k]["feature_sha256"] for k in ("train", "valid")}
            or scaler.metadata["source_metadata"]["source_sha256"] != dict(official["train"].source_sha256)
            or scaler.metadata["source_metadata"]["text_feature_sha256"] != cache_meta["train"]["feature_sha256"]
            or experiment["weight_mode"] != "sqrt"
            or experiment["weighted_scope"] != "optimizer classification CE only"
            or final_summary["normalization_sha256"] != sha(STATS)
            or final_summary["source_sha256"]["attachment3"] != dict(special.source_sha256)
            or old_provenance["seeds"] != list(SEEDS)
            or old_provenance["weights"] != {str(seed): 1.0 / 3 for seed in SEEDS}
            or old_provenance["official_test_rule"] != "arithmetic mean of frozen seed logits and intensity"):
        raise RuntimeError("Frozen sources or existing aggregation rule differ")
    sources = [record(STATS), record(TRAINED / "provenance.json"),
               record(TRAINED / "balanced_experiment.json"), record(FINAL / "summary.json"),
               record(PREVIOUS / "summary.json"), record(PREVIOUS / "provenance.json")]
    for seed in SEEDS:
        valid_path = TRAINED / "temporal" / f"seed_{seed}" / "valid_complete_predictions.npz"
        test_path = FINAL / f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz"
        saved_sources = old_provenance["source_files"][str(seed)]
        if sha(valid_path) != saved_sources["valid_prediction"] or sha(test_path) != saved_sources["test_prediction"]:
            raise RuntimeError(f"Frozen source prediction hash differs: {seed}")
        sources.extend((record(valid_path), record(test_path)))
    for name, expected in final_summary["output_sha256"].items():
        if sha(FINAL / name) != expected:
            raise RuntimeError(f"Original final output hash differs: {name}")
    for name, expected in old_summary["output_sha256"].items():
        if sha(PREVIOUS / name) != expected:
            raise RuntimeError(f"Original ensemble output hash differs: {name}")
    jobs = load_frozen_models(torch)
    official_packed = {kind: official_arrays(kind, aggregate, official[kind])
                       for kind in ("valid", "test")}
    valid_conditions, condition_sources = frozen_validation_conditions(aggregate, official_packed["valid"])
    sources.extend(condition_sources)
    old_test = aggregate.read_npz(PREVIOUS / "balanced_sqrt_ensemble_test_predictions.npz")
    expected_test = ensemble_prediction(official_packed["test"]["logits"],
                                        official_packed["test"]["intensity"])
    if not np.array_equal(expected_test[0], old_test["logits"]) or not np.array_equal(expected_test[1], old_test["intensity"]):
        raise RuntimeError("Fixed official aggregation does not reproduce v1.0 bit for bit")
    sources.extend(record(PREVIOUS / name) for name in ("balanced_sqrt_ensemble_test_predictions.npz", "attachment3_ensemble_predictions.csv"))
    print("Preflight OK: 728 valid + 727 test IDs, 30 attachment3 files, 3 frozen checkpoints")
    print("Original official-test logits/intensity reproduced bit for bit; no rule selection")
    print("Existing 18 validation missing conditions x 3 seeds: IDs/labels verified")
    if args.check:
        print("No inference, training, or output written")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    partial = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or partial.exists():
        raise FileExistsError(f"Refusing to overwrite {OUT} or its partial output")
    partial.mkdir(parents=True)
    protocol = {"schema": "q2_balanced_ensemble_correction_v1.1", "date": "2026-09-25",
                "correction_reason": "v1.0 used different classification aggregation for official and attachment3 inputs",
                "operator_all_splits": OPERATOR, "seeds": list(SEEDS),
                "weights": [1.0 / 3] * 3, "classification": "argmax(mean(seed_logits)); probabilities=softmax(mean(seed_logits))",
                "regression": "mean(seed_intensity)", "accumulation_dtype": "float32 (matching frozen official v1.0)",
                "feature_version": "aligned_50", "feature_dimensions": [768, 74, 35],
                "checkpoint_selection_split": "valid", "official_test_already_viewed": True,
                "one_shot_holdout_claim": False, "retraining": False, "weight_or_seed_tuning": False,
                "correction_uses_test_metric_comparison_to_choose_rule": False,
                "rule_basis": "preserve the existing official mean-logits rule and apply it consistently to attachment3",
                "attachment3_labels_available": False, "attachment3_accuracy_reported": False,
                "mask_contract": "content_mask excludes padding/special tokens; source_observed is loader observation; source_unavailable=content & ~source_observed; artificial_missing is zero for this inference; effective=content & source_observed & ~artificial_missing",
                "source_unavailable_caveat": "A zero-valued source observation does not prove the cause of unavailability.",
                "validation_robustness_scope": "18 existing saved post-BERT feature-masking conditions; no new masks or condition selection; not equivalent to missing raw text",
                "code": record(Path(__file__)), "sources": sources,
                "checkpoints": [{k: job[k] for k in ("seed", "checkpoint", "best_epoch", "config")} for job in jobs],
                "normalization": record(STATS), "bert_model": model_info,
                "source_sha256": {"official": dict(official["test"].source_sha256),
                                  "attachment3": dict(special.source_sha256)}}
    # Materialize the immutable correction rule before any new specialist inference.
    write_json(partial / "protocol.json", protocol)
    summaries = {}
    for kind, packed in official_packed.items():
        logits, intensity, probability = ensemble_prediction(packed["logits"], packed["intensity"])
        np.savez_compressed(partial / f"{kind}_seed_predictions.npz", **packed)
        np.savez_compressed(partial / f"balanced_sqrt_ensemble_{kind}_predictions.npz",
                            ids=packed["ids"], logits=logits, intensity=intensity,
                            true_class=packed["true_class"], true_intensity=packed["true_intensity"])
        summaries[kind] = {"classification": aggregate.classification_metrics(packed["true_class"], logits.argmax(axis=1)),
                           "regression": aggregate.regression_metrics(packed["true_intensity"], intensity)}
    robustness, robustness_rows = {}, []
    condition_dir = partial / "valid_missing"
    condition_dir.mkdir()
    for condition, packed in valid_conditions.items():
        logits, intensity, probability = ensemble_prediction(packed["logits"], packed["intensity"])
        np.savez_compressed(condition_dir / f"{condition}_seed_predictions.npz", **packed)
        np.savez_compressed(condition_dir / f"{condition}_ensemble_predictions.npz",
                            ids=packed["ids"], logits=logits, intensity=intensity,
                            true_class=packed["true_class"], true_intensity=packed["true_intensity"])
        cm = aggregate.classification_metrics(packed["true_class"], logits.argmax(axis=1))
        rm = aggregate.regression_metrics(packed["true_intensity"], intensity)
        robustness[condition] = {"classification": cm, "regression": rm}
        robustness_rows.append({"condition": condition, "n": len(packed["ids"]),
                                "accuracy": cm["accuracy"], "macro_f1": cm["macro_f1"],
                                "neutral_f1": cm["neutral_f1"], "mae": rm["mae"], "pearson_r": rm["pearson_r"],
                                "delta_macro_f1_from_complete": cm["macro_f1"] - summaries["valid"]["classification"]["macro_f1"],
                                "delta_mae_from_complete": rm["mae"] - summaries["valid"]["regression"]["mae"]})
    write_csv(partial / "valid_missing_ensemble_metrics.csv", robustness_rows)
    text_features = frozen_text_features(special, cache, torch, args.device)
    tensors, observed = prepare_special_tensors(special, text_features, scaler, torch, args.device)
    seed_logits, seed_intensity = [], []
    file_ids = np.asarray([path.stem for path in special.source_paths])
    for job in jobs:
        job["model"].to(args.device)
        with torch.inference_mode():
            logits, intensity = job["model"](*tensors)
        seed_logits.append(logits.cpu().numpy())
        seed_intensity.append(intensity.cpu().numpy())
        job["model"].cpu()
        print(f"Attachment3 frozen seed={job['seed']}: saved raw logits, n={len(special)}", flush=True)
    seed_logits, seed_intensity = np.stack(seed_logits), np.stack(seed_intensity)
    if not np.isfinite(seed_logits).all() or not np.isfinite(seed_intensity).all():
        raise RuntimeError("Nonfinite specialist predictions")
    # Ensure the recomputation uses the same frozen per-seed predictors as v1.0.
    with (FINAL / "attachment3_all_models.csv").open(encoding="utf-8-sig", newline="") as handle:
        previous_rows = {(int(row["seed"]), row["file_id"]): row for row in csv.DictReader(handle)}
    max_previous_probability_diff = 0.0
    max_previous_intensity_diff = 0.0
    for s, seed in enumerate(SEEDS):
        probabilities = softmax(seed_logits[s])
        for i, file_id in enumerate(file_ids):
            previous = previous_rows[(seed, file_id)]
            old_probs = [float(previous[f"{name.lower()}_probability"]) for name in CLASSES]
            delta_p = float(np.max(np.abs(probabilities[i] - old_probs)))
            delta_y = abs(float(seed_intensity[s, i]) - float(previous["intensity"]))
            max_previous_probability_diff = max(max_previous_probability_diff, delta_p)
            max_previous_intensity_diff = max(max_previous_intensity_diff, delta_y)
            if delta_p > 5e-6 or delta_y > 5e-6:
                raise RuntimeError(f"Frozen per-seed specialist inference changed: {seed}/{file_id}")
    np.savez_compressed(partial / "attachment3_seed_predictions.npz", ids=np.asarray(special.ids),
                        file_ids=file_ids, seeds=np.asarray(SEEDS), logits=seed_logits,
                        intensity=seed_intensity)
    logits, intensity, probability = ensemble_prediction(seed_logits, seed_intensity)
    np.savez_compressed(partial / "attachment3_ensemble_predictions.npz", ids=np.asarray(special.ids),
                        file_ids=file_ids, logits=logits, intensity=intensity, probabilities=probability)
    rows = [{"file_id": str(file_id), "model": "temporal_balanced_sqrt_ensemble_v1.1",
             "ensemble_rule": OPERATOR, "polarity": CLASSES[int(probability[i].argmax())],
             "intensity": float(intensity[i]), **{f"{name.lower()}_probability": float(probability[i, c])
                                                  for c, name in enumerate(CLASSES)}}
            for i, file_id in enumerate(file_ids)]
    write_csv(partial / "attachment3_ensemble_predictions.csv", rows)
    masks = {"ids": np.asarray(special.ids), "file_ids": file_ids, "content_mask": special.content_mask,
             **{f"source_observed_{m}": observed[m] for m in MODALITIES},
             **{f"source_unavailable_{m}": special.content_mask & ~observed[m] for m in MODALITIES},
             **{f"artificial_missing_{m}": np.zeros_like(special.content_mask) for m in MODALITIES},
             **{f"effective_{m}": observed[m] for m in MODALITIES}}
    np.savez_compressed(partial / "attachment3_masks.npz", **masks)
    audit_rows = []
    for i, file_id in enumerate(file_ids):
        audit_rows.append({"file_id": str(file_id), "content_positions": int(special.content_mask[i].sum()),
                           **{f"{m}_observed_positions": int(observed[m][i].sum()) for m in MODALITIES},
                           **{f"{m}_zero_content_positions": int((special.content_mask[i] & ~observed[m][i]).sum()) for m in MODALITIES}})
    write_csv(partial / "attachment3_observation_audit.csv", audit_rows)
    with (PREVIOUS / "attachment3_ensemble_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        old_special = {row["file_id"]: row for row in csv.DictReader(handle)}
    changed_classes = [row["file_id"] for row in rows if row["polarity"] != old_special[row["file_id"]]["polarity"]]
    summary = {"schema": "q2_balanced_ensemble_final_v1.1", "operator_all_splits": OPERATOR,
               "selection_split": "valid", "official_test_already_viewed": True, "one_shot_holdout_claim": False,
               "retraining": False, "weight_or_seed_tuning": False, "official_valid_n": 728,
               "official_test_n": 727, "specialist_n": 30, "specialist_labels_used": False,
               "validation": summaries["valid"], "official_test": summaries["test"],
               "validation_missing": robustness,
               "validation_missing_scope": "existing post-BERT feature masking; text conditions do not represent raw-text missingness",
               "official_test_arrays_bitwise_identical_to_v1_0": True,
               "attachment3_max_per_seed_probability_difference_from_v1_0": max_previous_probability_diff,
               "attachment3_max_per_seed_intensity_difference_from_v1_0": max_previous_intensity_diff,
               "attachment3_classes_changed_by_operator_correction": changed_classes,
               "output_sha256": {path.relative_to(partial).as_posix(): sha(path)
                                 for path in partial.rglob("*") if path.is_file()}}
    write_json(partial / "summary.json", summary)
    partial.rename(OUT)
    print(f"Corrected fixed ensemble saved: {OUT}")
    print(f"Attachment3 class changes from aggregation correction only: {len(changed_classes)} / 30")
    print("Official-test arrays unchanged bit for bit; no fitting or selection performed")


if __name__ == "__main__":
    main()
