"""Freeze and evaluate the validation-adopted text-safe model exactly once."""

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
TRAINED = Q2 / "q2-text-safe-balanced-sqrt-v1.0"
BASELINE = Q2 / "q2-text-safe-eval-balanced-sqrt-v1.0" / "report.json"
PROTOCOL = Q2 / "q2-text-safe-final-protocol-v1.0.json"
OUT = Q2 / "q2-text-safe-final-v1.0"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CLASSES = ("Negative", "Neutral", "Positive")
CONDITIONS = ("20pct_start", "20pct_middle", "20pct_end",
              "40pct_start", "40pct_middle", "40pct_end")


def load(filename):
    path = SRC / filename
    spec = importlib.util.spec_from_file_location("text_safe_final_" + path.stem.replace("-", "_"), path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def protocol_data():
    candidate = read_json(TRAINED / "summary.json")
    reference = read_json(BASELINE)
    if len(candidate) != 3 or reference["split"] != "official_valid":
        raise RuntimeError("Validation adoption evidence incomplete")
    new_gaps = [row["valid_input_text_gap"][condition]["classification"]["macro_f1"]
                for row in candidate for condition in CONDITIONS]
    old_gaps = [row["macro_f1"] for row in reference["rows"]
                if row["encoding"] == "input_level" and row["condition"] in CONDITIONS]
    new_complete = [row["valid_complete"]["classification"]["macro_f1"] for row in candidate]
    old_complete = [row["macro_f1"] for row in reference["rows"] if row["condition"] == "complete"]
    if len(new_gaps) != 18 or len(old_gaps) != 18 or len(new_complete) != 3 or len(old_complete) != 3:
        raise RuntimeError("Validation adoption rows missing")
    gap_gain = float(np.mean(new_gaps) - np.mean(old_gaps))
    complete_change = float(np.mean(new_complete) - np.mean(old_complete))
    if gap_gain < 0.02 or complete_change < -0.02:
        raise RuntimeError("Predeclared validation adoption rule not met")
    checkpoints = {}
    for seed, row in zip(SEEDS, candidate):
        if row["seed"] != seed or row["test_evaluated"] is not False:
            raise RuntimeError("Candidate seed/provenance mismatch")
        path = TRAINED / f"seed_{seed}" / "best.pt"
        digest = sha(path)
        if digest != row["checkpoint_sha256"]:
            raise RuntimeError(f"Candidate checkpoint checksum mismatch: {seed}")
        checkpoints[str(seed)] = digest
    return {"schema": "q2_text_safe_final_protocol_v1.0",
            "model": "input_level_text_gap_augmented_temporal_balanced_sqrt",
            "feature_version": "official_aligned_50", "selection_split": "official_valid",
            "official_test_n": 727, "attachment3_n": 30,
            "ensemble_classification": "mean of three seed softmax probability vectors, then argmax",
            "ensemble_regression": "arithmetic mean of three seed intensity predictions",
            "seeds": list(SEEDS), "checkpoint_sha256": checkpoints,
            "normalization_sha256": sha(STATS),
            "validation_adoption": {"rule": "six-condition mean input-level gap Macro-F1 gain >=0.02 and complete Macro-F1 change >=-0.02 versus balanced-sqrt-only, same seeds",
                                    "gap_macro_f1_gain": gap_gain,
                                    "complete_macro_f1_change": complete_change},
            "test_tuning": False, "specialist_labels_used": False}


def evaluate(args):
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    frozen = read_json(PROTOCOL)
    if frozen != protocol_data():
        raise RuntimeError("Frozen protocol differs from current validation evidence/checkpoints")
    if OUT.exists() or OUT.with_name(OUT.name + ".partial").exists():
        raise FileExistsError("Final output exists; refusing another test evaluation")
    temporal = load("2026-09-24_train-q2-temporal_v1.0.py")
    loader = load("2026-09-24_q2_data_v1.0.py")
    cache = load("2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = load("2026-09-24_q2-normalization_v1.0.py")
    robust = load("2026-09-24_train-q2-robust_v1.0.py")
    metrics = load("2026-09-24_q2-evaluation_v1.0.py")
    official = loader.load_official()
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[3], 3)
    test = official["test"]
    if len(test) != 727 or len(special) != 30:
        raise RuntimeError("Test or attachment-3 sample count changed")
    info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {split: cache.verify_cache(official[split], info)
                  for split in ("train", "valid", "test")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    if scaler.metadata["source_metadata"]["text_feature_sha256"] != cache_meta["train"]["feature_sha256"]:
        raise RuntimeError("Normalization was fitted on other text")
    models = []
    for seed in SEEDS:
        saved = torch.load(TRAINED / f"seed_{seed}" / "best.pt", map_location="cpu", weights_only=True)
        if (saved["seed"] != seed or saved["model"] != "text_safe_balanced_sqrt"
                or saved["normalization_sha256"] != sha(STATS)):
            raise RuntimeError(f"Checkpoint metadata mismatch: {seed}")
        config = temporal.TemporalConfig(**saved["config"])
        model = temporal.build_model(saved["train_priors"], config, torch).to(args.device).eval()
        model.load_state_dict(saved["model_state"], strict=True)
        models.append((seed, model, config))
    print("Frozen preflight OK: 3 checkpoints, 727 test, 30 attachment-3; no selection by test", flush=True)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import BertModel
    encoder = BertModel.from_pretrained(str(cache.MODEL), local_files_only=True,
                                        attn_implementation="eager").to(args.device).float().eval()
    encoder.requires_grad_(False)
    special_text = np.empty((len(special), 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(special), 10):
            end = min(start + 10, len(special))
            inputs = {key: torch.as_tensor(np.asarray(getattr(special, key)[start:end]).copy(),
                                           dtype=torch.long, device=args.device)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            special_text[start:end] = encoder(**inputs).last_hidden_state.cpu().numpy()
    del encoder
    raw = {"test": {"text": np.load(cache.OUT / "test.npy", mmap_mode="r", allow_pickle=False),
                    "audio": test.audio, "vision": test.vision},
           "attachment3": {"text": special_text, "audio": special.audio, "vision": special.vision}}
    tensors, observed = {}, {}
    for name, batch in (("test", test), ("attachment3", special)):
        arrays, masks = robust.normalize_once(batch, raw[name], scaler)
        observed[name] = masks
        tensors[name] = ({m: torch.as_tensor(arrays[m].copy(), device=args.device) for m in MODALITIES},
                         torch.as_tensor(batch.content_mask.copy(), dtype=torch.bool, device=args.device),
                         {m: torch.as_tensor(masks[m].copy(), dtype=torch.bool, device=args.device) for m in MODALITIES})
    partial = OUT.with_name(OUT.name + ".partial")
    partial.mkdir(parents=True)
    per_seed, probabilities, intensities, specialist = [], [], [], []
    for seed, model, config in models:
        tf, tc, tm = tensors["test"]
        test_tensors = (tf, tc, tm,
                        torch.as_tensor(test.classification_labels.copy(), dtype=torch.long, device=args.device),
                        torch.as_tensor(test.regression_labels.copy(), dtype=torch.float32, device=args.device))
        report, pred = temporal.evaluate(model, test_tensors, config, torch, metrics, predictions=True)
        path = partial / f"seed_{seed}_test_predictions.npz"
        np.savez_compressed(path, ids=np.asarray(test.ids), **pred)
        per_seed.append({"seed": seed, "prediction_sha256": sha(path), "report": report})
        exp = np.exp(pred["logits"] - pred["logits"].max(axis=1, keepdims=True))
        probabilities.append(exp / exp.sum(axis=1, keepdims=True))
        intensities.append(pred["intensity"])
        with torch.inference_mode():
            sf, sc, sm = tensors["attachment3"]
            logits, strength = model(sf, sc, sm)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            strength = strength.cpu().numpy()
        for i, path in enumerate(special.source_paths):
            specialist.append({"file_id": path.stem, "seed": seed,
                               "polarity": CLASSES[int(probs[i].argmax())],
                               "intensity": float(strength[i]),
                               **{f"{name}_probability": float(probs[i, j])
                                  for j, name in enumerate(("negative", "neutral", "positive"))}})
        print(f"seed={seed}: test and attachment 3 saved", flush=True)
    ensemble_prob = np.mean(probabilities, axis=0)
    ensemble_intensity = np.mean(intensities, axis=0)
    ensemble_class = ensemble_prob.argmax(axis=1)
    np.savez_compressed(partial / "ensemble_test_predictions.npz", ids=np.asarray(test.ids),
                        probabilities=ensemble_prob, intensity=ensemble_intensity,
                        true_class=test.classification_labels, true_intensity=test.regression_labels)
    ensemble_report = {"classification": metrics.classification_metrics(test.classification_labels, ensemble_class),
                       "regression": metrics.regression_metrics(test.regression_labels, ensemble_intensity)}
    ensemble_special = []
    for file_id in sorted({row["file_id"] for row in specialist}):
        rows = [row for row in specialist if row["file_id"] == file_id]
        if len(rows) != 3:
            raise RuntimeError("Attachment-3 ensemble missing seed")
        prob = np.mean([[row[f"{name}_probability"] for name in ("negative", "neutral", "positive")]
                        for row in rows], axis=0)
        ensemble_special.append({"file_id": file_id, "seed": "ensemble",
                                 "polarity": CLASSES[int(prob.argmax())],
                                 "intensity": float(np.mean([row["intensity"] for row in rows])),
                                 **{f"{name}_probability": float(prob[j])
                                    for j, name in enumerate(("negative", "neutral", "positive"))}})
    write_csv(partial / "attachment3_all_models.csv", specialist + ensemble_special)
    write_csv(partial / "attachment3_ensemble_predictions.csv", ensemble_special)
    audit = []
    for i, path in enumerate(special.source_paths):
        row = {"file_id": path.stem, "content_positions": int(special.content_mask[i].sum())}
        for modality in MODALITIES:
            mask = observed["attachment3"][modality][i]
            row[f"{modality}_observed_positions"] = int(mask.sum())
            row[f"{modality}_zero_content_positions"] = int((special.content_mask[i] & ~mask).sum())
        audit.append(row)
    write_csv(partial / "attachment3_observation_audit.csv", audit)
    summary = {"schema": "q2_text_safe_final_v1.0", "protocol_sha256": sha(PROTOCOL),
               "test_tuning": False, "specialist_labels_used": False,
               "official_test_n": len(test), "attachment3_n": len(special),
               "input": "complete original frozen BERT test cache; no synthetic test gaps",
               "source_sha256": {"official": dict(test.source_sha256),
                                 "attachment3": dict(special.source_sha256)},
               "normalization_sha256": sha(STATS), "test_cache_sha256": cache_meta["test"]["feature_sha256"],
               "per_seed": per_seed, "ensemble_report": ensemble_report,
               "output_sha256": {p.name: sha(p) for p in partial.iterdir() if p.is_file()}}
    (partial / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                          encoding="utf-8")
    partial.rename(OUT)
    print(f"One-time frozen evaluation saved: {OUT}")


def audit():
    metrics = load("2026-09-24_q2-evaluation_v1.0.py")
    summary = read_json(OUT / "summary.json")
    if summary["protocol_sha256"] != sha(PROTOCOL) or len(summary["per_seed"]) != 3:
        raise RuntimeError("Final provenance mismatch")
    seed_probabilities, seed_intensities = [], []
    reference_ids = reference_class = reference_intensity = None
    for row in summary["per_seed"]:
        path = OUT / f"seed_{row['seed']}_test_predictions.npz"
        if sha(path) != row["prediction_sha256"]:
            raise RuntimeError("Per-seed prediction checksum mismatch")
        with np.load(path, allow_pickle=False) as z:
            if len(z["ids"]) != 727 or len(set(z["ids"].astype(str))) != 727:
                raise RuntimeError("Per-seed IDs invalid")
            cls = metrics.classification_metrics(z["true_class"], z["logits"].argmax(axis=1))
            reg = metrics.regression_metrics(z["true_intensity"], z["intensity"])
            ids = z["ids"].astype(str).copy()
            true_class = z["true_class"].copy()
            true_intensity = z["true_intensity"].copy()
            logits = z["logits"].copy()
            exp = np.exp(logits - logits.max(axis=1, keepdims=True))
            seed_probabilities.append(exp / exp.sum(axis=1, keepdims=True))
            seed_intensities.append(z["intensity"].copy())
        if reference_ids is None:
            reference_ids, reference_class, reference_intensity = ids, true_class, true_intensity
        elif (not np.array_equal(ids, reference_ids)
              or not np.array_equal(true_class, reference_class)
              or not np.array_equal(true_intensity, reference_intensity)):
            raise RuntimeError("Seed predictions have different ID or label order")
        if cls != row["report"]["classification"] or reg != row["report"]["regression"]:
            raise RuntimeError("Per-seed metrics differ")
    with np.load(OUT / "ensemble_test_predictions.npz", allow_pickle=False) as z:
        if len(z["ids"]) != 727 or len(set(z["ids"].astype(str))) != 727:
            raise RuntimeError("Ensemble IDs invalid")
        if (not np.array_equal(z["ids"].astype(str), reference_ids)
                or not np.array_equal(z["true_class"], reference_class)
                or not np.array_equal(z["true_intensity"], reference_intensity)):
            raise RuntimeError("Ensemble ID or label order differs")
        np.testing.assert_array_equal(z["probabilities"], np.mean(seed_probabilities, axis=0))
        np.testing.assert_array_equal(z["intensity"], np.mean(seed_intensities, axis=0))
        cls = metrics.classification_metrics(z["true_class"], z["probabilities"].argmax(axis=1))
        reg = metrics.regression_metrics(z["true_intensity"], z["intensity"])
    if cls != summary["ensemble_report"]["classification"] or reg != summary["ensemble_report"]["regression"]:
        raise RuntimeError("Ensemble metrics differ")
    for filename, digest in summary["output_sha256"].items():
        if sha(OUT / filename) != digest:
            raise RuntimeError(f"Output checksum mismatch: {filename}")
    with (OUT / "attachment3_ensemble_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        ensemble_rows = list(csv.DictReader(handle))
        if len(ensemble_rows) != 30:
            raise RuntimeError("Attachment-3 ensemble row count invalid")
    with (OUT / "attachment3_all_models.csv").open(encoding="utf-8-sig", newline="") as handle:
        specialist_rows = list(csv.DictReader(handle))
    for ensemble_row in ensemble_rows:
        matches = [row for row in specialist_rows if row["file_id"] == ensemble_row["file_id"]
                   and row["seed"] != "ensemble"]
        if len(matches) != 3:
            raise RuntimeError("Attachment-3 ensemble lacks three matching seeds")
        for field in ("intensity", "negative_probability", "neutral_probability", "positive_probability"):
            np.testing.assert_allclose(float(ensemble_row[field]),
                                       np.mean([float(row[field]) for row in matches]), rtol=0, atol=0)
    print("Independent audit passed: 3 seeds, ensemble, 727 test IDs, 30 attachment-3 predictions")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--freeze", action="store_true")
    action.add_argument("--run", action="store_true")
    action.add_argument("--audit", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.freeze:
        frozen = protocol_data()
        if PROTOCOL.exists():
            if read_json(PROTOCOL) != frozen:
                raise RuntimeError("Existing freeze protocol differs")
        else:
            PROTOCOL.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
        print(f"Protocol frozen: {PROTOCOL}; validation gap gain={frozen['validation_adoption']['gap_macro_f1_gain']:.4f}")
    elif args.run:
        evaluate(args)
    else:
        audit()


if __name__ == "__main__":
    main()
