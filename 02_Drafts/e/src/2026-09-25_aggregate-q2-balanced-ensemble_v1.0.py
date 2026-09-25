"""Aggregate the frozen balanced-sqrt Q2 seeds without test-driven tuning.

The equal-weight rule is fixed from the validation protocol before reading
official-test labels.  Official-test predictions are aggregated by arithmetic
mean of logits and regression outputs.  Attachment-3 saved probabilities and
intensities are aggregated by arithmetic mean because that output contains no
logits or labels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
TRAINED = Q2 / "q2-temporal-balanced-sqrt-v1.0"
FINAL = Q2 / "q2-temporal-balanced-sqrt-final-v1.0"
OUT = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.0"
SEEDS = (20260924, 20260925, 20260926)
CLASSES = ("Negative", "Neutral", "Positive")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def classification_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("classification shape mismatch")
    matrix = np.zeros((3, 3), dtype=np.int64)
    for actual, predicted in zip(target, prediction):
        if not (0 <= actual < 3 and 0 <= predicted < 3):
            raise ValueError("class outside 0..2")
        matrix[actual, predicted] += 1
    per_class = []
    f1 = []
    for cls in range(3):
        tp = matrix[cls, cls]
        precision = tp / matrix[:, cls].sum() if matrix[:, cls].sum() else 0.0
        recall = tp / matrix[cls, :].sum() if matrix[cls, :].sum() else 0.0
        score = (2 * precision * recall / (precision + recall)
                 if precision + recall else 0.0)
        f1.append(score)
        per_class.append({"id": cls, "name": CLASSES[cls],
                          "support": int(matrix[cls, :].sum()),
                          "precision": float(precision),
                          "recall": float(recall), "f1": float(score)})
    return {"n": int(target.size), "accuracy": float(np.mean(target == prediction)),
            "macro_f1": float(np.mean(f1)),
            "neutral_f1": float(f1[1]), "neutral_recall": float(per_class[1]["recall"]),
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
            "per_class": per_class}


def regression_metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    target = np.asarray(target, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    if target.shape != prediction.shape or target.ndim != 1:
        raise ValueError("regression shape mismatch")
    if not np.isfinite(target).all() or not np.isfinite(prediction).all():
        raise ValueError("nonfinite regression values")
    if target.size < 2 or np.std(target) == 0 or np.std(prediction) == 0:
        pearson = None
        status = "undefined_constant"
    else:
        pearson = float(np.corrcoef(target, prediction)[0, 1])
        status = "defined"
    return {"n": int(target.size), "mae": float(np.mean(np.abs(target - prediction))),
            "pearson_r": pearson, "pearson_status": status}


def read_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as loaded:
        required = {"ids", "logits", "intensity", "true_class", "true_intensity"}
        if set(loaded.files) != required:
            raise ValueError(f"Unexpected keys in {path.name}: {loaded.files}")
        result = {key: loaded[key].copy() for key in loaded.files}
    if result["ids"].ndim != 1 or result["logits"].shape != (result["ids"].size, 3):
        raise ValueError(f"Invalid prediction shape: {path}")
    if result["intensity"].shape != (result["ids"].size,):
        raise ValueError(f"Invalid intensity shape: {path}")
    for key in ("logits", "intensity", "true_intensity"):
        if not np.isfinite(result[key]).all():
            raise ValueError(f"Nonfinite values in {path}: {key}")
    return result


def load_split(kind: str, base: Path) -> list[dict[str, np.ndarray]]:
    if kind == "valid":
        paths = [TRAINED / "temporal" / f"seed_{seed}" / "valid_complete_predictions.npz"
                 for seed in SEEDS]
    elif kind == "test":
        paths = [FINAL / f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz"
                 for seed in SEEDS]
    else:
        raise ValueError(kind)
    rows = [read_npz(path) for path in paths]
    reference = rows[0]
    for index, row in enumerate(rows[1:], start=1):
        if not np.array_equal(row["ids"], reference["ids"]):
            raise ValueError(f"{kind}: IDs differ at seed index {index}")
        for key in ("true_class", "true_intensity"):
            if not np.array_equal(row[key], reference[key]):
                raise ValueError(f"{kind}: labels differ at seed index {index}")
    ids = reference["ids"].astype(str)
    if len(np.unique(ids)) != ids.size:
        raise ValueError(f"{kind}: duplicate IDs")
    return rows


def summarize_npz(rows: list[dict[str, np.ndarray]]) -> dict:
    logits = np.mean([row["logits"] for row in rows], axis=0)
    intensity = np.mean([row["intensity"] for row in rows], axis=0)
    target_class = rows[0]["true_class"].astype(np.int64)
    target_intensity = rows[0]["true_intensity"].astype(np.float64)
    return {"classification": classification_metrics(target_class, logits.argmax(axis=1)),
            "regression": regression_metrics(target_intensity, intensity),
            "logits": logits, "intensity": intensity}


def read_special() -> dict[str, list[dict[str, str]]]:
    path = FINAL / "attachment3_all_models.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    grouped: dict[str, list[dict[str, str]]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") != "temporal_balanced_sqrt":
                raise ValueError("Unexpected attachment-3 model")
            grouped.setdefault(row["file_id"], []).append(row)
    if len(grouped) != 30:
        raise ValueError(f"Expected 30 attachment-3 files, got {len(grouped)}")
    for file_id, rows in grouped.items():
        if sorted(int(row["seed"]) for row in rows) != sorted(SEEDS):
            raise ValueError(f"{file_id}: expected exactly three frozen seeds")
    return grouped


def aggregate_special(grouped: dict[str, list[dict[str, str]]]) -> list[dict[str, object]]:
    output = []
    for file_id in sorted(grouped):
        rows = grouped[file_id]
        probabilities = np.asarray([
            [float(row["negative_probability"]), float(row["neutral_probability"]),
             float(row["positive_probability"])] for row in rows], dtype=np.float64)
        intensities = np.asarray([float(row["intensity"]) for row in rows], dtype=np.float64)
        if not np.isfinite(probabilities).all() or not np.isfinite(intensities).all():
            raise ValueError(f"Nonfinite attachment-3 values: {file_id}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Probabilities do not sum to one: {file_id}")
        mean_probability = probabilities.mean(axis=0)
        predicted = int(mean_probability.argmax())
        output.append({"file_id": file_id, "model": "temporal_balanced_sqrt_ensemble",
                       "ensemble_rule": "equal_mean_of_three_frozen_seed_probabilities",
                       "polarity": CLASSES[predicted], "intensity": float(intensities.mean()),
                       "negative_probability": float(mean_probability[0]),
                       "neutral_probability": float(mean_probability[1]),
                       "positive_probability": float(mean_probability[2])})
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()

    valid_rows = load_split("valid", TRAINED)
    test_rows = load_split("test", FINAL)
    special = read_special()
    valid = summarize_npz(valid_rows)
    test = summarize_npz(test_rows)
    print(f"Preflight OK: valid={len(valid_rows[0]['ids'])}, test={len(test_rows[0]['ids'])}, attachment3={len(special)}")
    print(f"Validation equal-weight Macro-F1: {valid['classification']['macro_f1']:.6f}")
    print(f"Validation equal-weight Neutral-F1: {valid['classification']['neutral_f1']:.6f}")
    if args.check:
        print("No aggregation output written")
        return
    if OUT.exists() or OUT.with_name(OUT.name + ".partial").exists():
        raise FileExistsError(f"Refusing to overwrite: {OUT}")
    partial = OUT.with_name(OUT.name + ".partial")
    partial.mkdir(parents=True)

    test_npz = partial / "balanced_sqrt_ensemble_test_predictions.npz"
    np.savez_compressed(test_npz, ids=test_rows[0]["ids"], logits=test["logits"],
                        intensity=test["intensity"], true_class=test_rows[0]["true_class"],
                        true_intensity=test_rows[0]["true_intensity"])
    special_rows = aggregate_special(special)
    special_csv = partial / "attachment3_ensemble_predictions.csv"
    columns = ("file_id", "model", "ensemble_rule", "polarity", "intensity",
               "negative_probability", "neutral_probability", "positive_probability")
    with special_csv.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(special_rows)
    audit_source = FINAL / "attachment3_observation_audit.csv"
    shutil.copy2(audit_source, partial / audit_source.name)

    provenance = {"schema": "q2_temporal_balanced_sqrt_equal_ensemble_v1.0",
                  "feature_version": "aligned_50", "selection_split": "valid",
                  "test_tuning": False, "specialist_labels_used": False,
                  "seeds": list(SEEDS), "weights": {str(seed): 1.0 / len(SEEDS) for seed in SEEDS},
                  "official_test_rule": "arithmetic mean of frozen seed logits and intensity",
                  "attachment3_rule": "arithmetic mean of frozen seed probabilities and intensity",
                  "selection_rule": "equal weights fixed before reading official-test labels",
                  "source_files": {str(seed): {
                      "valid_prediction": sha256(TRAINED / "temporal" / f"seed_{seed}" / "valid_complete_predictions.npz"),
                      "test_prediction": sha256(FINAL / f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz")}
                      for seed in SEEDS},
                  "attachment3_source": sha256(FINAL / "attachment3_all_models.csv")}
    (partial / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    summary = {"schema": "q2_temporal_balanced_sqrt_equal_ensemble_final_v1.0",
               "selection_split": "valid", "official_test_n": len(test_rows[0]["ids"]),
               "specialist_n": len(special), "test_tuning": False,
               "ensemble": provenance, "validation": {k: v for k, v in valid.items() if k not in ("logits", "intensity")},
               "official_test": {k: v for k, v in test.items() if k not in ("logits", "intensity")},
               "output_sha256": {p.name: sha256(p) for p in partial.iterdir()
                                 if p.is_file() and p.name != "summary.json"}}
    (partial / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    partial.rename(OUT)
    print(f"Ensemble output saved: {OUT}")
    print(f"Official-test ensemble Macro-F1: {test['classification']['macro_f1']:.6f}")


if __name__ == "__main__":
    main()
