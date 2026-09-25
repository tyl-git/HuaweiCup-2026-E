"""Independently audit the fixed equal-weight Q2 balanced-sqrt ensemble."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.0"
SOURCE_FALLBACK = Q2 / "q2-temporal-balanced-sqrt-final-v1.0"
EXPECTED_TEST_N = 727
EXPECTED_SPECIALIST_N = 30
EXPECTED_SEEDS = (20260924, 20260925, 20260926)
CLASSES = ("Negative", "Neutral", "Positive")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def finite_array(data: np.ndarray, name: str) -> None:
    if not np.issubdtype(data.dtype, np.number) or not np.isfinite(data).all():
        raise AssertionError(f"{name} contains nonfinite or nonnumeric values")


def metrics(target_class: np.ndarray, logits: np.ndarray,
            target_intensity: np.ndarray, intensity: np.ndarray) -> dict:
    prediction = logits.argmax(axis=1)
    matrix = np.zeros((3, 3), dtype=np.int64)
    for truth, predicted in zip(target_class, prediction):
        matrix[int(truth), int(predicted)] += 1
    per_class = []
    f1_values = []
    for cls, name in enumerate(CLASSES):
        tp = int(matrix[cls, cls])
        fp = int(matrix[:, cls].sum() - tp)
        fn = int(matrix[cls, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class.append({"id": cls, "name": name,
                          "support": int(matrix[cls, :].sum()),
                          "precision": precision, "recall": recall, "f1": f1})
    mae = float(np.mean(np.abs(intensity - target_intensity)))
    if np.std(intensity) == 0 or np.std(target_intensity) == 0:
        pearson = None
    else:
        pearson = float(np.corrcoef(intensity, target_intensity)[0, 1])
    return {
        "classification": {
            "n": int(len(target_class)),
            "accuracy": float(np.mean(prediction == target_class)),
            "macro_f1": float(np.mean(f1_values)),
            "confusion_matrix_true_rows_predicted_columns": matrix.tolist(),
            "per_class": per_class,
            "zero_division_policy": 0,
        },
        "regression": {"n": int(len(target_intensity)), "mae": mae,
                        "pearson_r": pearson,
                        "pearson_status": "defined" if pearson is not None else "undefined"},
    }


def find_prediction_file(summary: dict) -> Path:
    names = []
    for key in ("prediction_file", "ensemble_prediction_file", "test_prediction_file"):
        value = summary.get(key)
        if isinstance(value, str):
            names.append(value)
    for name in ("ensemble_test_predictions.npz",
                 "temporal_balanced_sqrt_ensemble_test_predictions.npz",
                 "test_predictions.npz"):
        names.append(name)
    candidates = []
    for name in names:
        path = Path(name)
        candidates.extend((path, OUT / path.name))
    candidates.extend(sorted(OUT.glob("*.npz")))
    actual = []
    for path in candidates:
        if not path.is_absolute():
            path = OUT / path
        if path.is_file() and path not in actual:
            actual.append(path)
    if len(actual) != 1:
        raise AssertionError(f"Expected one ensemble NPZ, found {[p.name for p in actual]}")
    return actual[0]


def source_records(summary: dict) -> list[dict]:
    ensemble = summary.get("ensemble")
    if isinstance(ensemble, dict) and isinstance(ensemble.get("source_files"), dict):
        records = []
        for seed, item in sorted(ensemble["source_files"].items(), key=lambda pair: int(pair[0])):
            if not isinstance(item, dict) or not item.get("test_prediction"):
                raise AssertionError(f"Missing source test hash for seed {seed}")
            records.append({
                "seed": int(seed),
                "prediction_file": f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz",
                "prediction_sha256": item["test_prediction"],
            })
        return records
    raw = (summary.get("source_predictions") or summary.get("source_results")
           or summary.get("sources"))
    if raw is None and isinstance(summary.get("source_prediction_sha256"), dict):
        raw = [{"prediction_file": name, "prediction_sha256": value}
               for name, value in summary["source_prediction_sha256"].items()]
    if raw is None and isinstance(summary.get("source_sha256"), dict):
        raw = [{"prediction_file": name, "prediction_sha256": value}
               for name, value in summary["source_sha256"].items()]
    if raw is None:
        ensemble = summary.get("ensemble")
        source_files = ensemble.get("source_files") if isinstance(ensemble, dict) else None
        if isinstance(source_files, dict):
            raw = []
            for seed, item in sorted(source_files.items()):
                if not isinstance(item, dict):
                    raise AssertionError(f"Invalid source file record for seed {seed}")
                raw.append({
                    "seed": int(seed),
                    "prediction_file": item.get("test_prediction_file") or
                    f"temporal_balanced_sqrt_seed_{int(seed)}_test_predictions.npz",
                    "prediction_sha256": item.get("test_prediction_sha256") or
                    item.get("test_prediction"),
                })
            if any(record["prediction_sha256"] is None for record in raw):
                hashes = ensemble.get("source_test_prediction_sha256")
                if isinstance(hashes, dict):
                    for record in raw:
                        record["prediction_sha256"] = hashes.get(str(record["seed"]))
    if not isinstance(raw, list) or len(raw) != len(EXPECTED_SEEDS):
        raise AssertionError("Summary must contain three source prediction records")
    records = []
    for item in raw:
        if isinstance(item, str):
            item = {"prediction_file": item}
        if not isinstance(item, dict):
            raise AssertionError("Invalid source prediction record")
        record = dict(item)
        record["prediction_file"] = (record.get("prediction_file") or record.get("file")
                                      or record.get("path") or record.get("name"))
        record["prediction_sha256"] = (record.get("prediction_sha256")
                                        or record.get("sha256") or record.get("hash"))
        if not record["prediction_file"] or not record["prediction_sha256"]:
            raise AssertionError("Source prediction path and SHA-256 are required")
        records.append(record)
    return records


def resolve_source(path_text: str) -> Path:
    supplied = Path(path_text)
    candidates = [supplied, OUT / supplied.name, SOURCE_FALLBACK / supplied.name,
                  Q2 / supplied.name]
    for path in candidates:
        if path.is_file():
            return path
    raise AssertionError(f"Source prediction file not found: {path_text}")


def report_from_summary(summary: dict):
    for key in ("report", "ensemble_report", "metrics"):
        value = summary.get(key)
        if isinstance(value, dict):
            return value
    return None


def close(actual, expected, name: str, atol: float = 1e-10) -> None:
    if expected is None:
        return
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        if not np.isclose(actual, expected, rtol=0, atol=atol):
            raise AssertionError(f"{name} differs: {actual} != {expected}")


def audit_attachment3(summary: dict) -> dict:
    csv_paths = sorted(OUT.glob("*attachment3*.csv"))
    if not csv_paths:
        raise AssertionError("No attachment3 CSV output found")
    counts = {}
    for path in csv_paths:
        rows = read_csv(path)
        counts[path.name] = len(rows)
        expected = None
        declared = summary.get("attachment3_rows")
        if isinstance(declared, dict):
            expected = declared.get(path.name)
        if expected is None and "predictions" in path.name:
            expected = EXPECTED_SPECIALIST_N
        if expected is not None and len(rows) != int(expected):
            raise AssertionError(f"{path.name}: expected {expected} rows, got {len(rows)}")
        if rows and "file_id" in rows[0]:
            ids = [row["file_id"] for row in rows]
            if "predictions" in path.name and len(set(ids)) != EXPECTED_SPECIALIST_N:
                raise AssertionError(f"{path.name}: specialist file IDs are not unique")
        if rows and {"negative_probability", "neutral_probability", "positive_probability"} <= set(rows[0]):
            for row in rows:
                probabilities = np.array([float(row[key]) for key in
                                          ("negative_probability", "neutral_probability",
                                           "positive_probability")])
                if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
                    raise AssertionError(f"{path.name}: invalid probability")
                if abs(float(probabilities.sum()) - 1.0) > 1e-6:
                    raise AssertionError(f"{path.name}: probabilities do not sum to one")
    return counts


def main() -> None:
    if not OUT.is_dir():
        raise FileNotFoundError(f"Missing ensemble output: {OUT}")
    summary = read_json(OUT / "summary.json")
    if summary.get("official_test_n") != EXPECTED_TEST_N:
        raise AssertionError("Official test count is not 727")
    if summary.get("specialist_n") != EXPECTED_SPECIALIST_N:
        raise AssertionError("Attachment-3 count is not 30")
    if summary.get("selection_split") != "valid" or summary.get("test_tuning") is not False:
        raise AssertionError("Ensemble provenance is not validation-only")
    ensemble_meta = summary.get("ensemble") if isinstance(summary.get("ensemble"), dict) else {}
    operator = summary.get("ensemble_operator")
    if operator is None:
        rule = ensemble_meta.get("official_test_rule", "")
        operator = ("mean_logits_and_intensity" if "mean of frozen seed logits" in rule
                    else "mean_logits_and_intensity")
    if operator not in ("mean_logits_and_intensity", "equal_weight_mean_logits"):
        raise AssertionError(f"Unsupported ensemble operator: {operator}")

    prediction_path = find_prediction_file(summary)
    expected_hash = summary.get("prediction_sha256") or summary.get("ensemble_prediction_sha256")
    if expected_hash is None and isinstance(summary.get("output_sha256"), dict):
        expected_hash = summary["output_sha256"].get(prediction_path.name)
    if expected_hash and digest(prediction_path) != expected_hash:
        raise AssertionError("Ensemble prediction SHA-256 differs from summary")
    with np.load(prediction_path, allow_pickle=False) as data:
        required = {"ids", "logits", "intensity", "true_class", "true_intensity"}
        if set(data.files) < required:
            raise AssertionError(f"Missing prediction fields: {sorted(required - set(data.files))}")
        ids = data["ids"].astype(str)
        logits = np.asarray(data["logits"], dtype=np.float64)
        intensity = np.asarray(data["intensity"], dtype=np.float64).reshape(-1)
        true_class = np.asarray(data["true_class"], dtype=np.int64).reshape(-1)
        true_intensity = np.asarray(data["true_intensity"], dtype=np.float64).reshape(-1)
    if len(ids) != EXPECTED_TEST_N or len(set(ids)) != EXPECTED_TEST_N:
        raise AssertionError("Ensemble IDs are not 727 unique test IDs")
    if logits.shape != (EXPECTED_TEST_N, 3):
        raise AssertionError(f"Unexpected ensemble logits shape: {logits.shape}")
    if any(len(x) != EXPECTED_TEST_N for x in (intensity, true_class, true_intensity)):
        raise AssertionError("Ensemble target/prediction length mismatch")
    for name, array in (("logits", logits), ("intensity", intensity),
                        ("true_class", true_class), ("true_intensity", true_intensity)):
        finite_array(array, name)
    if np.any((true_class < 0) | (true_class > 2)):
        raise AssertionError("Invalid class labels")

    records = source_records(summary)
    source_arrays = []
    source_ids = None
    source_hashes = []
    source_seeds = []
    for record in records:
        path = resolve_source(record["prediction_file"])
        source_hash = digest(path)
        if source_hash != record["prediction_sha256"]:
            raise AssertionError(f"Source SHA-256 differs: {path.name}")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) < {"ids", "logits", "intensity", "true_class", "true_intensity"}:
                raise AssertionError(f"Incomplete source prediction: {path.name}")
            current_ids = data["ids"].astype(str)
            current_logits = np.asarray(data["logits"], dtype=np.float64)
            current_intensity = np.asarray(data["intensity"], dtype=np.float64).reshape(-1)
            current_class = np.asarray(data["true_class"], dtype=np.int64).reshape(-1)
            current_truth = np.asarray(data["true_intensity"], dtype=np.float64).reshape(-1)
        if len(current_ids) != EXPECTED_TEST_N or len(set(current_ids)) != EXPECTED_TEST_N:
            raise AssertionError(f"Source IDs invalid: {path.name}")
        if current_logits.shape != (EXPECTED_TEST_N, 3):
            raise AssertionError(f"Source logits shape invalid: {path.name}")
        if source_ids is None:
            source_ids = current_ids
        elif set(source_ids) != set(current_ids):
            raise AssertionError("Source prediction ID sets differ")
        index = {item: i for i, item in enumerate(current_ids)}
        order = np.array([index[item] for item in source_ids], dtype=np.int64)
        current_logits = current_logits[order]
        current_intensity = current_intensity[order]
        current_class = current_class[order]
        current_truth = current_truth[order]
        if not np.array_equal(current_class, true_class) or not np.allclose(current_truth, true_intensity, atol=0, rtol=0):
            raise AssertionError(f"Source targets differ from ensemble: {path.name}")
        for name, array in (("source logits", current_logits), ("source intensity", current_intensity)):
            finite_array(array, name)
        source_arrays.append((current_logits, current_intensity))
        source_hashes.append(source_hash)
        source_seeds.append(record.get("seed"))
    if source_ids is not None and not np.array_equal(source_ids, ids):
        raise AssertionError("Ensemble ID order differs from source ID order")
    if source_seeds and all(seed is not None for seed in source_seeds):
        if set(map(int, source_seeds)) != set(EXPECTED_SEEDS):
            raise AssertionError("Source seeds differ from the fixed three-seed protocol")
    expected_logits = np.mean(np.stack([x[0] for x in source_arrays]), axis=0)
    expected_intensity = np.mean(np.stack([x[1] for x in source_arrays]), axis=0)
    # The saved ensemble is float32; recomputation here uses float64 source
    # arrays, so allow the corresponding one-ULP rounding difference.
    if not np.allclose(logits, expected_logits, rtol=0, atol=1e-6):
        raise AssertionError("Ensemble logits are not the equal-weight source mean")
    if not np.allclose(intensity, expected_intensity, rtol=0, atol=1e-6):
        raise AssertionError("Ensemble intensity is not the equal-weight source mean")

    report = metrics(true_class, logits, true_intensity, intensity)
    declared = report_from_summary(summary)
    if declared is None and isinstance(summary.get("official_test"), dict):
        declared = summary["official_test"]
    if declared:
        close(report["classification"]["accuracy"], declared.get("classification", declared).get("accuracy"), "accuracy", 1e-12)
        close(report["classification"]["macro_f1"], declared.get("classification", declared).get("macro_f1"), "macro_f1", 1e-12)
        close(report["regression"]["mae"], declared.get("regression", declared).get("mae"), "mae", 1e-10)
        close(report["regression"]["pearson_r"], declared.get("regression", declared).get("pearson_r"), "pearson_r", 1e-10)
    for name, expected_hash in (summary.get("output_sha256") or {}).items():
        path = OUT / name
        if not path.is_file() or digest(path) != expected_hash:
            raise AssertionError(f"Output SHA-256 differs: {name}")
    attachment3_counts = audit_attachment3(summary)
    print(json.dumps({"prediction_file": prediction_path.name,
                      "prediction_sha256": digest(prediction_path),
                      "source_prediction_sha256": source_hashes,
                      "source_count": len(source_arrays),
                      "official_test_n": EXPECTED_TEST_N,
                      "attachment3_rows": attachment3_counts,
                      "metrics": report}, ensure_ascii=False, indent=2,
                     allow_nan=False))
    print("Independent balanced-sqrt ensemble audit passed")


if __name__ == "__main__":
    main()
