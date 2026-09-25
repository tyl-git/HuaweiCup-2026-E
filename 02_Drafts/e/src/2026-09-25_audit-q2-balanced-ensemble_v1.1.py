"""Independently verify the consistent v1.1 frozen ensemble, without inference.

This audit does not import the producer, train a model, select a rule, or use
specialist labels. It checks source and output hashes, all seed IDs, direct
equal-logit means, metrics, masks, and bitwise preservation of the old test.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.1"
OLD = Q2 / "q2-temporal-balanced-sqrt-ensemble-final-v1.0"
TRAINED = Q2 / "q2-temporal-balanced-sqrt-v1.0"
FINAL = Q2 / "q2-temporal-balanced-sqrt-final-v1.0"
SEEDS = (20260924, 20260925, 20260926)
CLASSES = ("Negative", "Neutral", "Positive")
MODALITIES = ("text", "audio", "vision")
OPERATOR = "arithmetic_mean_logits_and_intensity_then_softmax"


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def npz(path):
    with np.load(path, allow_pickle=False) as value:
        return {name: value[name].copy() for name in value.files}


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def exact(a, b, message):
    require(a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b), message)


def probability(logits):
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=1, keepdims=True)


def metrics(target, logits, strength_true, strength):
    predicted = logits.argmax(axis=1)
    cm = np.bincount(3 * target.astype(np.int64) + predicted, minlength=9).reshape(3, 3)
    per_class = []
    for c in range(3):
        tp, real, found = int(cm[c, c]), int(cm[c].sum()), int(cm[:, c].sum())
        per_class.append({"id": c, "name": CLASSES[c], "support": real,
                          "precision": tp / found if found else 0.0,
                          "recall": tp / real if real else 0.0,
                          "f1": 2 * tp / (real + found) if real + found else 0.0})
    yt, yp = strength_true.astype(np.float64), strength.astype(np.float64)
    ct, cp = yt - yt.mean(), yp - yp.mean()
    denominator = np.sqrt(np.dot(ct, ct) * np.dot(cp, cp))
    return {"classification": {"n": len(target), "accuracy": float(np.trace(cm) / len(target)),
                                "macro_f1": float(np.mean([x["f1"] for x in per_class])),
                                "neutral_f1": per_class[1]["f1"], "neutral_recall": per_class[1]["recall"],
                                "confusion_matrix_true_rows_predicted_columns": cm.tolist(),
                                "per_class": per_class},
            "regression": {"n": len(target), "mae": float(np.mean(np.abs(yt - yp))),
                           "pearson_r": float(np.dot(ct, cp) / denominator) if denominator else None,
                           "pearson_status": "defined" if denominator else "undefined_constant"}}


def compare_metrics(actual, declared, label):
    for kind, names in (("classification", ("accuracy", "macro_f1", "neutral_f1", "neutral_recall")),
                        ("regression", ("mae", "pearson_r"))):
        for name in names:
            require(abs(actual[kind][name] - declared[kind][name]) < 1e-12,
                    f"{label}: {kind}/{name} mismatch")
    require(actual["classification"]["confusion_matrix_true_rows_predicted_columns"] ==
            declared["classification"]["confusion_matrix_true_rows_predicted_columns"], f"{label}: confusion matrix mismatch")


def check_arrays(packed, combined, n, label, sources=None):
    require(packed["seeds"].tolist() == list(SEEDS), f"{label}: seed order mismatch")
    require(packed["ids"].shape == (n,) and len(np.unique(packed["ids"])) == n, f"{label}: ID count/uniqueness")
    exact(packed["ids"], combined["ids"], f"{label}: ensemble ID order")
    for name, shape in (("logits", (3, n, 3)), ("intensity", (3, n))):
        require(packed[name].shape == shape and packed[name].dtype == np.float32, f"{label}: seed shape/dtype {name}")
        require(np.isfinite(packed[name]).all(), f"{label}: nonfinite {name}")
        # A fresh direct numpy reduction independently checks the producer's operator.
        expected = np.add.reduce(packed[name], axis=0) / np.float32(3)
        exact(expected, combined[name], f"{label}: ensemble is not exact frozen arithmetic mean {name}")
    if sources:
        for s, source_path in enumerate(sources):
            source = npz(source_path)
            exact(source["ids"], packed["ids"], f"{label}: source IDs at seed {SEEDS[s]}")
            for name in ("logits", "intensity"):
                exact(source[name], packed[name][s], f"{label}: source {name} at seed {SEEDS[s]}")
            for name in ("true_class", "true_intensity"):
                exact(source[name], packed[name], f"{label}: source labels at seed {SEEDS[s]}")
                exact(source[name], combined[name], f"{label}: ensemble labels")
    if "probabilities" in combined:
        require(np.allclose(probability(combined["logits"]), combined["probabilities"], rtol=0, atol=1e-15), f"{label}: softmax is not applied after logit average")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    summary = read_json(OUT / "summary.json")
    protocol = read_json(OUT / "protocol.json")
    require(protocol["operator_all_splits"] == summary["operator_all_splits"] == OPERATOR, "One consistent operator required")
    require(protocol["seeds"] == list(SEEDS) and protocol["weights"] == [1 / 3] * 3, "Fixed seeds/weights required")
    require(protocol["official_test_already_viewed"] and not protocol["one_shot_holdout_claim"], "Post-test status not disclosed")
    require(not protocol["retraining"] and not protocol["weight_or_seed_tuning"]
            and not protocol["correction_uses_test_metric_comparison_to_choose_rule"], "Correction must not fit/select")
    require(protocol["feature_version"] == "aligned_50" and protocol["feature_dimensions"] == [768, 74, 35], "Input interface differs")
    require(protocol["checkpoint_selection_split"] == "valid", "Wrong checkpoint selection split")
    require(not protocol["attachment3_labels_available"] and not protocol["attachment3_accuracy_reported"], "Specialist labels/accuracy forbidden")
    records = [protocol["code"], protocol["normalization"], *protocol["sources"]]
    records += [job["checkpoint"] for job in protocol["checkpoints"]]
    for rec in records:
        require(sha(ROOT / rec["path"]) == rec["sha256"], f"Source hash mismatch: {rec['path']}")
    for name, expected_hash in summary["output_sha256"].items():
        require(sha(OUT / name) == expected_hash, f"Output hash mismatch: {name}")
    for job in protocol["checkpoints"]:
        seed = job["seed"]
        recorded = read_json(TRAINED / "temporal" / f"seed_{seed}" / "metrics.json")
        require(recorded["checkpoint_sha256"] == job["checkpoint"]["sha256"], f"Seed {seed} checkpoint hash mismatch")
        require(recorded["normalization_sha256"] == protocol["normalization"]["sha256"], f"Seed {seed} normalization hash mismatch")
        require(recorded["best_epoch"] == job["best_epoch"] and recorded["config"] == job["config"], f"Seed {seed} metadata differs")
    reports = {}
    for kind, n in (("valid", 728), ("test", 727)):
        packed = npz(OUT / f"{kind}_seed_predictions.npz")
        combined = npz(OUT / f"balanced_sqrt_ensemble_{kind}_predictions.npz")
        sources = ([TRAINED / "temporal" / f"seed_{seed}" / "valid_complete_predictions.npz" for seed in SEEDS]
                   if kind == "valid" else [FINAL / f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz" for seed in SEEDS])
        check_arrays(packed, combined, n, kind, sources)
        reports[kind] = metrics(combined["true_class"], combined["logits"], combined["true_intensity"], combined["intensity"])
        compare_metrics(reports[kind], summary["validation" if kind == "valid" else "official_test"], kind)
        if kind == "test":
            old = npz(OLD / "balanced_sqrt_ensemble_test_predictions.npz")
            require(set(old) == set(combined), "Official-test fields changed")
            for key in old:
                exact(old[key], combined[key], f"Official-test {key} changed from v1.0")
    with (OUT / "valid_missing_ensemble_metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_metrics = {row["condition"]: row for row in csv.DictReader(handle)}
    conditions = [f"{m}_missing_{fraction}pct_{position}" for m in MODALITIES
                  for fraction in (20, 40) for position in ("start", "middle", "end")]
    require(set(summary["validation_missing"]) == set(conditions) == set(csv_metrics), "18 exact validation conditions required")
    for condition in conditions:
        packed = npz(OUT / "valid_missing" / f"{condition}_seed_predictions.npz")
        combined = npz(OUT / "valid_missing" / f"{condition}_ensemble_predictions.npz")
        paths = [TRAINED / "temporal" / f"seed_{seed}" / f"valid_{condition}_predictions.npz" for seed in SEEDS]
        check_arrays(packed, combined, 728, condition, paths)
        report = metrics(combined["true_class"], combined["logits"], combined["true_intensity"], combined["intensity"])
        compare_metrics(report, summary["validation_missing"][condition], condition)
        for kind, names in (("classification", ("accuracy", "macro_f1", "neutral_f1")), ("regression", ("mae", "pearson_r"))):
            for name in names:
                require(abs(report[kind][name] - float(csv_metrics[condition][name])) < 1e-12, f"CSV metric differs: {condition}/{name}")
        require(abs(report["classification"]["macro_f1"] - reports["valid"]["classification"]["macro_f1"]
                    - float(csv_metrics[condition]["delta_macro_f1_from_complete"])) < 1e-12, f"CSV delta F1 differs: {condition}")
        require(abs(report["regression"]["mae"] - reports["valid"]["regression"]["mae"]
                    - float(csv_metrics[condition]["delta_mae_from_complete"])) < 1e-12, f"CSV delta MAE differs: {condition}")
    packed = npz(OUT / "attachment3_seed_predictions.npz")
    combined = npz(OUT / "attachment3_ensemble_predictions.npz")
    check_arrays(packed, combined, 30, "attachment3")
    exact(packed["file_ids"], combined["file_ids"], "Attachment3 file ID mismatch")
    require(set(combined) == {"ids", "file_ids", "logits", "intensity", "probabilities"}, "Specialist contains unsupported or label fields")
    with (OUT / "attachment3_ensemble_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 30 and [row["file_id"] for row in rows] == combined["file_ids"].tolist(), "Attachment3 CSV ID order/count")
    for i, row in enumerate(rows):
        require(row["ensemble_rule"] == OPERATOR, "Attachment3 CSV operator")
        require(row["polarity"] == CLASSES[combined["logits"][i].argmax()], "Attachment3 class differs")
        require(float(row["intensity"]) == float(combined["intensity"][i]), "Attachment3 intensity differs")
        for c, label in enumerate(CLASSES):
            require(abs(float(row[f"{label.lower()}_probability"]) - combined["probabilities"][i, c]) < 1e-15, "Attachment3 probability differs")
    with (FINAL / "attachment3_all_models.csv").open(encoding="utf-8-sig", newline="") as handle:
        old_seed = {(int(row["seed"]), row["file_id"]): row for row in csv.DictReader(handle)}
    for s, seed in enumerate(SEEDS):
        probs = probability(packed["logits"][s])
        for i, file_id in enumerate(packed["file_ids"]):
            old_row = old_seed[(seed, file_id)]
            require(abs(float(packed["intensity"][s, i]) - float(old_row["intensity"])) <= 5e-6, "Original per-seed intensity changed")
            for c, label in enumerate(CLASSES):
                require(abs(probs[i, c] - float(old_row[f"{label.lower()}_probability"])) <= 5e-6, "Original per-seed predictor changed")
    masks = npz(OUT / "attachment3_masks.npz")
    exact(masks["ids"], combined["ids"], "Mask IDs mismatch")
    content = masks["content_mask"]
    require(content.shape == (30, 50) and content.dtype == bool, "Content mask schema")
    for m in MODALITIES:
        observed = masks[f"source_observed_{m}"]
        artificial = masks[f"artificial_missing_{m}"]
        unavailable = masks[f"source_unavailable_{m}"]
        effective = masks[f"effective_{m}"]
        require(all(x.shape == content.shape and x.dtype == bool for x in (observed, artificial, unavailable, effective)), f"{m} mask schema")
        require(not np.any(observed & ~content) and not artificial.any(), f"{m}: padding or artificial gap error")
        exact(unavailable, content & ~observed, f"{m}: source unavailable semantics")
        exact(effective, content & observed & ~artificial, f"{m}: effective semantics")
    report = {"schema": "independent_balanced_ensemble_audit_v1.1", "status": "passed",
              "source_records_verified": len(records), "output_files_verified": len(summary["output_sha256"]),
              "valid_n": 728, "official_test_n": 727, "attachment3_n": 30, "validation_missing_conditions": 18,
              "operator_all_splits": OPERATOR, "official_test_arrays_bitwise_identical_to_v1_0": True,
              "source_seed_ID_order_verified": True, "checkpoint_and_normalization_hashes_verified": True,
              "mask_semantics_verified": True, "official_test_already_viewed": True,
              "official_test": reports["test"], "validation": reports["valid"],
              "auditor_sha256": sha(Path(__file__)), "summary_sha256": sha(OUT / "summary.json")}
    if args.write_report:
        report_path = OUT / "independent_audit.json"
        if report_path.exists():
            require(read_json(report_path) == report, "Existing independent audit differs")
        else:
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent v1.1 ensemble audit passed: valid, test, 18 gaps, attachment3; test unchanged")


if __name__ == "__main__":
    main()
