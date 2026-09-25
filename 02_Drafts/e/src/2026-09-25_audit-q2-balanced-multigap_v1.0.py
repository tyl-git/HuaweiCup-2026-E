"""Independent numerical/mask audit of the 24 fixed validation local gaps."""
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
OUT = Q2 / "q2-balanced-multigap-stress-valid-v1.0"
SEEDS = [20260924, 20260925, 20260926]
MODALITIES = ("text", "audio", "vision")
GROUPS = (("text", "audio"), ("text", "vision"), ("audio", "vision"), MODALITIES)


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def require(value, message):
    if not value:
        raise AssertionError(message)


def metric(target, logits, truth, strength):
    target = target.astype(np.int64)
    cm = np.bincount(3 * target + logits.argmax(axis=1), minlength=9).reshape(3, 3)
    f1 = [2 * cm[c, c] / (cm[c].sum() + cm[:, c].sum())
          if cm[c].sum() + cm[:, c].sum() else 0.0 for c in range(3)]
    truth, strength = truth.astype(np.float64), strength.astype(np.float64)
    return {"accuracy": float(np.trace(cm) / len(target)), "macro_f1": float(np.mean(f1)),
            "neutral_f1": float(f1[1]), "mae": float(np.abs(truth - strength).mean()),
            "pearson_r": float(np.corrcoef(truth, strength)[0, 1]), "confusion": cm.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    protocol, summary = read(OUT / "protocol.json"), read(OUT / "summary.json")
    expected_conditions = [{"name": f"{'_'.join(group)}_missing_{percent}pct_{position}",
                            "modalities": list(group), "fraction": percent / 100, "position": position}
                           for group in GROUPS for percent in (20, 40) for position in ("start", "middle", "end")]
    require(protocol["conditions"] == expected_conditions, "Predeclared condition set differs")
    require(protocol["conditions_frozen_before_task_predictions"] and protocol["evaluation_split"] == "valid"
            and protocol["n"] == 728 and protocol["seeds"] == SEEDS
            and protocol["weights"] == [1 / 3] * 3 and protocol["no_training_or_tuning"]
            and not protocol["official_test_predictions_used"] and not protocol["one_shot_holdout_claim"], "Invalid protocol")
    for item in protocol["sources"] + [job["checkpoint"] for job in protocol["checkpoint_records"]]:
        require(sha(ROOT / item["path"]) == item["sha256"], f"Source hash differs: {item['path']}")
    for name, digest in summary["output_sha256"].items():
        require(sha(OUT / name) == digest, f"Output hash differs: {name}")
    reference_record = summary["complete_reference"]
    require(sha(ROOT / reference_record["path"]) == reference_record["sha256"], "Complete reference changed")
    reference = npz(ROOT / reference_record["path"])
    predictions = npz(OUT / "predictions.npz")
    masks = npz(OUT / "condition_masks.npz")
    require(predictions["seeds"].tolist() == SEEDS, "Prediction seed order")
    for value in (predictions, masks):
        require(value["conditions"].tolist() == [c["name"] for c in expected_conditions], "Condition order")
        require(value["ids"].shape == (728,) and np.unique(value["ids"]).size == 728
                and np.array_equal(value["ids"], reference["ids"]), "Validation IDs differ")
    for name in ("true_class", "true_intensity"):
        require(np.array_equal(predictions[name], reference[name]), f"Validation {name} differs")
    require(predictions["seed_logits"].shape == (24, 3, 728, 3)
            and predictions["seed_intensity"].shape == (24, 3, 728), "Prediction shapes")
    for name in ("logits", "intensity"):
        source = predictions[f"seed_{name}"]
        require(source.dtype == np.float32 and np.isfinite(source).all(), f"Invalid seed {name}")
        require(np.array_equal(np.add.reduce(source, axis=1) / np.float32(3), predictions[name]), f"Ensemble {name} differs")
    content = masks["content_mask"]
    require(content.dtype == bool and content.shape == (728, 50), "Content mask shape/dtype")
    require(masks["interval_mask"].dtype == bool and masks["interval_mask"].shape == (24, 728, 50), "Interval mask shape/dtype")
    for c, condition in enumerate(expected_conditions):
        independent_intervals = np.zeros_like(content)
        bounds = np.full((728, 2), -1, dtype=np.int64)
        for i, row in enumerate(content):
            eligible = np.flatnonzero(row)
            if eligible.size == 0:
                continue
            require(eligible[-1] - eligible[0] + 1 == eligible.size, "Noncontiguous content")
            width = min(eligible.size, max(1, int(np.ceil(eligible.size * condition["fraction"]))))
            begin_index = {"start": 0, "middle": (eligible.size - width) // 2,
                           "end": eligible.size - width}[condition["position"]]
            start, end = int(eligible[begin_index]), int(eligible[begin_index + width - 1]) + 1
            independent_intervals[i, start:end] = True
            bounds[i] = (start, end)
        require(np.array_equal(independent_intervals, masks["interval_mask"][c])
                and np.array_equal(bounds, masks["interval_bounds"][c]), f"Shared interval mismatch: {condition['name']}")
        require(not np.any(independent_intervals & ~content), "Gap includes padding")
        for m in MODALITIES:
            observed = masks[f"source_observed_{m}"]
            require(observed.dtype == bool and observed.shape == content.shape
                    and not np.any(observed & ~content), f"Original observation mask invalid: {m}")
            require(np.array_equal(masks[f"source_unavailable_{m}"], content & ~observed), "Source zero mask invalid")
            expected_missing = independent_intervals & observed if m in condition["modalities"] else np.zeros_like(content)
            require(np.array_equal(expected_missing, masks[f"missing_{m}"][c]), "Artificial missing differs")
            require(np.array_equal(content & observed & ~expected_missing, masks[f"effective_{m}"][c]), "Effective mask differs")
    with (OUT / "metrics.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == 24 and [r["condition"] for r in rows] == [c["name"] for c in expected_conditions], "Metrics CSV condition mismatch")
    complete_metric = metric(reference["true_class"], reference["logits"], reference["true_intensity"], reference["intensity"])
    compact = []
    for c, condition in enumerate(expected_conditions):
        result = metric(predictions["true_class"], predictions["logits"][c], predictions["true_intensity"], predictions["intensity"][c])
        stated = summary["results"][condition["name"]]
        for kind, names in (("classification", ("accuracy", "macro_f1", "neutral_f1")), ("regression", ("mae", "pearson_r"))):
            for name in names:
                require(abs(result[name] - stated[kind][name]) < 1e-12
                        and abs(result[name] - float(rows[c][name])) < 1e-12, f"Metric differs: {condition['name']}/{name}")
        require(result["confusion"] == stated["classification"]["confusion_matrix_true_rows_predicted_columns"], "Confusion differs")
        for column, name in (("delta_macro_f1_from_complete", "macro_f1"), ("delta_mae_from_complete", "mae")):
            require(abs(result[name] - complete_metric[name] - float(rows[c][column])) < 1e-12, "Delta metric differs")
        for m in MODALITIES:
            require(int(rows[c][f"{m}_removed_observed_positions"]) == int(masks[f"missing_{m}"][c].sum()), "Removed count differs")
            newly_empty = masks[f"source_observed_{m}"].any(axis=1) & ~masks[f"effective_{m}"][c].any(axis=1)
            require(int(rows[c][f"{m}_samples_newly_fully_unavailable"]) == int(newly_empty.sum()), "Newly empty count differs")
        compact.append({"condition": condition["name"], **{key: result[key] for key in ("macro_f1", "mae")}})
    report = {"schema": "q2_multigap_independent_audit_v1.0", "status": "passed", "n": 728,
              "condition_count": 24, "seed_count": 3, "prediction_arrays_verified": True,
              "all_conditions_use_same_predictor": True, "shared_intervals_independently_reconstructed": True,
              "padding_and_source_masks_preserved": True, "metrics_independently_recomputed": True,
              "split": "valid", "no_training_or_tuning": True, "summary_sha256": sha(OUT / "summary.json"),
              "auditor_sha256": sha(Path(__file__)), "metrics": compact}
    if args.write_report:
        path = OUT / "independent_audit.json"
        if path.exists():
            require(read(path) == report, "Existing audit report differs")
        else:
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    print("Independent simultaneous-local-gap stress audit passed: 24 conditions x 3 frozen seeds")


if __name__ == "__main__":
    main()
