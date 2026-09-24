"""Independently verify saved Q2 temporal test and specialist outputs."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "03_Results/e/question-two/q2-temporal-final-v1.0"


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    assert summary["official_test_n"] == 727 and summary["specialist_n"] == 30
    assert summary["specialist_labels_used"] is False
    assert len(summary["results"]) == 9
    expected = {(model, seed) for model in ("temporal", "no_temporal", "no_gap_signal")
                for seed in (20260924, 20260925, 20260926)}
    assert {(row["model"], row["seed"]) for row in summary["results"]} == expected
    for row in summary["results"]:
        path = OUT / f"{row['model']}_seed_{row['seed']}_test_predictions.npz"
        assert digest(path) == row["prediction_sha256"]
        with np.load(path, allow_pickle=False) as data:
            ids, logits = data["ids"], data["logits"]
            target, scores = data["true_class"], data["intensity"]
            truth = data["true_intensity"]
            assert len(ids) == len(set(ids)) == len(target) == 727
            assert logits.shape == (727, 3) and scores.shape == truth.shape == (727,)
            assert all(np.isfinite(a).all() for a in (logits, scores, truth))
            predicted = logits.argmax(axis=1)
            accuracy = np.mean(predicted == target)
            f1 = []
            for cls in range(3):
                tp = np.sum((predicted == cls) & (target == cls))
                fp = np.sum((predicted == cls) & (target != cls))
                fn = np.sum((predicted != cls) & (target == cls))
                f1.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
            mae = np.mean(np.abs(scores - truth))
            pearson = np.corrcoef(scores, truth)[0, 1]
            report = row["report"]
            assert abs(accuracy - report["classification"]["accuracy"]) < 1e-12
            assert abs(np.mean(f1) - report["classification"]["macro_f1"]) < 1e-12
            assert abs(mae - report["regression"]["mae"]) < 1e-7
            assert abs(pearson - report["regression"]["pearson_r"]) < 1e-6
    all_rows = read_csv(OUT / "attachment3_all_models.csv")
    chosen = read_csv(OUT / "attachment3_predictions.csv")
    audit = read_csv(OUT / "attachment3_observation_audit.csv")
    expected_files = {f"附件3_{i:02}" for i in range(1, 31)}
    assert len(all_rows) == 270 and len(chosen) == len(audit) == 30
    assert {row["file_id"] for row in chosen} == expected_files
    assert {row["file_id"] for row in audit} == expected_files
    assert chosen == [row for row in all_rows if row["model"] == "temporal"
                      and row["seed"] == "20260926"]
    for row in all_rows:
        assert row["polarity"] in ("Negative", "Neutral", "Positive")
        prob = np.array([float(row[f"{name}_probability"])
                         for name in ("negative", "neutral", "positive")])
        assert np.isfinite(prob).all() and np.isfinite(float(row["intensity"]))
        assert np.all((prob >= 0) & (prob <= 1)) and abs(prob.sum() - 1) < 1e-6
        assert row["polarity"] == ("Negative", "Neutral", "Positive")[prob.argmax()]
    for path, expected_hash in summary["output_sha256"].items():
        assert digest(OUT / path) == expected_hash
    print("Independent Q2 audit passed: 9 test prediction files, 270 specialist predictions, 30 selected rows")


if __name__ == "__main__":
    main()
