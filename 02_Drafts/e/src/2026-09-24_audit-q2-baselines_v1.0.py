"""Independently audit the Q2 baseline train/valid artifacts without inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
RESULTS = ROOT / "03_Results" / "e" / "question-two"
EXPERIMENT = RESULTS / "q2-baselines-v1.0"
REPORT = RESULTS / "2026-09-24_q2-baseline-review_v1.0"
MODES = {"text": ("text",), "audio": ("audio",),
         "vision": ("vision",), "fusion": ("text", "audio", "vision")}
SEEDS = (20260924, 20260925, 20260926)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def require(condition, detail: str):
    if not condition:
        raise RuntimeError(detail)


def close(actual, expected, detail: str, atol: float = 1e-6):
    require(np.isclose(actual, expected, rtol=0, atol=atol),
            f"{detail}: actual {actual}, recorded {expected}")


def compare_metrics(actual: dict, recorded: dict, detail: str):
    require(actual.keys() == recorded.keys(), f"{detail}: metric fields changed")
    for key in ("n", "confusion_matrix_true_rows_predicted_columns",
                "zero_division_policy", "pearson_status"):
        if key in actual:
            require(actual[key] == recorded[key], f"{detail}: {key} mismatch")
    for key in ("accuracy", "macro_f1", "mae", "pearson_r"):
        if key in actual:
            if actual[key] is None:
                require(recorded[key] is None, f"{detail}: {key} should be null")
            else:
                close(actual[key], recorded[key], f"{detail}: {key}", 1e-10)
    if "per_class" in actual:
        require(len(actual["per_class"]) == len(recorded["per_class"]) == 3,
                f"{detail}: per-class count")
        for a, b in zip(actual["per_class"], recorded["per_class"]):
            require((a["id"], a["name"], a["support"]) ==
                    (b["id"], b["name"], b["support"]), f"{detail}: per-class identity")
            for key in ("precision", "recall", "f1"):
                close(a[key], b[key], f"{detail}: class {a['id']} {key}", 1e-10)


def evaluate_npz(path: Path, split: str, batch, modalities, report, metrics, priors):
    require(sha256(path) == report["prediction_files_sha256"][split],
            f"Prediction hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        require(set(z.files) == {"ids", "available", "logits", "intensity",
                                 "true_class", "true_intensity"},
                f"Unexpected NPZ fields: {path}")
        ids = z["ids"]
        available = z["available"]
        logits = z["logits"]
        intensity = z["intensity"]
        true_class = z["true_class"]
        true_intensity = z["true_intensity"]
    n = len(batch)
    require(ids.shape == (n,) and np.array_equal(ids, np.asarray(batch.ids)),
            f"Sample ID/order mismatch: {path}")
    expected_available = np.column_stack([
        batch.observation_masks[m].any(axis=1) for m in modalities])
    require(available.shape == (n, len(modalities)) and available.dtype == bool and
            np.array_equal(available, expected_available), f"Availability mismatch: {path}")
    require(logits.shape == (n, 3) and intensity.shape == (n,) and
            np.isfinite(logits).all() and np.isfinite(intensity).all(),
            f"Prediction shape/finiteness mismatch: {path}")
    require(np.array_equal(true_class, batch.classification_labels) and
            np.array_equal(true_intensity, batch.regression_labels),
            f"Label mismatch: {path}")
    empty = ~available.any(axis=1)
    require(int(empty.sum()) == report["empty_observation_samples"][split],
            f"Empty-observation count mismatch: {path}")
    if empty.any():
        np.testing.assert_allclose(logits[empty],
                                   np.broadcast_to(np.log(priors["class_probabilities"]),
                                                   (int(empty.sum()), 3)), rtol=0, atol=1e-7)
        np.testing.assert_allclose(intensity[empty], priors["intensity_mean"],
                                   rtol=0, atol=1e-7)
    classification = metrics.classification_metrics(true_class, logits.argmax(axis=1))
    regression = metrics.regression_metrics(true_intensity, intensity)
    compare_metrics(classification, report[split]["classification"], f"{path}: classification")
    compare_metrics(regression, report[split]["regression"], f"{path}: regression")
    shifted = logits.astype(np.float64) - logits.max(axis=1, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=1))
    ce = float(np.mean(logsumexp - shifted[np.arange(n), true_class]))
    l1 = float(np.mean(np.abs(intensity.astype(np.float64) - true_intensity)))
    close(ce, report[split]["cross_entropy"], f"{path}: cross entropy", 1e-6)
    close(l1, report[split]["l1"], f"{path}: L1", 1e-6)
    config = report["config"]
    close(config["classification_loss_weight"] * ce +
          config["regression_loss_weight"] * l1,
          report[split]["selection_loss"], f"{path}: selection loss", 2e-6)
    return {"classification": classification, "regression": regression,
            "empty_observation_samples": int(empty.sum())}


def audit_history(history: list[dict], report: dict, checkpoint: dict, detail: str):
    config = report["config"]
    require(len(history) == report["epochs_run"] and 1 <= len(history) <= config["max_epochs"],
            f"{detail}: history length")
    best, best_epoch, stale = float("inf"), 0, 0
    for epoch, row in enumerate(history, 1):
        require(row["epoch"] == epoch, f"{detail}: epoch numbering")
        selection = (config["classification_loss_weight"] * row["valid_cross_entropy"] +
                     config["regression_loss_weight"] * row["valid_l1"])
        close(selection, row["valid_selection_loss"], f"{detail}: epoch {epoch} loss", 1e-10)
        if selection < best - 1e-8:
            best, best_epoch, stale = selection, epoch, 0
        else:
            stale += 1
        if epoch < len(history):
            require(stale < config["patience"], f"{detail}: continued after early stop")
    require(best_epoch == report["best_epoch"] == checkpoint["epoch"],
            f"{detail}: best epoch mismatch")
    require(stale >= config["patience"] or len(history) == config["max_epochs"],
            f"{detail}: training ended before early stop or max epoch")
    close(best, report["valid"]["selection_loss"], f"{detail}: best validation loss", 1e-8)
    best_row = history[best_epoch - 1]
    close(best_row["valid_cross_entropy"], report["valid"]["cross_entropy"],
          f"{detail}: best cross entropy", 1e-8)
    close(best_row["valid_l1"], report["valid"]["l1"],
          f"{detail}: best L1", 1e-8)


def summary_row(report: dict) -> dict:
    return {"model": report["model"], "seed": report["seed"],
            "best_epoch": report["best_epoch"],
            "accuracy": report["valid"]["classification"]["accuracy"],
            "macro_f1": report["valid"]["classification"]["macro_f1"],
            "mae": report["valid"]["regression"]["mae"],
            "pearson_r": report["valid"]["regression"]["pearson_r"]}


def mean_sd(values):
    a = np.asarray(values, dtype=np.float64)
    return {"mean": float(a.mean()), "sample_sd": float(a.std(ddof=1)),
            "n_seeds": len(a)}


def audit() -> dict:
    import torch

    loader = module("q2_audit_loader", "2026-09-24_q2_data_v1.0.py")
    metrics = module("q2_audit_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    provenance = read_json(EXPERIMENT / "provenance.json")
    require(provenance["model_types"] == list(MODES) and provenance["seeds"] == list(SEEDS),
            "Unexpected experiment matrix")
    require(provenance["fit_split"] == "train" and provenance["selection_split"] == "valid" and
            not provenance["test_evaluated"] and not provenance["specialists_evaluated"],
            "Experiment split declaration mismatch")
    for filename, expected_hash in provenance["dependency_sha256"].items():
        require((SRC / filename).is_file() and sha256(SRC / filename) == expected_hash,
                f"Training dependency changed: {filename}")
    source = loader.DEFAULT_OFFICIAL
    require(sha256(source) == provenance["official_source_sha256"],
            "Official source hash mismatch")
    require(sha256(RESULTS / "2026-09-24_q2-normalization_v1.0.npz") ==
            provenance["normalization_sha256"], "Normalizer hash mismatch")
    for split in ("train", "valid"):
        cache_file = RESULTS / "bert-cache-v1.0" / f"{split}.npy"
        require(sha256(cache_file) == provenance["bert_cache_sha256"][split],
                f"BERT cache hash mismatch: {split}")
    batches = loader.load_official(source)
    del batches["test"]
    constant = read_json(RESULTS / "2026-09-24_q2-constant-baselines_v1.0.json")
    require(constant["source_sha256"] == provenance["official_source_sha256"] and
            constant["fit_split"] == "train" and not constant["test_evaluated"],
            "Constant baseline source/split mismatch")
    counts = np.bincount(batches["train"].classification_labels, minlength=3)
    majority = int(counts.argmax())
    training_scores = batches["train"].regression_labels.astype(np.float64)
    constant_values = {"mean_intensity": float(training_scores.mean()),
                       "median_intensity": float(np.median(training_scores))}
    require(majority == constant["constants"]["majority_class"] and
            counts.tolist() == constant["constants"]["class_counts"],
            "Constant class prior mismatch")
    for split, batch in batches.items():
        compare_metrics(metrics.classification_metrics(
            batch.classification_labels, np.full(len(batch), majority)),
            constant["results"][split]["majority_class"], f"{split}: constant class")
        for key, value in constant_values.items():
            compare_metrics(metrics.regression_metrics(
                batch.regression_labels, np.full(len(batch), value, dtype=np.float64)),
                constant["results"][split][key], f"{split}: constant {key}")
    expected_keys = {(mode, seed) for mode in MODES for seed in SEEDS}
    existing_keys = {(p.parent.parent.name, int(p.parent.name.removeprefix("seed_")))
                     for p in EXPERIMENT.glob("*/seed_*/metrics.json")}
    require(existing_keys == expected_keys, "Missing/extra experiment runs")
    written_summary = read_json(EXPERIMENT / "summary.json")
    require(len(written_summary) == len(expected_keys), "Summary run count mismatch")
    summary_keys = {(r["model"], r["seed"]) for r in written_summary}
    require(summary_keys == expected_keys, "Summary run identities mismatch")
    recalculated = []
    run_audits = []
    for name, modalities in MODES.items():
        for seed in SEEDS:
            folder = EXPERIMENT / name / f"seed_{seed}"
            report = read_json(folder / "metrics.json")
            detail = f"{name}/{seed}"
            require(report["model"] == name and report["modalities"] == list(modalities) and
                    report["seed"] == seed and report["config"] == provenance["config"] and
                    report["normalization_sha256"] == provenance["normalization_sha256"],
                    f"{detail}: run metadata mismatch")
            require(report["checkpoint_selection"] ==
                    "minimum_valid_cross_entropy_plus_l1" and
                    report["checkpoint_reload_predictions_identical"] and
                    not report["test_evaluated"] and not report["specialists_evaluated"],
                    f"{detail}: run boundary/checkpoint declaration mismatch")
            checkpoint_file = folder / "best.pt"
            require(sha256(checkpoint_file) == report["checkpoint_sha256"],
                    f"{detail}: checkpoint hash mismatch")
            checkpoint = torch.load(checkpoint_file, map_location="cpu", weights_only=True)
            require(checkpoint["model"] == name and
                    tuple(checkpoint["modalities"]) == modalities and
                    checkpoint["seed"] == seed and checkpoint["config"] == report["config"] and
                    checkpoint["train_priors"] == report["train_priors"] and
                    checkpoint["normalization_sha256"] == provenance["normalization_sha256"] and
                    bool(checkpoint["model_state"]), f"{detail}: checkpoint metadata mismatch")
            priors = report["train_priors"]
            train_classes = batches["train"].classification_labels
            require(priors["class_counts"] == np.bincount(train_classes, minlength=3).tolist(),
                    f"{detail}: class prior mismatch")
            np.testing.assert_allclose(priors["class_probabilities"], counts / counts.sum(),
                                       rtol=0, atol=1e-12)
            close(priors["intensity_mean"],
                  float(np.mean(batches["train"].regression_labels, dtype=np.float64)),
                  f"{detail}: intensity prior", 1e-10)
            history = read_json(folder / "history.json")
            audit_history(history, report, checkpoint, detail)
            split_audits = {split: evaluate_npz(
                folder / f"{split}_predictions.npz", split, batches[split],
                modalities, report, metrics, priors) for split in ("train", "valid")}
            row = summary_row(report)
            recorded_row = next(r for r in written_summary
                                if (r["model"], r["seed"]) == (name, seed))
            require(row == recorded_row, f"{detail}: summary mismatch")
            recalculated.append(row)
            run_audits.append({"model": name, "seed": seed,
                              "best_epoch": report["best_epoch"],
                              "epochs_run": report["epochs_run"],
                              "checkpoint_sha256": report["checkpoint_sha256"],
                              "prediction_files_sha256": report["prediction_files_sha256"],
                              "history_sha256": sha256(folder / "history.json"),
                              "metrics_sha256": sha256(folder / "metrics.json"),
                              "recomputed_metrics": split_audits})
    aggregates = {}
    for name in MODES:
        rows = [r for r in recalculated if r["model"] == name]
        aggregates[name] = {key: mean_sd([r[key] for r in rows])
                            for key in ("accuracy", "macro_f1", "mae", "pearson_r",
                                        "best_epoch")}
        aggregates[name]["neutral_recall"] = mean_sd([
            read_json(EXPERIMENT / name / f"seed_{seed}" / "metrics.json")
            ["valid"]["classification"]["per_class"][1]["recall"] for seed in SEEDS])
    return {"schema": "q2-baseline-independent-audit-v1.0",
            "status": "verified", "runs": len(recalculated),
            "splits_evaluated": ["train", "valid"], "test_evaluated": False,
            "specialists_evaluated": False,
            "source_sha256": provenance["official_source_sha256"],
            "normalization_sha256": provenance["normalization_sha256"],
            "audit_script_sha256": sha256(Path(__file__)),
            "provenance_sha256": sha256(EXPERIMENT / "provenance.json"),
            "summary_sha256": sha256(EXPERIMENT / "summary.json"),
            "dependency_sha256": provenance["dependency_sha256"],
            "checks": {"prediction_npz_files": 24, "checkpoint_files": 12,
                       "history_files": 12, "metrics_files": 12,
                       "sample_ids_labels_and_availability": "matched_official_train_valid",
                       "constant_baselines_recomputed": True,
                       "checkpoint_inference_rerun": False,
                       "checkpoint_reload_identity": "verified_training_record_only",
                       "loss_absolute_tolerance": 2e-6,
                       "metric_absolute_tolerance": 1e-10},
            "baseline_valid": {
                "majority_accuracy": constant["results"]["valid"]["majority_class"]["accuracy"],
                "majority_macro_f1": constant["results"]["valid"]["majority_class"]["macro_f1"],
                "mean_mae": constant["results"]["valid"]["mean_intensity"]["mae"],
                "median_mae": constant["results"]["valid"]["median_intensity"]["mae"]},
            "sample_standard_deviation_ddof": 1,
            "seed_statistics_are_not_confidence_intervals": True,
            "aggregates": aggregates, "runs_detail": recalculated, "run_audits": run_audits,
            "empty_observation_samples": {name: {
                split: read_json(EXPERIMENT / name / f"seed_{SEEDS[0]}" / "metrics.json")
                ["empty_observation_samples"][split] for split in ("train", "valid")}
                for name in MODES}}


def render_md(result: dict) -> str:
    baseline = result["baseline_valid"]
    lines = ["# 问题二基线训练独立验收", "",
             "对 4 种模型、每种 3 个种子的训练/验证预测逐样本复算，并核对样本顺序、原始标签、文件 SHA-256、检查点、最优轮次和提前停止。共 12 次运行通过。官方 test 和专项附件未评价。", "",
             "| 模型 | Accuracy | Macro-F1 | MAE | Pearson r | 最优轮次 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name in MODES:
        agg = result["aggregates"][name]
        values = [f"{agg[key]['mean']:.4f} ± {agg[key]['sample_sd']:.4f}"
                  for key in ("accuracy", "macro_f1", "mae", "pearson_r", "best_epoch")]
        lines.append(f"| {name} | " + " | ".join(values) + " |")
    lines += ["", "上表是 valid 的三个固定种子均值 ± 样本标准差（ddof=1），不是置信区间；每组样本数均为 728。检查点根据 valid 的交叉熵加 L1 最小值选择，不能把这些 valid 数值当作最终泛化成绩。", "",
              f"常数参考：多数类 Accuracy {baseline['majority_accuracy']:.4f}、Macro-F1 {baseline['majority_macro_f1']:.4f}；训练集均值强度 MAE {baseline['mean_mae']:.4f}，中位数强度 MAE {baseline['median_mae']:.4f}。", "",
              "融合模型的平均 Accuracy 和 Macro-F1 略高于文本模型，但平均 MAE 略差，Pearson r 接近；不能据此宣称三模态在所有目标上均有增益。音频单模态接近多数类，两个种子在 valid 完全预测为 Positive。", "",
              "Neutral 类依然薄弱：音频平均召回率为 0，视觉几乎为 0；文本约 0.089，融合约 0.170。视觉单模态有 110 个 train 和 15 个 valid 样本无有效观测，使用预先固定的训练集先验；融合至少有其他模态，故无全模态空观测。", "",
              "最优轮次大多在第 1 至 2 轮（视觉有一次第 8 轮），随后达到提前停止条件。这提示当前简单池化模型易迅速过拟合；下一阶段的缺失模态鲁棒训练应继续报告三种子分布、Neutral 指标和强度指标，并单独对照文本基线。", "",
              "验收边界：本脚本从保存的预测复算指标，验证 24 个预测文件、12 个检查点与训练记录的指纹和对应关系，没有重新执行模型推理。训练脚本记录的检查点重载预测一致性保留为历史证据。官方单个 PKL 由现有加载器做结构检查后立即丢弃 test 对象；不计算 test 性能，不读取专项样本。", "",
              f"数据 SHA-256：`{result['source_sha256']}`；标准化包 SHA-256：`{result['normalization_sha256']}`。机器可读核查见同名 JSON，原始逐次记录在 `q2-baselines-v1.0`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Recompute audit and compare existing reports without writing")
    args = parser.parse_args()
    result = audit()
    outputs = {Path(f"{REPORT}.json"):
               json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
               Path(f"{REPORT}.md"): render_md(result)}
    for path, content in outputs.items():
        if args.check:
            require(path.read_text(encoding="utf-8") == content,
                    f"Audit export mismatch: {path}")
        else:
            path.write_text(content, encoding="utf-8")
    print(f"Q2 baseline audit verified: {result['runs']} train/valid runs; "
          f"reports {'checked' if args.check else 'written'}")


if __name__ == "__main__":
    main()
