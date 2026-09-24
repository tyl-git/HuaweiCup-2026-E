"""Evaluate frozen Q2 fusion baselines under fixed contiguous validation gaps.

This is a validation-only robustness audit.  It reloads the existing fusion
checkpoints, applies deterministic contiguous masks to valid features before
pooling, and never trains or evaluates the official test/specialist splits.
"""

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
BASELINES = RESULTS / "q2-baselines-v1.0"
OUT = RESULTS / "q2-missing-baseline-v1.0"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
TEXT_CACHE = RESULTS / "bert-cache-v1.0" / "valid.npy"
SEEDS = (20260924, 20260925, 20260926)
FRACTIONS = (0.2, 0.4)
POSITIONS = ("start", "middle", "end")
MODALITIES = ("text", "audio", "vision")


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pool_normalized(features, masks, content_mask):
    """Mask-aware mean pooling for already standardized arrays."""
    pooled, available = {}, {}
    for modality, values in features.items():
        mask = np.asarray(masks[modality], dtype=bool) & np.asarray(content_mask, dtype=bool)
        counts = mask.sum(axis=1, dtype=np.int32)
        numerators = np.einsum("nt,ntd->nd", mask, values, dtype=np.float64)
        result = (numerators / np.maximum(counts[:, None], 1)).astype(np.float32)
        result[counts == 0] = 0
        pooled[modality] = result
        available[modality] = counts > 0
        if not np.isfinite(result).all():
            raise RuntimeError(f"Nonfinite pooled features for {modality}")
    return pooled, available


def checkpoint_model(path, baseline, torch, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    config = baseline.TrainingConfig(**saved["config"])
    model = baseline.build_model(tuple(saved["modalities"]), saved["train_priors"], config, torch)
    model.load_state_dict(saved["model_state"])
    model.to(device).eval()
    return model, saved


def condition_key(modality: str, fraction: float, position: str) -> str:
    return f"{modality}_missing_{int(round(fraction * 100)):02d}pct_{position}"


def render_report(metadata, rows, aggregates):
    lines = [
        "# Q2 验证集连续局部缺失鲁棒性审计",
        "",
        "本报告重新评估已经保存的三模态 fusion baseline checkpoint，不重新训练模型。加载官方单个 PKL 后只保留 valid，不评价 test 或附件三/四。缺失区间在固定 50 个序列位置中的内容坐标上构造，比例以该模态原本有观测的位置为分母；同一条件和 sample ID 对三个训练种子使用完全相同的区间计划。",
        "",
        f"完整输入基准：{metadata['valid_samples']} 条 valid 样本，fusion seeds={', '.join(map(str, SEEDS))}。固定缺失计划种子：{metadata['plan_seed']}。",
        "",
        "| 缺失模态 | 比例 | 区间 | Accuracy | macro-F1 | MAE | Pearson r | ΔAccuracy | Δmacro-F1 | ΔMAE | Δr |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in aggregates:
        def fmt(value):
            return "undefined" if value is None else f"{value:.4f}"
        lines.append(
            f"| {item['modality']} | {item['fraction']:.1f} | {item['position']} | "
            f"{fmt(item['accuracy_mean'])}±{item['accuracy_std']:.4f} | "
            f"{fmt(item['macro_f1_mean'])}±{item['macro_f1_std']:.4f} | "
            f"{fmt(item['mae_mean'])}±{item['mae_std']:.4f} | "
            f"{fmt(item['pearson_mean'])}±{item['pearson_std']:.4f} | "
            f"{item['delta_accuracy_mean']:+.4f} | {item['delta_macro_f1_mean']:+.4f} | "
            f"{item['delta_mae_mean']:+.4f} | {fmt(item['delta_pearson_mean'])} |"
        )
    lines += [
        "",
        "完整输入与缺失输入都按同一组验证标签计算三分类 Accuracy、固定三类 macro-F1、强度 MAE 和 Pearson r。表中均值±样本标准差（ddof=1）跨三个训练种子，并非置信区间；Δ 是每个种子相对于其自身完整输入的配对差，正的 ΔMAE 表示误差增加。每条样本删除位置数向上取整，因此实际总体比例高于目标比例，详见计划 summary。",
        "",
        "这里模拟的是特征行缺失。文本在完整句子经过冻结 BERT 编码后遮掉部分向量，剩余上下文向量可能仍含被遮词的信息；结果不等价于原始转写缺失，也不能作为因果模态贡献。音视频局部缺失影响小，可能与均值池化及模型更依赖文本有关，不能据此推断音视频没有情感信息。",
        "",
        "本轮只描述现有简单融合基线在遮挡下的性能变化，尚未宣称缺失模态鲁棒训练已经完成。后续鲁棒模型应在训练集施加同协议增强，并在相同计划下比较。",
        "",
        f"结果 JSON：`{metadata['result_file']}`；缺失计划：`{metadata['plan_file']}`。",
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--plan-seed", type=int, default=20260930)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--refresh", action="store_true", help="Regenerate these derived evaluation outputs")
    group.add_argument("--check", action="store_true", help="Recompute and compare without writing")
    args = parser.parse_args()
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if OUT.exists() and not (args.refresh or args.check):
        raise FileExistsError(f"Prior evaluation exists; use --check: {OUT}")

    loader = load_module("q2_missing_loader", "2026-09-24_q2_data_v1.0.py")
    norm = load_module("q2_missing_norm", "2026-09-24_q2-normalization_v1.0.py")
    missing = load_module("q2_missing_intervals", "2026-09-24_q2-missing-intervals_v1.0.py")
    baseline = load_module("q2_missing_baseline", "2026-09-24_train-q2-baselines_v1.0.py")
    metrics = load_module("q2_missing_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    provenance = json.loads((BASELINES / "provenance.json").read_text(encoding="utf-8"))
    for filename, expected in provenance["dependency_sha256"].items():
        if sha256(SRC / filename) != expected:
            raise RuntimeError(f"Baseline dependency changed: {filename}")
    if sha256(STATS) != provenance["normalization_sha256"]:
        raise RuntimeError("Baseline normalization changed")
    if sha256(TEXT_CACHE) != provenance["bert_cache_sha256"]["valid"]:
        raise RuntimeError("Baseline valid BERT cache changed")
    baseline.set_seed(SEEDS[0], torch)

    batches = loader.load_official()
    valid = batches.pop("valid")
    del batches
    if set(valid.source_sha256.values()) != {provenance["official_source_sha256"]}:
        raise RuntimeError("Baseline official input changed")
    standardizer = norm.MultimodalStandardizer.load(STATS)
    text = np.load(TEXT_CACHE, mmap_mode="r", allow_pickle=False)
    raw = {"text": text, "audio": valid.audio, "vision": valid.vision}
    normalized = standardizer.transform(raw, valid.content_mask, valid.observation_masks)
    complete_features = dict(normalized.features)
    complete_masks = dict(normalized.observation_masks)
    complete_pool, complete_available = pool_normalized(
        complete_features, complete_masks, valid.content_mask)

    plan_records = []
    plans = {}
    for modality in MODALITIES:
        for fraction in FRACTIONS:
            for position in POSITIONS:
                key = condition_key(modality, fraction, position)
                plan = missing.make_interval_plan(
                    valid.content_mask, valid.observation_masks, valid.ids,
                    modalities=(modality,), fraction=fraction, position=position,
                    seed=args.plan_seed)
                plans[key] = plan
                plan_records.append({
                    "key": key, "modality": modality, "fraction": fraction,
                    "position": position, "summary": plan.summary(),
                    "intervals": plan.intervals[modality].tolist(),
                })

    condition_data = {}
    for record in plan_records:
        key = record["key"]
        masked = missing.apply_interval_plan(
            complete_features, valid.content_mask, complete_masks,
            valid.ids, plans[key])
        pooled, available = pool_normalized(
            masked.features, masked.observation_masks, valid.content_mask)
        condition_data[key] = (pooled, available)

    results = []
    predictions = {}
    device = torch.device(args.device)
    for seed in SEEDS:
        checkpoint = BASELINES / "fusion" / f"seed_{seed}" / "best.pt"
        recorded = json.loads(checkpoint.with_name("metrics.json").read_text(encoding="utf-8"))
        if sha256(checkpoint) != recorded["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch for {seed}")
        model, saved = checkpoint_model(checkpoint, baseline, torch, device)
        if saved["normalization_sha256"] != provenance["normalization_sha256"] or saved["seed"] != seed:
            raise RuntimeError(f"Checkpoint contract mismatch for {seed}")
        complete_tensors = baseline.tensors_for(
            valid, complete_pool, complete_available, ("text", "audio", "vision"), torch, device)
        complete_report, complete_predictions = baseline.evaluate(model, complete_tensors, baseline.TrainingConfig(**saved["config"]), torch, metrics, return_predictions=True)
        with np.load(checkpoint.with_name("valid_predictions.npz"), allow_pickle=False) as previous:
            np.testing.assert_array_equal(previous["ids"], np.asarray(valid.ids))
            for field, values in complete_predictions.items():
                np.testing.assert_allclose(values, previous[field], atol=1e-6, rtol=1e-6)
        predictions[f"seed_{seed}_complete.npz"] = complete_predictions
        for record in plan_records:
            pooled, available = condition_data[record["key"]]
            tensors = baseline.tensors_for(
                valid, pooled, available, ("text", "audio", "vision"), torch, device)
            report, masked_predictions = baseline.evaluate(model, tensors, baseline.TrainingConfig(**saved["config"]), torch, metrics, return_predictions=True)
            predictions[f"seed_{seed}_{record['key']}.npz"] = masked_predictions
            c = complete_report["classification"]
            m = report["classification"]
            cr = complete_report["regression"]
            mr = report["regression"]
            results.append({
                "model": "fusion", "seed": seed, "condition": record["key"],
                "modality": record["modality"], "fraction": record["fraction"],
                "position": record["position"], "n": int(m["n"]),
                "complete": complete_report, "masked": report,
                "delta": {
                    "accuracy": float(m["accuracy"] - c["accuracy"]),
                    "macro_f1": float(m["macro_f1"] - c["macro_f1"]),
                    "mae": float(mr["mae"] - cr["mae"]),
                    "pearson_r": (None if mr["pearson_r"] is None or cr["pearson_r"] is None
                                  else float(mr["pearson_r"] - cr["pearson_r"])),
                },
                "checkpoint_sha256": sha256(checkpoint),
            })
        print(f"seed {seed}: evaluated complete + {len(plan_records)} missing conditions", flush=True)

    aggregate = []
    for record in plan_records:
        rows = [row for row in results if row["condition"] == record["key"]]
        def vals(path):
            value = []
            for row in rows:
                current = row
                for part in path:
                    current = current[part]
                if current is not None:
                    value.append(float(current))
            return np.asarray(value, dtype=np.float64)
        def summary(path):
            value = vals(path)
            return (float(value.mean()), float(value.std(ddof=1))) if len(value) > 1 else (None, None)
        am, ass = summary(("masked", "classification", "accuracy"))
        fm, fss = summary(("masked", "classification", "macro_f1"))
        mm, mss = summary(("masked", "regression", "mae"))
        pm, pss = summary(("masked", "regression", "pearson_r"))
        dam, dass = summary(("delta", "accuracy"))
        dfm, dfss = summary(("delta", "macro_f1"))
        dmm, dmss = summary(("delta", "mae"))
        dpm, dpss = summary(("delta", "pearson_r"))
        aggregate.append({
            "condition": record["key"], "modality": record["modality"],
            "fraction": record["fraction"], "position": record["position"],
            "accuracy_mean": am, "accuracy_std": ass,
            "macro_f1_mean": fm, "macro_f1_std": fss,
            "mae_mean": mm, "mae_std": mss,
            "pearson_mean": pm, "pearson_std": pss,
            "delta_accuracy_mean": dam, "delta_accuracy_std": dass,
            "delta_macro_f1_mean": dfm, "delta_macro_f1_std": dfss,
            "delta_mae_mean": dmm, "delta_mae_std": dmss,
            "delta_pearson_mean": dpm, "delta_pearson_std": dpss,
        })

    if not args.check:
        OUT.mkdir(parents=True, exist_ok=args.refresh)
    plan_path = OUT / "2026-09-24_q2-missing-plan_v1.0.json"
    result_path = OUT / "2026-09-24_q2-missing-baseline-results_v1.0.json"
    report_path = OUT / "2026-09-24_q2-missing-baseline-results_v1.0.md"
    plan_payload = {
        "schema": "q2_contiguous_missing_plan_v1.0", "split": "valid",
        "sample_ids": list(valid.ids), "seed": args.plan_seed,
        "fractions": list(FRACTIONS), "positions": list(POSITIONS),
        "modalities": list(MODALITIES), "conditions": plan_records,
        "original_masks": {m: valid.observation_masks[m].tolist() for m in MODALITIES},
    }
    plan_text = json.dumps(plan_payload, ensure_ascii=False, indent=2) + "\n"
    prediction_hashes = {}
    for filename, values in predictions.items():
        path = OUT / filename
        if args.check:
            with np.load(path, allow_pickle=False) as saved_predictions:
                np.testing.assert_array_equal(saved_predictions["ids"], np.asarray(valid.ids))
                for field, array in values.items():
                    np.testing.assert_allclose(saved_predictions[field], array, atol=1e-6, rtol=1e-6)
        else:
            np.savez_compressed(path, ids=np.asarray(valid.ids), **values)
        prediction_hashes[filename] = sha256(path)
    metadata = {
        "schema": "q2_missing_baseline_results_v1.0", "split": "valid",
        "valid_samples": len(valid), "plan_seed": args.plan_seed,
        "conditions": len(plan_records), "seeds": list(SEEDS), "model": "fusion",
        "official_source_sha256": dict(valid.source_sha256),
        "normalization_sha256": sha256(STATS), "bert_cache_sha256": sha256(TEXT_CACHE),
        "plan_file": str(plan_path), "result_file": str(result_path),
        "test_evaluated": False, "specialists_evaluated": False,
        "sample_standard_deviation_ddof": 1,
        "device": args.device, "torch": torch.__version__, "numpy": np.__version__,
        "baseline_complete_predictions_reproduced": True,
        "prediction_files_sha256": prediction_hashes,
        "plan_sha256": hashlib.sha256(plan_text.encode("utf-8")).hexdigest(),
        "dependency_sha256": {name: sha256(SRC / name) for name in (
            Path(__file__).name, "2026-09-24_q2_data_v1.0.py",
            "2026-09-24_q2-normalization_v1.0.py",
            "2026-09-24_q2-missing-intervals_v1.0.py",
            "2026-09-24_train-q2-baselines_v1.0.py",
            "2026-09-24_q2-evaluation_v1.0.py")},
        "checkpoint_sha256": {str(seed): sha256(BASELINES / "fusion" / f"seed_{seed}" / "best.pt") for seed in SEEDS},
    }
    payload = {"metadata": metadata, "aggregate": aggregate, "per_seed": results}
    outputs = {plan_path: plan_text,
               result_path: json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
               report_path: render_report(metadata, results, aggregate)}
    for path, content in outputs.items():
        if args.check:
            if path.read_text(encoding="utf-8") != content:
                raise RuntimeError(f"Evaluation export mismatch: {path}")
        else:
            path.write_text(content, encoding="utf-8")
    print(f"Saved: {result_path}")
    print(f"Saved: {report_path}")
    print(f"Saved: {plan_path}")


if __name__ == "__main__":
    main()
