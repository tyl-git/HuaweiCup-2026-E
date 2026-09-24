"""Read-only audit of the official aligned_50.pkl and label.xlsx."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "01_Source" / "E" / "E题数据" / "附件2-数据集特征文件"
OUTPUT = ROOT / "03_Results" / "e" / "question-two"
SPLITS = ("train", "valid", "test")
MODALITIES = {"text": 768, "audio": 74, "vision": 35}
EXPECTED_KEYS = {"raw_text", "audio", "vision", "id", "text_bert", "classification_labels", "regression_labels", "text"}
TOKENIZER = Path(r"D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")


def small_examples(values: list, maximum: int = 10) -> list:
    return values[:maximum]


def inspect_split(name: str, data: dict, spreadsheet_rows: dict, tokenizer) -> tuple[dict, list]:
    n = len(data["id"])
    if set(data) != EXPECTED_KEYS:
        raise ValueError(f"Unexpected keys in {name}: {set(data)}")
    for key, dim in MODALITIES.items():
        if data[key].shape != (n, 50, dim):
            raise ValueError(f"Unexpected shape for {name}/{key}: {data[key].shape}")
    for key in ("raw_text", "classification_labels", "regression_labels"):
        if data[key].shape != (n,):
            raise ValueError(f"Unexpected shape for {name}/{key}: {data[key].shape}")
    if data["text_bert"].shape != (n, 3, 50):
        raise ValueError(f"Unexpected text_bert shape in {name}")
    ids = [str(item) for item in data["id"]]
    attention = np.asarray(data["text_bert"][:, 1, :])
    token_ids = np.asarray(data["text_bert"][:, 0, :])
    token_types = np.asarray(data["text_bert"][:, 2, :])
    active = attention == 1
    special = active & np.isin(token_ids, [101, 102])
    content = active & ~special
    lengths = active.sum(axis=1)
    expected_attention = np.arange(50)[None, :] < lengths[:, None]
    classification = np.asarray(data["classification_labels"])
    regression = np.asarray(data["regression_labels"])
    label_cross = Counter((int(c), int(np.sign(r))) for c, r in zip(classification, regression))
    excel_mismatches = []
    excel_missing = []
    records = []
    for i, sample_id in enumerate(ids):
        row = spreadsheet_rows.get((name, sample_id))
        if row is None:
            excel_missing.append(sample_id)
        else:
            excel_text, excel_label, annotation = row
            mismatch = []
            if str(data["raw_text"][i]) != excel_text:
                mismatch.append("raw_text")
            if not np.isclose(regression[i], excel_label, atol=1e-6, rtol=0):
                mismatch.append("regression_label")
            inferred_annotation = "Positive" if regression[i] > 0 else "Negative" if regression[i] < 0 else "Neutral"
            if annotation != inferred_annotation:
                mismatch.append("annotation")
            if mismatch:
                excel_mismatches.append({"id": sample_id, "fields": mismatch})
        records.append((name, sample_id, str(data["raw_text"][i])))

    report = {
        "count": n,
        "keys": sorted(data),
        "missing_expected_keys": sorted(EXPECTED_KEYS - set(data)),
        "extra_keys": sorted(set(data) - EXPECTED_KEYS),
        "ids_unique": len(set(ids)),
        "duplicate_ids_within_split": small_examples([key for key, count in Counter(ids).items() if count > 1]),
        "raw_text_unique": len(set(str(x) for x in data["raw_text"])),
        "raw_text_empty": sum(not str(x).strip() for x in data["raw_text"]),
        "fields": {},
        "bert_tokens": {
            "shape": list(data["text_bert"].shape),
            "dtype": str(data["text_bert"].dtype),
            "attention_values": [int(x) for x in np.unique(attention)],
            "token_type_values": [int(x) for x in np.unique(token_types)],
            "non_prefix_attention_samples": int(np.any(active != expected_attention, axis=1).sum()),
            "length_min": int(lengths.min()),
            "length_median": float(np.median(lengths)),
            "length_max": int(lengths.max()),
            "length_50_count": int((lengths == 50).sum()),
            "content_steps": int(content.sum()),
            "special_steps": int(special.sum()),
            "inactive_token_id_nonzero": int(np.count_nonzero(token_ids[~active])),
            "active_token_id_zero": int(np.count_nonzero(token_ids[active] == 0)),
            "inactive_token_type_nonzero": int(np.count_nonzero(token_types[~active])),
        },
        "labels": {
            "classification_values": {str(int(key)): int(value) for key, value in sorted(Counter(classification).items())},
            "regression_sign_counts": {str(key): int(value) for key, value in sorted(Counter(np.sign(regression)).items())},
            "classification_vs_regression_sign": {f"{key[0]}:{key[1]}": int(value) for key, value in sorted(label_cross.items())},
            "regression_min": float(regression.min()),
            "regression_median": float(np.median(regression)),
            "regression_mean": float(regression.mean()),
            "regression_std": float(regression.std()),
            "regression_quartiles": np.quantile(regression, [0.25, 0.5, 0.75]).tolist(),
            "regression_max": float(regression.max()),
            "nonfinite_classification": int((~np.isfinite(classification)).sum()),
            "nonfinite_regression": int((~np.isfinite(regression)).sum()),
        },
        "excel": {
            "missing_count": len(excel_missing),
            "missing_examples": small_examples(excel_missing),
            "mismatch_count": len(excel_mismatches),
            "mismatch_examples": small_examples(excel_mismatches),
        },
    }
    raw_texts = [str(x) for x in data["raw_text"]]
    raw_encoded = tokenizer(raw_texts, padding=False, truncation=False, verbose=False)
    encoded = tokenizer(raw_texts, max_length=50, padding="max_length", truncation=True, return_tensors="np")
    expected_bert = np.stack([encoded["input_ids"], encoded["attention_mask"], encoded["token_type_ids"]], axis=1)
    raw_lengths = np.array([len(x) for x in raw_encoded["input_ids"]])
    report["bert_tokens"]["tokenizer_reproduction_mismatch_samples"] = int(np.any(data["text_bert"] != expected_bert, axis=(1, 2)).sum())
    report["bert_tokens"]["raw_token_length_min"] = int(raw_lengths.min())
    report["bert_tokens"]["raw_token_length_max"] = int(raw_lengths.max())
    report["bert_tokens"]["raw_token_length_gt_50_count"] = int((raw_lengths > 50).sum())
    report["bert_tokens"]["raw_token_length_eq_50_count"] = int((raw_lengths == 50).sum())
    for key, dim in MODALITIES.items():
        arr = np.asarray(data[key])
        finite = np.isfinite(arr)
        all_zero = np.all(arr == 0, axis=2)
        report["fields"][key] = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "expected_dimension": dim,
            "nonfinite_values": int((~finite).sum()),
            "nonfinite_active_values": int((~finite & active[:, :, None]).sum()),
            "zero_steps_total": int(all_zero.sum()),
            "zero_steps_inside_bert_mask": int((all_zero & active).sum()),
            "zero_steps_at_special_tokens": int((all_zero & special).sum()),
            "zero_steps_at_content_tokens": int((all_zero & content).sum()),
            "zero_steps_outside_bert_mask": int((all_zero & ~active).sum()),
            "nonzero_steps_outside_bert_mask": int((~all_zero & ~active).sum()),
            "samples_all_zero_inside_bert_mask": int(np.all(all_zero | ~active, axis=1).sum()),
            "samples_all_zero_at_content_tokens": int(np.all(all_zero | ~content, axis=1).sum()),
            "samples_all_zero_at_content_tokens_ids": [ids[i] for i in np.flatnonzero(np.all(all_zero | ~content, axis=1))],
            "finite_min": float(arr[finite].min()),
            "finite_max": float(arr[finite].max()),
        }
    return report, records


def render_markdown(result: dict) -> str:
    lines = [
        "# 附件二官方对齐数据审计",
        "",
        "数据：`aligned_50.pkl` 与 `label.xlsx`。原始材料只读；这是数据质量与输入协议审计，不含训练或预测结果。",
        "",
        "| 划分 | 样本 | BERT 长度 min / median / max | 长度为 50 | 分类 0 / 1 / 2 |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in SPLITS:
        split = result["splits"][name]
        token = split["bert_tokens"]
        label = split["labels"]["classification_values"]
        lines.append(
            f"| {name} | {split['count']} | {token['length_min']} / {token['length_median']:g} / {token['length_max']} | "
            f"{token['length_50_count']} | {label.get('0',0)} / {label.get('1',0)} / {label.get('2',0)} |"
        )
    lines += ["", "## 关键检查", ""]
    for name in SPLITS:
        split = result["splits"][name]
        lines.append(
            f"- `{name}`：Excel 缺行 {split['excel']['missing_count']}，原文/数值标签/annotation 不一致 "
            f"{split['excel']['mismatch_count']}；attention 非前缀 {split['bert_tokens']['non_prefix_attention_samples']}。"
        )
        lines.append(
            f"  音频在 BERT 有效位置的全零步 {split['fields']['audio']['zero_steps_inside_bert_mask']}，"
            f"视觉 {split['fields']['vision']['zero_steps_inside_bert_mask']}；"
            f"BERT 无效位置的非零音频步 {split['fields']['audio']['nonzero_steps_outside_bert_mask']}，"
            f"非零视觉步 {split['fields']['vision']['nonzero_steps_outside_bert_mask']}。"
        )
    lines += [
        f"- 跨划分完全相同 ID：{result['cross_split']['exact_id_duplicate_count']}。",
        f"- 跨划分相同视频 ID：{result['cross_split']['base_video_overlap_count']}。此项按 `$_$` 前的视频 ID 核对。",
        f"- 跨划分完全相同原文：{result['cross_split']['raw_text_duplicate_count']} 种：`Alright`、`Okay`。仅是常用短句相同，不能据此认定样本泄漏。",
        f"- Excel 总行数：{result['spreadsheet']['data_rows']}；ID+划分重复：{result['spreadsheet']['duplicate_key_count']}；"
        f"Excel 中未出现在特征包的行：{result['spreadsheet']['unmatched_row_count']}。",
        "",
        "## 使用边界",
        "",
        "`text_bert` 第二维依次含 token ID、attention mask、token type ID；有效长度含 `[CLS]` 和 `[SEP]`，应取 attention mask，不能从 768 维文本嵌入是否为零推断。"
        "全零音频/视觉步只是数值现象，不一定可判定为真正缺失；正式模型应在 train 上拟合标准化参数，valid 用于选择方案，test 留作最终评估。"
        "附件二的 768/74/35 与问题一自提的 768/25/49 维不直接混接。",
        "",
        "分类标签的数值与回归正负号对照、每模态有限值和零步细节见同名 JSON。",
    ]
    lines += ["", "## 特殊 token、截断与有限值", "",
              "全部样本分类映射为 `0=负面`、`1=中性`、`2=正面`，与回归标签正负号一致。"
              "三模态全部数值有限；文本 padding 向量仍非零。以下从内容 token 中剔除了 `[CLS]`、`[SEP]`。",
              "", "| 划分 | 内容 token | 音频全零内容 token | 视觉全零内容 token | 全零视觉样本 | 原文编码超过 50 token |", "|---|---:|---:|---:|---:|---:|"]
    for name in SPLITS:
        s = result["splits"][name]
        lines.append(f"| {name} | {s['bert_tokens']['content_steps']} | {s['fields']['audio']['zero_steps_at_content_tokens']} | "
                     f"{s['fields']['vision']['zero_steps_at_content_tokens']} | {s['fields']['vision']['samples_all_zero_at_content_tokens']} | "
                     f"{s['bert_tokens']['raw_token_length_gt_50_count']} |")
    lines += ["", "本地固定版本 BERT tokenizer 对全部原文重新分词（仅分词，无模型推理）。按照 `max_length=50`、尾部截断、尾部 padding "
              "重建的 token ID、attention mask、token type ID 与原包完全匹配。原文超过 50 token 的样本已截断；长度恰为 50 的样本不一定被截断。",
              "", "另取 train 首条重新运行同一 BERT 最后一层，活动 token 与预计算 `text` 最大绝对差约 6.2e-6；"
              "这一检查只覆盖一条样本，详情见 `2026-09-24_official-bert-comparison_v1.0.json`。",
              "", "此审计没有恢复原始时间戳或重新提取特征，不能据文件名或共同长度证明每个位置的时序对齐精度。",
              "", "## 复跑", "",
              "在项目根目录运行，复用已装主环境和本地 tokenizer，不需要下载或安装。默认只覆盖本脚本的 JSON/MD 两份派生报告；`--check` 只比较现有报告，不写文件。",
              "", "```powershell",
              '& "D:\\06_Apps\\python-envs\\huaweicup-e\\Scripts\\python.exe" -X utf8 "02_Drafts\\e\\src\\2026-09-24_audit-attachment-two_v1.0.py" --check',
              "```", "",
              "校验来源包括特征包 SHA-256、Excel 的 video_id/clip_id/mode 组合键、原文精确比较、标签绝对误差不超过 1e-6、"
              "三模态结构/有限值/全零步、attention mask 与本地 tokenizer 重建对照。"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with existing reports without writing")
    parser.add_argument("--tokenizer", type=Path, default=TOKENIZER, help="local BERT tokenizer snapshot; no network downloads")
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True, use_fast=True)
    workbook = load_workbook(SOURCE / "label.xlsx", read_only=True, data_only=True)
    rows = list(workbook["label"].values)
    workbook.close()
    if rows[0] != ("video_id", "clip_id", "text", "label", "annotation", "mode"):
        raise ValueError(f"Unexpected spreadsheet header: {rows[0]}")
    excel = {}
    duplicate_excel_keys = []
    for video_id, clip_id, text, label, annotation, mode in rows[1:]:
        key = (str(mode), f"{video_id}$_${clip_id}")
        if key in excel:
            duplicate_excel_keys.append(key)
        excel[key] = (str(text), float(label), str(annotation))
    with (SOURCE / "aligned_50.pkl").open("rb") as handle:
        data = pickle.load(handle)
    if sorted(data) != sorted(SPLITS):
        raise ValueError(f"Unexpected splits: {sorted(data)}")
    splits = {}
    records = []
    for name in SPLITS:
        splits[name], subset = inspect_split(name, data[name], excel, tokenizer)
        records.extend(subset)
    id_locations = defaultdict(set)
    video_locations = defaultdict(set)
    text_locations = defaultdict(set)
    for split, sample_id, raw_text in records:
        id_locations[sample_id].add(split)
        video_locations[sample_id.split("$_$")[0]].add(split)
        text_locations[raw_text].add(split)
    overlapping_ids = sorted(k for k, v in id_locations.items() if len(v) > 1)
    overlapping_videos = sorted(k for k, v in video_locations.items() if len(v) > 1)
    overlapping_texts = sorted(k for k, v in text_locations.items() if len(v) > 1)
    unmatched_excel = sorted(set(excel) - {(s, i) for s, i, _ in records})
    source_path = SOURCE / "aligned_50.pkl"
    with source_path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    report = {
        "source": str(source_path.relative_to(ROOT)),
        "source_sha256": digest,
        "tokenizer_snapshot": str(args.tokenizer),
        "splits": splits,
        "cross_split": {
            "exact_id_duplicate_count": len(overlapping_ids),
            "exact_id_examples": small_examples(overlapping_ids),
            "base_video_overlap_count": len(overlapping_videos),
            "base_video_examples": small_examples(overlapping_videos),
            "raw_text_duplicate_count": len(overlapping_texts),
            "raw_text_examples": small_examples(overlapping_texts),
        },
        "spreadsheet": {
            "data_rows": len(rows) - 1,
            "duplicate_key_count": len(duplicate_excel_keys),
            "duplicate_key_examples": small_examples(duplicate_excel_keys),
            "unmatched_row_count": len(unmatched_excel),
            "unmatched_row_examples": small_examples(unmatched_excel),
        },
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    md_text = render_markdown(report)
    paths = {
        OUTPUT / "2026-09-24_attachment-two-audit_v1.0.json": json_text,
        OUTPUT / "2026-09-24_attachment-two-audit_v1.0.md": md_text,
    }
    if args.check:
        for path, expected in paths.items():
            if path.read_text(encoding="utf-8") != expected:
                raise ValueError(f"Report differs from current data: {path}")
        print("Audit reports verified")
    else:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        for path, contents in paths.items():
            path.write_text(contents, encoding="utf-8")
            print(f"Saved: {path}")
    print("Rows:", {name: splits[name]["count"] for name in SPLITS})


if __name__ == "__main__":
    main()
