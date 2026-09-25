"""Create a non-destructive, evidence-scoped human-QC companion for 100 Q1 NPZ.

No source feature, label, sample, word, or timestamp is changed. --check is
read-only. The review log is a human observation record, not an instruction file.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE_REL = Path("02_Drafts/e/mosei-multimodal-aligned-v1.0")
LOG_REL = Path("03_Results/e/question-one/2026-09-25_q1_manual-review-log_v1.0.tsv")
TABLE_REL = Path("03_Results/e/question-one/2026-09-24_q1_sample-summary_v1.0.tsv")
OUTPUT_REL = Path("03_Results/e/question-one/q1-human-qc-overlay-v1.0")
SCHEMA = "q1_human_qc_overlay_v1.0"
EXPECTED_REVIEWED = {"sample_0010", "sample_0011", "sample_0042", "sample_0043", "sample_0088"}
MASKS = ("timing", "text", "audio", "vision", "all_modalities")


def sha(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def json_text(value) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"


def table_text(rows: list[dict]) -> str:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def readme() -> str:
    return r'''# 问题一人工复核附加层 v1.0

本目录只为既有 100 条 NPZ 添加质量状态和使用掩码，不替换原始数据，也不改写文本、情感标签、特征、词数或时间戳。原文件保留在 `02_Drafts/e/mosei-multimodal-aligned-v1.0/`。本目录不涉及问题二/三的官方特征、训练或预测。

## 人工证据与结论边界

唯一人工证据来源是随附 `manual_review_evidence.tsv`（原日志的逐字节副本）。用户播放观察指出 `sample_0010/0011/0042/0043/0088` 没有人说话；其中两条日志还记录了数字静音检测。这里将其解释为“该次人工观察不能支持这些原转写词的语音边界”，不据此改写原转写、划定新的时间或给情感重新打标签。

这 5 条的全部原始词标记为 `unsupported_for_speech`。其余 95 条在**本附加层证据范围内**是 `not_adjudicated_in_this_overlay`，绝不是自动判定已人工合格。其他地方存在的样例粗粒度听审记录不等于逐词边界真值，本层也不覆盖、否认那些记录。没有说话不等于没有环境声音；检测到脸也不证明这张脸正在说话。

## 文件

- `masks/sample_XXXX.npz`：与原 NPZ 一一对应，保持原词顺序；包括原自动 mask 的副本、人工状态和派生使用 mask。
- `sample_quality.tsv`：100 条样本质量状态、词数、源文件 SHA-256、人工证据字段。
- `summary.json`：严格按下述定义计算的数量；**所有数量都不是对齐准确率**。
- `provenance.json`：原 100 NPZ、原统计表、人工日志、生成脚本与本目录输出的 SHA-256。
- `manual_review_evidence.tsv`：本版本依赖的人工日志快照。

## 必须分开的三类概念

1. `automatic_*_mask`：完全复制原文件，表示原算法的可用性判定，不能当成人工语音/时间确认。
2. `human_timing_support_state`：逐词三态，`-1=未在本层裁定`、`0=人工观察不支持语音时间`、`1=人工明确支持`。当前证据只产生 0 和 -1。
3. 使用策略：
   - `review_filtered_timing_candidate_mask = automatic_timing_mask & (human_timing_support_state != 0)`：保留尚无人工反证的自动**候选**；不会使剩余候选变成人工真值。
   - `review_filtered_audio_word_candidate_mask` / `review_filtered_vision_word_candidate_mask`：原模态 mask 与上述时间候选 mask 的交集。
   - `review_filtered_multimodal_candidate_mask`：原三模态同时可用 mask 与上述时间候选 mask 的交集。
   - `strict_human_confirmed_timing_mask = automatic_timing_mask & (human_timing_support_state == 1)`：只允许正向人工时间确认；本证据集没有这种确认，因此当前全为 False。

另外保存 `timing_support_known_mask = state != -1` 和 `timing_supported_mask = state == 1`；后者的 False 既可能未知，也可能明确不支持，**必须结合前者或三态字段解释**。`source_exact_word_boundaries_manually_verified` 只转存旧文件自带状态，不扩展为新证据。

## 下游怎么使用

允许展示自动候选的工具，应使用 `review_filtered_timing_candidate_mask` 控制词时间高亮/跳转，并清楚标注“自动候选，未逐词人工确认”。5 条无说话观察样本的语音词时间全部关闭；仍可播放完整原视频、查看原转写及原特征，并提示不支持词级语音证据。视觉帧和环境声音特征可以作为原始信号保留，但不能再凭这些词时间宣称与口语对应。

只接受人工核实时间的统计，应使用 `strict_human_confirmed_timing_mask`，不能使用“未被否定”代替“已被证实”。原有自动复核标记仍需显示；本层不会自动关闭它们。

```python
from pathlib import Path
import hashlib
import numpy as np

root = Path(r"D:\01_Projects\HuaweiCup-2026")  # 改成项目或克隆根目录
sample = "sample_0010"
original = root / "02_Drafts/e/mosei-multimodal-aligned-v1.0" / f"{sample}.npz"
companion = root / "03_Results/e/question-one/q1-human-qc-overlay-v1.0/masks" / f"{sample}.npz"
with np.load(original, allow_pickle=False) as data, np.load(companion, allow_pickle=False) as qc:
    with original.open("rb") as stream:
        assert hashlib.file_digest(stream, "sha256").hexdigest() == str(qc["source_npz_sha256"].item())
    np.testing.assert_array_equal(data["word_indices"], qc["word_indices"])
    np.testing.assert_array_equal(data["words"], qc["words"])
    candidate = qc["review_filtered_timing_candidate_mask"]
    # 仅产生显示/计算视图，不写回 data，更不删除原样本。
    display_times = data["word_times_s"].copy()
    display_times[~candidate] = np.nan
    audio_candidate_mask = qc["review_filtered_audio_word_candidate_mask"]
    strict_confirmed = qc["strict_human_confirmed_timing_mask"]
```

## 重建、验证与撤回

在 PowerShell 中，Python 环境已有 NumPy，无须新增安装：

```powershell
$py = 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe'
$script = 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-25_build-q1-human-qc-overlay_v1.0.py'
& $py -X utf8 $script --check
```

`--check` 逐项重算状态、检查所有输入/输出哈希，完全只读。首次创建使用 `--build`，已存在目录会拒绝覆盖；需要试重建时加 `--output <新目录>`。新增人工记录必须另建新版本，不能修改本版本证据后悄悄复用已有掩码。

撤回本层只需让消费程序停止读取本目录，改回原自动候选视图；不需要还原原始文件，因为从未修改。撤回会重新显示已知不可靠的时间候选，必须同时保留相关警告。本脚本不提供删除原数据的操作。
'''


def expected_inputs(root: Path):
    log_path, table_path = root / LOG_REL, root / TABLE_REL
    evidence = read_tsv(log_path)
    by_review = {row["sample_id"]: row for row in evidence}
    if len(by_review) != len(evidence) or set(by_review) != EXPECTED_REVIEWED:
        raise RuntimeError("Review log IDs changed: use a new overlay version after inspecting new evidence")
    for row in evidence:
        if row["human_observation"] != "no_person_speaking" or row["timing_status"] != "unsupported_for_speech":
            raise RuntimeError("This overlay version supports only the explicitly recorded no-speech observations")
    old_table = read_tsv(table_path)
    by_sample = {row["sample"]: row for row in old_table}
    expected = {f"sample_{i:04d}" for i in range(1, 101)}
    source = root / SOURCE_REL
    files = sorted(source.glob("*.npz"))
    if set(by_sample) != expected or {p.stem for p in files} != expected or len(files) != 100:
        raise RuntimeError("Expected exactly the unchanged 100 source samples and table rows")
    hashes = {LOG_REL.as_posix(): sha(log_path), TABLE_REL.as_posix(): sha(table_path)}
    entries, totals, sample_rows = [], {f"automatic_{m}_words": 0 for m in MASKS}, []
    totals.update({"source_words": 0, "human_timing_unsupported_words": 0,
                   "human_timing_unknown_words": 0, "human_timing_supported_words": 0,
                   "review_filtered_timing_candidate_words": 0,
                   "review_filtered_audio_word_candidate_words": 0,
                   "review_filtered_vision_word_candidate_words": 0,
                   "review_filtered_multimodal_candidate_words": 0,
                   "strict_human_confirmed_timing_words": 0})
    unsupported_counts = {}
    for path in files:
        sample = path.stem
        digest = sha(path)
        hashes[path.relative_to(root).as_posix()] = digest
        if by_sample[sample]["merged_feature_sha256"] != digest:
            raise RuntimeError(f"{sample}: original feature differs from its archived Q1 summary hash")
        with np.load(path, allow_pickle=False) as z:
            n = int(z["sequence_length"].item())
            if str(z["sample_id"].item()) != sample or z["words"].shape != (n,):
                raise RuntimeError(f"{sample}: source identity/word shape mismatch")
            masks = {m: z[f"{m}_mask"].copy() for m in MASKS}
            if any(a.shape != (n,) or a.dtype != np.bool_ for a in masks.values()):
                raise RuntimeError(f"{sample}: invalid source mask schema")
            for name, width in (("text", 768), ("audio", 25), ("vision", 49)):
                if z[name].shape != (n, width):
                    raise RuntimeError(f"{sample}: unexpected {name} feature shape")
            if z["word_indices"].shape != (n,) or z["word_times_s"].shape != (n, 2):
                raise RuntimeError(f"{sample}: invalid word identity/time schema")
            reviewed = sample in by_review
            state = np.full(n, 0 if reviewed else -1, dtype=np.int8)
            candidate = masks["timing"] & (state != 0)
            payload = {
                "schema": np.asarray(SCHEMA), "sample_id": np.asarray(sample),
                "source_npz_sha256": np.asarray(digest),
                "source_npz_relative_path": np.asarray(path.relative_to(root).as_posix()),
                "sequence_length": np.asarray(n, dtype=np.int64),
                "words": z["words"].copy(), "word_indices": z["word_indices"].copy(),
                "automatic_word_times_s": z["word_times_s"].copy(),
                **{f"automatic_{m}_mask": a for m, a in masks.items()},
                "human_timing_support_state": state,
                "timing_support_known_mask": state != -1,
                "timing_supported_mask": state == 1,
                "review_filtered_timing_candidate_mask": candidate,
                "review_filtered_audio_word_candidate_mask": masks["audio"] & candidate,
                "review_filtered_vision_word_candidate_mask": masks["vision"] & candidate,
                "review_filtered_multimodal_candidate_mask": masks["all_modalities"] & candidate,
                "strict_human_confirmed_timing_mask": masks["timing"] & (state == 1),
                "human_review_status": np.asarray("unsupported_for_speech" if reviewed else "not_adjudicated_in_this_overlay"),
                "manual_evidence_present": np.asarray(reviewed),
                "source_needs_review": z["needs_review"].copy(),
                "source_alignment_review_flags": z["alignment_review_flags"].copy(),
                "source_merge_review_flags": z["merge_review_flags"].copy(),
                "source_exact_word_boundaries_manually_verified": z["exact_word_boundaries_manually_verified"].copy(),
            }
            record = by_review.get(sample, {})
            row = {"sample_id": sample, "source_words": n,
                   "source_npz_sha256": digest, "human_review_status": payload["human_review_status"].item(),
                   "manual_evidence_present": str(reviewed).lower(),
                   "manual_review_date": record.get("review_date", ""),
                   "manual_human_observation": record.get("human_observation", ""),
                   "manual_timing_status": record.get("timing_status", "not_adjudicated_in_this_overlay"),
                   "manual_audio_status": record.get("audio_status", "not_adjudicated_in_this_overlay"),
                   "manual_vision_status": record.get("vision_status", "not_adjudicated_in_this_overlay"),
                   "manual_note": record.get("reviewer_note", ""),
                   "original_needs_review": str(bool(z["needs_review"].item())).lower(),
                   "source_exact_word_boundaries_manually_verified": str(bool(z["exact_word_boundaries_manually_verified"].item())).lower(),
                   **{f"automatic_{m}_words": int(a.sum()) for m, a in masks.items()},
                   "human_timing_unsupported_words": int((state == 0).sum()),
                   "human_timing_unknown_words": int((state == -1).sum()),
                   "human_timing_supported_words": int((state == 1).sum()),
                   "review_filtered_timing_candidate_words": int(candidate.sum()),
                   "review_filtered_audio_word_candidate_words": int(payload["review_filtered_audio_word_candidate_mask"].sum()),
                   "review_filtered_vision_word_candidate_words": int(payload["review_filtered_vision_word_candidate_mask"].sum()),
                   "review_filtered_multimodal_candidate_words": int(payload["review_filtered_multimodal_candidate_mask"].sum()),
                   "strict_human_confirmed_timing_words": int(payload["strict_human_confirmed_timing_mask"].sum()),
                   "original_features_retained": "true"}
            for key in totals:
                totals[key] += row[key]
            if reviewed:
                unsupported_counts[sample] = n
            entries.append((sample, payload))
            sample_rows.append(row)
    summary = {"schema": SCHEMA, "evidence_date": "2026-09-25", "samples": len(entries),
               "source_sample_count_unchanged": True, "original_feature_files_modified": 0,
               "manual_unsupported_samples": len(by_review),
               "not_adjudicated_in_this_overlay_samples": len(entries) - len(by_review),
               "positively_timing_supported_samples_in_this_overlay": 0,
               "unsupported_word_counts_by_sample": unsupported_counts, "word_counts": totals,
               "count_semantics": "Availability and review-filter counts only; not alignment accuracy. Unknown does not mean verified.",
               "review_scope": "Only the five no-person-speaking observations in manual_review_evidence.tsv; no new listening, boundaries, or emotion labels.",
               "existing_statistics_policy": "Original Q1 summary and all 100 original NPZ retained byte-for-byte; new counts are a separate companion policy.",
               "word_state_values": {"-1": "not_adjudicated_in_this_overlay", "0": "unsupported_for_speech", "1": "positively_supported_by_human"},
               "derivation": {"candidate": "automatic_timing AND human_state != 0", "strict_confirmed": "automatic_timing AND human_state == 1"}}
    return hashes, entries, sample_rows, summary


def verify(root: Path, output: Path) -> dict:
    hashes, entries, rows, summary = expected_inputs(root)
    manifest = json.loads((output / "provenance.json").read_text(encoding="utf-8"))
    if manifest["source_sha256"] != hashes or manifest["generator_sha256"] != sha(Path(__file__).resolve()):
        raise RuntimeError("Input or generator fingerprint changed; do not silently reuse this overlay")
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    if actual != set(manifest["output_sha256"]) | {"provenance.json"}:
        raise RuntimeError("Unexpected or missing overlay file")
    for relative, digest in manifest["output_sha256"].items():
        if sha(output / relative) != digest:
            raise RuntimeError(f"Overlay hash changed: {relative}")
    if json.loads((output / "summary.json").read_text(encoding="utf-8")) != summary:
        raise RuntimeError("Summary disagrees with independently rebuilt source/evidence state")
    if (output / "sample_quality.tsv").read_text(encoding="utf-8-sig") != table_text(rows):
        raise RuntimeError("Sample table differs from evidence")
    if (output / "manual_review_evidence.tsv").read_bytes() != (root / LOG_REL).read_bytes():
        raise RuntimeError("Evidence snapshot differs from original log")
    if (output / "README.md").read_text(encoding="utf-8") != readme():
        raise RuntimeError("Usage definitions differ from generator")
    for sample, payload in entries:
        with np.load(output / "masks" / f"{sample}.npz", allow_pickle=False) as z:
            if set(z.files) != set(payload):
                raise RuntimeError(f"{sample}: companion fields differ")
            for key, expected in payload.items():
                np.testing.assert_array_equal(z[key], expected, err_msg=f"{sample}/{key}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--build", action="store_true")
    action.add_argument("--check", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / OUTPUT_REL
    if args.build:
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite overlay: {output}")
        hashes, entries, rows, summary = expected_inputs(root)
        output.mkdir(parents=True)
        (output / "masks").mkdir()
        for sample, payload in entries:
            np.savez_compressed(output / "masks" / f"{sample}.npz", **payload)
        (output / "manual_review_evidence.tsv").write_bytes((root / LOG_REL).read_bytes())
        (output / "sample_quality.tsv").write_text(table_text(rows), encoding="utf-8-sig", newline="")
        (output / "summary.json").write_text(json_text(summary), encoding="utf-8", newline="")
        (output / "README.md").write_text(readme(), encoding="utf-8", newline="")
        if {relative: sha(root / relative) for relative in hashes} != hashes:
            raise RuntimeError("Source files changed while building; inspect partial overlay before reuse")
        manifest = {"schema": SCHEMA, "generator": "02_Drafts/e/src/" + Path(__file__).name,
                    "generator_sha256": sha(Path(__file__).resolve()), "source_sha256": hashes,
                    "output_sha256": {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob("*")) if p.is_file()},
                    "preservation": "Read-only original inputs; overlays are consumed explicitly and can be disabled without restoring originals."}
        (output / "provenance.json").write_text(json_text(manifest), encoding="utf-8", newline="")
    summary = verify(root, output)
    print(json_text({"output": str(output), "samples": summary["samples"],
                     "unsupported_samples": summary["manual_unsupported_samples"],
                     "counts": summary["word_counts"]}), end="")
    print("Q1 human-QC overlay verified: 100 sources unchanged; no labels/features/timestamps overwritten")


if __name__ == "__main__":
    main()
