# 问题一人工复核附加层 v1.0

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
