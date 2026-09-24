# HuaweiCup 2026 E

## 写问题一论文，从这里开始

先读[论文写作指南](02_Drafts/e/2026-09-24_q1-paper-writing-guide_v1.0.md)：它按正文小节列出该讲什么、放什么公式、用哪张图和哪份数据。再读[数学方法稿](02_Drafts/e/2026-09-24_q1-model-method_v1.0.md)与[复现说明](02_Drafts/e/2026-09-24_q1-reproduction_v1.0.md)。

[论文图表材料](03_Results/e/question-one/paper-assets-v1.0/q1_assets_notes_v1.0.md)已包含流程图、计数图（PNG/SVG）、1,932 行逐词表和统计 JSON；[典型样本图](02_Drafts/e/2026-09-24_q1-typical-example_v1.0.png)也可直接取用。下载仓库后，复用已有 Python + NumPy + Matplotlib 环境运行 `python 02_Drafts/e/src/2026-09-24_export-q1-paper-assets_v1.0.py` 即可重新导出这些便携图表（不含依赖原始中间数据的典型样本图）。

统计是历史自动掩码及静音排除后的候选数，不能写成对齐准确率。复核状态、原始数据缺口和如何取材均已在指南说明。

This repository contains the current, reviewable E-problem work from the local project. The primary completed material is Question One: extraction of text, acoustic and visual features and their word-level temporal alignment. Question Two planning, audit reports and selected training summaries are included as supporting material.

## Start here

1. Read [`01_Source/E/2026-09-24_e-problem-statement.docx`](01_Source/E/2026-09-24_e-problem-statement.docx) for the problem statement.
2. Read [`02_Drafts/e/2026-09-24_q1-paper-writing-guide_v1.0.md`](02_Drafts/e/2026-09-24_q1-paper-writing-guide_v1.0.md) for the paper structure, equations, figures, tables and evidence paths.
3. Read [`02_Drafts/e/2026-09-24_q1-model-method_v1.0.md`](02_Drafts/e/2026-09-24_q1-model-method_v1.0.md) for the full mathematical method.
4. Read [`02_Drafts/e/2026-09-24_q1-reproduction_v1.0.md`](02_Drafts/e/2026-09-24_q1-reproduction_v1.0.md) for the local environment, scripts, checks and reproduction boundaries.

## Question One evidence

- [`03_Results/e/question-one/2026-09-24_q1_results-index_v1.0.md`](03_Results/e/question-one/2026-09-24_q1_results-index_v1.0.md) is the result index and interpretation boundary.
- [`03_Results/e/question-one/2026-09-24_q1_audit_v1.0.json`](03_Results/e/question-one/2026-09-24_q1_audit_v1.0.json) records the full audit counts and source fingerprints.
- [`03_Results/e/question-one/2026-09-24_q1_sample-summary_v1.0.tsv`](03_Results/e/question-one/2026-09-24_q1_sample-summary_v1.0.tsv) and the appendix contain the 100-sample summary.
- [`02_Drafts/e/2026-09-24_q1-typical-example_v1.0.md`](02_Drafts/e/2026-09-24_q1-typical-example_v1.0.md) and its PNG show a traceable ten-word example.
- `02_Drafts/e/mosei-multimodal-aligned-v1.0/` contains the 100 compact word-level NPZ outputs (about 7.4 MiB). They preserve words, times, masks, features, coverage and provenance fields.

The historical v1.0 audit contains 1,932 source words, 1,923 automatically timed/audio-mask words, and 1,828 words with all three historical masks true. These are data-availability counts, not alignment accuracy. Sixty-five samples remain in the automatic review queue. A later scan found `sample_0042` and `sample_0043` to be digitally silent; the paper guide explains the resulting sensitivity counts (1,899 and 1,821) and why the old NPZ files were not silently rewritten.

## Code and other material

- `02_Drafts/e/src/` contains the saved extraction, alignment, audit, merge and Question Two scripts.
- `02_Drafts/e/qt-showcase/` contains the local Qt viewer source. It needs the local staged videos and intermediate media files, which are intentionally not in this public repository.
- `03_Results/e/question-two/` contains selected audit, normalization, baseline and robust-training reports. Model weights, test predictions and large caches are excluded.
- `requirements/` information is kept in `02_Drafts/e/2026-09-23_environment_requirements_v1.2.txt` and `2026-09-23_extraction_requirements_v1.0.txt`.

## Reproduction boundary

This is a compact research record, not a zero-download one-click reproduction bundle. The original videos, multi-gigabyte PKL attachments, decoded WAVs, full eGeMAPS/OpenFace/BERT intermediate directories, Hugging Face model caches, Python environments and training checkpoints are not uploaded. Prepare the original competition attachments and the locally documented model/software versions before running the scripts. Current scripts also contain machine-specific paths; update those paths for another machine and run the documented `--check` commands first.

The source attachments are treated as read-only. No source transcript or sentiment label is changed by the saved results. Automatic CTC times and masks require targeted human review; the Qt viewer is an inspection aid, not a human-ground-truth generator.
