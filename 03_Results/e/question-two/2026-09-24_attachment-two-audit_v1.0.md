# 附件二官方对齐数据审计

数据：`aligned_50.pkl` 与 `label.xlsx`。原始材料只读；这是数据质量与输入协议审计，不含训练或预测结果。

| 划分 | 样本 | BERT 长度 min / median / max | 长度为 50 | 分类 0 / 1 / 2 |
|---|---:|---:|---:|---:|
| train | 3395 | 3 / 22 / 50 | 253 | 967 / 758 / 1670 |
| valid | 728 | 3 / 23.5 / 50 | 47 | 206 / 184 / 338 |
| test | 727 | 4 / 23 / 50 | 52 | 207 / 158 / 362 |

## 关键检查

- `train`：Excel 缺行 0，原文/数值标签/annotation 不一致 0；attention 非前缀 0。
  音频在 BERT 有效位置的全零步 6855，视觉 11039；BERT 无效位置的非零音频步 0，非零视觉步 0。
- `valid`：Excel 缺行 0，原文/数值标签/annotation 不一致 0；attention 非前缀 0。
  音频在 BERT 有效位置的全零步 1502，视觉 2436；BERT 无效位置的非零音频步 0，非零视觉步 0。
- `test`：Excel 缺行 0，原文/数值标签/annotation 不一致 0；attention 非前缀 0。
  音频在 BERT 有效位置的全零步 1454，视觉 2478；BERT 无效位置的非零音频步 0，非零视觉步 0。
- 跨划分完全相同 ID：0。
- 跨划分相同视频 ID：0。此项按 `$_$` 前的视频 ID 核对。
- 跨划分完全相同原文：2 种：`Alright`、`Okay`。仅是常用短句相同，不能据此认定样本泄漏。
- Excel 总行数：4850；ID+划分重复：0；Excel 中未出现在特征包的行：0。

## 使用边界

`text_bert` 第二维依次含 token ID、attention mask、token type ID；有效长度含 `[CLS]` 和 `[SEP]`，应取 attention mask，不能从 768 维文本嵌入是否为零推断。全零音频/视觉步只是数值现象，不一定可判定为真正缺失；正式模型应在 train 上拟合标准化参数，valid 用于选择方案，test 留作最终评估。附件二的 768/74/35 与问题一自提的 768/25/49 维不直接混接。

分类标签的数值与回归正负号对照、每模态有限值和零步细节见同名 JSON。

## 特殊 token、截断与有限值

全部样本分类映射为 `0=负面`、`1=中性`、`2=正面`，与回归标签正负号一致。三模态全部数值有限；文本 padding 向量仍非零。以下从内容 token 中剔除了 `[CLS]`、`[SEP]`。

| 划分 | 内容 token | 音频全零内容 token | 视觉全零内容 token | 全零视觉样本 | 原文编码超过 50 token |
|---|---:|---:|---:|---:|---:|
| train | 76882 | 65 | 4249 | 110 | 234 |
| valid | 17172 | 46 | 980 | 15 | 44 |
| test | 16855 | 0 | 1024 | 28 | 49 |

本地固定版本 BERT tokenizer 对全部原文重新分词（仅分词，无模型推理）。按照 `max_length=50`、尾部截断、尾部 padding 重建的 token ID、attention mask、token type ID 与原包完全匹配。原文超过 50 token 的样本已截断；长度恰为 50 的样本不一定被截断。

另取 train 首条重新运行同一 BERT 最后一层，活动 token 与预计算 `text` 最大绝对差约 6.2e-6；这一检查只覆盖一条样本，详情见 `2026-09-24_official-bert-comparison_v1.0.json`。

此审计没有恢复原始时间戳或重新提取特征，不能据文件名或共同长度证明每个位置的时序对齐精度。

## 复跑

在项目根目录运行，复用已装主环境和本地 tokenizer，不需要下载或安装。默认只覆盖本脚本的 JSON/MD 两份派生报告；`--check` 只比较现有报告，不写文件。

```powershell
& "D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe" -X utf8 "02_Drafts\e\src\2026-09-24_audit-attachment-two_v1.0.py" --check
```

校验来源包括特征包 SHA-256、Excel 的 video_id/clip_id/mode 组合键、原文精确比较、标签绝对误差不超过 1e-6、三模态结构/有限值/全零步、attention mask 与本地 tokenizer 重建对照。
