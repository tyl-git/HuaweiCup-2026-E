# HuaweiCup 2026 E

本仓库保存 E 题三问的代码、紧凑结果和论文证据。先读[三问论文写作与证据指南](02_Drafts/e/2026-09-24_e-paper-guide_v1.0.md)和[成果总览](03_Results/e/2026-09-24_e-results-summary_v1.0.md)，题面为[归档文档](01_Source/E/2026-09-24_e-problem-statement.docx)。这是研究记录，不是正式竞赛提交包。

| 问题 | 已完成成果 | 结果入口 |
| --- | --- | --- |
| 一：三模态特征提取与词级时序对齐 | 附件一 100 条片段的词级 `768/25/49` 维 NPZ、自动质量审计、流程图和典型样本图 | [问题一指南](02_Drafts/e/2026-09-24_q1-paper-writing-guide_v1.0.md)、[100 条 NPZ](02_Drafts/e/mosei-multimodal-aligned-v1.0/)、[图表](03_Results/e/question-one/paper-assets-v1.0/) |
| 二：局部连续缺失下的情感识别 | 官方 `aligned_50` 的 `3395/728/727` 划分；三种子时序模型、基线和消融；727 条官方 test 与附件三 30 条无标签预测 | [测试与专项结果](03_Results/e/question-two/q2-temporal-final-v1.0/summary.json)、[附件三预测](03_Results/e/question-two/q2-temporal-final-v1.0/attachment3_predictions.csv)、[训练统计](03_Results/e/question-two/2026-09-24_q2-normalization_v1.0.npz) |
| 三：可解释预测 | 附件四 20 条极性/强度、模态影响、1,662 条位置遮挡、自动音视频时间候选；100 条固定 valid 的遮挡对照 | [预测](03_Results/e/question-three/q3-explanations-v1.0/attachment4_predictions.csv)、[解释](03_Results/e/question-three/q3-explanations-v1.0/attachment4_explanations.csv)、[时间质量](03_Results/e/question-three/q3-evidence-time-v1.0/attachment4_time_quality.csv) |

问题一自提特征与问题二、三的官方 `768/74/35` 维特征来自不同附件，不能混作同一模型输入。问题二的主模型由验证集选择为 `temporal/20260926`，其官方 test Accuracy `0.7015`、Macro-F1 `0.6189`、MAE `0.6156`、Pearson r `0.6938`；三种子均值及消融见论文指南与[论文图表](03_Results/e/paper-assets-v1.0/)。主[checkpoint](03_Results/e/question-two/q2-temporal-v1.0/temporal/seed_20260926/best.pt)与训练统计包已归档。附件三、四没有情感真值，因此不报告其预测准确率。

## 复现入口

- [问题一方法与版本](02_Drafts/e/2026-09-24_q1-reproduction_v1.0.md)，[问题一结果边界](03_Results/e/question-one/2026-09-24_q1_results-index_v1.0.md)。
- [问题二输入接口](02_Drafts/e/2026-09-24_q2-input-evaluation-protocol_v1.0.md)、[训练脚本](02_Drafts/e/src/2026-09-24_train-q2-temporal_v1.0.py)、[官方测试脚本](02_Drafts/e/src/2026-09-24_eval-q2-temporal-final_v1.0.py)、[独立审计脚本](02_Drafts/e/src/2026-09-24_audit-q2-temporal-final_v1.0.py)。
- [问题三解释脚本](02_Drafts/e/src/2026-09-24_eval-q3-explanations_v1.0.py)、[自动时间回映脚本](02_Drafts/e/src/2026-09-24_map-q3-evidence-time_v1.0.py)、[图表导出脚本](02_Drafts/e/src/2026-09-24_export-q23-paper-assets_v1.0.py)。
- Python 包版本见[主环境清单](02_Drafts/e/2026-09-23_environment_requirements_v1.2.txt)及[提取环境清单](02_Drafts/e/2026-09-23_extraction_requirements_v1.0.txt)。脚本中仍有本机绝对路径；在其他机器上要配置原附件、FFmpeg、OpenFace 与冻结 BERT/CTC 权重路径。预训练工具的 snapshot 和文件哈希见结果 JSON。

原始视频/PKL、音频、整套 BERT 中间缓存和其他大体积训练产物不上传。问题一 65 条自动复核标记尚未逐条人工验收；历史三模态 mask 数量不等于对齐准确率。问题三的秒数与视频帧是自动候选，20 条中 8 条带复核标记、9 个标点位置无法回映，均不得当成人工确认的精确证据。正式竞赛附件另需满足题面的 50 MB、匿名和文件清单要求。
