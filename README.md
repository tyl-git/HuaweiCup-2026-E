# HuaweiCup 2026 E

本仓库保存 E 题三问的代码、紧凑结果和论文证据。先读[三问论文写作与证据指南](02_Drafts/e/2026-09-24_e-paper-guide_v1.0.md)和[成果总览](03_Results/e/2026-09-24_e-results-summary_v1.0.md)，题面为[归档文档](01_Source/E/2026-09-24_e-problem-statement.docx)。这是研究记录，不是正式竞赛提交包。

| 问题 | 已完成成果 | 结果入口 |
| --- | --- | --- |
| 一：三模态特征提取与词级时序对齐 | 附件一 100 条片段的词级 `768/25/49` 维 NPZ、自动质量审计、只读人工 QC 叠加层、流程图和典型样本图 | [问题一指南](02_Drafts/e/2026-09-24_q1-paper-writing-guide_v1.0.md)、[100 条 NPZ](02_Drafts/e/mosei-multimodal-aligned-v1.0/)、[人工 QC 叠加层](03_Results/e/question-one/q1-human-qc-overlay-v1.0/)、[图表](03_Results/e/question-one/paper-assets-v1.0/) |
| 二：局部连续缺失下的情感识别 | 官方 `aligned_50` 的 `3395/728/727` 划分；类别均衡 temporal、三种子等权集成、基线、消融和 24 种多模态压力测试；727 条官方 test 与附件三 30 条无标签预测 | [统一集成审计](03_Results/e/question-two/q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json)、[改进证据](03_Results/e/question-two/q2-improvement-evidence-v1.1/report.json)、[多模态压力测试](03_Results/e/question-two/q2-balanced-multigap-stress-valid-v1.0/summary.json)、[改进记录](docs/2026-09-25_model-improvement-summary_v1.0.md)、[附件三预测](03_Results/e/question-two/q2-temporal-balanced-sqrt-ensemble-final-v1.1/attachment3_ensemble_predictions.csv)、[训练统计](03_Results/e/question-two/2026-09-24_q2-normalization_v1.0.npz) |
| 三：可解释预测 | 附件四 20 条极性/强度、模态影响、1,662 条位置遮挡、自动音视频时间候选；100 条固定 valid 的遮挡对照。当前推荐解释与问题二一致，使用冻结的 balanced 三 seed ensemble；旧 temporal/20260926 结果保留为历史参考。 | [集成解释](03_Results/e/question-three/q3-balanced-ensemble-explanations-v1.1/attachment4_explanations.csv)、[集成解释审计图](03_Results/e/question-three/q3-balanced-ensemble-figures-v1.1/summary.json)、[旧参考解释](03_Results/e/question-three/q3-explanations-v1.0/attachment4_explanations.csv)、[时间质量](03_Results/e/question-three/q3-evidence-time-v1.0/attachment4_time_quality.csv) |

问题一自提特征与问题二、三的官方 `768/74/35` 维特征来自不同附件，不能混作同一模型输入。问题二当前推荐最终结果是三个类别均衡 temporal checkpoint 的固定等权集成：官方 test Accuracy `0.690509`、Macro-F1 `0.647076`、MAE `0.606778`、Pearson r `0.703049`；模型和权重按 valid 固定。测试集在历史开发过程中已经被查看过，因此仓库不声称一次性盲测或全程独立 holdout；本轮统一集成算子只是审计和复现修正，没有依据 test 反选模型、seed 或权重。输入级文本缺失对照表明，post-BERT 清零会高估文本缺失鲁棒性；额外的卷积、focal-loss 和 checkpoint 选择候选均未达到预设晋级线，主结果维持冻结。结果与原因见[模型搜索审计](03_Results/e/question-two/q2-model-search-v1.0.json)及[改进记录](docs/2026-09-25_model-improvement-summary_v1.0.md)。附件三、四没有情感真值，因此不报告其预测准确率。

## 复现入口

- [问题一方法与版本](02_Drafts/e/2026-09-24_q1-reproduction_v1.0.md)，[问题一结果边界](03_Results/e/question-one/2026-09-24_q1_results-index_v1.0.md)。
- [问题二输入接口](02_Drafts/e/2026-09-24_q2-input-evaluation-protocol_v1.0.md)、[类别均衡训练](02_Drafts/e/src/2026-09-25_train-q2-temporal-balanced_v1.0.py)、[统一固定集成](02_Drafts/e/src/2026-09-25_aggregate-q2-balanced-ensemble_v1.1.py)、[独立审计](02_Drafts/e/src/2026-09-25_audit-q2-balanced-ensemble_v1.1.py)、[多模态压力测试](02_Drafts/e/src/2026-09-25_eval-q2-balanced-multigap_v1.0.py)、[模型搜索审计](03_Results/e/question-two/q2-model-search-v1.0.json)、[改进证据](03_Results/e/question-two/q2-improvement-evidence-v1.1/report.json)。候选脚本仅用于 train/valid 复现实验：[类别均衡 checkpoint 选择](02_Drafts/e/src/2026-09-25_train-q2-balanced-selection_v1.0.py)、[focal loss](02_Drafts/e/src/2026-09-25_train-q2-focal-balanced_v1.0.py)。
- [问题三解释脚本](02_Drafts/e/src/2026-09-24_eval-q3-explanations_v1.0.py)、[自动时间回映脚本](02_Drafts/e/src/2026-09-24_map-q3-evidence-time_v1.0.py)、[图表导出脚本](02_Drafts/e/src/2026-09-24_export-q23-paper-assets_v1.0.py)。
- Python 包版本见[主环境清单](02_Drafts/e/2026-09-23_environment_requirements_v1.2.txt)及[提取环境清单](02_Drafts/e/2026-09-23_extraction_requirements_v1.0.txt)。脚本中仍有本机绝对路径；在其他机器上要配置原附件、FFmpeg、OpenFace 与冻结 BERT/CTC 权重路径。预训练工具的 snapshot 和文件哈希见结果 JSON。

原始视频/PKL、音频、整套 BERT 中间缓存和其他大体积训练产物不上传。问题一 65 条自动复核标记尚未逐条人工验收；历史三模态 mask 数量不等于对齐准确率。问题三的秒数与视频帧是自动候选，20 条中 8 条带复核标记、9 个标点位置无法回映，均不得当成人工确认的精确证据。正式竞赛附件另需满足题面的 50 MB、匿名和文件清单要求。
