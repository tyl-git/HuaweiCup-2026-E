# E题模型改进与最终复现记录

日期：2026-09-25

本文记录在统一 `aligned_50` 输入接口下完成的模型改进、验证规则和最终结果。所有模型只使用题目给出的官方训练集、验证集和测试集；没有引入其他情感数据集。训练集用于拟合标准化和类别权重，验证集用于选择 epoch 与冻结集成规则。官方测试集在历史开发中已经被查看，因此不能称为全程未触碰的一次性 holdout；本轮没有用测试结果调整权重、epoch 或 seed。

## 1. 改进目标

原始时序模型对 Positive 类较强，但 Neutral 类 F1 偏低。改进优先解决类别不均衡，同时保持题目要求的局部连续缺失建模、三层 mask 语义和可复现评估。

## 2. 第一轮改进：类别均衡损失

训练标签计数为 `[967, 758, 1670]`（Negative、Neutral、Positive）。在训练阶段将分类交叉熵乘以训练集标签计算的平方根逆频率权重：

$$w_c=\frac{1/\sqrt{\pi_c}}{\frac{1}{3}\sum_j1/\sqrt{\pi_j}}.$$

得到权重 `[1.037908, 1.172297, 0.789794]`。验证集仍使用普通 CE+L1 选择 checkpoint，因此类别权重没有改变模型选择口径，也没有使用测试集调参。

### 验证集结果

三种子 `20260924/20260925/20260926` 的类别均衡模型平均 Accuracy 约为 `0.6342`，Macro-F1 约为 `0.6099`，明显高于原始 temporal 模型平均 Macro-F1 约 `0.5742`。Neutral F1 由约 `0.35` 提升到约 `0.46`，说明改进主要解决了中性类召回不足；这不是所有指标都同步提升，应按多指标报告。

### 官方测试集三种子

| seed | Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson r |
|---:|---:|---:|---:|---:|---:|
| 20260924 | 0.6891 | 0.6523 | 0.4620 | 0.6029 | 0.7101 |
| 20260925 | 0.6795 | 0.6362 | 0.4371 | 0.6237 | 0.6845 |
| 20260926 | 0.6836 | 0.6289 | 0.3944 | 0.6220 | 0.6893 |

测试结果只用于最终报告，不能反过来选择 seed。inverse 频率权重也做过独立训练，但验证结果波动更大且回归指标更差，因此没有作为主模型。

## 3. 固定三种子等权集成

依据验证集结果固定三个类别均衡 temporal checkpoint 等权集成；测试集历史上已被查看，本轮没有根据其结果修改集成规则：

$$z=\frac{1}{3}\sum_{s=1}^{3}z_s,\qquad \hat y=\arg\max z.$$

回归强度同样取三个 checkpoint 输出的算术平均。该规则不是根据测试成绩调出的，脚本会检查 ID 顺序、标签一致性、有限值和来源哈希。

最终集成测试结果。v1.1 对 valid、test、附件三统一使用三个 seed 的 logits 算术平均后 softmax，回归强度仍取算术平均；test 数组与 v1.0 逐元素一致，附件三的 30 个预测类别均未改变：

| 指标 | 数值 |
|---|---:|
| Accuracy | **0.690509** |
| Macro-F1 | **0.647076** |
| Neutral F1 | 0.445183 |
| Neutral Recall | 0.424051 |
| MAE | **0.606778** |
| Pearson r | **0.703049** |

这是当前问题二的推荐最终结果。Neutral 仍然是主要误差来源，论文中必须同时报告混淆矩阵和逐类 F1，不能只报告 Accuracy。

## 4. 文本缺失协议修正

原有缺失评估是在冻结 BERT 输出后把对应行清零。这个做法会保留被删除词对上下文 token 的影响，因此可能高估文本缺失鲁棒性。

新增的严格对照在 BERT 编码前把同一连续 token 区间替换为 `[MASK]`，再冻结 BERT 编码，最后才应用有效观测 mask。验证集结果显示，真实输入级缺失比旧的 post-BERT 清零更困难：

| 缺失条件 | 输入级 Macro-F1 相对旧方法变化 | 输入级 Accuracy 相对旧方法变化 |
|---|---:|---:|
| 20% start | -0.0480 | -0.0472 |
| 20% middle | -0.0370 | -0.0353 |
| 20% end | -0.0358 | -0.0403 |
| 40% start | -0.0772 | -0.0691 |
| 40% middle | -0.0762 | -0.0668 |
| 40% end | -0.0877 | -0.0746 |

因此：历史 post-BERT 文本缺失结果只能作为旧协议的补充审计，不能宣称等价于原始文本局部缺失。附件3的正式口径仍是原始特征中局部连续时间段全零，并且必须分别保存 `content_mask`、原始 `observed/source mask`、注入的 `missing mask` 和最终 `effective mask`。

## 5. 结果边界

### 第二轮：输入级文本缺失增强

在冻结 BERT 前，对训练集的连续文本 token 区间使用 `[MASK]`，并在编码后把同一区间标为缺失；其余音视频和标准化沿用原协议。独立训练的三个 seed 只按完整输入 valid 的普通 CE+L1 保存 checkpoint。事先设定的验证准入规则是：六种输入级文本缺口的平均 Macro-F1 比同 seed 类别均衡模型至少提高 `0.02`，且完整输入平均 Macro-F1 下降不超过 `0.02`。实际为 `+0.025418` 和 `-0.008733`。该方案作为鲁棒性补充保留；其完整 test 结果是历史开发结果，不能据此重新选择模型或权重。

| 方案 | 官方 test Accuracy | Macro-F1 | Neutral F1 | MAE | Pearson r |
|---|---:|---:|---:|---:|---:|
| 类别均衡三 seed 集成（推荐主结果） | 0.690509 | 0.647076 | 0.445183 | 0.606778 | 0.703049 |
| 输入级文本增强三 seed 集成（鲁棒性补充） | 0.680880 | 0.631360 | 0.408027 | 0.611846 | 0.702636 |

第二轮提高了验证集真实文本缺口鲁棒性，却没有提高完整官方 test 指标。它说明缺失鲁棒性与完整输入性能存在权衡，不应因为已看见 test 结果再改变集成权重或反选 seed。附件3的 30 条特征中 `text_zero_content_positions` 均为 0，不能把第二轮写成“直接修复附件3文本缺失”；附件3主要表现为音频/视觉局部零区间。

第二轮的冻结协议为 `03_Results/e/question-two/q2-text-safe-final-protocol-v1.0.json`，独立审计覆盖三个 seed 的 checkpoint/预测哈希、727 个唯一 test ID、30 条附件3预测以及集成逐元素平均。

### 后续审计与同时缺失压力测试

一致化集成见 `03_Results/e/question-two/q2-temporal-balanced-sqrt-ensemble-final-v1.1/`。从已保存的单 seed 输出生成的配对比较、混淆矩阵、学习曲线和 239 个原视频为抽样单位的 1000 次 valid bootstrap，见 `q2-improvement-evidence-v1.1/`。类别均衡减原 temporal 的 valid Macro-F1 差值百分位区间为 `[+0.0160,+0.0596]`，Neutral F1 为 `[+0.0616,+0.1621]`；区间仅条件于这些已训练模型和复用的验证数据，不是确认性泛化置信区间。

另以共享 content 时间段注入 24 种两模态/三模态局部连续缺失，见 `q2-balanced-multigap-stress-valid-v1.0/`。完整 valid Macro-F1 为 `0.617352`；文本+视觉 40% 末段缺失降至 `0.572036`，三模态 40% 中段缺失为 `0.579666`。模型明显依赖文本，不能宣传在所有缺失组合下都保持性能。本次共享区间按 content 长度计算；旧单模态测试按各模态原始可观察位置数计算，两组干预不可直接逐项对比。文本缺失仍是 BERT 后特征遮挡，输入级重编码实验另列。

论文图：`03_Results/e/paper-assets-v1.0/q2_balanced_and_text_safe_test_v1.0.png` 对比完整输入 test 指标，其中原模型是三 seed **指标均值**，两个改进模型是三 seed **预测集成**，图中已逐项标注，不能称完全同一汇总方式。`q2_text_gap_protocol_comparison_v1.0.png` 在同一批 valid 样本、同一文本区间和同一类别均衡 checkpoint 上比较两种缺失注入阶段。两图同时提供 SVG，生成脚本为 `02_Drafts/e/src/2026-09-25_plot-q2-improvements_v1.0.py`。

### 额外模型搜索：未晋级候选

在冻结主模型和指标口径后，又完成三类 train/valid-only 比较：每模态独立时间卷积、加权验证损失选择 checkpoint、以及加权 focal loss（固定 γ=2）。所有候选沿用官方 `aligned_50`、同一有效长度与缺失 mask、train-only 标准化、相同训练数据增强和三个 seed；没有评估 test 或附件样本。晋级线预先固定为三 seed 集成 valid Macro-F1 至少提高 `0.01`，且 MAE 不恶化超过 `0.01`。

| 候选 | Valid Accuracy | Macro-F1 | Neutral F1 | MAE | 对照结论 |
|---|---:|---:|---:|---:|---|
| 锁定 balanced temporal 集成 | 0.64011 | 0.61735 | 0.47439 | 0.56105 | 当前主结果 |
| 模态独立时间卷积，三 seed 集成 | 0.64560 | 0.61770 | 0.45882 | 0.56438 | Macro-F1 仅 `+0.00035`，未达门槛，Neutral F1 与 MAE 变差 |
| Balanced-selection，三 seed 集成 | 0.64011 | 0.61735 | 0.47439 | 0.56105 | 与锁定集成持平；不晋级 |
| Weighted focal loss，三 seed 集成 | 0.63462 | 0.61248 | 0.45822 | 0.55742 | MAE 改善 `0.00363`，Macro-F1 下降 `0.00487`，不晋级 |

Focal 候选三个单模型 Macro-F1 均值为 `0.59583`（样本标准差 `0.01279`），Neutral F1 均值为 `0.44089`（标准差 `0.03390`），seed 间波动没有支持稳定改进。模态卷积候选增加了参数与分支，但三 seed 集成几乎没有 Macro-F1 收益；这与验证集上的有限样本和过拟合风险一致。Balanced-selection 使用加权 CE+L1 选择 epoch，虽是合理的受控消融，但集成类别结果与锁定基准完全相同，不能据此宣称性能提升。

logit 类别偏置在整份 valid 上曾提高 Macro-F1，但五折 out-of-fold Macro-F1 只有 `0.59989` 且折间偏置不稳定，因此不采用。候选的紧凑指标和拒绝理由保存在[模型搜索审计 JSON](../03_Results/e/question-two/q2-model-search-v1.0.json)；完整 checkpoint、epoch 记录和 NPZ 预测保存在运行机器对应的 `03_Results/e/question-two/` 目录中，因体积和数据管理策略未放进仓库。它们支持当前停止扩大模型搜索：现有 valid 只有 728 条且按原视频分组后有效独立单位更少，test 也有历史查看，继续追逐同一 valid 分数会扩大选择偏差。除非出现有明确题意依据、事先固定假设并可用新增独立数据验证的方案，否则主模型维持冻结。

- Q1 自提特征是 `768/25/49`，Q2/Q3使用题目附件的 `aligned_50` 官方 `768/74/35`，两者不能混写成同一输入。
- 附件3和附件4没有情感真值，只报告预测和掩码/解释审计，不报告准确率。
- gate 或 attention 只用于产生候选证据；问题三的证据必须用位置遮挡后预测变化验证。
- 自动 CTC 时间是候选定位，不是人工精确时间真值。带 `review` 的样本在论文中必须保留该状态。
- GitHub 中不包含原始视频、PKL、BERT 大缓存和大体积训练中间文件；复现需要按 README 获取题目附件并配置本机路径。

## 6. 复现入口

- 类别均衡训练：`02_Drafts/e/src/2026-09-25_train-q2-temporal-balanced_v1.0.py`
- 文本输入级缓存：`02_Drafts/e/src/2026-09-25_q2-text-safe-prep_v1.0.py`
- 文本协议对照：`02_Drafts/e/src/2026-09-25_eval-q2-text-safe_v1.0.py`
- 三种子一致化集成：`02_Drafts/e/src/2026-09-25_aggregate-q2-balanced-ensemble_v1.1.py`
- 集成结果：`03_Results/e/question-two/q2-temporal-balanced-sqrt-ensemble-final-v1.1/summary.json`
- 固定模型配对证据：`02_Drafts/e/src/2026-09-25_build-q2-evidence_v1.1.py`
- 同时缺失压力测试及审计：`02_Drafts/e/src/2026-09-25_eval-q2-balanced-multigap_v1.0.py`、`2026-09-25_audit-q2-balanced-multigap_v1.0.py`
- 文本对照结果：`03_Results/e/question-two/q2-text-safe-eval-v1.0/report.json`
- 输入级增强训练：`02_Drafts/e/src/2026-09-25_train-q2-text-safe-balanced_v1.0.py`
- 输入级增强最终评估与审计：`02_Drafts/e/src/2026-09-25_eval-q2-text-safe-final_v1.0.py`
- 输入级增强结果：`03_Results/e/question-two/q2-text-safe-final-v1.0/summary.json`
- 模型搜索候选：`02_Drafts/e/src/2026-09-25_train-q2-balanced-selection_v1.0.py`、`02_Drafts/e/src/2026-09-25_train-q2-focal-balanced_v1.0.py`
- 紧凑模型搜索结果：`03_Results/e/question-two/q2-model-search-v1.0.json`
