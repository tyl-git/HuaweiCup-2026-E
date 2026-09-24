# 问题二：用户手动运行的缺失增强与动态融合实验

日期：2026-09-24。所有训练由用户手动启动；助手准备代码、做不更新模型参数的检查，并核查用户运行后保存的结果。复用现有 `huaweicup-e` 环境，无需安装软件或依赖。原始 `01_Source` 只读。

## 目前完成的工作

text、audio、vision、fusion 各三个随机种子的第一轮训练已完成。12 次运行的预测、标签顺序、指标、检查点、早停和来源指纹通过独立审计。普通 fusion 还完成了 18 个 valid 特征缺失条件的评估；三种子的完整输入预测已与第一次训练产物核对。

这些是验证集结果，不是最终测试成绩。完整输入时 fusion 的平均 Accuracy 为 0.6168、macro-F1 为 0.5320、MAE 为 0.6021；text 分别为 0.6085、0.4987、0.5986。融合对分类略有帮助，回归 MAE 未改善。Neutral 召回低和过拟合是后续需要观察的问题。

## 本轮实验设计

| 实验 | 融合结构 | 训练时缺失增强 | 状态与目的 |
|---|---|---|---|
| fusion（已有） | 各模态编码后拼接 | 无 | 已完成，作为对照 |
| fusion_aug | 与已有 fusion 相同 | 有 | 首先运行，单独检验缺失增强 |
| gated_clean | 按模态可用性屏蔽并动态加权 | 无 | 检验融合结构变化 |
| gated_aug | 同 gated_clean | 有 | 检验两者结合 |

每个新增实验使用种子 20260924、20260925、20260926。共需新训练 9 次，分三条命令启动也可以。已有 fusion 不重跑，不覆盖。

数据仍为 train 3395 / valid 728；冻结 BERT 文本缓存及官方音频/视觉输入维度为 768/74/35。只用现有 train 统计量标准化一次，再按独立观测掩码池化。缺失后重新对剩余可观测位置取平均：

\[
\bar x_m=\frac{\sum_t o_{m,t}x_{m,t}}{\max(1,\sum_t o_{m,t})}.
\]

增强按样本和 epoch 确定，固定哈希与随机种子可复现。每条训练样本约 50% 概率保持完整，否则从三种模态选择一种，在其原本有观测的位置中删除目标 20% 或 40% 的连续区间；区间位置从起始、中间、末尾、随机选取。条件均匀抽取，不保证有限样本中数量恰好相等。原先无观测的位置不重复计为新增缺失，实际比例按删除数量与原观测数量记录。验证集只使用事先固定的缺失计划，不参与增强抽样。

动态融合先对各模态投影得到向量 h，再由小型门控网络给出分数，对有观测模态做 softmax 加权。完全无观测模态权重为零；全部无观测时回退到训练集类别先验及平均强度。门控权重是模型内部权重，不能直接当作因果解释。

基础超参数沿用基线：AdamW、学习率 0.001、weight decay 0.0001、batch 64、dropout 0.2、最多 40 轮、patience 8。优化目标为三分类交叉熵加回归 L1，最佳 checkpoint 仍按完整 valid 的同一损失最小值选取。此次不同时修改类别权重或标签规则，以免混淆增强和结构的作用。

每个最佳 checkpoint 在完整 valid 和 18 个单模态局部缺失条件下评估：3 模态 × 2 比例 × 3 位置。所有模型使用同一批 728 条样本、相同区间和掩码。报告 Accuracy、macro-F1、各类指标、MAE 和 Pearson r，并相对于同种子的完整输入记录配对变化。跨三个种子报告均值和样本标准差（ddof=1），不把它写为置信区间。

## 在 PowerShell 启动

交付前已通过语法检查、合成数据自测和真实 train/valid 数据的 CUDA 前向预检（`fusion_aug`）。预检没有优化器更新，也没有创建训练结果目录。需要自行复查时可运行以下命令；已经通过检查后无需重复运行：

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_train-q2-robust_v1.0.py' --self-test
```

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_train-q2-robust_v1.0.py' --preflight --device cuda --model fusion_aug
```

第一步只启动简单融合加缺失增强，共三个种子：

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_train-q2-robust_v1.0.py' --run --device cuda --model fusion_aug
```

结束后核查该组结果，再依次进行下面两组。每条命令是独立的一组训练，不要同时开多个终端运行它们。

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_train-q2-robust_v1.0.py' --run --device cuda --model gated_clean
```

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_train-q2-robust_v1.0.py' --run --device cuda --model gated_aug
```

`--model all` 可从尚未开始的新实验目录依次运行全部九次；已按组运行时继续用各组命令。脚本保护已有结果，不自动覆盖，目前不支持中断后的自动续训：任意种子中途退出后，直接重跑同一组会拒绝覆盖已有目录。此时应先保留日志和已有结果，核查状态后再安排恢复。

结果目录：`D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-robust-v1.0`。每个 variant/seed 目录保存最佳权重、逐轮记录、完整及缺失验证预测、指标、增强记录和来源指纹。输出中显示正常的逐轮验证损失即可；不到 40 轮早停是预期行为。具体耗时尚未实测。

按组运行后分别生成 `summary_fusion_aug.json`、`summary_gated_clean.json`、`summary_gated_aug.json`；一次运行全部实验则生成 `summary_all.json`。一组的三个种子及其验证评估全部结束后，最后输出 `Robust train/valid runs complete; test and specialists untouched`。先运行第一组并反馈末尾日志，核查产物后继续下一组。

## 论文如何使用，以及仍需补齐的内容

本轮用于回答“性能变化来自缺失增强还是融合结构”。先形成四组消融表，分别列完整输入表现和 18 条件的结果，不只挑有利条件。即使某个指标下降，也如实报告。若模型只改善分类而不改善回归，按任务分别解释。

文本是在完整句子经 BERT 编码后遮挡特征行，剩余上下文向量可能包含被遮词的信息。因此只能称为特征层的局部缺失，不能声称重现了原始转写缺失。音视频缺失影响小也不代表它们不存在情感信息。

此轮尚不覆盖双模态同时缺失或整模态缺失的充分实验。模型确定前还需补充对应对照、Neutral 类表现、过拟合分析与复现核查。只有方案冻结后才统一编码和评估官方 test、推理附件三/四；不能通过专项预测反向调参。问题三的模态贡献与局部证据需有遮挡对照支持，门控权重本身不构成完整解释。
