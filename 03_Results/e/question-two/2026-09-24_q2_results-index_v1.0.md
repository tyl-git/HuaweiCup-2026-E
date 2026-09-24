# 问题二：训练输入与基线成果索引

更新日期：2026-09-24。已完成数据审计、统一输入、训练/验证文本编码、训练集标准化、常数参考基线，以及单模态和简单三模态融合的第一轮训练与独立验收。原始 `01_Source` 未改写；全部复用现有 Python、PyTorch、CUDA 与本地 BERT，无新增安装或模型下载。

## 当前成果

| 项目 | 当前结果 | 阅读入口 |
|---|---|---|
| 官方数据审计 | 3,395 / 728 / 727；与 Excel 样本、原文、标签一致；官方划分间 ID 与视频 ID 无重合 | [附件二审计](2026-09-24_attachment-two-audit_v1.0.md) |
| 专项结构 | 附件三 30 个、附件四 20 个对齐文件核查通过；附件三无预计算文本，附件四 `13.pkl` 视觉全零 | [专项接口](2026-09-24_special-input-contract_v1.0.md) |
| 统一只读加载器 | 4,850 官方样本及 50 专项文件全量通过；9 项异常边界测试通过 | [加载器](../../../02_Drafts/e/src/2026-09-24_q2_data_v1.0.py) |
| 冻结 BERT 编码 | train/valid 共 4,123 条完成，原位置保留；缓存带源文件、token、样本顺序、权重与配置指纹 | [编码报告](2026-09-24_q2-text-cache_v1.0.md) |
| 标准化 | 只由 train 有效观测拟合，valid 只应用；缺失零占位与独立掩码保留 | [验收报告](2026-09-24_q2-normalization_v1.0.md) |
| 评价与常数参考 | valid 多数类 Accuracy 0.464286、macro-F1 0.211382；均值强度 MAE 0.780847、中位数 MAE 0.769002；常数预测 Pearson 不可定义 | [常数基线](2026-09-24_q2-constant-baselines_v1.0.md) |
| 单模态与融合基线 | text / audio / vision / fusion 各 3 个固定种子，共 12 次 train/valid 运行；逐样本指标、checkpoint、预测文件和 SHA-256 独立验收通过；test 与专项未评估 | [基线结果报告](../../../02_Drafts/e/2026-09-24_q2-baseline-results_v1.0.md) |
| 普通融合局部缺失评估 | 18 个固定条件 × 3 个种子；728 条 valid 保留；完整输入重载预测通过核对；逐样本预测、原始掩码与区间已保存 | [缺失评估报告](q2-missing-baseline-v1.0/2026-09-24_q2-missing-baseline-results_v1.0.md) |

输入定义见[输入与评价协议](../../../02_Drafts/e/2026-09-24_q2-input-evaluation-protocol_v1.0.md)。三个模态为 768/74/35 维，固定 50 个序列位置。50 不是有效词数，也不是秒级视频时间；内容掩码由官方 BERT attention 去除首尾特殊 token 后得到。

## 使用方式与验收

加载器返回 `input_ids / attention_mask / token_type_ids`、音视频、内容掩码和三种模态观测掩码。`reference_text` 只用于核对官方预计算值；正式文本输入读取经过指纹核验的统一 BERT 缓存。训练标准化统计包为 [NPZ](2026-09-24_q2-normalization_v1.0.npz)，对应 [JSON](2026-09-24_q2-normalization_v1.0.json) 记录来源、算法、训练位置数和指纹。[验收 JSON](2026-09-24_q2-normalization-check_v1.0.json)记录变换后有限值、零占位与掩码检查。

从任意 PowerShell 工作目录运行以下只读验收命令。标准化验收会重新计算 train 的统计量与已有包比较，既不覆盖文件，也不使用 valid/test 拟合。

```powershell
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_q2_data_v1.0.py'
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_prepare-q2-text_v1.0.py' --check
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_prepare-q2-normalization_v1.0.py' --check
& 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe' -X utf8 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e\src\2026-09-24_q2-evaluation_v1.0.py' --check
```

加载器测试覆盖浮点 token 非整数、attention 内部空洞、形状颠倒、非有限值、标签越界/矛盾、文本 padding 非零、全视觉缺失和数组只读。标准化自测覆盖 train-only 拟合、排除填充极值、全模态缺失、常量维度、变换不修改统计、真实零值仍有观测以及文件/来源清单校验。

## 下一阶段

基线已完成，当前结果显示 text 是最强单模态，fusion 对分类有小幅帮助，但回归 MAE 没有改善；Neutral 类仍弱，vision 存在无有效观测样本。音频的标准化后数值存在长尾（train 最大绝对值约 59.76，valid 约 169.52）；当前未据 valid 重拟合或截断。若采用裁剪或鲁棒变换，规则必须由 train 制定并保存。

现已完成普通融合的验证集连续局部缺失实验：text/audio/vision 分别采用目标 20%/40%，起始/中间/末尾各一段。目标 40% 文本起始缺失使平均 Accuracy 下降约 0.0238、macro-F1 下降约 0.0287、MAE 增加约 0.0214；音视频变化较小且方向不一致。实际比例因样本取整高于目标比例。文本掩码作用于 BERT 输出向量，保留向量含完整句子上下文，不等价于原始转写缺失。表中标准差统一采用 ddof=1。

下一轮由用户手动运行三组消融：简单融合加缺失增强、动态融合无增强、动态融合加增强。每组固定三个种子，和现有简单融合无增强基线比较，分别报告完整输入和相同 18 个缺失条件。训练尚未启动。所有专项文件目前只做结构审计；官方 test 尚未做性能评估。固定最终方案后统一编码、标准化并评估 test，再进行附件三/四的任务推理。

## 与问题一的边界

问题一对应视频特征提取和词级时序对齐，维数为 768/25/49。深入复核证实 `sample_0042`、`sample_0043` 的音轨整段数字静音；历史 1,828 个三模态 mask 通过词中，排除静音相关词后为 1,821 个候选词，并非对齐准确率。旧 NPZ 和 Qt 仍使用历史掩码，后续需要新版质量修订；人工复核未被自动判定完成。详见[问题一成果索引](../question-one/2026-09-24_q1_results-index_v1.0.md)。
