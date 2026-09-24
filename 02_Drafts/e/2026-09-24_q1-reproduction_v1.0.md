# 问题一：多模态特征提取与词级时序对齐复现说明

版本：v1.0；核对日期：2026-09-24。本文对应现有 100 条视频的 `mosei-multimodal-aligned-v1.0` 结果，说明结果读取、计算流程和复现边界。这里的特征与质量标记不构成情绪预测结果。

## 1. 可复现范围与数据位置

项目根目录为 `D:\01_Projects\HuaweiCup-2026`；以下相对路径以 `02_Drafts\e` 为起点。原题及原始数据在 `01_Source\E` 中，复现流程只读原始材料。暂存视频与原视频通过 SHA-256 校验一致，英文暂存文件名用于避开 OpenFace 对中文路径的兼容性问题。

| 文件或目录 | 用途 |
|---|---|
| `2026-09-23_mosei_sample-manifest_v1.0.csv` | 100 行原始样本清单；保留原表行号、视频与片段 ID、原文本、标签及文本复核备注 |
| `2026-09-23_mosei-stage-map_v1.0.csv` | 原样本到 `sample_0001`—`sample_0100` 的映射，以及原视频/暂存视频哈希 |
| `mosei-stage-v1.0` | 100 个英文名暂存视频，内容与原视频相同 |
| `mosei-audio-v1.0` | 16 kHz、单声道、PCM 16-bit WAV |
| `mosei-audio-egemaps-v1.0` | 逐音频窗口的 25 维 eGeMAPSv02 LLD CSV |
| `mosei-openface-pilot-v1.0` | 全部 100 条视频的 OpenFace CSV 与详情；名称虽含 pilot，现已覆盖全量 |
| `mosei-word-times-v1.0` | 每条样本的词时刻 CSV、CTC 对齐详情 JSON、CTC log-probability NPZ |
| `mosei-text-bert-v1.0` | BERT token/word 特征、token 到原词映射和模型标识 |
| `mosei-multimodal-aligned-v1.0` | 正式使用的 100 个词级三模态压缩 NPZ |
| `pilot-sample` | `sample_0013` 的先行验证产物，批处理回归检查依赖这些文件 |
| `src` | 已保存的预检、暂存、音频、CTC、BERT、合并脚本 |
| `qt-showcase` | 本地 Qt 交互展示入口；读取结果，不参与特征计算 |
| `2026-09-24_q1-typical-example_v1.0.md` / `.json` / `.png` | `sample_0013` 的逐词来源、可视化与数值重算证据 |
| `03_Results/e/question-one` | 100 行样本汇总、65 行待复核队列、数据字段说明、附录表和审计 JSON；路径相对项目根目录 |

已保存的脚本支持对现有中间数据进行校验、增量续跑和重建合并结果。当前仓库尚未保存最初清单构建、原始 OpenFace 批量驱动和 OpenFace QC 清单生成的独立脚本；这些环节有清单、逐视频产物或日志，但不能据此宣称从原始附件开始“一键全流程复现”。

## 2. 已有环境与模型

复用已有两个环境，不重复安装。主环境为 `D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe`，提取环境为 `D:\06_Apps\python-envs\huaweicup-e-extract\Scripts\python.exe`。

| 组件 | 已核对版本/配置 | 作用 |
|---|---|---|
| Python | 3.11.16，Windows 64-bit | 两个环境的运行时 |
| PyTorch / torchaudio | 2.7.1+cu128 / 2.7.1+cu128 | GPU 推理与 CTC 强制对齐 |
| Transformers | 4.57.1 | 加载本地 BERT 和 Wav2Vec2 |
| NumPy / pandas | 2.4.6 / 3.0.6 | 数组、CSV 与 NPZ |
| openSMILE Python 包 | 2.6.0；`eGeMAPSv02` / `LowLevelDescriptors` | 25 维逐窗口低层声学描述子 |
| FFmpeg / FFprobe | 9.0.2-essentials_build-www.gyan.dev | WAV 解码、真实视频帧时间戳 |
| PySide6 | 6.11.2 | 可选的本地 Qt 结果浏览 |
| OpenFace | 当前可执行文件未提供 FileVersion/ProductVersion | 视线、姿态和面部动作特征 |

完整 Python 包列表分别见 `2026-09-23_environment_requirements_v1.2.txt` 和 `2026-09-23_extraction_requirements_v1.0.txt`。主环境清单早于 Qt 安装，未包含后来加入的 PySide6 等 Qt 依赖；它是计算环境快照，不能视为当前全部展示依赖的完整锁定文件。PyTorch 的 `+cu128` 构建需要对应软件源，不能假定普通 PyPI 直接满足该版本。

模型从本机缓存离线读取，两个脚本均使用 `local_files_only=True`：

| 模型 | 固定 revision | 本机路径 |
|---|---|---|
| `google-bert/bert-base-uncased` | `86b5e0934494bd15c9632b12f734a8a67f723594` | `D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594` |
| `facebook/wav2vec2-base-960h` | `22aad52d435eb6dbaf354bdad9b0da84ce7d6156` | `D:\06_Apps\huggingface-data\hub\models--facebook--wav2vec2-base-960h\snapshots\22aad52d435eb6dbaf354bdad9b0da84ce7d6156` |

OpenFace 位于 `D:\06_Apps\openface`。2026-09-24 读取的 `FeatureExtraction.exe` SHA-256 为：

```text
a29ba49cfc59039bfe5e2f141898b2a110da420f6f520d6a923a86ac78cd96ae
```

这是当前二进制的身份指纹，不是对原始下载发行版本的推断。原下载包版本、完整模型文件哈希与当时命令行尚未形成完整版本记录。现有日志可确认 CLNF landmark 跟踪、MTCNN 人脸检测及动态 AU 模块被加载；`*_of_details.txt` 记录 Gaze=1、AUs=1、Pose=1，普通 Landmarks 2D/3D=0、Shape parameters=0。所用相机参数也保留在各视频详情中。

## 3. 复现操作顺序

在 PowerShell 中先设置绝对路径。只复制代码块内容，不复制提示符 `PS ...>` 或 Markdown 标记。

```powershell
$q1Draft = 'D:\01_Projects\HuaweiCup-2026\02_Drafts\e'
$q1MainPython = 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe'
$q1ExtractPython = 'D:\06_Apps\python-envs\huaweicup-e-extract\Scripts\python.exe'
$q1Source = Join-Path $q1Draft 'src'
```

### 3.1 首选：检查已经生成的结果

现有工程的主要验收入口是 `2026-09-24_build-q1-tables_v1.0.py --check`。它只读原始 Excel、100 个源视频和暂存视频、上游质量报告及最终 NPZ，复核身份、哈希、词序、掩码、汇聚权重和音频/视觉词向量数值，并与已有五份导出结果逐字比较。所需组件是 Python 3.11+、NumPy、pandas、openpyxl；**不依赖 Qt、GPU、模型推理或视频解码**。音频 `--check` 另核对 FFmpeg/FFprobe 和暂存视频。

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_extract-batch-audio_v1.0.py') --check
if ($LASTEXITCODE -ne 0) { throw 'Audio preflight failed' }
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_build-q1-tables_v1.0.py') --check
if ($LASTEXITCODE -ne 0) { throw 'Question-one audit failed' }
```

第一次需要从现有输入生成五份结果表时，去掉 `--check` 运行同一脚本；默认运行会重新生成并替换它自己的五份派生报告，`--check` 则只读比较、不会写入。五份文件为 `2026-09-24_q1_sample-summary_v1.0.tsv`、`2026-09-24_q1_review-queue_v1.0.tsv`、`2026-09-24_q1_sample-appendix_v1.0.md`、`2026-09-24_q1_table-schema_v1.0.md`、`2026-09-24_q1_audit_v1.0.json`。复核队列里的 `pending` 表示尚待人耳/画面检查，P1/P2 只是派工顺序。

可选：Qt 程序另有不启动窗口的 `--audit`，用于展示端的数据读取检查；它需要 PySide6，不能代替上述跨文件审计。

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Draft 'qt-showcase\2026-09-24_multimodal-viewer_v1.0.py') --audit
```

如需从现有逐窗口、逐帧特征重新计算并检查合并逻辑，运行：

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_merge-batch-modalities_v1.0.py') --check
if ($LASTEXITCODE -ne 0) { throw 'Multimodal reconstruction failed' }
```

合并 `--check` 不写批量结果，但会读取 100 条视频的实际帧 PTS、重算词级汇聚，并先比对 pilot 音频/视觉特征；它不是只读一下文件名。它检查的是重建逻辑与中间输入，不能替代对已有最终 NPZ 的哈希或数组审计。

典型样本报告已有单独生成脚本，读取保存的 BERT token 特征、声学窗口、OpenFace 帧与最终 NPZ，独立重算特征映射，并生成 Markdown、JSON、PNG，不调用模型。此脚本需要主环境中的 Matplotlib。当前产物对应 `sample_0013`：

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_build-q1-example_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'Typical example reconstruction failed' }
```

### 3.2 从已保留清单开始重建的阶段入口

下表按依赖顺序列出现有脚本。执行前保留配套清单、`pilot-sample` 回归参照和模型缓存。原始数据位置或软件位置改变时，需要在脚本中的已明确常量处配置新路径。

| 顺序 | 解释器 | 脚本 | 核心输入与输出 |
|---|---|---|---|
| 1 | 主环境 | `2026-09-23_batch-preflight_v1.0.py` | 100 行 manifest、原视频、pilot QC、本地 BERT tokenizer；只作预检 |
| 2 | 主环境 | `2026-09-23_stage-videos_v1.0.py` | 由 manifest 顺序生成英文暂存名和 stage map，并逐条哈希校验 |
| 3 | 主环境 | `2026-09-24_extract-batch-audio_v1.0.py` | stage map → WAV 与音频质量表 |
| 4 | 提取环境 | `2026-09-24_extract-batch-egemaps_v1.0.py` | WAV → 音频窗口 CSV 与质量表 |
| 5 | OpenFace 本体 | 原驱动脚本未保存 | 暂存视频 → 现有 OpenFace CSV、详情和日志；QC 表亦作为已有输入保留 |
| 6 | 主环境 | `2026-09-24_align-batch-words_v1.0.py` | 原文本与 WAV → CTC 词时刻、概率和复核标记 |
| 7 | 主环境 | `2026-09-24_extract-batch-bert_v1.0.py` | 原文本与词索引 → BERT token/word 特征 |
| 8 | 主环境 | `2026-09-24_merge-batch-modalities_v1.0.py` | 对齐词时刻 + 三种特征 → 最终词级 NPZ 与质量表 |

步骤 1、2 的保存命令如下。**已有 stage map 时，步骤 2 会主动报 `FileExistsError`，这是防止无意重编号；当前工程不需要重做它，也不要为绕过保护而删除映射。**

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-23_batch-preflight_v1.0.py')
# 仅首次构建、stage map 尚不存在时运行下面一条：
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-23_stage-videos_v1.0.py')
```

其余已保存的计算入口如下；每一步成功后再运行下一步。该命令组用于重建/续跑说明，本次文档整理没有重新运行这些昂贵阶段。

```powershell
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_extract-batch-audio_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'Audio extraction failed' }
& $q1ExtractPython -X utf8 (Join-Path $q1Source '2026-09-24_extract-batch-egemaps_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'eGeMAPS extraction failed' }
# 此时必须已有 100 个 OpenFace CSV 及 2026-09-23_mosei-openface-qc_v1.0.csv。
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_align-batch-words_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'Word alignment failed' }
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_extract-batch-bert_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'BERT extraction failed' }
& $q1MainPython -X utf8 (Join-Path $q1Source '2026-09-24_merge-batch-modalities_v1.0.py')
if ($LASTEXITCODE -ne 0) { throw 'Multimodal merge failed' }
```

音频、eGeMAPS、CTC、BERT 和合并脚本提供 `--check`；这些入口不写批量输出，但检查范围不同：

- 音频 `--check`：校验 FFmpeg/FFprobe 可用性、100 条 stage map 与暂存视频哈希，不解码生成 WAV。
- eGeMAPS `--check`：校验 100 个 WAV 的来源和格式，并重新提取 pilot 声学特征与已存参照比较。
- CTC `--check`：检查 100 个 WAV/原文、预检文本规范化，再加载缓存模型重做 pilot 强制对齐；不是重算或逐一核验 100 份已存词时刻的入口。
- BERT `--check`：检查全部原词、字符区间、已有词时刻表与 tokenizer 映射，再重算 pilot BERT 特征作回归比较；不重新提取全部 100 份 BERT 向量。
- 合并 `--check`：重建 100 条词级汇聚，但不写批量结果。另有 `--pilot-check`，仅检查 pilot 音频/视觉的重建一致性。

增量运行会验证已存文件的来源、哈希或数组内容；不匹配时保留旧文件并报错。方法调整应新建版本，不能直接把旧结果当成新算法输出。正式运行会更新阶段质量 CSV 和方法 JSON，执行前应确认运行版本。GPU/CPU 选择、脚本 SHA-256 等属于部分阶段的运行签名；换硬件、改脚本后旧输出不一定能通过续跑签名检查。

### 3.3 实际提取和对齐参数

音频调用的关键 FFmpeg 参数为：

```text
-nostdin -hide_banner -v error -xerror -y -i <staged_video>
-map 0:a:0 -vn -ac 1 -ar 16000 -c:a pcm_s16le <temporary_wav>
```

`-y` 只用于脚本管理的 `.partial.wav` 临时输出。没有裁剪、降噪、响度归一化或变速。音轨起点、视频轨起点及两者差值记录在音频质量表中。

openSMILE 使用默认 `eGeMAPSv02` 低层描述子，共 25 维；它不同于整段 88 维 functionals。使用库返回的真实 `[start_s,end_s)` 窗口，保留窗口重叠与尾部窗口。有限的零值或库的有限哨兵值不会仅因等于零就被当成缺失。

CTC 使用 Wav2Vec2 的对数概率和 `torchaudio.functional.forced_align`、`merge_tokens`。模型处于 `eval`/推理模式，不在这 100 条数据上训练。卷积总步长为 320 个音频采样点，时间步为 320/16000=0.02 秒；词首/词尾由归属该词的首末字符 bin 得到，不用“音频长度÷输出帧数”强行缩放。

原文按非空白片段 `\S+` 编号，保留标点和原拼写。CTC 目标转大写、规范化弯引号、去标点、拆分连字符并临时展开数字；展开后的字符仍映射回原词。数字读法、说话人括注和原文疑点会保留复核标记。`sample_0001` 的原词 `answers` 没有依据听感改为 `matters`。无法构成可对齐发音的独立标点或说话人信息仍占据原词行，但不提供伪造时间。

BERT 使用最后一层 `last_hidden_state`，原词的所有 WordPiece token 求均值；特殊 token 的 `word_ids=-1`，不参与任何词均值。模型使用 float32、`eval` 和推理模式；无训练、无截断，最大输入为 84 个 BERT token。单条序列不做固定 50 长度处理。

视觉使用有效帧条件 `success==1`、`confidence>=0.8`、全部 49 维有限。视觉区间从 FFprobe 的真实视频帧 PTS 生成，按呈现顺序与 OpenFace 第 1…N 行匹配，并检查总数一致；前一帧的结束时刻是下一帧 PTS，末帧采用已报告正时长。转换到音频零点的公式是 `video_pts - video_stream_start - audio_vs_video_start`。

两种时序模态均以“源窗口/帧区间与词区间的交叠时长”为权重，并按有效交叠总时长归一化。覆盖率使用交叠区间的并集长度除以词长，避免音频重叠窗口重复计时。只要存在正的有效交叠，模态 mask 就为真；部分覆盖单独保留，不能把 mask 为真解释成覆盖完整。

## 4. 最终 NPZ 数据字典

每条样本一个文件，使用 `numpy.load(..., allow_pickle=False)`。共 50 个字段，无 Python object 数组。记 `N` 为原文非空白词数，`A` 为音频窗口数，`V` 为视频帧数；N 在 5—65 之间，**变长、无填充、无重采样、无归一化**。相同词的三种特征使用相同行号。

| 字段 | 形状/类型 | 含义 |
|---|---|---|
| `text`, `audio`, `vision` | `(N,768)`, `(N,25)`, `(N,49)`；float32 | 词级特征；音频/视觉无观测行使用零占位 |
| `words` | `(N,)`；Unicode | 原词原拼写与标点 |
| `word_indices` | `(N,)`；int64 | 从 1 开始的原词编号，非 NumPy 行号 |
| `word_char_spans` | `(N,2)`；int64 | 原转写字符串的零基半开字符区间，可用 `text[a:b]` 还原词 |
| `word_times_s` | `(N,2)`；float64 | 以解码音频起点为零的 `[start,end)` 秒；未计时词为 NaN |
| `sequence_length` | 标量 int64 | N，不是有效模态数 |
| `timing_mask` | `(N,)`；bool | 是否有可用的自动词时间区间 |
| `text_mask`, `audio_mask`, `vision_mask` | 各 `(N,)`；bool | 对应模态是否有可用特征；视觉可部分覆盖 |
| `all_modalities_mask` | `(N,)`；bool | `timing_mask & text_mask & audio_mask & vision_mask` |
| `audio_frame_count`, `vision_frame_count` | 各 `(N,)`；int64 | 对该词有正有效交叠、参与汇聚的源窗口/帧个数 |
| `audio_window_coverage`, `vision_coverage` | 各 `(N,)`；float64 | 词区间被有效源区间覆盖的比例，取值 0—1 |
| `audio_source_weights`, `vision_source_weights` | `(N,A)`, `(N,V)`；float32 | 源行到词行的归一化权重；有效词行权重和约 1，无观测行全零 |
| `audio_source_intervals_s`, `vision_source_intervals_s` | `(A,2)`, `(V,2)`；float64 | 与上述权重列一一对应的真实源区间，单位秒 |
| `vision_source_valid_mask` | `(V,)`；bool | 原始视觉帧通过成功/置信度/有限值检查的标记 |
| `audio_feature_names`, `vision_feature_names` | `(25,)`, `(49,)`；Unicode | 特征列名称和顺序；BERT 的 768 维为潜在表征，无逐维物理名称 |
| `sample_id`, `video_id`, `clip_id` | 字符串标量 | 暂存 ID、原始视频 ID、原始片段 ID；原片段 ID 按字符串读取 |
| `source_video_relpath` | 字符串标量 | 相对于原数据集中 `label-100.xlsx` 同目录的原视频路径 |
| `transcript` | 字符串标量 | 原始转写全文 |
| `label`, `label_original`, `annotation` | float64 标量、字符串、字符串 | 随样本保留的原标注及数值原表示；不是模型预测 |
| `transcript_review`, `transcript_qc_note` | 字符串标量 | 原始清单的文本复核状态和备注 |
| `alignment_status`, `alignment_review_flags` | 字符串标量 | `auto_checked`/`needs_review` 与分号分隔的 CTC 阶段标记 |
| `word_review_flags` | `(N,)`；Unicode | 每个原词的复核原因 |
| `word_ctc_probability` | `(N,)`；float64 | 对齐路径上归属于该词字符的平均概率；未计时词 NaN；不是人工实测准确率 |
| `merge_review_flags`, `needs_review` | 字符串标量、bool 标量 | 汇聚缺失/覆盖标记，以及对齐或合并任一阶段需复核的联合状态 |
| `exact_word_boundaries_manually_verified` | bool 标量 | 当前均为 False，不能把自动对齐成功称为人工精确边界认证 |
| `audio_pooling`, `vision_pooling` | 字符串标量 | 采用的汇聚方法说明 |
| `vision_confidence_threshold` | float64 标量 | 0.8 |
| `time_origin` | 字符串标量 | 统一时间原点说明 |
| `policy_json`, `provenance_json` | JSON 字符串标量 | 策略、上游文件 SHA-256、脚本 SHA-256、对齐记录 |
| `text_model_id`, `text_model_revision`, `text_pooling` | 字符串标量 | BERT 模型、revision 与词均值池化方法 |

`provenance_json` 保存的是相对 `02_Drafts/e` 的上游路径及哈希。合并文件含有原始标签只用于溯源；问题一计算流程未用情绪标签训练特征模型。

BERT 中间 NPZ 还保留 `token_features`、`input_ids`、`attention_mask`、`token_type_ids`、`tokens`、`word_ids`。其中 `word_ids` 对原词采用零基编号，最终 `word_indices` 采用一基编号，两者不要混淆。该中间文件用于追溯 token→word 映射，读取最终词级特征不强制依赖它。

### 4.1 可独立运行的 NumPy 读取例子

只需要 Python 与 NumPy；不需要 GPU、Qt 或模型权重。把下面代码保存为临时 `.py` 文件，或在 UTF-8 Python 环境中运行。它只读最终 NPZ，并保留无时间词和缺失模态。

```python
from pathlib import Path
import numpy as np

root = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e")
path = root / "mosei-multimodal-aligned-v1.0" / "sample_0013.npz"
with np.load(path, allow_pickle=False) as z:
    n = int(z["sequence_length"].item())
    words = z["words"]
    times = z["word_times_s"]
    timed = z["timing_mask"]
    assert len(words) == n
    for key, dim in (("text", 768), ("audio", 25), ("vision", 49)):
        assert z[key].shape == (n, dim)
        assert np.isfinite(z[key]).all()
    assert np.isfinite(times[timed]).all()
    assert np.isnan(times[~timed]).all()
    assert np.array_equal(
        z["all_modalities_mask"],
        timed & z["text_mask"] & z["audio_mask"] & z["vision_mask"],
    )
    for i, word in enumerate(words):
        span = f"{times[i, 0]:.3f}-{times[i, 1]:.3f}s" if timed[i] else "untimed"
        status = tuple(bool(z[k][i]) for k in ("text_mask", "audio_mask", "vision_mask"))
        print(int(z["word_indices"][i]), word, span, status)
    # 展示/统计时显式保留缺失；不要据零值判断模态有效性。
    audio_for_display = z["audio"].copy()
    audio_for_display[~z["audio_mask"]] = np.nan
    vision_for_display = z["vision"].copy()
    vision_for_display[~z["vision_mask"]] = np.nan
    print("sample:", z["sample_id"].item(), "words:", n)
    print("valid in all modalities:", int(z["all_modalities_mask"].sum()))
```

无有效视觉帧的 4 个样本是 `sample_0010`、`sample_0011`、`sample_0042`、`sample_0088`。它们保留原文本、音频、词序与样本身份，`vision_mask` 全 False；零占位不等于没有表情或中性情绪。9 个未计时词也仍在结果中，不能删除后沿用旧行号。

## 5. 验收依据与人工复核边界

计算结果以阶段质量表和合并表为依据：100 个样本、1,932 个原始词、1,923 个有自动词时刻的词、1,828 个三模态同时有效词；65 个样本带复核标记，其中 CTC 阶段 63 个、合并阶段 19 个，两组有交集。`status=ok` 表示该阶段文件生成/核验通过，不表示不存在复核问题。

CTC 复核条件包括 ASR WER>0.35、词平均路径概率<0.20、词跨度>2秒、至少4个对齐字符被压缩到≤20ms，以及数字读法/文本差异等规则。它们用于安排检查顺序，不能作为对齐准确率。已有人耳确认的 pilot 文本与短语层面停顿、语序、语速记录在 `pilot-sample/2026-09-23_pilot-qc_v1.0.json`；精确词边界未逐词人工核定。

新一轮人工复核应独立记录样本 ID、原词编号、疑点、听看结论和是否修改，不直接覆盖 `01_Source`。若需调整读法、文本或词时刻，应保留原版本，并同步重建 BERT 映射（当文本改变时）、相关音视频词汇聚、质量清单和 Qt 展示数据，避免不同版本错配。

现有 `mosei-multimodal-method`、`mosei-word-alignment-method`、`mosei-text-bert-method` JSON 记录策略、模型版本和脚本哈希。跨 GPU、驱动或算子版本可能出现微小浮点差异；当前回归检查有数值容差，而已存合并 NPZ 复用检查逐数组相等，不能承诺不同机器逐字节一致。

## 6. 提交体积规划与后续补齐项

按题目附件总量不超过 50 MB 规划，最终压缩包必须实测字节数；为避免 MB/MiB 换算争议，可用 50,000,000 字节作为上限。现有最终 100 个合并 NPZ 合计约 7.37 MiB，保留了全部掩码、词映射及汇聚权重，适合作为核心结果。建议一并提交 100 行汇总表、必要词级明细、方法 JSON、模型与复现说明、计算脚本及少量典型结果图。

原始视频约 98.81 MiB，OpenFace 逐帧 CSV/详情/日志合计约 58.88 MiB，不能把这些目录整份塞入限额附件。原视频、模型权重、Python 环境与软件本体不随结果包重复打包；引用题目原附件与固定模型 revision。完整中间产物保留在本地，用于追溯和必要时复算。Qt 本地展示依赖视频和配套展示数据；轻量提交包不能宣称开箱即可播放所有视频。

在声明“从原始附件完整重建”前仍需补齐：

1. 将从 `label-100.xlsx` 构建 manifest 的规则保存为独立入口。新审计脚本已核对现有 100 行原文、标签和原表行号；仍缺的是首次创建清单的入口，而不是现有清单的身份核验。
2. 保存 OpenFace 原批处理调用或明确标注的新复现驱动，补上二进制来源版本、模型文件指纹及 QC 生成入口；不能用推测的命令覆盖历史说明。
3. 处理对 `pilot-sample` 的固定依赖：提交必要的小型回归参照，或新增明确的首次构建模式；现有脚本不支持直接省略这些文件。
4. 正式外发前将硬编码项目/工具/模型路径整理为配置项，并在新位置验证；本说明所列命令针对当前本机路径。
5. 补齐高风险人工复核结果。Qt 同步播放是检查工具，不自动完成听辨或发声人身份确认。

本轮说明未修改原始材料、未重新训练模型、未重新批量提取特征，也未把待复核标记改写成已确认。
