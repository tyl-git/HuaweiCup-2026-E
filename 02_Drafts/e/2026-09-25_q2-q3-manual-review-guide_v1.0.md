# 问题二与问题三人工复查指南

日期：2026-09-25  
适用范围：华为杯 E 题当前 Q2（多模态情感识别与局部缺失鲁棒性）和 Q3（模型解释与证据定位）结果。  
目标：确认结果文件、输入接口、缺失掩码、模型指标和解释证据都能独立复核，并区分“自动审计通过”和“人工确认”。

## 0. 先理解复查边界

问题二和问题三使用官方 aligned_50 输入接口，三模态维度为：文本 768、音频 74、视觉 35。问题一自提的 768/25/49 特征不能混入 Q2/Q3 的复查或论文表格。

附件三的缺失含义是某一模态在局部连续时间段被置为全零。它不是整段样本的模态删除。复查时必须区分四种状态：

| 名称 | 含义 |
|---|---|
| content_mask | 真实内容位置，排除 padding；文本还排除 CLS/SEP |
| source_observed | 原始文件中该位置是否有有效观测 |
| injected_missing | 为附件三专项测试人为注入的连续缺失区间 |
| effective_observed | 送入模型后仍可使用的观测，通常是 source_observed AND NOT injected_missing |

没有标签的附件三和附件四不能报告准确率、F1 或“预测正确率”。它们只能用于预测展示、缺失行为审计和解释证据分析。

## 1. 环境和路径

在 PowerShell 中先执行：

    $py = 'D:\06_Apps\python-envs\huaweicup-e\Scripts\python.exe'
    $root = 'D:\01_Projects\HuaweiCup-2026'
    $draft = "$root\02_Drafts\e"
    $src = "$draft\src"
    $results = "$root\03_Results\e"
    $q2 = "$results\question-two"
    $q3 = "$results\question-three"
    Set-Location $root

不要重新安装包，也不要修改原始数据、checkpoint、预测 NPZ 或自动报告。人工记录另存为新文件。

## 2. 问题二自动复查

### 2.0 自动核对与人工播放的区别

以下内容不需要人工逐行完成，程序已经可以自动检查：样本数量、表格行数、ID 唯一性、标签是否为 0/1/2、预测是否有限、预测标签是否等于官方标签、附件三预测文件是否完整。

候选 CSV 只是“告诉你应该看哪些样本”的抽样清单，不是人工复查本身。真正的人工复查是打开对应原视频，听和看原始片段，判断转写、语气、画面、时间和模型错误是否有明显原因。人工不需要改官方标签，也不需要重新计算 Accuracy。

当前有一个数据边界：官方 Q2 `aligned_50.pkl` 只保存样本 ID、文本、特征和标签，加载器没有为 727 条 official test 样本保存直接的视频路径。因此，Q2 的候选表不能保证每一条都能直接播放。能找到对应原视频的样本才进行听看；找不到时记为“视频未定位”，保留为 `review`，不能把它当成程序失败。Q3 附件四结果包含 `video_file`，可以直接按时间做播放复查。

### 2.1 数据接口和边界测试

运行两个只读测试：

    & $py -X utf8 "$src\2026-09-24_test-q2-data_v1.0.py" -v
    & $py -X utf8 "$src\2026-09-24_test-q2-missing-intervals_v1.0.py" -v

应看到所有测试通过。测试覆盖以下边界：

- train/valid/test 样本数分别为 3395、728、727；
- 三模态张量形状和 dtype 正确；
- padding 位置不会被误当成有效观测；
- 全视觉缺失样本仍保留为缺失；
- 非有限值、错误标签、形状错误会被拒绝；
- 附件三缺失区间是连续位置，缺失前后位置不被误清零；
- source_observed、injected_missing 和 effective_observed 的语义不混淆。

如果这里失败，不要继续解释指标；先记录失败测试名和 traceback。

### 2.2 冻结评估方案检查

正式评估脚本提供 --check 只读模式：

    & $py -X utf8 "$src\2026-09-24_eval-q2-temporal-final_v1.0.py" --check --device cuda

预期输出应包含：

    Preflight OK: official test=727, attachment 3=30, frozen checkpoints=9

这一步只确认样本数量、checkpoint 哈希、配置和输入来源，不重新训练，也不写预测。若 checkpoint metadata mismatch，停止并报告具体模型/seed；不要删除或覆盖 checkpoint。

### 2.3 独立审计预测文件

    & $py -X utf8 "$src\2026-09-24_audit-q2-temporal-final_v1.0.py"

当前应通过：

- 9 个正式 test 预测文件；
- 270 条附件三全模型/seed 专项预测；
- 30 条选定附件三行；
- 所有类别、概率、强度和 ID 均可读取。

结果目录：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-temporal-final-v1.0\

重点文件：

- summary.json：模型、seed、checkpoint、指标和选择规则；
- temporal_seed_20260926_test_predictions.npz：当前 valid 选择的主模型测试预测；
- attachment3_predictions.csv：选定主模型的附件三预测；
- attachment3_all_models.csv：所有模型和 seed 的附件三预测；
- attachment3_observation_audit.csv：局部缺失掩码审计。

### 2.4 手工复算文件数量、ID 和数值范围

下面命令只读检查主模型 NPZ：

    @'
    from pathlib import Path
    import numpy as np

    p = Path(r"D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-temporal-final-v1.0\temporal_seed_20260926_test_predictions.npz")
    z = np.load(p, allow_pickle=False)
    print('keys:', z.files)
    for k in z.files:
        a = z[k]
        finite = bool(np.isfinite(a).all()) if np.issubdtype(a.dtype, np.number) else 'n/a'
        print(k, a.shape, a.dtype, 'finite=', finite)
    ids = z['ids'].astype(str)
    logits = z['logits']
    intensity = z['intensity']
    print('n=', len(ids), 'unique_ids=', len(set(ids)))
    print('class_range=', logits.argmax(axis=1).min(), logits.argmax(axis=1).max())
    print('intensity_range=', float(intensity.min()), float(intensity.max()))
    '@ | & $py -X utf8 -

判定：n=727、ID 唯一、数值全部 finite、预测类别只在 0/1/2。正式 test 的 ID 顺序必须与缓存中的 test 顺序一致，不能按文件名重新排序后再比较。

### 2.5 查看正式测试指标

打开：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-final-test-v1.0\
    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-temporal-final-v1.0\summary.json

当前固定主模型为 temporal / seed_20260926，seed 由 valid 选择，不能看完 test 后重新挑选。当前主模型指标为：

| 指标 | 当前值 |
|---|---:|
| Accuracy | 0.7015 |
| Macro-F1 | 0.6189 |
| MAE | 0.6156 |
| Pearson r | 0.6938 |

论文中应同时报告三个 seed 的均值和样本标准差，不要只报最好 seed。还要报告混淆矩阵和每类指标。当前需要特别关注 Neutral 类召回率约 0.2405：这说明总体 Accuracy 不能代表三类均衡识别能力。

### 2.6 正式测试人工抽样

从主模型预测中各抽取以下四类样本：

1. 高置信正确：最大 softmax 概率高，且预测类别与真值一致；
2. 高置信错误：最大概率高但类别错误；
3. Neutral 真值样本：分别抽取预测为 Negative、Neutral、Positive 的样本；
4. 强度绝对误差最大的 5 个样本。

建议总计 20 条。对每条记录：test_id、真实类别、预测类别、最大概率、真实强度、预测强度、绝对误差、错误类型、人工备注。人工判断重点是：

- 是否处于情绪边界；
- 语义是否含否定、转折或讽刺；
- 视频中是否存在多人或非说话人；
- 音频是否有明显噪声、截断或静音；
- 视觉是否全缺失或仅局部有效。

高置信错误不是程序错误的同义词，只有在 ID、标签、输入和预测都对应错误时才记录为模型错误。

这里的人工操作应当按以下顺序进行，而不是只看 CSV：

1. 根据候选表中的 `id` 找到对应原始视频或 Qt 查看器中的样本；如果没有视频路径，记录“视频未定位”；
2. 播放完整片段，先听说话内容，再看说话人的表情和动作；
3. 对照转写，记录“听到的内容是否基本一致”；
4. 用普通语言记录片段整体情绪是“明显负面、明显正面、接近中性或不确定”；
5. 再查看官方 `true_class` 和模型 `pred_class`，说明模型错误是否可能由语义模糊、否定/转折、噪声、静音或视觉缺失造成；
6. 如果没有对应视频，记录“原视频未能定位”，判定为 `review`，不要猜测。

人工听看得到的是错误原因线索，不是新的标签真值。即使你觉得视频更像正面，也不能修改官方标签；只在备注中写“人工听感偏正面”或“内容不确定”。

如果不想手工从 727 行中筛选，可以先生成抽样候选表。该命令只读取预测 NPZ，生成一个新的草稿 CSV：

    @'
    from pathlib import Path
    import csv
    import numpy as np

    src = Path(r"D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-temporal-final-v1.0\temporal_seed_20260926_test_predictions.npz")
    dst = Path(r"D:\01_Projects\HuaweiCup-2026\02_Drafts\e\2026-09-25_q2-manual-review-candidates_v1.0.csv")
    z = np.load(src, allow_pickle=False)
    ids = z['ids'].astype(str)
    logits = z['logits'].astype(np.float64)
    true_class = z['true_class'].astype(int)
    pred_class = logits.argmax(axis=1).astype(int)
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    prob = exp / exp.sum(axis=1, keepdims=True)
    confidence = prob.max(axis=1)
    abs_error = np.abs(z['intensity'] - z['true_intensity'])
    rows = []
    for i in np.argsort(-confidence):
        if pred_class[i] == true_class[i]:
            rows.append(('high_conf_correct', i))
            if sum(k == 'high_conf_correct' for k, _ in rows) == 5: break
    for i in np.argsort(-confidence):
        if pred_class[i] != true_class[i]:
            rows.append(('high_conf_error', i))
            if sum(k == 'high_conf_error' for k, _ in rows) == 5: break
    for i in np.flatnonzero((true_class == 1) & (pred_class != 1))[:5]:
        rows.append(('neutral_error', int(i)))
    for i in np.argsort(-abs_error)[:5]:
        rows.append(('largest_intensity_error', int(i)))
    with dst.open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['category','row','id','true_class','pred_class','confidence','true_intensity','pred_intensity','abs_error'])
        for category, i in rows:
            w.writerow([category, i, ids[i], int(true_class[i]), int(pred_class[i]), float(confidence[i]), float(z['true_intensity'][i]), float(z['intensity'][i]), float(abs_error[i])])
    print('Saved:', dst)
    '@ | & $py -X utf8 -

打开该 CSV 后按 category 逐条找到同一个 ID 的原始样本，不能把抽样候选表本身当作新的评估结果。

### 2.7 附件三局部连续缺失复查

查看：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-attachment3-v1.0\attachment3_observation_audit.csv
    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-two\q2-attachment3-v1.0\attachment3_predictions.csv

关键字段：

    file_id
    content_positions
    text_observed_positions
    text_zero_content_positions
    audio_observed_positions
    audio_zero_content_positions
    vision_observed_positions
    vision_zero_content_positions

随机抽查至少 6 个文件，覆盖文本、音频、视觉三种模态以及不同缺失起点。每个文件按以下顺序检查：

1. 读取 content_positions，确认实际内容长度；
2. 查看被置零的连续区间是否位于内容范围内；
3. 确认缺失区间前后仍有有效位置；
4. 确认未被注入的模态没有被同时清零；
5. 确认 padding 不计入缺失比例；
6. 将结果与 attachment3_predictions.csv 的同一 file_id 对上。

判定规则：

- 局部连续全零、其他区间保留：pass；
- 整个模态全零但专项条件不是整段缺失：fail；
- 只能从 CSV 推断、无法定位原始区间：review；
- 无标签附件三只记录预测和输入掩码，不记录“预测正确”。

### 2.8 缺失鲁棒性图和消融结果

打开：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\paper-assets-v1.0\q2_test_ablation_v1.0.png
    D:\01_Projects\HuaweiCup-2026\03_Results\e\paper-assets-v1.0\q2_missing_heatmap_v1.0.png

阅读顺序：先看完整输入，再看单模态基线、融合基线、时间模块、门控/增强消融，最后看附件三局部缺失热图。重点不是某一格是否上升，而是：

- 缺失比例从 20% 到 40% 时性能是否平稳；
- 缺失起点在 start/middle/end 时是否有系统差异；
- 文本缺失是否比音频或视觉更敏感；
- 结果是否在三个 seed 上方向一致。

不要把“局部缺失下性能变化很小”写成“模型能处理整模态完全缺失”，因为当前实验没有覆盖整模态删除。

## 3. 问题三自动复查

### 3.1 解释 checkpoint 只读核验

当前 Q3 使用 valid 选择的冻结 temporal checkpoint。先找到主 checkpoint：

    $ckpt = "$q2\q2-temporal-v1.0\temporal\seed_20260926\best.pt"
    Test-Path $ckpt

运行解释脚本的检查模式：

    & $py -X utf8 "$src\2026-09-24_eval-q3-explanations_v1.0.py" --check --checkpoint $ckpt --device cuda

此模式应确认 checkpoint 哈希、模型配置、valid/附件四数量和输入来源，不重新训练、不选择 checkpoint、不写新解释结果。

### 3.2 核对解释结果文件

目录：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-three\q3-explanations-v1.0\

文件与用途：

- summary.json：解释方法、checkpoint 和统计摘要；
- attachment4_predictions.csv：附件四 20 条预测；
- attachment4_explanations.csv：每条样本的主模态、gate/attention 候选证据；
- attachment4_position_effects.csv：逐位置遮挡后的概率和强度变化。

必须确认：

- 附件四为 20 条，不能把解释样本数写成 test 样本数；
- 每个候选位置都有原始模态和位置索引；
- 遮挡前后使用同一个冻结模型；
- 解释结果没有使用附件四标签调参；
- primary_modality 是候选主模态，不是因果结论。

### 3.3 逐样本遮挡核验

当前统计：gate×time 候选位置平均预测类概率下降约 0.04269，随机位置约 0.00738，候选强于随机约 0.69。这支持“候选位置更有影响”，但不能单独证明因果解释。

至少人工检查 4 条样本，其中必须包括样本 04：

1. 在 attachment4_explanations.csv 找到样本及其候选位置；
2. 在 attachment4_position_effects.csv 找到相同 sample_id、modality、position_0based；
3. 比较 candidate_top1、class_probability_delta 和 intensity_delta；
4. 确认遮挡后概率变化方向与摘要统计一致；
5. 对照原始文本或音频，确认被遮挡位置确实存在内容；
6. 若变化接近 0，记录“候选未产生明显影响”，不要强行解释。

样本 04 中，文本词 not 的单位置遮挡使预测类别概率下降约 0.23752，可作为论文中的完整案例，但必须同时展示遮挡前后预测和词位置，不能只放 gate 权重。

### 3.4 模态影响统计的正确读法

当前附件四主模态统计为：文本 16/20、音频 4/20、视觉 0/20。这只能说明当前解释规则下，候选敏感位置主要落在文本和音频；不能写成“视觉对情感没有作用”。视觉可能被模型用于非 top-1 证据，或被输入质量、全缺失样本比例影响。

论文中可写：

> gate/attention 用于生成候选证据，随后通过单位置遮挡重算预测验证其敏感性；因此报告的是遮挡敏感性，而非将注意力权重直接等同于解释。

### 3.5 时间映射和视频人工复核

先运行两个只读检查：

    & $py -X utf8 "$src\2026-09-24_map-q3-evidence-time_v1.0.py" --check --device cuda
    & $py -X utf8 "$src\2026-09-24_localize-q3-evidence_v1.0.py" --check --device cuda

结果目录：

    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-three\q3-evidence-time-v1.0\
    D:\01_Projects\HuaweiCup-2026\03_Results\e\question-three\q3-localization-v1.0\

重点文件：

- attachment4_evidence_times.csv：候选证据到秒数的映射；
- attachment4_word_times.csv：离线 CTC 词时间；
- attachment4_time_quality.csv：时间质量和需复核状态；
- attachment4_localized_evidence.csv：包含文本片段、字符范围和视频文件的定位结果。

当前有 12 条 automatic candidate、8 条 review。对 8 条 review 逐条播放原视频，记录：

- 证据词是否真的被说出；
- 秒数是否大致落在该词的发音区间；
- 视频帧是否显示说话人或相关视觉事件；
- ASR 是否存在明显识别错误；
- 音频时长、视频时长和帧数是否一致。

人工确认之前，时间只能写成“自动估计候选时间”；不能写成“人工精确定位”。

先把需要人工播放的行列出来：

    Import-Csv "$q3\q3-evidence-time-v1.0\attachment4_time_quality.csv" |
      Where-Object { $_.alignment_status -match 'review' } |
      Select-Object file_id, video_file, video_duration_s, transcript_words, timed_words, asr_wer, mean_ctc_probability, alignment_status, error |
      Format-Table -AutoSize

查看样本 04 的文本候选和遮挡影响：

    Import-Csv "$q3\q3-explanations-v1.0\attachment4_explanations.csv" |
      Where-Object { $_.sample_id -match '(^|/)04$|sample_04' } |
      Format-List
    Import-Csv "$q3\q3-explanations-v1.0\attachment4_position_effects.csv" |
      Where-Object { $_.sample_id -match '(^|/)04$|sample_04' -and $_.modality -eq 'text' } |
      Sort-Object { [double]$_.class_probability_delta } |
      Select-Object -First 10 |
      Format-Table -AutoSize

### 3.6 Q3 的通过标准

满足以下条件才可将单条解释记为 pass：

1. 候选位置在原始内容中存在；
2. 遮挡后预测概率或强度确实发生变化，且变化记录可复算；
3. 时间映射没有明显落在视频外或静音区；
4. 若时间质量标为 review，已人工播放确认；
5. 没有把 gate/attention 单独称为最终解释。

如果只有候选权重、没有遮挡变化，记为 review；如果位置不存在、索引错位或时间明显越界，记为 fail。

## 4. 统一人工记录表

建议新建一个独立 TSV，不覆盖自动结果：

    sample_id    question    file_or_time    check_item    observation    judgment    evidence    note

字段说明：

- sample_id：正式 test、附件三或附件四 ID；
- question：Q2 或 Q3；
- file_or_time：CSV 文件名、行号、视频秒数或帧号；
- check_item：输入、mask、预测、遮挡、时间映射等；
- observation：只写观察到的事实；
- judgment：只能填 pass、review、fail；
- evidence：截图名、CSV 行号或命令输出；
- note：论文表述限制或后续处理。

推荐记录顺序：

1. 自动测试和独立审计；
2. Q2 正式 test 预测 20 条抽样；
3. Q2 附件三 6 条局部缺失抽样；
4. Q3 遮挡案例 4 条；
5. Q3 时间 review 样本 8 条；
6. 汇总 pass/review/fail 数量和未解决问题。

## 5. 论文中可以写什么

可以写：

- 主模型和 seed 是根据 valid 结果预先固定；
- test 结果按 Accuracy、Macro-F1、MAE、Pearson r 报告，并给出多 seed 均值/标准差；
- 附件三采用局部连续时间段全零的缺失设置，并显式保存多种 mask；
- 附件四解释先用 gate/attention 产生候选，再用遮挡实验验证预测敏感性；
- 证据时间由离线 CTC/词时间映射自动估计，部分样本经过人工复核。

不能写：

- 仅凭 attention/gate 权重就声称“模型解释”；
- 附件三证明了整模态完全缺失鲁棒性；
- 无标签附件三/四有准确率或解释准确率；
- 没有人工检查的时间全部精确到词级真值；
- 只报告三个 seed 中最高的 test 结果而不报告波动；
- 将问题一的 768/25/49 特征与问题二、三的 768/74/35 特征混为同一输入。

## 6. 最终复查完成标志

问题二完成：两个数据测试通过、正式评估 --check 通过、独立审计通过、20 条 test 抽样有记录、6 条附件三局部缺失有记录、三 seed 结果和混淆矩阵已保存。  
问题三完成：解释 --check 通过、4 条遮挡案例有记录、8 条时间 review 样本已播放、自动候选与人工确认状态已区分。

所有未实际查看或无法独立复算的项目都保留为 review，不要为了让清单全绿而改写原始结果。
