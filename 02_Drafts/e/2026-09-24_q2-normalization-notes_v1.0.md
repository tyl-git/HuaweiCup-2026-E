# 问题二训练集标准化模块

模块：`src/2026-09-24_q2-normalization_v1.0.py`。它只做数值标准化，不加载或修改原始 PKL，不执行文本编码。文本输入必须来自训练、验证、测试和专项一致的冻结 BERT 编码流程；不能把加载器的 `reference_text` 混作专项所需的统一文本定义。

## 统计口径

对每个模态 m，仅将满足下列条件的序列位置加入拟合：

`effective_mask[m] = content_mask & observation_masks[m] & any(raw_features[m] != 0, axis=-1)`。

`content_mask` 排除 CLS、SEP 与 PAD；调用者可再通过观测掩码明确屏蔽缺失位置。单个维度等于 0 是合法观测，只有整行全零才从拟合集合中排除。三种模态独立统计，不能用三模态交集掩码缩小各自的拟合集。

均值和总体标准差按训练内容位置计算（`ddof=0`），并用 float64 分块合并均值与平方偏差。每维尺度为 `max(std, 1e-6)`；常量维度不会除以零。输出有效位置为 `(x-mean)/scale` 的 float32 值，其他位置为精确全零。模块不截断大值，也不以验证集均值重新居中。

标准化后，真实观测可能恰好得到全零向量。因此模型必须继续使用返回的 `observation_masks`；不能重新用标准化后的非零性推断缺失。

## 与统一加载器衔接

```python
features = {
    "text": frozen_bert_train,  # [N, 50, 768]，统一编码结果
    "audio": train.audio,
    "vision": train.vision,
}
normalizer = normalization.MultimodalStandardizer()
normalizer.fit(
    features,
    train.content_mask,
    train.observation_masks,
    split=train.split,
    source_metadata={
        "source_sha256": dict(train.source_sha256),
        "text_encoding": train_bert_cache_manifest,
    },
)
normalized_train = normalizer.transform(
    features, train.content_mask, train.observation_masks,
)
normalizer.save(results_dir / "2026-09-24_q2-normalization-stats_v1.0.npz")
```

`fit` 只接受显式 `split="train"`，同一实例禁止再次拟合。该检查依赖调用者如实提供训练划分；模块同时保存精确输入指纹，供上层管线核验来源。验证/测试/专项调用 `transform`，不调用 `fit`。各模态在训练中完全没有可观测位置时，模块报错并要求解决输入问题；验证/测试中的整模态缺失可以正常返回零占位和全假观测掩码。

`transform` 返回 `NormalizedFeatures`，字段为 `features`、`content_mask`、`observation_masks` 和 `raw_nonzero_masks`。所有返回数组设为只读；输入数组不被修改。

## 保存、载入与验收

`save(path.npz)` 同时创建同名 `.json`，不覆盖已有文件，也拒绝写入 `01_Source`。NPZ 存储每个模态的 mean/std/scale；JSON 记录计数、维度、安全下界、来源元数据、原始数组/掩码 SHA-256、统计 NPZ SHA-256 以及规范化 JSON 内容 SHA-256。`load` 核对两层指纹及统计形状、有限值和尺度规则。校验用于发现不一致和意外修改，不是数字签名。

`--self-test` 使用合成数据覆盖训练划分限制、非内容位置的极值排除、显式观测掩码、整行零与单维零区别、常量维度、缺失保持零、有效零仍保留掩码、验证分布偏移不重新拟合、输入只读、NPZ/JSON 往返及损坏检测。`--inspect <统计文件.npz>` 只读校验并输出清单。

实际训练统计量和其验收结果由主预处理管线生成并记录；模块自测不能替代真实数据验收。
