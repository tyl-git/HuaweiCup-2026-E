"""Fit train-only statistics on verified BERT/official inputs and audit transforms."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import importlib.util
import sys

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_Results/e/question-two'
STATS = OUT / '2026-09-24_q2-normalization_v1.0.npz'


def module(name, filename):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(filename))
    result = importlib.util.module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def render_report(report):
    lines = ['# 训练集标准化参数与输入验收', '',
             '均值、总体标准差仅由官方 train 中的内容且有观测位置拟合；文本来自统一冻结 BERT 缓存，音频和视觉来自官方附件二。保存 float64 统计量，输出为 float32。验证集只应用训练统计量，test 和专项集没有参与拟合或本次变换。', '',
             '| 模态 | 维数 | 训练参与位置 | 标准差设下界的维数 | train 缺整模态样本 | valid 缺整模态样本 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, stats in report['statistics'].items():
        lines.append(f"| {key} | {stats['dimensions']} | {stats['count']} | {stats['floored_scale_dimensions']} | {report['transforms']['train'][key]['samples_without_observation']} | {report['transforms']['valid'][key]['samples_without_observation']} |")
    lines += ['', '已验收：缺失、特殊 token 和 padding 的输出严格为零；独立观测掩码保留；全部输出有限；训练统计重算一致；保存后载入一致。标准化后真实观测可能恰为零，模型必须使用独立掩码，不可再按数值是否为零重新判缺失。', '',
              '音频/视觉的全零原始行只按“无非零观测”处理，不能由此断言发生了现实遮挡。训练集 110 条全视觉无观测样本和验证集 15 条仍保留，不删除、不伪造视觉特征。', '',
              '生成物包括同名 NPZ 统计包、JSON 来源清单和 `q2-normalization-check` 验收 JSON。标准化数组在验收时计算但不重复保存；后续训练按缓存与统计包即时变换。', '',
              '本轮尚未训练情感模型、开展局部缺失增强实验或生成专项预测。', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Recompute and compare existing statistics/reports, without writing')
    args = parser.parse_args()
    loader = module('q2_data_normalization_runner', '2026-09-24_q2_data_v1.0.py')
    cache = module('q2_text_normalization_runner', '2026-09-24_prepare-q2-text_v1.0.py')
    norm = module('q2_normalization_runner', '2026-09-24_q2-normalization_v1.0.py')
    batches = loader.load_official()
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {key: cache.verify_cache(batches[key], model_info) for key in ('train', 'valid')}
    features = {key: {'text': np.load(cache.OUT / f'{key}.npy', mmap_mode='r', allow_pickle=False),
                      'audio': batches[key].audio, 'vision': batches[key].vision}
                for key in ('train', 'valid')}
    train = batches['train']
    provenance = {'source_sha256': dict(train.source_sha256),
                  'text_cache_contract': cache_meta['train']['contract'],
                  'text_feature_sha256': cache_meta['train']['feature_sha256']}
    fitted = norm.MultimodalStandardizer().fit(
        features['train'], train.content_mask, train.observation_masks,
        split=train.split, source_metadata=provenance)
    print('Train-only statistics fitted for 3395 samples', flush=True)
    if not args.check and not STATS.exists() and not STATS.with_suffix('.json').exists():
        fitted.save(STATS)
    stored = norm.MultimodalStandardizer.load(STATS)
    # Creation time differs on repeat fits; all algorithm/provenance fields must match.
    for key, value in fitted.metadata.items():
        if key != 'created_utc' and stored.metadata[key] != value:
            raise RuntimeError(f'Stored normalization metadata changed: {key}')
    for modality, stats in fitted.statistics.items():
        previous = stored.statistics[modality]
        if previous.count != stats.count:
            raise RuntimeError(f'{modality}: train count mismatch')
        for field in ('mean', 'std', 'scale'):
            np.testing.assert_array_equal(getattr(previous, field), getattr(stats, field))
    report = {'fit_split': 'train', 'train_samples': len(train),
              'statistics': fitted.metadata['statistics'],
              'statistics_file_sha256': cache.sha(STATS),
              'cache_feature_sha256': {key: value['feature_sha256'] for key, value in cache_meta.items()},
              'test_transformed': False, 'specialists_transformed': False, 'transforms': {}}
    for key in ('train', 'valid'):
        batch = batches[key]
        result = stored.transform(features[key], batch.content_mask, batch.observation_masks)
        report['transforms'][key] = {}
        for modality, array in result.features.items():
            observed = result.observation_masks[modality]
            np.testing.assert_array_equal(observed, batch.observation_masks[modality])
            if np.any(array[~observed] != 0) or not np.isfinite(array).all():
                raise RuntimeError(f'{key}/{modality}: mask/finite assertion failed')
            active = array[observed].astype(np.float64)
            active_mean, active_std = active.mean(axis=0), active.std(axis=0)
            if key == 'train':
                np.testing.assert_allclose(active_mean, 0, atol=2e-7)
                stats = stored.statistics[modality]
                np.testing.assert_allclose(active_std, stats.std / stats.scale, atol=2e-7)
            report['transforms'][key][modality] = {
                'shape': list(array.shape), 'observed_positions': int(observed.sum()),
                'samples_without_observation': int((~observed.any(axis=1)).sum()),
                'all_finite': True, 'masked_output_exact_zero': True, 'observation_mask_unchanged': True,
                'max_absolute_observed_mean': float(np.abs(active_mean).max()),
                'max_absolute_value': float(np.abs(active).max()),
            }
        del result
        print(f'{key}: transformed features, finite values and independent masks verified', flush=True)
    outputs = {
        OUT / '2026-09-24_q2-normalization-check_v1.0.json': json.dumps(report, ensure_ascii=False, indent=2)+'\n',
        OUT / '2026-09-24_q2-normalization_v1.0.md': render_report(report),
    }
    for path, content in outputs.items():
        if args.check:
            if path.read_text(encoding='utf-8') != content:
                raise RuntimeError(f'Export mismatch: {path.name}')
        else:
            path.write_text(content, encoding='utf-8')
    print('Normalization artifact and reports verified' if args.check else 'Normalization artifact and reports saved', flush=True)


if __name__ == '__main__':
    main()
