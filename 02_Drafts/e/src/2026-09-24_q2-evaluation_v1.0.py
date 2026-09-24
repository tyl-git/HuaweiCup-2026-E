"""Fixed three-class/regression metrics and train-only constant baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / '03_Results' / 'e' / 'question-two'


def regression_metrics(target, prediction):
    y = np.asarray(target, dtype=np.float64).reshape(-1)
    p = np.asarray(prediction, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not y.size:
        raise ValueError('Regression target/prediction lengths must match and be nonempty')
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError('Regression values must be finite')
    yd, pd = y - y.mean(), p - p.mean()
    denom = np.linalg.norm(yd) * np.linalg.norm(pd)
    constant = np.all(y == y[0]) or np.all(p == p[0])
    r = None if y.size < 2 or constant or denom == 0 else float(np.clip(yd @ pd / denom, -1, 1))
    return {'n': int(y.size), 'mae': float(np.mean(np.abs(y-p))), 'pearson_r': r,
            'pearson_status': 'undefined_constant_or_too_few_values' if r is None else 'defined'}


def classification_metrics(target, prediction):
    y, p = np.asarray(target).reshape(-1), np.asarray(prediction).reshape(-1)
    if y.shape != p.shape or not y.size:
        raise ValueError('Classification lengths must match and be nonempty')
    if not np.isin(y, [0, 1, 2]).all() or not np.isin(p, [0, 1, 2]).all():
        raise ValueError('Expected class IDs 0=negative, 1=neutral, 2=positive')
    cm = np.bincount(3*y.astype(np.int64)+p.astype(np.int64), minlength=9).reshape(3, 3)
    support, predicted, tp = cm.sum(1), cm.sum(0), cm.diagonal()
    prec = np.divide(tp, predicted, out=np.zeros(3), where=predicted != 0)
    recall = np.divide(tp, support, out=np.zeros(3), where=support != 0)
    f1 = np.divide(2*tp, support+predicted, out=np.zeros(3), where=support+predicted != 0)
    return {'n': int(y.size), 'accuracy': float(tp.sum()/y.size), 'macro_f1': float(f1.mean()),
            'confusion_matrix_true_rows_predicted_columns': cm.tolist(),
            'per_class': [{'id': i, 'name': n, 'support': int(support[i]),
                           'precision': float(prec[i]), 'recall': float(recall[i]), 'f1': float(f1[i])}
                          for i, n in enumerate(('Negative', 'Neutral', 'Positive'))],
            'zero_division_policy': 0}


def self_test():
    c = classification_metrics([0, 0, 1, 2], [0, 1, 1, 2])
    assert c['confusion_matrix_true_rows_predicted_columns'] == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert c['accuracy'] == 0.75 and abs(c['macro_f1'] - 7/9) < 1e-12
    r = regression_metrics([-1, 0, 1], [-2, 0, 2])
    assert abs(r['mae'] - 2/3) < 1e-12 and abs(r['pearson_r'] - 1) < 1e-12
    assert regression_metrics([-1, 0, 1], [0.1, 0.1, 0.1])['pearson_r'] is None
    assert regression_metrics([0], [1])['pearson_r'] is None
    assert classification_metrics([1], [1])['macro_f1'] == 1/3
    for fn, y, p in ((regression_metrics, [0], [float('nan')]),
                      (classification_metrics, [0], [3]), (regression_metrics, [0, 1], [0])):
        try:
            fn(y, p)
        except ValueError:
            continue
        raise AssertionError('Invalid metric inputs were accepted')
    print('Metric self-tests passed')


def constant_baselines():
    paths = list((ROOT/'01_Source'/'E').rglob('aligned_50.pkl'))
    if len(paths) != 1:
        raise RuntimeError(f'Expected one official aligned_50.pkl, got {len(paths)}')
    path = paths[0]
    with path.open('rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    with path.open('rb') as handle:
        data = pickle.load(handle)
    train_cls = np.asarray(data['train']['classification_labels']).reshape(-1)
    train_y = np.asarray(data['train']['regression_labels'], dtype=np.float64).reshape(-1)
    classification_metrics(train_cls, train_cls)
    regression_metrics(train_y, train_y)
    counts = np.bincount(train_cls.astype(np.int64), minlength=3)
    majority = int(counts.argmax())
    mean, median = float(train_y.mean()), float(np.median(train_y))
    result = {'source_relpath': str(path.relative_to(ROOT)), 'source_sha256': digest,
              'fit_split': 'train', 'evaluated_splits': ['train', 'valid'],
              'test_evaluated': False, 'specialists_evaluated': False,
              'constants': {'majority_class': majority, 'class_counts': counts.tolist(),
                            'intensity_mean': mean, 'intensity_median': median},
              'results': {}}
    for split in ('train', 'valid'):
        c = np.asarray(data[split]['classification_labels']).reshape(-1)
        y = np.asarray(data[split]['regression_labels']).reshape(-1)
        result['results'][split] = {
            'majority_class': classification_metrics(c, np.full(c.shape, majority)),
            'mean_intensity': regression_metrics(y, np.full(y.shape, mean, dtype=np.float64)),
            'median_intensity': regression_metrics(y, np.full(y.shape, median, dtype=np.float64))}
    return result


def render_report(result):
    c = result['constants']
    lines = ['# 问题二常数参考基线', '',
             '这组基线只从附件二训练集计算多数类别、平均强度和中位数强度，不读取特征做预测。用于检验评价代码和建立最低比较参照，不代表已训练多模态模型。', '',
             f"训练集多数类为 {c['majority_class']}（Positive），平均强度 {c['intensity_mean']:.6f}，中位数强度 {c['intensity_median']:.6f}。类别并列时固定取编号较小者；F1 为固定三类的 macro-F1，无预测类别的分母为零时取 0。", '',
             '| 划分 | 样本 | 多数类 Accuracy | 多数类 macro-F1 | 均值 MAE | 中位数 MAE | Pearson r |',
             '|---|---:|---:|---:|---:|---:|---|']
    for split, r in result['results'].items():
        m = r['majority_class']
        lines.append(f"| {split} | {m['n']} | {m['accuracy']:.6f} | {m['macro_f1']:.6f} | {r['mean_intensity']['mae']:.6f} | {r['median_intensity']['mae']:.6f} | 不可定义（常数预测） |")
    lines += ['', '均值与中位数都是预先设定的参考方法；这里只报告两者，未根据验证结果调整其值。每类指标及混淆矩阵保存在同名 JSON。官方 test 和附件三、四未做预测或性能评估。', '',
              '验证通过：手算混淆矩阵与 macro-F1、线性预测的 MAE/Pearson、常数预测的不可定义相关系数、非法输入拒绝。', '',
              f"源数据 SHA-256：`{result['source_sha256']}`。", '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true', help='Only verify metric edge cases')
    parser.add_argument('--check', action='store_true', help='Recompute baselines and compare reports without writing')
    args = parser.parse_args()
    self_test()
    if args.self_test:
        return
    result = constant_baselines()
    outputs = {OUT/'2026-09-24_q2-constant-baselines_v1.0.json': json.dumps(result, ensure_ascii=False, indent=2)+'\n',
               OUT/'2026-09-24_q2-constant-baselines_v1.0.md': render_report(result)}
    if not args.check:
        OUT.mkdir(parents=True, exist_ok=True)
    for path, content in outputs.items():
        if args.check:
            if path.read_text(encoding='utf-8') != content:
                raise RuntimeError(f'Export mismatch: {path.name}')
        else:
            path.write_text(content, encoding='utf-8')
    print('Constant baseline reports verified' if args.check else 'Constant baseline reports written')
    print(json.dumps(result['results']['valid'], ensure_ascii=False))


if __name__ == '__main__':
    main()
