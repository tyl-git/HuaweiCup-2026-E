"""Verify and illustrate the saved pilot without rerunning extraction models."""
from pathlib import Path
import csv
import hashlib
import json
import wave

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

DRAFT = Path(__file__).resolve().parents[1]
PREFIX = '2026-09-24_q1-typical-example_v1.0'
SAMPLE = 'sample_0013'


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [x.strip() for x in reader.fieldnames]
        return list(reader)


def main():
    paths = {
        'merged': DRAFT / f'mosei-multimodal-aligned-v1.0/{SAMPLE}.npz',
        'bert': DRAFT / f'mosei-text-bert-v1.0/{SAMPLE}.npz',
        'audio': DRAFT / f'mosei-audio-egemaps-v1.0/{SAMPLE}.csv',
        'vision': DRAFT / f'mosei-openface-pilot-v1.0/{SAMPLE}.csv',
        'waveform': DRAFT / f'mosei-audio-v1.0/{SAMPLE}.wav',
        'human_qc': DRAFT / 'pilot-sample/2026-09-23_pilot-qc_v1.0.json',
    }
    with np.load(paths['merged'], allow_pickle=False) as handle:
        data = {k: handle[k] for k in handle.files}
    with np.load(paths['bert'], allow_pickle=False) as handle:
        bert = {k: handle[k] for k in handle.files}
    audio_rows, visual_rows = load_rows(paths['audio']), load_rows(paths['vision'])
    source = {
        'audio': np.array([[float(r[n]) for n in data['audio_feature_names']] for r in audio_rows]),
        'vision': np.array([[float(r[n]) for n in data['vision_feature_names']] for r in visual_rows]),
    }
    intervals = data['word_times_s']
    words = data['words'].tolist()
    assert data['sample_id'].item() == SAMPLE
    assert data['all_modalities_mask'].all()
    checks = {}
    for modality in ('audio', 'vision'):
        src_intervals = data[f'{modality}_source_intervals_s']
        if modality == 'audio':
            np.testing.assert_allclose(src_intervals,
                [[float(r['start_s']), float(r['end_s'])] for r in audio_rows], atol=1e-10)
            valid = np.isfinite(source[modality]).all(axis=1)
        else:
            valid = np.array([int(r['success']) == 1 and float(r['confidence']) >= .8
                              for r in visual_rows]) & np.isfinite(source[modality]).all(axis=1)
            np.testing.assert_array_equal(valid, data['vision_source_valid_mask'])
        # Recompute geometry directly; do not call the original merge implementation.
        overlap = np.maximum(0, np.minimum(intervals[:, 1, None], src_intervals[None, :, 1])
                               - np.maximum(intervals[:, 0, None], src_intervals[None, :, 0]))
        overlap[:, ~valid] = 0
        overlap[overlap < 1e-12] = 0
        expected_weights = overlap / overlap.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(data[f'{modality}_source_weights'], expected_weights,
                                   rtol=1e-6, atol=1e-7)
        reconstruction = expected_weights @ source[modality]
        np.testing.assert_allclose(reconstruction, data[modality], rtol=2e-5, atol=2e-5)
        checks[f'{modality}_maximum_absolute_reconstruction_error'] = float(
            np.max(np.abs(reconstruction - data[modality])))
    expected_text = np.stack([bert['token_features'][bert['word_ids'] == i].mean(axis=0)
                              for i in range(len(words))])
    np.testing.assert_allclose(expected_text, data['text'], rtol=1e-5, atol=1e-6)
    checks['text_maximum_absolute_reconstruction_error'] = float(
        np.max(np.abs(expected_text - data['text'])))

    records = []
    for i, word in enumerate(words):
        record = {'word_index': i+1, 'word': word, 'start_s': float(intervals[i, 0]),
                  'end_s': float(intervals[i, 1]),
                  'bert_token_indices_0based': np.flatnonzero(bert['word_ids'] == i).tolist(),
                  'bert_input_ids': bert['input_ids'][bert['word_ids'] == i].tolist()}
        for modality in ('audio', 'vision'):
            weights = data[f'{modality}_source_weights'][i]
            indices = np.flatnonzero(weights)
            record[f'{modality}_rows_1based'] = (indices+1).tolist()
            record[f'{modality}_weights'] = weights[indices].astype(float).tolist()
            record[f'{modality}_coverage'] = float(data[
                'audio_window_coverage' if modality == 'audio' else 'vision_coverage'][i])
        records.append(record)
    result = {'sample': SAMPLE, 'video_id': data['video_id'].item(),
              'clip_id': data['clip_id'].item(), 'transcript': data['transcript'].item(),
              'checks': checks, 'words': records,
              'source_sha256': {p.relative_to(DRAFT).as_posix(): digest(p) for p in paths.values()},
              'validation_scope': 'Numerical reconstruction and provenance; not measured word-boundary accuracy',
              'human_qc': json.loads(paths['human_qc'].read_text(encoding='utf-8-sig'))}
    (DRAFT / (PREFIX + '.json')).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')

    with wave.open(str(paths['waveform']), 'rb') as stream:
        assert stream.getnchannels() == 1 and stream.getsampwidth() == 2
        rate = stream.getframerate()
        waveform = np.frombuffer(stream.readframes(stream.getnframes()), dtype='<i2') / 32768.0
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(4, 1, figsize=(13, 8.2), sharex=True,
                            gridspec_kw={'height_ratios': [1.0, 0.75, 1.1, 1.1]})
    fig.patch.set_facecolor('white')
    fig.suptitle('同一句话的三模态如何落到同一个词时间轴', x=.08, ha='left',
                 fontsize=18, fontweight='bold', color='#182d40')
    fig.text(.08, .926, 'sample_0013  |  -THoVjtIkeU / 2  |  10 个词全部保留  |  特征维度：文本 768 · 声音 25 · 视觉 49',
             fontsize=10, color='#536373')
    axs[0].plot(np.arange(len(waveform))[::16]/rate, waveform[::16], color='#7890a4', lw=.65)
    axs[0].set_ylabel('声音波形\n振幅')
    axs[1].set_ylim(-.15, 1.75)
    axs[1].set_yticks([])
    axs[1].set_ylabel('原始词\nCTC 时间')
    for i, (word, (start, end)) in enumerate(zip(words, intervals)):
        y = (i % 2) * .85
        axs[1].broken_barh([(start, end-start)], (y, .55), facecolors='#e1edf7', edgecolors='#4f86ac')
        if end-start < .07:
            axs[1].annotate(word, ((start+end)/2, y+.55), xytext=(3, 8),
                            textcoords='offset points', ha='left', va='bottom', fontsize=9)
        else:
            axs[1].text((start+end)/2, y+.275, word, ha='center', va='center', fontsize=9)
    audio_index = data['audio_feature_names'].tolist().index('Loudness_sma3')
    visual_index = data['vision_feature_names'].tolist().index('AU12_r')
    for ax, modality, index, color, ylabel in (
        (axs[2], 'audio', audio_index, '#15856f', '声音：响度\nLoudness_sma3'),
        (axs[3], 'vision', visual_index, '#a369b0', '视觉：嘴角拉升\nAU12 强度')):
        src_times = data[f'{modality}_source_intervals_s'].mean(axis=1)
        ax.plot(src_times, source[modality][:, index], color=color, alpha=.65, lw=1,
                label='原始窗口 / 帧特征')
        for i, (start, end) in enumerate(intervals):
            ax.hlines(data[modality][i, index], start, end, color='#182d40', lw=2.5,
                       label='交叠加权后的词级值' if i == 0 else None)
        ax.set_ylabel(ylabel)
        ax.legend(loc='upper right', fontsize=8, frameon=False, ncol=2)
    for ax in axs:
        for start, end in intervals:
            ax.axvline(start, color='#c5d1da', lw=.65, ls=':')
            ax.axvline(end, color='#c5d1da', lw=.65, ls=':')
        ax.set_xlim(0, len(waveform)/rate)
        ax.grid(axis='y', color='#ecf0f3', lw=.6)
    axs[-1].set_xlabel('相对解码音频起点的时间（秒）')
    fig.text(.08, .035, '说明：图中仅选取两个可解释维度示意；词间停顿不强行分配给词。响度与 AU12 均不是情感预测分数。',
             fontsize=9, color='#536373')
    fig.text(.08, .012, '人工已确认字幕与整体节奏；逐词精确边界尚未人工核验。', fontsize=9, color='#536373')
    fig.subplots_adjust(left=.12, right=.98, top=.88, bottom=.10, hspace=.19)
    fig.savefig(DRAFT / (PREFIX+'.png'), dpi=180, facecolor='white')
    plt.close(fig)

    lines = ['# 问题一典型样本验证', '',
        f'样本 `{SAMPLE}`，原视频标识 `{data["video_id"].item()}/{data["clip_id"].item()}`。', '',
        f'> {data["transcript"].item()}', '',
        '本例重新从已保存的 BERT token 向量、声音窗口和 OpenFace 帧特征计算词级向量，'
        '检查它们与全量合并文件一致。该检查验证映射和数值计算，不能衡量词边界是否等于人工真值。', '',
        f'![典型样本共同时间轴]({PREFIX}.png)', '',
        '## 逐词对应关系', '',
        '表中行号从 1 开始，不计 CSV 表头。BERT token 位置从 0 开始，包含序列特殊 token 的位置；'
        '特殊 token 本身不参与词向量平均。声音窗口彼此可重叠，窗口数不同于互不重叠的采样数。', '',
        '| 原词 | 自动区间 / 秒 | BERT token 位置 | 声音窗口行号（数量） | 视觉帧号（数量） | 声音 / 视觉覆盖率 |',
        '|---|---|---|---|---|---|']
    for r in records:
        a, v = r['audio_rows_1based'], r['vision_rows_1based']
        lines.append(f'| {r["word"]} | {r["start_s"]:.3f}–{r["end_s"]:.3f} | '
                     f'{", ".join(map(str,r["bert_token_indices_0based"]))} | '
                     f'{a[0]}–{a[-1]}（{len(a)}） | {v[0]}–{v[-1]}（{len(v)}） | '
                     f'{r["audio_coverage"]:.0%} / {r["vision_coverage"]:.0%} |')
    social = records[2]
    indices = np.array(social['vision_rows_1based'])-1
    lines += ['', '## 以 social 为例', '',
        '`social` 是原文第 3 个词，字符区间 `[8,14)`，自动时间为 `[1.200,1.440)` 秒。'
        f'它映射到 BERT token 位置 {social["bert_token_indices_0based"]}、词表 ID {social["bert_input_ids"]}，'
        '得到一个 768 维上下文向量；同一时间区间内的 25 个声音窗口汇聚为 25 维向量，8 帧视觉汇聚为 49 维向量。', '',
        '| 视频帧号 | 共同时间轴上的帧区间 / 秒 | 对 social 的归一化权重 |', '|---|---|---|']
    for j in indices:
        left, right = data['vision_source_intervals_s'][j]
        lines.append(f'| {j+1} | {left:.6f}–{right:.6f} | {data["vision_source_weights"][2,j]:.6f} |')
    lines += ['', '每帧的贡献取决于它与词区间交叠的时长，边界帧只贡献相交部分。'
        '同一词的有效权重和为 1。全量权重及声音窗口行号保存在配套 JSON，可回溯所有参与汇聚的观测。', '',
        '## 数值验证与人工检查范围', '',
        '| 重算项目 | 最大绝对误差 |', '|---|---|']
    for key, value in checks.items():
        lines.append(f'| {key.split("_")[0]} | {value:.8g} |')
    lines += ['', '数值比较采用浮点容差：文本 `rtol=1e-5, atol=1e-6`，声音/视觉 `rtol=2e-5, atol=2e-5`；'
        '权重另与区间交叠公式重算结果比较。10 个词均有有效三模态特征，声音与视觉覆盖率均为 100%。', '',
        '用户先前已听音确认转写及整体停顿、语序、语速相符；人工记录明确写明 `exact_word_boundaries=not_manually_verified`。'
        '因此，本例只能支持“端到端映射可追溯、数值重算一致、整体节奏经听音确认”，不能支持“所有词边界精确无误”。', '',
        '## 复现与来源', '',
        '运行 `src/2026-09-24_build-q1-example_v1.0.py` 可重建本报告、图及 JSON；'
        '不调用模型、不改动原始特征。来源文件 SHA-256 保存在配套 JSON 中。', '',
        '- `mosei-multimodal-aligned-v1.0/sample_0013.npz`',
        '- `mosei-text-bert-v1.0/sample_0013.npz`',
        '- `mosei-audio-egemaps-v1.0/sample_0013.csv`',
        '- `mosei-openface-pilot-v1.0/sample_0013.csv`',
        '- `mosei-audio-v1.0/sample_0013.wav`',
        '- `pilot-sample/2026-09-23_pilot-qc_v1.0.json`', '']
    (DRAFT / (PREFIX+'.md')).write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'sample': SAMPLE, 'words': len(words), 'checks': checks,
                      'outputs': [PREFIX+ext for ext in ('.md', '.json', '.png')]}, indent=2))


if __name__ == '__main__':
    main()
