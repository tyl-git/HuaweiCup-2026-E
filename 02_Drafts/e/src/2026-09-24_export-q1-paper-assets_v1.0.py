"""Export question-one paper tables and figures from compact, saved results.

No source media, model, or training dependency is required. The silence-excluded
columns are posthoc sensitivity indicators; the historical NPZ masks are not changed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np


SCRIPT_VERSION = '1.0'
NPZ_REL = Path('02_Drafts/e/mosei-multimodal-aligned-v1.0')
AUDIT_REL = Path('03_Results/e/question-one/2026-09-24_q1_audit_v1.0.json')
SILENCE_REL = Path('02_Drafts/e/2026-09-24_q1-p1-review-evidence_v1.0.json')
DEFAULT_OUTPUT_REL = Path('03_Results/e/question-one/paper-assets-v1.0')
FIELDS = [
    'sample', 'video_id', 'clip_id', 'word_index', 'word', 'char_start', 'char_end',
    'start_s', 'end_s', 'timing_mask', 'text_mask', 'audio_mask', 'vision_mask',
    'all_modalities_mask', 'audio_window_coverage', 'vision_coverage',
    'audio_window_count', 'vision_frame_count', 'needs_review',
    'alignment_review_flags', 'merge_review_flags', 'word_review_flags',
    'verified_digital_silence', 'candidate_timing_excluding_silence',
    'candidate_audio_excluding_silence',
    'candidate_all_modalities_excluding_silence',
]


def sha256(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def scalar(value):
    return np.asarray(value).item()


def flag(value: bool) -> str:
    return 'true' if value else 'false'


def time_cell(value: float) -> str:
    return 'NaN' if math.isnan(value) else format(value, '.9g')


def load_inputs(root: Path):
    audit_path = root / AUDIT_REL
    silence_path = root / SILENCE_REL
    audit = json.loads(audit_path.read_text(encoding='utf-8-sig'))
    evidence = json.loads(silence_path.read_text(encoding='utf-8-sig'))
    scan = evidence['all_audio_silence_scan']
    records = scan['records']
    if len(records) != 100 or len({r['sample'] for r in records}) != 100:
        raise ValueError('Silence scan does not cover 100 distinct samples')
    silent = {r['sample'] for r in records if r['all_samples_zero']}
    if silent != set(scan['all_zero_samples']) or silent != {'sample_0042', 'sample_0043'}:
        raise ValueError('Verified digital-silence sample set changed')
    if any((r['nonzero_samples'] != 0) != (not r['all_samples_zero']) for r in records):
        raise ValueError('Silence scan record values disagree')
    return audit, scan, silent, audit_path, silence_path


def export_rows(root: Path, silent: set[str]):
    npz_dir = root / NPZ_REL
    files = sorted(npz_dir.glob('sample_*.npz'))
    expected = {f'sample_{number:04d}.npz' for number in range(1, 101)}
    if len(files) != 100 or {p.name for p in files} != expected:
        raise ValueError('Expected exactly sample_0001.npz through sample_0100.npz')

    totals = Counter()
    rows = []
    review_samples = 0
    no_visual_samples = 0
    for path in files:
        sample = path.stem
        with np.load(path, allow_pickle=False) as z:
            words = z['words']
            n = len(words)
            indices = z['word_indices']
            spans = z['word_char_spans']
            times = z['word_times_s']
            masks = {name: z[name].astype(bool) for name in
                     ('timing_mask', 'text_mask', 'audio_mask', 'vision_mask', 'all_modalities_mask')}
            if (int(scalar(z['sequence_length'])) != n or times.shape != (n, 2)
                    or spans.shape != (n, 2) or not np.array_equal(indices, np.arange(1, n + 1))):
                raise ValueError(f'Word axis/schema mismatch: {sample}')
            if any(mask.shape != (n,) for mask in masks.values()):
                raise ValueError(f'Mask length mismatch: {sample}')
            joint = masks['timing_mask'] & masks['text_mask'] & masks['audio_mask'] & masks['vision_mask']
            if not np.array_equal(joint, masks['all_modalities_mask']):
                raise ValueError(f'Joint mask mismatch: {sample}')
            if (not np.isfinite(times[masks['timing_mask']]).all()
                    or not np.isnan(times[~masks['timing_mask']]).all()):
                raise ValueError(f'Timing/NaN mismatch: {sample}')
            if np.any(times[masks['timing_mask'], 1] <= times[masks['timing_mask'], 0]):
                raise ValueError(f'Non-positive word duration: {sample}')
            if z['text'].shape != (n, 768) or z['audio'].shape != (n, 25) or z['vision'].shape != (n, 49):
                raise ValueError(f'Feature dimensions changed: {sample}')
            if not np.isfinite(z['audio_window_coverage']).all() or not np.isfinite(z['vision_coverage']).all():
                raise ValueError(f'Non-finite coverage: {sample}')

            sample_silent = sample in silent
            sample_review = bool(scalar(z['needs_review']))
            review_samples += sample_review
            no_visual_samples += not bool(z['vision_source_valid_mask'].any())
            totals.update(samples=1, source_words=n)
            for name, mask in masks.items():
                totals[f'{name}_words'] += int(mask.sum())
            totals['candidate_timed_words_excluding_silence'] += int(masks['timing_mask'].sum()) * (not sample_silent)
            totals['candidate_audio_words_excluding_silence'] += int(masks['audio_mask'].sum()) * (not sample_silent)
            totals['candidate_all_modalities_words_excluding_silence'] += int(joint.sum()) * (not sample_silent)

            for i in range(n):
                row = {
                    'sample': sample,
                    'video_id': str(scalar(z['video_id'])),
                    'clip_id': str(scalar(z['clip_id'])),
                    'word_index': int(indices[i]),
                    'word': str(words[i]),
                    'char_start': int(spans[i, 0]),
                    'char_end': int(spans[i, 1]),
                    'start_s': time_cell(float(times[i, 0])),
                    'end_s': time_cell(float(times[i, 1])),
                    'audio_window_coverage': format(float(z['audio_window_coverage'][i]), '.9g'),
                    'vision_coverage': format(float(z['vision_coverage'][i]), '.9g'),
                    'audio_window_count': int(z['audio_frame_count'][i]),
                    'vision_frame_count': int(z['vision_frame_count'][i]),
                    'needs_review': flag(sample_review),
                    'alignment_review_flags': str(scalar(z['alignment_review_flags'])),
                    'merge_review_flags': str(scalar(z['merge_review_flags'])),
                    'word_review_flags': str(z['word_review_flags'][i]),
                    'verified_digital_silence': flag(sample_silent),
                    'candidate_timing_excluding_silence': flag(bool(masks['timing_mask'][i]) and not sample_silent),
                    'candidate_audio_excluding_silence': flag(bool(masks['audio_mask'][i]) and not sample_silent),
                    'candidate_all_modalities_excluding_silence': flag(bool(joint[i]) and not sample_silent),
                }
                row.update({name: flag(bool(mask[i])) for name, mask in masks.items()})
                rows.append(row)
    totals['needs_review_samples'] = review_samples
    totals['no_valid_visual_samples'] = no_visual_samples
    return rows, totals, files


def validate_counts(totals: Counter, audit: dict, scan: dict):
    required_counts = {
        'samples': 100, 'source_words': 1932, 'timing_mask_words': 1923,
        'text_mask_words': 1932, 'audio_mask_words': 1923, 'vision_mask_words': 1828,
        'all_modalities_mask_words': 1828, 'needs_review_samples': 65,
        'no_valid_visual_samples': 4, 'candidate_timed_words_excluding_silence': 1899,
        'candidate_audio_words_excluding_silence': 1899,
        'candidate_all_modalities_words_excluding_silence': 1821,
    }
    for key, value in required_counts.items():
        if totals[key] != value:
            raise ValueError(f'Historical v1.0 dataset changed at {key}: {totals[key]} != {value}')
    expected = audit['totals']
    mapping = {
        'samples': 'samples',
        'source_words': 'source_words',
        'timing_mask_words': 'timed_words',
        'text_mask_words': 'text_valid_words',
        'audio_mask_words': 'audio_valid_words',
        'vision_mask_words': 'vision_valid_words',
        'all_modalities_mask_words': 'all_modalities_valid_words',
        'needs_review_samples': 'needs_review_samples',
        'no_valid_visual_samples': 'no_valid_visual_samples',
    }
    for actual, key in mapping.items():
        if totals[actual] != expected[key]:
            raise ValueError(f'{actual}: NPZ={totals[actual]}, audit={expected[key]}')
    checks = {
        'historical_timing_and_audio_mask_words': totals['timing_mask_words'],
        'historical_all_modalities_mask_words': totals['all_modalities_mask_words'],
        'candidate_timed_words_excluding_verified_digital_silence': totals['candidate_timed_words_excluding_silence'],
        'candidate_all_modalities_words_excluding_verified_digital_silence': totals['candidate_all_modalities_words_excluding_silence'],
    }
    for key, value in checks.items():
        if scan[key] != value:
            raise ValueError(f'{key}: NPZ={value}, silence evidence={scan[key]}')
    if totals['source_words'] - totals['timing_mask_words'] != expected['untimed_words']:
        raise ValueError('Untimed count disagrees with audit')
    if (totals['timing_mask_words'] - totals['candidate_timed_words_excluding_silence']
            != scan['all_zero_sample_timed_words']
            or totals['all_modalities_mask_words'] - totals['candidate_all_modalities_words_excluding_silence']
            != scan['all_zero_sample_all_modalities_mask_words']):
        raise ValueError('Silence-exclusion deltas disagree with evidence')


def write_figures(output: Path, totals: Counter):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, Rectangle
    except ImportError:
        return False

    labels = ['Source words', 'Historical timed', 'Timed, silence excluded',
              'Historical three-modality', 'Three-modality, silence excluded']
    values = [totals['source_words'], totals['timing_mask_words'],
              totals['candidate_timed_words_excluding_silence'],
              totals['all_modalities_mask_words'],
              totals['candidate_all_modalities_words_excluding_silence']]
    colors = ['#36646e', '#478f9a', '#9abac1', '#b76045', '#d99b83']
    fig, ax = plt.subplots(figsize=(10.8, 4.6), layout='constrained')
    y = np.arange(len(labels))
    ax.barh(y, values, height=0.64, color=colors)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.13)
    ax.set_xlabel('Word count (not alignment accuracy)')
    ax.set_title('Question 1: Historical masks and silence sensitivity')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='x', alpha=0.18)
    ax.set_axisbelow(True)
    for pos, value in enumerate(values):
        ax.text(value + 14, pos, f'{value:,}', va='center', fontsize=10)
    for ext in ('png', 'svg'):
        fig.savefig(output / f'q1_coverage_bars_v1.0.{ext}', dpi=180, facecolor='white')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.3), layout='constrained')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.3)
    ax.axis('off')
    boxes = [
        (0.3, 2.45, 'Source video\ntranscript', '#e5eef0'),
        (2.75, 2.45, 'WAV / video / PTS\noriginal transcript', '#e5eef0'),
        (5.2, 3.25, 'CTC word times', '#dce8e0'),
        (5.2, 2.15, 'BERT 768-D', '#dce8e0'),
        (5.2, 1.05, 'eGeMAPS 25-D\nOpenFace 49-D', '#dce8e0'),
        (8.1, 2.45, 'Word mapping\n+ overlap pooling', '#f4e8dd'),
        (10.3, 2.45, 'NPZ + masks\nreview flags', '#f4e8dd'),
    ]
    for x, y0, label, color in boxes:
        ax.add_patch(Rectangle((x, y0), 1.65, 0.76, facecolor=color, edgecolor='#40515a', linewidth=1.2))
        ax.text(x + 0.825, y0 + 0.38, label, ha='center', va='center', fontsize=9)
    arrows = [((1.95, 2.83), (2.75, 2.83)), ((4.4, 2.83), (5.12, 3.63)),
              ((4.4, 2.83), (5.12, 2.53)), ((4.4, 2.83), (5.12, 1.43)),
              ((6.85, 3.63), (8.1, 2.83)), ((6.85, 2.53), (8.1, 2.83)),
              ((6.85, 1.43), (8.1, 2.83)), ((9.75, 2.83), (10.3, 2.83))]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=12,
                                     linewidth=1.2, color='#40515a'))
    ax.set_title('Question 1: Feature extraction and word-level temporal alignment', pad=15)
    ax.text(6, 0.35, 'Saved historical v1.0 results; automatic times are not human-verified.',
            ha='center', va='center', fontsize=9, color='#59656a')
    for ext in ('png', 'svg'):
        fig.savefig(output / f'q1_pipeline_v1.0.{ext}', dpi=180, facecolor='white')
    plt.close(fig)
    return True


def write_notes(output: Path, charts_written: bool):
    notes = '''# Question-one paper assets

This directory is regenerated from the compact export's saved NPZ files, audit
JSON and digital-silence evidence. No source media, model, GPU, Qt, network,
training or feature extraction is used. Original inputs are read only.

## Run after cloning

Required: Python 3.11+ and NumPy. Matplotlib is optional for PNG/SVG figures.
Reuse an environment that already has these packages.

```powershell
python 02_Drafts/e/src/2026-09-24_export-q1-paper-assets_v1.0.py
```

The default root is derived from the script location, not the working directory.
Optional overrides are `--root PATH` and `--output-dir PATH`. The default output
is `03_Results/e/question-one/paper-assets-v1.0` below that root. A rerun replaces
only this exporter's named derived assets. It does not alter the historical NPZ,
audit or silence evidence. Inputs with changed historical counts are rejected.

## Files and interpretation

| File | Meaning |
|---|---|
| `q1_words_v1.0.tsv` | 1,932 original whitespace-separated word units; UTF-8 TSV, one row per unit |
| `q1_summary_v1.0.json` | Checked historical counts, posthoc sensitivity counts and SHA-256 provenance |
| `q1_coverage_bars_v1.0.png` / `.svg` | Word counts under historical masks and after digital-silence exclusion |
| `q1_pipeline_v1.0.png` / `.svg` | Diagram of the historical extraction/alignment method, not operations rerun by this exporter |

`start_s`/`end_s` are historical automatic CTC intervals in seconds; literal
`NaN` is preserved for the nine untimed units. Word and character indices are
retained: word indices start at 1; character spans use zero-based, end-exclusive
Python string positions. `word` preserves spelling and punctuation.

The five historical mask columns are copied without modification. The audio
mask means an eGeMAPS window overlaps the automatic interval; it does not mean
speech was detected. Coverage columns are fractions in [0,1], not accuracy.
Window/frame counts describe observations used in pooling; overlapping audio
windows are not independent samples. False masks identify missing observations,
not observed neutral emotion.

`verified_digital_silence` is true on all 24 word rows from sample_0042 and
sample_0043, using the saved full-waveform scan. The three
`candidate_*_excluding_silence` columns apply the historical mask AND NOT that
sample-level silence flag. They are posthoc sensitivity indicators, not a v1.1
realignment. Original times and masks remain visible. Nonzero audio does not
establish speech presence, transcript correctness or accurate word boundaries.

Historical counts are 1,923 timed/audio-mask words and 1,828 joint-mask words.
Silence-excluded candidates are 1,899 and 1,821. None is an alignment accuracy
or a number of human-verified boundaries. The 65 flagged samples still require
review; unflagged samples have not thereby been manually verified.

The NPZ hashes in the summary allow each row's source to be identified. This
compact export checks saved word axes, masks, times, shapes and count agreement;
it cannot repeat the full upstream-media audit or feature reconstruction.
'''
    notes += '\nFigure generation in this run: ' + ('PNG and SVG generated.\n' if charts_written else
                'skipped because Matplotlib is unavailable; the TSV and JSON were generated.\n')
    (output / 'q1_assets_notes_v1.0.md').write_text(notes, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3],
                        help='Root of the compact project export')
    parser.add_argument('--output-dir', type=Path,
                        help='Destination; default: <root>/03_Results/e/question-one/paper-assets-v1.0')
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output_dir or root / DEFAULT_OUTPUT_REL).resolve()
    audit, scan, silent, audit_path, silence_path = load_inputs(root)
    rows, totals, files = export_rows(root, silent)
    validate_counts(totals, audit, scan)
    if (output == root or output in (audit_path.parent, silence_path.parent)
            or output.is_relative_to(root / NPZ_REL)):
        raise ValueError('Output must be a dedicated directory, not a source directory')
    output.mkdir(parents=True, exist_ok=True)

    word_path = output / 'q1_words_v1.0.tsv'
    with word_path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        'schema_version': SCRIPT_VERSION,
        'scope': 'Saved Question 1 features and word-level alignment; no sentiment predictions',
        'historical_v1_0': {
            'samples': totals['samples'],
            'source_words': totals['source_words'],
            'untimed_words': totals['source_words'] - totals['timing_mask_words'],
            'timed_words': totals['timing_mask_words'],
            'text_mask_words': totals['text_mask_words'],
            'audio_mask_words': totals['audio_mask_words'],
            'vision_mask_words': totals['vision_mask_words'],
            'all_modalities_mask_words': totals['all_modalities_mask_words'],
            'needs_review_samples': totals['needs_review_samples'],
            'no_valid_visual_samples': totals['no_valid_visual_samples'],
        },
        'posthoc_digital_silence_sensitivity': {
            'verified_all_zero_audio_samples': sorted(silent),
            'historical_timed_words_in_silent_samples': scan['all_zero_sample_timed_words'],
            'historical_all_modalities_words_in_silent_samples': scan['all_zero_sample_all_modalities_mask_words'],
            'candidate_timed_words_excluding_silent_samples': totals['candidate_timed_words_excluding_silence'],
            'candidate_audio_words_excluding_silent_samples': totals['candidate_audio_words_excluding_silence'],
            'candidate_all_modalities_words_excluding_silent_samples': totals['candidate_all_modalities_words_excluding_silence'],
        },
        'provenance': {
            'audit_relpath': AUDIT_REL.as_posix(), 'audit_sha256': sha256(audit_path),
            'silence_evidence_relpath': SILENCE_REL.as_posix(), 'silence_evidence_sha256': sha256(silence_path),
            'npz_directory_relpath': NPZ_REL.as_posix(), 'npz_files': len(files),
            'npz_sha256_by_sample': {path.stem: sha256(path) for path in files},
        },
        'interpretation': [
            'Historical masks indicate computable overlap, not correct speech or alignment.',
            'Audio mask can be true on digital silence because eGeMAPS windows exist.',
            'Silence-excluded counts are sensitivity candidates, not revised NPZ masks or verified word boundaries.',
            'Review flags do not mean a sample has been confirmed wrong; unflagged samples are not human-verified.',
        ],
    }
    (output / 'q1_summary_v1.0.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    charts_written = write_figures(output, totals)
    write_notes(output, charts_written)
    print(f'Root: {root}')
    print(f'Words: {len(rows)} in {word_path}')
    print(f'Historical timed / three-modality: {totals["timing_mask_words"]} / {totals["all_modalities_mask_words"]}')
    print(f'Silence-excluded candidates: {totals["candidate_timed_words_excluding_silence"]} / '
          f'{totals["candidate_all_modalities_words_excluding_silence"]}')
    print(f'Figures: {"PNG + SVG" if charts_written else "skipped (matplotlib unavailable)"}')
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
