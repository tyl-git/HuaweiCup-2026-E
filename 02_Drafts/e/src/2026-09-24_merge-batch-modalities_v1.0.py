"""Pool frame features onto original word intervals, retaining all quality masks.

--pilot-check: reconstruct existing pilot audio/vision arrays without BERT or output.
--check: validate and reconstruct all 100 samples without writing batch output.
No normalization, fixed-length padding, resampling, or sample deletion is applied.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

DRAFT = Path(__file__).resolve().parents[1]
WORDS = DRAFT / 'mosei-word-times-v1.0'
TEXT = DRAFT / 'mosei-text-bert-v1.0'
AUDIO = DRAFT / 'mosei-audio-egemaps-v1.0'
VISION = DRAFT / 'mosei-openface-pilot-v1.0'
OUTPUT = DRAFT / 'mosei-multimodal-aligned-v1.0'
REPORT = DRAFT / '2026-09-24_mosei-multimodal-check_v1.0.csv'
METHOD = DRAFT / '2026-09-24_mosei-multimodal-method_v1.0.json'
FFPROBE = Path(r'D:\06_Apps\ffmpeg\bin\ffprobe.exe')
VISION_FIELDS = ['gaze_0_x', 'gaze_0_y', 'gaze_0_z', 'gaze_1_x', 'gaze_1_y', 'gaze_1_z',
    'gaze_angle_x', 'gaze_angle_y', 'pose_Tx', 'pose_Ty', 'pose_Tz', 'pose_Rx', 'pose_Ry', 'pose_Rz',
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r', 'AU07_r', 'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r',
    'AU15_r', 'AU17_r', 'AU20_r', 'AU23_r', 'AU25_r', 'AU26_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU04_c',
    'AU05_c', 'AU06_c', 'AU07_c', 'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 'AU20_c',
    'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']
POLICY = {
    'schema_version': '1.0',
    'word_axis': 'All original whitespace tokens, 1-based index, unchanged spelling/punctuation',
    'untimed_words': 'Retained; NaN timestamps, zero audio/vision, false audio/vision masks',
    'audio_pooling': 'Normalized intersection duration with actual openSMILE intervals',
    'audio_coverage': 'Union of intersected valid audio windows divided by word duration',
    'vision_pooling': 'Normalized intersection duration with valid frame display intervals',
    'vision_intervals': 'Actual ffprobe PTS_i to PTS_(i+1); last frame ends at PTS+reported duration',
    'vision_validity': 'OpenFace success==1, confidence>=0.8, all 49 features finite',
    'vision_coverage': 'Union of intersected valid frame intervals divided by word duration',
    'vision_mask': 'True if any positive valid-frame overlap; partial coverage retained separately',
    'invalid_features': 'Zero placeholders and false modality mask; zero does not mean observed neutral behavior',
    'ctc_review': 'Existing alignment flags retained; timed does not mean manually verified',
    'time_origin': 'Decoded audio start at zero; actual video PTS converted using source stream start offset',
    'sequence_length': 'Variable original length; no resampling, truncation, padding or normalization',
    'visual_scope': 'OpenFace tracks one detected face; speaker identity is not verified',
}
REPORT_FIELDS = ['sample', 'video_id', 'clip_id', 'source_words', 'timed_words', 'untimed_words',
    'text_valid_words', 'audio_valid_words', 'vision_valid_words', 'all_modalities_valid_words',
    'audio_min_coverage', 'vision_min_coverage', 'vision_partial_words', 'vision_frame_count',
    'vision_valid_frames', 'alignment_status', 'alignment_review_flags', 'merge_review_flags',
    'needs_review', 'status', 'output_relpath', 'output_sha256', 'error']


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        reader.fieldnames = [key.strip() for key in reader.fieldnames]
        return list(reader)


def table_by_sample(path):
    entries = rows(path)
    table = {row['sample']: row for row in entries}
    if len(entries) != 100 or set(table) != {f'sample_{i:04d}' for i in range(1, 101)}:
        raise ValueError(f'Expected 100 unique sample rows: {path.name}')
    return table


def coverage_union(intervals, start, end):
    """Avoid double-counting overlapping openSMILE windows."""
    covered, right = 0.0, start
    for left, stop in intervals:
        left, stop = max(float(left), start), min(float(stop), end)
        if stop > max(left, right):
            covered += stop - max(left, right)
            right = stop
    return float(np.clip(covered / (end - start), 0.0, 1.0))


def pool(values, intervals, valid_frames, word_times, timed):
    values = np.asarray(values, dtype=np.float64)
    intervals = np.asarray(intervals, dtype=np.float64)
    valid_frames = np.asarray(valid_frames, dtype=bool)
    if (values.ndim != 2 or intervals.shape != (len(values), 2)
            or valid_frames.shape != (len(values),) or not np.isfinite(intervals).all()
            or np.any(intervals[:, 1] <= intervals[:, 0])
            or np.any(np.diff(intervals[:, 0]) <= 0)):
        raise ValueError('Invalid feature/time interval arrays')
    if not np.isfinite(values[valid_frames]).all():
        raise ValueError('Valid feature rows contain non-finite values')
    n = len(word_times)
    output = np.zeros((n, values.shape[1]), dtype=np.float32)
    mask = np.zeros(n, dtype=bool)
    counts = np.zeros(n, dtype=np.int64)
    coverage = np.zeros(n, dtype=np.float64)
    weights = np.zeros((n, len(values)), dtype=np.float32)
    for i in np.flatnonzero(timed):
        start, end = word_times[i]
        overlap = np.maximum(0.0, np.minimum(intervals[:, 1], end) - np.maximum(intervals[:, 0], start))
        overlap[~valid_frames] = 0
        overlap[overlap < 1e-12] = 0
        good = overlap > 0
        if not good.any():
            continue
        normalized = overlap[good] / overlap[good].sum()
        output[i] = normalized @ values[good]
        weights[i, good] = normalized
        mask[i] = True
        counts[i] = good.sum()
        coverage[i] = coverage_union(intervals[good], start, end)
    if not np.isfinite(output).all():
        raise ValueError('Pooled features contain non-finite values')
    return output, mask, counts, coverage, weights


def load_audio(path, expected_names):
    rs = rows(path)
    if not rs or list(rs[0]) != ['start_s', 'end_s'] + expected_names:
        raise ValueError('Audio column names/order differ from pilot')
    intervals = np.array([[float(r['start_s']), float(r['end_s'])] for r in rs])
    values = np.array([[float(r[n]) for n in expected_names] for r in rs], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Audio features contain non-finite values')
    return intervals, values


def frame_intervals(video, audio_record):
    command = [str(FFPROBE), '-v', 'error', '-threads', '1', '-select_streams', 'v:0', '-show_frames',
        '-show_entries', 'stream=start_time:frame=best_effort_timestamp_time,duration_time,pkt_duration_time',
        '-of', 'json', str(video)]
    data = json.loads(subprocess.run(command, check=True, capture_output=True, text=True, encoding='utf-8').stdout)
    frames = data['frames']
    starts = np.array([float(x['best_effort_timestamp_time']) for x in frames], dtype=np.float64)
    if not len(starts) or np.any(np.diff(starts) <= 0):
        raise ValueError('Video timestamps are not strictly increasing')
    stream_start = float(data['streams'][0]['start_time'])
    if abs(stream_start - float(audio_record['video_stream_start_seconds'])) > 1e-6:
        raise ValueError('Video stream start differs from audio provenance')
    final_duration = float(frames[-1].get('duration_time', frames[-1].get('pkt_duration_time', 0)))
    if not np.isfinite(final_duration) or final_duration <= 0:
        raise ValueError('Final video frame has no positive duration')
    # Offset converts video PTS into the same zero as decoded WAV and CTC times.
    starts -= stream_start + float(audio_record['audio_vs_video_start_seconds'])
    ends = np.r_[starts[1:], starts[-1] + final_duration]
    return np.column_stack((starts, ends))


def load_vision(path, video, audio_record, expected_qc=None):
    rs = rows(path)
    required = {'frame', 'timestamp', 'confidence', 'success', *VISION_FIELDS}
    if not rs or not required.issubset(rs[0]):
        raise ValueError('OpenFace feature columns missing')
    intervals = frame_intervals(video, audio_record)
    if len(rs) != len(intervals) or [int(r['frame']) for r in rs] != list(range(1, len(rs)+1)):
        raise ValueError('OpenFace rows do not match decoded video frame order/count')
    values = np.array([[float(r[n]) for n in VISION_FIELDS] for r in rs], dtype=np.float64)
    confidence = np.array([float(r['confidence']) for r in rs])
    success = np.array([int(r['success']) == 1 for r in rs])
    if not np.isfinite(confidence).all():
        raise ValueError('Non-finite OpenFace confidence')
    eligible = success & (confidence >= .8)
    if expected_qc is not None and (len(rs) != int(expected_qc['frames'])
            or int(eligible.sum()) != int(expected_qc['valid_frames'])):
        raise ValueError('OpenFace counts differ from existing QC report')
    valid = eligible & np.isfinite(values).all(axis=1)
    return intervals, values, valid


def pilot_check(audio_names):
    pilot = DRAFT / 'pilot-sample'
    audio_record = table_by_sample(DRAFT / '2026-09-24_mosei-audio-check_v1.0.csv')['sample_0013']
    ai, av = load_audio(AUDIO / 'sample_0013.csv', audio_names)
    vi, vv, valid = load_vision(VISION / 'sample_0013.csv', DRAFT / audio_record['stage_relpath'], audio_record)
    for modality, values, intervals, frame_mask in [('audio', av, ai, np.ones(len(ai), bool)), ('vision', vv, vi, valid)]:
        with np.load(pilot / f'2026-09-23_pilot-{modality}-word-features_v1.0.npz', allow_pickle=False) as z:
            actual = pool(values, intervals, frame_mask, z['word_times_s'], np.ones(len(z['words']), bool))
            expected = z[f'{modality}_features']
            np.testing.assert_allclose(actual[0], expected, rtol=2e-5, atol=2e-5)
            np.testing.assert_array_equal(actual[1], z[f'{modality}_mask'])
            np.testing.assert_allclose(actual[3], z['audio_window_coverage' if modality == 'audio' else 'vision_coverage'], atol=1e-8)
            np.testing.assert_allclose(actual[4], z['source_weights'], atol=2e-7)
            print(f'Pilot {modality} reconstruction: OK; max absolute difference={np.max(np.abs(actual[0]-expected)):.6g}', flush=True)


def load_words(sample, source, alignment):
    path, details_path = WORDS / (sample + '.csv'), WORDS / (sample + '.json')
    rs = rows(path)
    details = json.loads(details_path.read_text(encoding='utf-8-sig'))
    matches = list(re.finditer(r'\S+', source['text']))
    n = len(rs)
    if n != len(matches) or int(alignment['source_words']) != n or details['original_text'] != source['text']:
        raise ValueError('Word rows differ from original transcript')
    words = np.array([r['word'] for r in rs])
    spans = np.array([[int(r['char_start']), int(r['char_end'])] for r in rs], dtype=np.int64)
    times = np.full((n, 2), np.nan, dtype=np.float64)
    timed = np.zeros(n, dtype=bool)
    for i, (r, m) in enumerate(zip(rs, matches)):
        if int(r['word_index']) != i+1 or r['word'] != m.group() or tuple(spans[i]) != m.span():
            raise ValueError('Word identity/order/character span mismatch')
        if bool(r['start_s']) != bool(r['end_s']):
            raise ValueError('Partially missing word interval')
        if r['start_s']:
            times[i] = float(r['start_s']), float(r['end_s'])
            timed[i] = True
        if timed[i] != (r['alignable'].lower() == 'true'):
            raise ValueError('Alignable word lacks timing or untimed word has timing')
    if (not np.isfinite(times[timed]).all() or np.any(times[timed, 0] < 0)
            or np.any(times[timed, 1] <= times[timed, 0])
            or np.any(times[timed, 1] > float(details['audio_duration_s']) + 1/16000)
            or np.any(times[timed][1:, 0] < times[timed][:-1, 1] - 1e-9)
            or int(timed.sum()) != int(alignment['aligned_words'])):
        raise ValueError('Invalid/overlapping word timestamps')
    return rs, details, words, spans, times, timed


def merge(sample, source, alignment, audio_record, feature_record, visual_qc, audio_names):
    wr, details, words, spans, times, timed = load_words(sample, source, alignment)
    n = len(words)
    text_path = TEXT / (sample + '.npz')
    with np.load(text_path, allow_pickle=False) as z:
        text = z['word_features'].copy()
        text_mask = z['text_valid_mask'].astype(bool)
        if (text.shape != (n, 768) or text_mask.shape != (n,) or not np.isfinite(text).all()
                or not np.array_equal(z['words'], words) or not np.array_equal(z['word_indices'], np.arange(1, n+1))
                or not np.array_equal(z['word_char_spans'], spans) or str(z['transcript']) != source['text']):
            raise ValueError('BERT features disagree with word identity/schema')
        if (str(z['alignment_sha256']) != digest(WORDS/(sample+'.csv'))
                or str(z['alignment_status']) != alignment['status']
                or str(z['alignment_review_flags']) != alignment['review_flags']):
            raise ValueError('BERT word/alignment provenance mismatch')
        text_metadata = {f'text_{key}': z[key].copy() for key in ('model_id', 'model_revision', 'pooling')}
    for record in (alignment, audio_record, feature_record):
        if any(record[key] != source[key] for key in ('video_id', 'clip_id')):
            raise ValueError('Source identity differs between reports')
    if audio_record['status'] != 'ok' or feature_record['status'] != 'ok' or alignment['status'] not in ('auto_checked', 'needs_review'):
        raise ValueError('Input report contains a failed/unrecognized status')
    audio_path, vision_path = AUDIO / (sample + '.csv'), VISION / (sample + '.csv')
    video_path = DRAFT / source['stage_relpath']
    if digest(audio_path) != feature_record['feature_sha256'] or digest(video_path) != source['staged_sha256']:
        raise ValueError('Audio feature or staged video hash mismatch')
    if details['audio_sha256'] != audio_record['audio_sha256'] or feature_record['audio_sha256'] != audio_record['audio_sha256']:
        raise ValueError('Audio/word/feature provenance mismatch')
    ai, av = load_audio(audio_path, audio_names)
    vi, vv, valid = load_vision(vision_path, video_path, audio_record, visual_qc)
    audio, am, ac, aw, aweights = pool(av, ai, np.ones(len(av), bool), times, timed)
    vision, vm, vc, vw, vweights = pool(vv, vi, valid, times, timed)
    joint = text_mask & am & vm & timed
    merge_flags = []
    if np.any(timed & ~am): merge_flags.append('audio_missing_words')
    if np.any(timed & (aw < 1-1e-6)): merge_flags.append('audio_partial_coverage')
    if np.any(timed & ~vm): merge_flags.append('vision_missing_words')
    if np.any(vm & (vw < 1-1e-6)): merge_flags.append('vision_partial_coverage')
    if not valid.any(): merge_flags.append('no_valid_visual_frames')
    if np.any(~text_mask): merge_flags.append('text_missing_words')
    provenance = {p.relative_to(DRAFT).as_posix(): digest(p) for p in [
        text_path, audio_path, vision_path, video_path, WORDS/(sample+'.csv'), WORDS/(sample+'.json')]}
    provenance['script_sha256'] = digest(Path(__file__))
    provenance['alignment_record'] = alignment
    needs_review = alignment['status'] == 'needs_review' or bool(merge_flags)
    payload = dict(text=text.astype(np.float32), audio=audio, vision=vision, words=words,
        word_indices=np.arange(1, n+1, dtype=np.int64), word_char_spans=spans, word_times_s=times,
        sequence_length=np.array(n), timing_mask=timed, text_mask=text_mask, audio_mask=am, vision_mask=vm,
        all_modalities_mask=joint, audio_frame_count=ac, vision_frame_count=vc,
        audio_window_coverage=aw, vision_coverage=vw, audio_source_weights=aweights, vision_source_weights=vweights,
        audio_source_intervals_s=ai, vision_source_intervals_s=vi, vision_source_valid_mask=valid,
        audio_feature_names=np.array(audio_names), vision_feature_names=np.array(VISION_FIELDS),
        video_id=np.array(source['video_id']), clip_id=np.array(source['clip_id']), sample_id=np.array(sample),
        source_video_relpath=np.array(source['video_relpath']), transcript=np.array(source['text']),
        label=np.array(float(source['label'])), label_original=np.array(source['label']), annotation=np.array(source['annotation']),
        transcript_review=np.array(source['qc_status']), transcript_qc_note=np.array(source['qc_note']),
        alignment_status=np.array(alignment['status']), alignment_review_flags=np.array(alignment['review_flags']),
        word_review_flags=np.array([r['review_flags'] for r in wr]),
        word_ctc_probability=np.array([float(r['mean_ctc_probability']) if r['mean_ctc_probability'] else np.nan for r in wr]),
        merge_review_flags=np.array(';'.join(merge_flags)), needs_review=np.array(needs_review),
        exact_word_boundaries_manually_verified=np.array(False),
        audio_pooling=np.array(POLICY['audio_pooling']), vision_pooling=np.array(POLICY['vision_pooling']),
        vision_confidence_threshold=np.array(.8), time_origin=np.array(POLICY['time_origin']),
        policy_json=np.array(json.dumps(POLICY, sort_keys=True)),
        provenance_json=np.array(json.dumps(provenance, sort_keys=True)), **text_metadata)
    record = dict(sample=sample, video_id=source['video_id'], clip_id=source['clip_id'], source_words=n,
        timed_words=int(timed.sum()), untimed_words=int((~timed).sum()), text_valid_words=int(text_mask.sum()),
        audio_valid_words=int(am.sum()), vision_valid_words=int(vm.sum()), all_modalities_valid_words=int(joint.sum()),
        audio_min_coverage=f'{aw[timed].min():.6f}', vision_min_coverage=f'{vw[timed].min():.6f}',
        vision_partial_words=int(np.sum(vm & (vw < 1-1e-6))), vision_frame_count=len(valid), vision_valid_frames=int(valid.sum()),
        alignment_status=alignment['status'], alignment_review_flags=alignment['review_flags'],
        merge_review_flags=';'.join(merge_flags), needs_review=str(needs_review).lower(), status='ok',
        output_relpath=(OUTPUT/(sample+'.npz')).relative_to(DRAFT).as_posix(), output_sha256='', error='')
    return payload, record


def save_payload(path, payload):
    if path.exists():
        with np.load(path, allow_pickle=False) as previous:
            if set(previous.files) != set(payload):
                raise ValueError('Existing output schema differs; left unchanged')
            for key, value in payload.items():
                old = previous[key]
                numeric = old.dtype.kind in 'fc' and value.dtype.kind in 'fc'
                equal = np.array_equal(old, value, equal_nan=True) if numeric else np.array_equal(old, value)
                if not equal:
                    raise ValueError(f'Existing output differs at {key}; left unchanged')
        return False
    temporary = path.with_suffix('.partial.npz')
    np.savez_compressed(temporary, **payload)
    with np.load(temporary, allow_pickle=False) as z:
        for key, value in payload.items():
            if z[key].shape != value.shape:
                raise ValueError('Serialized shape mismatch: ' + key)
    temporary.rename(path)
    return True


def save_report(records):
    temporary = REPORT.with_suffix('.partial.csv')
    with temporary.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--pilot-check', action='store_true')
    args = parser.parse_args()
    with np.load(DRAFT/'pilot-sample/2026-09-23_pilot-audio-word-features_v1.0.npz', allow_pickle=False) as z:
        audio_names = z['feature_names'].tolist()
    if len(audio_names) != 25 or len(VISION_FIELDS) != 49:
        raise ValueError('Unexpected modality dimensions')
    pilot_check(audio_names)
    if args.pilot_check:
        return 0
    stage = rows(DRAFT/'2026-09-23_mosei-stage-map_v1.0.csv')
    expected = {f'sample_{i:04d}' for i in range(1, 101)}
    if len(stage) != 100 or {Path(r['stage_relpath']).stem for r in stage} != expected:
        raise ValueError('Expected 100 unique stage-map rows')
    alignment = table_by_sample(DRAFT/'2026-09-24_mosei-word-alignment-check_v1.0.csv')
    audio = table_by_sample(DRAFT/'2026-09-24_mosei-audio-check_v1.0.csv')
    features = table_by_sample(DRAFT/'2026-09-24_mosei-audio-egemaps-check_v1.0.csv')
    visual = table_by_sample(DRAFT/'2026-09-23_mosei-openface-qc_v1.0.csv')
    if not args.check:
        OUTPUT.mkdir(exist_ok=True)
    results, created, reused = [], 0, 0
    for index, source in enumerate(stage, 1):
        sample = Path(source['stage_relpath']).stem
        try:
            payload, record = merge(sample, source, alignment[sample], audio[sample], features[sample], visual[sample], audio_names)
            if not args.check:
                path = OUTPUT/(sample+'.npz')
                if save_payload(path, payload): created += 1
                else: reused += 1
                record['output_sha256'] = digest(path)
            results.append(record)
        except Exception as error:
            record = dict.fromkeys(REPORT_FIELDS, '')
            record.update(sample=sample, video_id=source['video_id'], clip_id=source['clip_id'], status='failed', error=str(error))
            results.append(record)
            print(f'Failed: {sample}: {error}', flush=True)
        if not args.check:
            save_report(results)
        if index % 10 == 0:
            print(f'Checked: {index} / 100', flush=True)
    good = [r for r in results if r['status'] == 'ok']
    summary = dict(samples_verified=len(good), failed=100-len(good), newly_created=created, reused=reused,
        source_words=sum(r['source_words'] for r in good), timed_words=sum(r['timed_words'] for r in good),
        all_modalities_valid_words=sum(r['all_modalities_valid_words'] for r in good),
        alignment_review_samples=sum(r['alignment_status']=='needs_review' for r in good),
        merge_review_samples=sum(bool(r['merge_review_flags']) for r in good),
        needs_review_samples=sum(r['needs_review']=='true' for r in good),
        no_valid_visual_samples=sum(r['vision_valid_frames']==0 for r in good))
    if not args.check:
        info = {**POLICY, 'summary': summary, 'script_sha256': digest(Path(__file__))}
        temporary = METHOD.with_suffix('.partial.json')
        temporary.write_text(json.dumps(info, indent=2), encoding='utf-8')
        temporary.replace(METHOD)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print('Read-only preflight complete' if args.check else f'Saved: {OUTPUT}\nQuality report: {REPORT}')
    return 0 if len(good)==100 else 1


if __name__ == '__main__':
    sys.exit(main())
