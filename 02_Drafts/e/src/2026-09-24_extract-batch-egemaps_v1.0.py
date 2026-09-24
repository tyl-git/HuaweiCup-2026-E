"""Extract eGeMAPSv02 LLD features; --check validates inputs and the pilot only."""
import argparse
import csv
import hashlib
import sys
import wave
from io import StringIO
from pathlib import Path

import numpy as np
import opensmile
import pandas as pd

DRAFT = Path(__file__).resolve().parents[1]
AUDIO = DRAFT / 'mosei-audio-v1.0'
OUTPUT = DRAFT / 'mosei-audio-egemaps-v1.0'
REPORT = DRAFT / '2026-09-24_mosei-audio-egemaps-check_v1.0.csv'
AUDIO_REPORT = DRAFT / '2026-09-24_mosei-audio-check_v1.0.csv'
CONFIG = 'eGeMAPSv02/LowLevelDescriptors/defaults'
FIELDS = [
    'sample', 'video_id', 'clip_id', 'audio_relpath', 'feature_relpath',
    'audio_sha256', 'feature_sha256', 'opensmile_version', 'configuration',
    'frames', 'feature_columns', 'first_start_s', 'last_end_s',
    'median_step_s', 'min_window_s', 'max_window_s', 'audio_duration_s',
    'audio_vs_video_start_seconds', 'finite_values', 'transcript_qc', 'status', 'error',
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_report(rows):
    temporary = REPORT.with_suffix('.partial.csv')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows.values())
    temporary.replace(REPORT)


def extract(smile, audio_path):
    native = smile.process_file(str(audio_path))
    # Keep the library's actual intervals, including overlapping/tail windows.
    frame = native.reset_index(drop=True)
    frame.insert(0, 'end_s', native.index.get_level_values('end').total_seconds())
    frame.insert(0, 'start_s', native.index.get_level_values('start').total_seconds())
    return frame


def validate_features(frame, feature_names, duration):
    if list(frame.columns) != ['start_s', 'end_s'] + feature_names:
        raise ValueError('Feature names/order differ from the pilot configuration')
    if frame.empty or not np.isfinite(frame.to_numpy(dtype=np.float64)).all():
        raise ValueError('Empty table or non-finite timestamps/features')
    start = frame['start_s'].to_numpy(dtype=float)
    end = frame['end_s'].to_numpy(dtype=float)
    steps = np.diff(start)
    if np.any(start < 0) or np.any(end <= start):
        raise ValueError('Invalid feature time interval')
    if np.any(steps <= 0) or np.any(np.diff(end) < -1e-8):
        raise ValueError('Feature times are not ordered')
    if abs(start[0]) > 1e-8 or end.max() > duration + 1 / 16000 + 1e-8:
        raise ValueError('Feature intervals lie outside decoded audio duration')
    # Zero pitch/jitter and finite sentinel values are retained.
    return {
        'frames': len(frame), 'feature_columns': len(feature_names),
        'first_start_s': f'{start[0]:.6f}', 'last_end_s': f'{end[-1]:.6f}',
        'median_step_s': f'{np.median(steps):.6f}' if len(steps) else '',
        'min_window_s': f'{np.min(end - start):.6f}',
        'max_window_s': f'{np.max(end - start):.6f}', 'finite_values': 'true',
    }


def check_inputs():
    entries = read_rows(AUDIO_REPORT)
    if len(entries) != 100 or any(row['status'] != 'ok' for row in entries):
        raise ValueError('Audio quality report must contain 100 successful rows')
    expected = {f'sample_{i:04d}' for i in range(1, 101)}
    if {row['sample'] for row in entries} != expected:
        raise ValueError('Audio report contains missing or duplicate sample IDs')
    for row in entries:
        path = (DRAFT / row['audio_relpath']).resolve()
        if path.parent != AUDIO.resolve() or path.name != row['sample'] + '.wav':
            raise ValueError('Unexpected audio path: ' + str(path))
        if digest(path) != row['audio_sha256']:
            raise ValueError('Audio hash mismatch: ' + path.name)
        with wave.open(str(path), 'rb') as stream:
            if (stream.getframerate(), stream.getnchannels(), stream.getsampwidth(),
                    stream.getcomptype()) != (16000, 1, 2, 'NONE'):
                raise ValueError('Unexpected audio format: ' + path.name)
            count = stream.getnframes()
        if count != int(row['audio_samples']) or count <= 0:
            raise ValueError('Audio sample count mismatch: ' + path.name)
        if abs(count / 16000 - float(row['audio_duration_seconds'])) > 1e-6:
            raise ValueError('Audio duration mismatch: ' + path.name)
    print('Input WAV files verified: 100 / 100', flush=True)
    return entries


def check_pilot(smile):
    pilot = DRAFT / 'pilot-sample'
    # Container metadata may differ, so compare decoded PCM data.
    with wave.open(str(pilot / '2026-09-23_pilot-audio_v1.0.wav'), 'rb') as old:
        with wave.open(str(AUDIO / 'sample_0013.wav'), 'rb') as current:
            duration = current.getnframes() / current.getframerate()
            if (old.getparams() != current.getparams()
                    or old.readframes(old.getnframes()) != current.readframes(current.getnframes())):
                raise ValueError('Pilot audio differs from sample_0013')
    expected = pd.read_csv(pilot / '2026-09-23_pilot-audio-egemaps_v1.0.csv')
    actual = extract(smile, AUDIO / 'sample_0013.wav')
    if actual.shape != expected.shape or list(actual.columns) != list(expected.columns):
        raise ValueError('Pilot feature shape or column names changed')
    np.testing.assert_allclose(actual.to_numpy(), expected.to_numpy(), rtol=1e-5, atol=1e-6)
    validate_features(actual, smile.feature_names, duration)
    validate_features(pd.read_csv(StringIO(actual.to_csv(index=False))), smile.feature_names, duration)
    print(f'Pilot regression: OK ({len(actual)} frames, 25 features)', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='No batch output written')
    args = parser.parse_args()
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
    )
    if opensmile.__version__ != '2.6.0' or smile.num_features != 25:
        raise ValueError('This batch requires the verified openSMILE 2.6.0 / 25-D setup')
    entries = check_inputs()
    check_pilot(smile)
    print(f'openSMILE: {opensmile.__version__}; feature columns: 25', flush=True)
    if args.check:
        print('Preflight OK (no batch features written)')
        return 0

    OUTPUT.mkdir(exist_ok=True)
    previous = {row['sample']: row for row in read_rows(REPORT)} if REPORT.exists() else {}
    checkpoint = {row['sample']: previous[row['sample']]
                  for row in entries if row['sample'] in previous}
    results = {}
    created = reused = failed = 0
    for index, row in enumerate(entries, 1):
        sample = row['sample']
        audio_path = DRAFT / row['audio_relpath']
        destination = OUTPUT / (sample + '.csv')
        duration = int(row['audio_samples']) / 16000
        record = {key: '' for key in FIELDS}
        for key in ('sample', 'video_id', 'clip_id', 'audio_relpath', 'audio_sha256',
                    'audio_vs_video_start_seconds', 'transcript_qc'):
            record[key] = row[key]
        record.update(feature_relpath=destination.relative_to(DRAFT).as_posix(),
                      opensmile_version=opensmile.__version__, configuration=CONFIG,
                      audio_duration_s=f'{duration:.6f}')
        try:
            existing = destination.exists()
            old = previous.get(sample, {})
            if existing:
                if (old.get('status') != 'ok' or old.get('audio_sha256') != row['audio_sha256']
                        or old.get('opensmile_version') != opensmile.__version__
                        or old.get('configuration') != CONFIG
                        or old.get('feature_sha256') != digest(destination)):
                    raise ValueError('Existing feature CSV lacks matching provenance; left unchanged')
                frame = pd.read_csv(destination)
            else:
                frame = extract(smile, audio_path)
            record.update(validate_features(frame, smile.feature_names, duration))
            if not existing:
                temporary = OUTPUT / (sample + '.partial.csv')
                frame.to_csv(temporary, index=False, encoding='utf-8-sig')
                validate_features(pd.read_csv(temporary), smile.feature_names, duration)
                # Windows rename fails if the final path already exists.
                temporary.rename(destination)
            record.update(feature_sha256=digest(destination), status='ok')
            if existing:
                reused += 1
            else:
                created += 1
        except Exception as error:
            failed += 1
            record.update(status='failed', error=str(error))
            print(f'Failed: {sample}: {error}', flush=True)
        results[sample] = record
        checkpoint[sample] = record
        save_report(checkpoint)
        if index % 10 == 0:
            print(f'Checked: {index} / 100', flush=True)
    save_report(results)
    total_frames = sum(int(row['frames']) for row in results.values() if row['status'] == 'ok')
    print(f'Feature files verified: {created + reused} / 100')
    print(f'Newly extracted: {created}; reused: {reused}; failed: {failed}')
    print('Feature columns: 25')
    print(f'Total feature frames: {total_frames}')
    print(f'Feature directory: {OUTPUT}')
    print(f'Quality report: {REPORT}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
