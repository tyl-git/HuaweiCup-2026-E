"""Extract 100 staged clips as mono PCM16 WAV; --check performs no writes."""

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path


DRAFT = Path(__file__).resolve().parents[1]
MAPPING = DRAFT / '2026-09-23_mosei-stage-map_v1.0.csv'
OUTPUT = DRAFT / 'mosei-audio-v1.0'
REPORT = DRAFT / '2026-09-24_mosei-audio-check_v1.0.csv'
FFMPEG = Path(r'D:\06_Apps\ffmpeg\bin\ffmpeg.exe')
FFPROBE = FFMPEG.with_name('ffprobe.exe')
FIELDS = [
    'sample', 'video_id', 'clip_id', 'stage_relpath', 'audio_relpath',
    'source_sha256', 'audio_sha256', 'sample_rate', 'channels',
    'sample_width_bytes', 'audio_samples', 'audio_duration_seconds',
    'audio_stream_start_seconds', 'video_stream_start_seconds',
    'audio_vs_video_start_seconds', 'transcript_qc', 'status', 'error',
]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(command):
    result = subprocess.run(
        [str(value) for value in command], capture_output=True,
        text=True, encoding='utf-8', errors='replace', check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    return result.stdout


def inspect_wav(path):
    with wave.open(str(path), 'rb') as stream:
        rate, channels, width = (
            stream.getframerate(), stream.getnchannels(), stream.getsampwidth()
        )
        frames = stream.getnframes()
        if (rate, channels, width, stream.getcomptype()) != (16000, 1, 2, 'NONE'):
            raise ValueError(f'Unexpected WAV format: {path.name}')
        if frames <= 0:
            raise ValueError(f'Empty WAV: {path.name}')
        size = 0
        while block := stream.readframes(65536):
            size += len(block)
        if size != frames * channels * width:
            raise ValueError(f'Truncated WAV: {path.name}')
    return {
        'sample_rate': rate, 'channels': channels, 'sample_width_bytes': width,
        'audio_samples': frames, 'audio_duration_seconds': f'{frames / rate:.6f}',
    }


def source_timing(path):
    info = json.loads(run([
        FFPROBE, '-v', 'error', '-show_entries',
        'stream=codec_type,start_time', '-of', 'json', path,
    ]))
    streams = info.get('streams', [])
    audio = [s for s in streams if s['codec_type'] == 'audio']
    video = [s for s in streams if s['codec_type'] == 'video']
    if not audio or not video:
        raise ValueError('Source must contain both an audio and a video stream')
    audio_start = audio[0].get('start_time', '')
    video_start = video[0].get('start_time', '')
    try:
        offset = f'{float(audio_start) - float(video_start):.6f}'
    except (ValueError, TypeError):
        offset = ''
    return {
        'audio_stream_start_seconds': audio_start,
        'video_stream_start_seconds': video_start,
        'audio_vs_video_start_seconds': offset,
    }


def save_report(records):
    temporary = REPORT.with_suffix('.partial.csv')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records.values())
    temporary.replace(REPORT)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Read-only input check')
    args = parser.parse_args()
    for program in (FFMPEG, FFPROBE):
        if not program.is_file():
            raise FileNotFoundError(program)
        run([program, '-version'])
    with MAPPING.open(encoding='utf-8-sig', newline='') as stream:
        entries = list(csv.DictReader(stream))
    if len(entries) != 100:
        raise ValueError(f'Expected 100 mapping rows; found {len(entries)}')
    staged_root = (DRAFT / 'mosei-stage-v1.0').resolve()
    seen = set()
    for entry in entries:
        source = (DRAFT / entry['stage_relpath']).resolve()
        if not source.is_relative_to(staged_root) or not source.is_file():
            raise ValueError(f'Invalid staged path: {source}')
        if source.stem in seen:
            raise ValueError(f'Duplicate sample: {source.stem}')
        seen.add(source.stem)
        if digest(source) != entry['staged_sha256']:
            raise ValueError(f'Staged video hash mismatch: {source.name}')
    print('Staged videos verified: 100 / 100', flush=True)
    print('FFmpeg and FFprobe: OK', flush=True)
    if args.check:
        print('Preflight OK (no audio extracted)')
        return 0

    previous = {}
    if REPORT.exists():
        with REPORT.open(encoding='utf-8-sig', newline='') as stream:
            previous = {r['sample']: r for r in csv.DictReader(stream)}
    OUTPUT.mkdir(exist_ok=True)
    records = {}
    # Keep prior rows in the on-disk report if the user interrupts a resumed run.
    checkpoint = dict(previous)
    extracted = reused = failed = 0
    for index, entry in enumerate(entries, 1):
        source = (DRAFT / entry['stage_relpath']).resolve()
        destination = OUTPUT / f'{source.stem}.wav'
        record = {key: '' for key in FIELDS}
        record.update(
            sample=source.stem, video_id=entry['video_id'], clip_id=entry['clip_id'],
            stage_relpath=entry['stage_relpath'],
            audio_relpath=destination.relative_to(DRAFT).as_posix(),
            source_sha256=entry['staged_sha256'], transcript_qc=entry['qc_status'],
        )
        try:
            old = previous.get(source.stem, {})
            if destination.exists():
                if (old.get('status') != 'ok'
                        or old.get('source_sha256') != entry['staged_sha256']
                        or old.get('audio_sha256') != digest(destination)):
                    raise ValueError('Existing WAV lacks matching provenance; left unchanged')
                record.update(inspect_wav(destination))
                for name in ('audio_stream_start_seconds', 'video_stream_start_seconds',
                             'audio_vs_video_start_seconds'):
                    record[name] = old.get(name, '')
                reused += 1
            else:
                record.update(source_timing(source))
                # No trimming, denoising, loudness normalization, or tempo change.
                # A retry may replace only this script's incomplete temporary file.
                temporary = OUTPUT / f'{source.stem}.partial.wav'
                run([
                    FFMPEG, '-nostdin', '-hide_banner', '-v', 'error', '-xerror', '-y',
                    '-i', source, '-map', '0:a:0', '-vn', '-ac', '1', '-ar', '16000',
                    '-c:a', 'pcm_s16le', temporary,
                ])
                record.update(inspect_wav(temporary))
                temporary.rename(destination)
                extracted += 1
            record.update(status='ok', audio_sha256=digest(destination))
        except Exception as error:
            failed += 1
            record.update(status='failed', error=str(error))
            print(f'Failed: {source.stem}: {error}', flush=True)
        records[source.stem] = record
        checkpoint[source.stem] = record
        save_report(checkpoint)
        if index % 10 == 0:
            print(f'Checked: {index} / 100', flush=True)
    save_report(records)
    total_seconds = sum(float(r['audio_duration_seconds'])
                        for r in records.values() if r['status'] == 'ok')
    print(f'Audio files verified: {extracted + reused} / 100')
    print(f'Newly extracted: {extracted}; reused: {reused}; failed: {failed}')
    print('Target format: 16000 Hz / mono / PCM 16-bit')
    print(f'Total audio duration: {total_seconds:.2f} seconds')
    print(f'Audio directory: {OUTPUT}')
    print(f'Quality report: {REPORT}')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
