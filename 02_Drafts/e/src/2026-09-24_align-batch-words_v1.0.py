"""Offline CTC word alignment. --check tests inputs and the pilot without batch output."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import unicodedata
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

DRAFT = Path(__file__).resolve().parents[1]
OUTPUT = DRAFT / 'mosei-word-times-v1.0'
REPORT = DRAFT / '2026-09-24_mosei-word-alignment-check_v1.0.csv'
RUN_INFO = DRAFT / '2026-09-24_mosei-word-alignment-method_v1.0.json'
MODEL = Path(r'D:\06_Apps\huggingface-data\hub\models--facebook--wav2vec2-base-960h\snapshots\22aad52d435eb6dbaf354bdad9b0da84ce7d6156')
NUMBERS = {
    '2008': 'TWO THOUSAND EIGHT', '10TH': 'TENTH',
    '1500': 'ONE THOUSAND FIVE HUNDRED', '100000': 'ONE HUNDRED THOUSAND',
    '20000': 'TWENTY THOUSAND', '50': 'FIFTY', '1989': 'NINETEEN EIGHTY NINE',
}
WORD_FIELDS = ['word_index', 'word', 'char_start', 'char_end', 'alignment_text',
               'alignable', 'start_s', 'end_s', 'mean_ctc_probability', 'review_flags']
SUMMARY_FIELDS = ['sample', 'video_id', 'clip_id', 'source_words', 'alignable_words',
                  'aligned_words', 'untimed_words', 'mean_ctc_probability',
                  'asr_wer', 'first_start_s', 'last_end_s', 'review_flags',
                  'status', 'words_relpath', 'details_relpath', 'error']
POLICY = {
    'version': '1.0', 'model_snapshot': str(MODEL),
    'timing': 'CTC bin [start,end) * convolution stride / 16000; no duration/T scaling',
    'source_words': 'Original whitespace tokens, 1-based index; original text unchanged',
    'normalization': 'Uppercase; curly apostrophes normalized; punctuation stripped; hyphens split',
    'speaker_annotation': 'sample_0006 bracketed label masked as presumed metadata, requires review',
    'numbers': NUMBERS, 'numeric_readings': 'Provisional; all numeric samples require listening review',
    'untimed_words': 'Pure punctuation and speaker metadata have blank times and alignable=false',
    'qc_rules': 'Review if ASR WER>0.35, word mean CTC probability<0.20, long span>2s, or >=4 letters in <=20ms',
    'qc_interpretation': 'Heuristics for triage only; probabilities/WER are not measured alignment accuracy',
    'original_transcript_issue': 'sample_0001 source answers retained; listener suggested matters; unverified',
}


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows, fields):
    temporary = path.with_suffix('.partial.csv')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path, value):
    temporary = path.with_suffix('.partial.json')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def normalize(sample, text):
    rows, flags = [], set()
    annotation = [(m.start(), m.end()) for m in re.finditer(r'\[President Ronald Reagan:\]', text)]
    if annotation:
        if sample != 'sample_0006':
            raise ValueError('Unexpected speaker annotation')
        flags.add('speaker_annotation_review')
    for index, match in enumerate(re.finditer(r'\S+', text), 1):
        word = match.group()
        token_flags = set()
        normalized = ''
        if any(match.start() < stop and match.end() > start for start, stop in annotation):
            token_flags.add('speaker_annotation_excluded')
        else:
            normalized = unicodedata.normalize('NFKC', word).upper()
            normalized = normalized.replace('\u2019', "'").replace('\u2018', "'")
            normalized = re.sub(r'[-\u2013\u2014]', ' ', normalized)
            normalized = re.sub(r'[^A-Z0-9\s\']', '', normalized)
            normalized = ' '.join(part.strip("'") for part in normalized.split()).strip()
            expanded = []
            for part in normalized.split():
                if re.search(r'\d', part):
                    if part not in NUMBERS:
                        raise ValueError(f'Unsupported number: {sample} {word}')
                    expanded.append(NUMBERS[part])
                    token_flags.add('numeric_reading_review')
                    flags.add('numeric_reading_review')
                else:
                    expanded.append(part)
            normalized = ' '.join(expanded)
            if not normalized:
                token_flags.add('punctuation_only')
        if not re.fullmatch(r"[A-Z' ]*", normalized):
            raise ValueError(f'Unsupported CTC text: {word}')
        rows.append(dict(word_index=index, word=word, char_start=match.start(), char_end=match.end(),
                         alignment_text=normalized, alignable=bool(normalized), start_s='', end_s='',
                         mean_ctc_probability='', review_flags=';'.join(sorted(token_flags))))
    if sample == 'sample_0001':
        flags.add('transcript_review_needed')
    if sample == 'sample_0023':
        flags.add('ellipsis_title_review')
    if not any(row['alignable'] for row in rows):
        raise ValueError('No alignable words')
    return rows, flags


def load_inputs():
    sources = read_csv(DRAFT / '2026-09-23_mosei-stage-map_v1.0.csv')
    audio_rows = read_csv(DRAFT / '2026-09-24_mosei-audio-check_v1.0.csv')
    expected = {f'sample_{i:04d}' for i in range(1, 101)}
    audio = {r['sample']: r for r in audio_rows}
    if len(sources) != 100 or len(audio_rows) != 100 or set(audio) != expected:
        raise ValueError('Expected 100 unique source/audio rows')
    items = {}
    for source in sources:
        sample = Path(source['stage_relpath']).stem
        record = audio[sample]
        if record['status'] != 'ok' or any(source[k] != record[k] for k in ('video_id', 'clip_id')):
            raise ValueError('Source/audio identity mismatch: ' + sample)
        path = (DRAFT / record['audio_relpath']).resolve()
        if path != (DRAFT / 'mosei-audio-v1.0' / (sample + '.wav')).resolve():
            raise ValueError('Unexpected audio path')
        if digest(path) != record['audio_sha256']:
            raise ValueError('WAV hash mismatch: ' + sample)
        normalize(sample, source['text'])
        items[sample] = (source, record, path)
    if set(items) != expected:
        raise ValueError('Missing or duplicate source samples')
    print('Input WAV files and transcripts verified: 100 / 100', flush=True)
    return items


def wav_signal(path):
    with wave.open(str(path), 'rb') as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getcomptype()) != (1, 2, 16000, 'NONE'):
            raise ValueError('Expected 16kHz mono PCM16 WAV')
        count = stream.getnframes()
        signal = np.frombuffer(stream.readframes(count), dtype='<i2').astype(np.float32) / 32768.0
    if len(signal) != count or not count:
        raise ValueError('Truncated or empty WAV')
    return signal


def word_error_rate(reference, hypothesis):
    ref, hyp = reference.split(), hypothesis.split()
    previous = list(range(len(hyp) + 1))
    for i, a in enumerate(ref, 1):
        current = [i]
        for j, b in enumerate(hyp, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(ref))


def align(sample, item, processor, model, device):
    source, audio, path = item
    words, flags = normalize(sample, source['text'])
    vocab = processor.tokenizer.get_vocab()
    target, owners = [], []
    for i, row in enumerate(words):
        if not row['alignable']:
            continue
        if target:
            target.append(vocab['|'])
            owners.append(None)
        for character in row['alignment_text']:
            target.append(vocab['|' if character == ' ' else character])
            owners.append(None if character == ' ' else i)
    signal = wav_signal(path)
    if len(signal) != int(audio['audio_samples']):
        raise ValueError('Audio sample count changed')
    batch = processor(signal, sampling_rate=16000, return_tensors='pt')
    with torch.inference_mode():
        logits = model(batch.input_values.to(device)).logits.float().cpu()
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite CTC emissions')
    seconds_per_bin = math.prod(model.config.conv_stride) / 16000
    log_probs = logits.log_softmax(-1)
    target_tensor = torch.tensor([target], dtype=torch.int64)
    repeats = sum(a == b for a, b in zip(target, target[1:]))
    if len(target) + repeats > logits.shape[1]:
        raise ValueError('Audio has too few CTC bins for the provided transcript')
    paths, scores = torchaudio.functional.forced_align(log_probs, target_tensor, blank=processor.tokenizer.pad_token_id)
    spans = torchaudio.functional.merge_tokens(paths[0], scores[0].exp(), blank=processor.tokenizer.pad_token_id)
    if [span.token for span in spans] != target:
        raise ValueError('Aligned character sequence differs from target')
    grouped = {i: [] for i, word in enumerate(words) if word['alignable']}
    for owner, span in zip(owners, spans):
        if owner is not None:
            grouped[owner].append(span)
    last_end, probabilities = 0.0, []
    for i, token_spans in grouped.items():
        start = token_spans[0].start * seconds_per_bin
        end = token_spans[-1].end * seconds_per_bin
        probability = sum(s.score * len(s) for s in token_spans) / sum(len(s) for s in token_spans)
        if not (last_end <= start < end <= len(signal) / 16000 + 1e-8):
            raise ValueError('Invalid or nonmonotonic word times')
        last_end = end
        probabilities.append(probability)
        row_flags = set(filter(None, words[i]['review_flags'].split(';')))
        if probability < .20:
            row_flags.add('low_ctc_probability')
        if end - start > 2:
            row_flags.add('long_word_span')
        if len(token_spans) >= 4 and end - start <= .020001:
            row_flags.add('compressed_word_span')
        flags.update(row_flags - {'punctuation_only'})
        words[i].update(start_s=round(start, 6), end_s=round(end, 6),
                        mean_ctc_probability=round(probability, 6), review_flags=';'.join(sorted(row_flags)))
    normalized = ' '.join(w['alignment_text'] for w in words if w['alignable'])
    asr = processor.batch_decode(logits.argmax(-1))[0]
    wer = word_error_rate(normalized, asr)
    if wer > .35:
        flags.add('asr_disagreement')
    timed = [w for w in words if w['alignable']]
    summary = dict(sample=sample, video_id=source['video_id'], clip_id=source['clip_id'],
                   source_words=len(words), alignable_words=len(timed), aligned_words=len(timed),
                   untimed_words=len(words)-len(timed), mean_ctc_probability=round(float(np.mean(probabilities)), 6),
                   asr_wer=round(wer, 6), first_start_s=timed[0]['start_s'], last_end_s=timed[-1]['end_s'],
                   review_flags=';'.join(sorted(flags)), status='needs_review' if flags else 'auto_checked',
                   words_relpath=(OUTPUT / (sample+'.csv')).relative_to(DRAFT).as_posix(),
                   details_relpath=(OUTPUT / (sample+'.json')).relative_to(DRAFT).as_posix(), error='')
    details = dict(sample=sample, original_text=source['text'], alignment_text=normalized,
                   original_label=source['label'], original_annotation=source['annotation'],
                   original_transcript_qc=source['qc_status'], original_qc_note=source['qc_note'],
                   audio_sha256=audio['audio_sha256'], audio_duration_s=len(signal)/16000,
                   audio_vs_video_start_seconds=audio['audio_vs_video_start_seconds'],
                   greedy_asr=asr, ctc_step_s=seconds_per_bin, ctc_bins=logits.shape[1],
                   target_token_ids=target, target_source_word_indices=[None if i is None else i+1 for i in owners],
                   ctc_token_start_bins=[s.start for s in spans], ctc_token_end_bins=[s.end for s in spans],
                   words=words, summary=summary,
                   timing_note='Automatic 20ms-bin estimates; exact word boundaries not manually verified')
    return words, details, log_probs.squeeze(0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    items = load_inputs()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Loading cached CTC model; device: ' + device, flush=True)
    torch.manual_seed(0)
    processor = Wav2Vec2Processor.from_pretrained(str(MODEL), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(MODEL), local_files_only=True).to(device).eval()
    pilot_result = align('sample_0013', items['sample_0013'], processor, model, device)
    previous = read_csv(DRAFT / 'pilot-sample' / '2026-09-23_pilot-word-times_v1.0.csv')
    pilot_words = pilot_result[0]
    if [w['alignment_text'] for w in pilot_words] != [w['word'].upper() for w in previous]:
        raise ValueError('Pilot word identity mismatch')
    delta = max(abs(float(a[k])-float(b[k])) for a,b in zip(pilot_words,previous) for k in ('start_s','end_s'))
    print(f'Pilot: {len(pilot_words)} words; maximum boundary difference: {delta:.6f}s', flush=True)
    if delta > .040001:
        raise ValueError('Pilot differs by more than two CTC bins; inspect before batch')
    if args.check:
        print('Preflight OK (no batch output written)', flush=True)
        return 0
    OUTPUT.mkdir(exist_ok=True)
    provenance = dict(POLICY, torch_version=torch.__version__, torchaudio_version=torchaudio.__version__,
                      device=device, script_sha256=digest(Path(__file__)))
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    summaries = {}
    for index, (sample, item) in enumerate(items.items(), 1):
        csv_path, details_path = OUTPUT/(sample+'.csv'), OUTPUT/(sample+'.json')
        try:
            if csv_path.exists() or details_path.exists():
                old = json.loads(details_path.read_text(encoding='utf-8')) if details_path.exists() else {}
                if (old.get('run_signature') != signature or old.get('audio_sha256') != item[1]['audio_sha256']
                        or old.get('original_text') != item[0]['text']
                        or not csv_path.exists() or old.get('csv_sha256') != digest(csv_path)):
                    raise ValueError('Existing alignment differs from provenance; left unchanged')
                summary = old['summary']
            else:
                words, details, emissions = pilot_result if sample == 'sample_0013' else align(sample, item, processor, model, device)
                # Cache emissions to review alternate number readings without repeating GPU inference.
                temporary = OUTPUT/(sample+'.partial.npz')
                np.savez_compressed(temporary, log_probs=emissions)
                temporary.replace(OUTPUT/(sample+'.npz'))
                write_csv(csv_path, words, WORD_FIELDS)
                details.update(run_signature=signature, csv_sha256=digest(csv_path))
                write_json(details_path, details)
                summary = details['summary']
            summaries[sample] = summary
        except Exception as error:
            summaries[sample] = dict(sample=sample, status='failed', error=str(error))
            print(f'Failed: {sample}: {error}', flush=True)
        write_csv(REPORT, summaries.values(), SUMMARY_FIELDS)
        if index % 10 == 0:
            print(f'Checked: {index} / 100', flush=True)
    failed = sum(s['status']=='failed' for s in summaries.values())
    review = sum(s['status']=='needs_review' for s in summaries.values())
    aligned = sum(int(s.get('aligned_words',0)) for s in summaries.values())
    write_json(RUN_INFO, dict(provenance, run_signature=signature, pilot_max_boundary_difference_s=delta,
                             completed=100-failed, failed=failed, needs_review=review, aligned_words=aligned))
    print(f'Alignment files generated: {100-failed} / 100')
    print(f'Failed: {failed}; needs review: {review}; auto-checked: {100-failed-review}')
    print(f'Aligned source words: {aligned}')
    print(f'Output directory: {OUTPUT}')
    print(f'Quality report: {REPORT}')
    return int(failed > 0)


if __name__ == '__main__':
    sys.exit(main())
