"""Offline BERT extraction preserving source whitespace word IDs; --check writes nothing."""
import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer

DRAFT = Path(__file__).resolve().parents[1]
MODEL = Path(r'D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594')
OUTPUT = DRAFT / 'mosei-text-bert-v1.0'
REPORT = DRAFT / '2026-09-24_mosei-text-bert-check_v1.0.csv'
METHOD = DRAFT / '2026-09-24_mosei-text-bert-method_v1.0.json'
FIELDS = ['sample', 'source_words', 'bert_tokens', 'hidden_size', 'finite_values',
          'alignment_join', 'alignment_status', 'alignment_review_flags', 'output_sha256',
          'transcript_sha256', 'alignment_sha256', 'run_signature', 'status', 'error']
POOLING = 'last_hidden_state; mean per source whitespace word including punctuation'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_report(data):
    temporary = REPORT.with_suffix('.partial.csv')
    with temporary.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(data.values())
    temporary.replace(REPORT)


def prepare_inputs(tokenizer, model):
    manifest = rows(DRAFT / '2026-09-23_mosei-stage-map_v1.0.csv')
    review = rows(DRAFT / '2026-09-24_mosei-word-alignment-check_v1.0.csv')
    alignment = {row['sample']: row for row in review}
    expected = {f'sample_{i:04d}' for i in range(1, 101)}
    if len(manifest) != 100 or len(review) != 100 or set(alignment) != expected:
        raise ValueError('Expected 100 unique source and alignment rows')
    prepared = {}
    for item in manifest:
        sample = Path(item['stage_relpath']).stem
        text = item['text']
        spans = list(re.finditer(r'\S+', text))
        words = [s.group() for s in spans]
        if alignment[sample]['status'] == 'failed':
            raise ValueError('Failed alignment: ' + sample)
        word_path = DRAFT / 'mosei-word-times-v1.0' / (sample + '.csv')
        aligned = rows(word_path)
        if len(aligned) != len(words):
            raise ValueError('Source/alignment word count mismatch: ' + sample)
        for i, (word, match, row) in enumerate(zip(words, spans, aligned), 1):
            if (word != row['word'] or int(row['word_index']) != i
                    or (int(row['char_start']), int(row['char_end'])) != match.span()):
                raise ValueError('Source/alignment word identity mismatch: ' + sample)
        encoded = tokenizer(words, is_split_into_words=True, return_tensors='pt', truncation=False)
        raw_ids = tokenizer(text, truncation=False)['input_ids']
        if encoded.input_ids[0].tolist() != raw_ids:
            raise ValueError('Pretokenization changed original BERT tokens: ' + sample)
        if encoded.input_ids.shape[1] > model.config.max_position_embeddings:
            raise ValueError('Transcript exceeds BERT capacity: ' + sample)
        ids = encoded.word_ids()
        if set(x for x in ids if x is not None) != set(range(len(words))):
            raise ValueError('BERT tokenizer lost a source word: ' + sample)
        prepared[sample] = (item, words, spans, aligned, word_path, encoded, ids, alignment[sample])
    if set(prepared) != expected:
        raise ValueError('Missing or duplicate source samples')
    print('Source/alignment/token mappings verified: 100 / 100', flush=True)
    print('Maximum BERT tokens: ' + str(max(p[5].input_ids.shape[1] for p in prepared.values())), flush=True)
    return prepared


def extract(tokenizer, model, prepared, device):
    item, words, spans, aligned, path, encoded, ids, review = prepared
    with torch.inference_mode():
        hidden = model(**{key: value.to(device) for key, value in encoded.items()}).last_hidden_state[0].float().cpu().numpy()
    word_ids = np.array([-1 if i is None else i for i in ids], dtype=np.int64)
    features = np.stack([hidden[word_ids == i].mean(axis=0) for i in range(len(words))])
    input_ids = encoded.input_ids[0].numpy()
    payload = dict(
        token_features=hidden, word_features=features, input_ids=input_ids,
        attention_mask=encoded.attention_mask[0].numpy(),
        token_type_ids=encoded.token_type_ids[0].numpy(), word_ids=word_ids,
        tokens=np.array(tokenizer.convert_ids_to_tokens(input_ids)),
        words=np.array(words), word_indices=np.arange(1, len(words)+1, dtype=np.int64),
        word_char_spans=np.array([s.span() for s in spans], dtype=np.int64),
        text_valid_mask=np.ones(len(words), dtype=bool),
        transcript=np.array(item['text']), model_id=np.array('google-bert/bert-base-uncased'),
        model_revision=np.array(MODEL.name), pooling=np.array(POOLING),
        alignment_sha256=np.array(digest(path)), alignment_status=np.array(review['status']),
        alignment_review_flags=np.array(review['review_flags']),
    )
    return payload


def validate(payload, prepared):
    item, words, spans, aligned, path, encoded, ids, review = prepared
    if payload['words'].tolist() != words or payload['transcript'].item() != item['text']:
        raise ValueError('Stored word/transcript identity mismatch')
    if payload['word_features'].shape != (len(words), 768):
        raise ValueError('Invalid word feature shape')
    if payload['token_features'].shape != (len(ids), 768):
        raise ValueError('Invalid token feature shape')
    if not all(np.isfinite(payload[key]).all() for key in ('word_features', 'token_features')):
        raise ValueError('Non-finite BERT feature value')
    if not np.array_equal(payload['word_indices'], np.arange(1, len(words)+1)):
        raise ValueError('Incorrect source word index')
    if not np.array_equal(payload['word_ids'], [-1 if i is None else i for i in ids]):
        raise ValueError('Incorrect token ownership')
    if not np.array_equal(payload['word_char_spans'], [s.span() for s in spans]):
        raise ValueError('Incorrect original character spans')
    if not np.array_equal(payload['input_ids'], encoded.input_ids[0].numpy()):
        raise ValueError('Incorrect BERT input token IDs')
    if payload['text_valid_mask'].shape != (len(words),) or not payload['text_valid_mask'].all():
        raise ValueError('Missing text features')
    if payload['alignment_sha256'].item() != digest(path):
        raise ValueError('Alignment mapping changed')
    for i in range(len(words)):
        np.testing.assert_allclose(payload['word_features'][i],
                                   payload['token_features'][payload['word_ids']==i].mean(axis=0),
                                   rtol=1e-5, atol=1e-6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(MODEL), local_files_only=True).to(device).eval()
    if not tokenizer.is_fast or model.config.hidden_size != 768:
        raise ValueError('Expected fast BERT tokenizer and hidden size 768')
    print(f'BERT: {MODEL.name}; hidden size: 768; device: {device}', flush=True)
    prepared = prepare_inputs(tokenizer, model)
    pilot = extract(tokenizer, model, prepared['sample_0013'], device)
    validate(pilot, prepared['sample_0013'])
    with np.load(DRAFT / 'pilot-sample' / '2026-09-23_pilot-text-bert_v1.0.npz', allow_pickle=False) as old:
        for key in ('input_ids', 'attention_mask', 'word_ids'):
            np.testing.assert_array_equal(pilot[key], old[key])
        for key in ('token_features', 'word_features'):
            np.testing.assert_allclose(pilot[key], old[key], rtol=1e-4, atol=1e-5)
        difference = float(np.max(np.abs(pilot['word_features']-old['word_features'])))
    print(f'Pilot regression: OK (10 words, 768 dimensions; max difference {difference:.8g})', flush=True)
    if args.check:
        print('Preflight OK (no batch output written)')
        return 0

    OUTPUT.mkdir(exist_ok=True)
    provenance = dict(model_id='google-bert/bert-base-uncased', revision=MODEL.name, model_path=str(MODEL),
                      pooling=POOLING, model_mode='eval; float32; no truncation; no training',
                      torch_version=torch.__version__, transformers_version=transformers.__version__,
                      device=device, source_word_index='1-based original whitespace word; punctuation included',
                      timestamp_policy='Inherited alignment flags remain unresolved; BERT adds no timing assurance',
                      script_sha256=digest(Path(__file__)), pilot_max_difference=difference)
    signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    previous = {r['sample']: r for r in rows(REPORT)} if REPORT.exists() else {}
    checkpoint = {s: previous[s] for s in prepared if s in previous}
    result = {}
    created = reused = failed = 0
    for index, (sample, entry) in enumerate(prepared.items(), 1):
        path = OUTPUT / (sample + '.npz')
        item, words, spans, aligned, align_path, encoded, ids, review = entry
        record = {key: '' for key in FIELDS}
        record.update(sample=sample, transcript_sha256=hashlib.sha256(item['text'].encode()).hexdigest(),
                      alignment_sha256=digest(align_path), run_signature=signature,
                      alignment_status=review['status'], alignment_review_flags=review['review_flags'])
        try:
            exists = path.exists()
            if exists:
                old = previous.get(sample, {})
                if (old.get('status') != 'ok' or any(old.get(k) != record[k] for k in
                        ('transcript_sha256', 'alignment_sha256', 'run_signature'))
                        or old.get('output_sha256') != digest(path)):
                    raise ValueError('Existing NPZ lacks matching provenance; left unchanged')
                with np.load(path, allow_pickle=False) as saved:
                    payload = {key: saved[key] for key in saved.files}
            else:
                payload = pilot if sample == 'sample_0013' else extract(tokenizer, model, entry, device)
            validate(payload, entry)
            if not exists:
                temporary = OUTPUT / (sample + '.partial.npz')
                np.savez_compressed(temporary, **payload)
                with np.load(temporary, allow_pickle=False) as saved:
                    validate(saved, entry)
                temporary.rename(path)
            record.update(source_words=len(words), bert_tokens=len(ids), hidden_size=768,
                          finite_values='true', alignment_join='exact', output_sha256=digest(path), status='ok')
            if exists:
                reused += 1
            else:
                created += 1
        except Exception as error:
            record.update(status='failed', error=str(error))
            failed += 1
            print(f'Failed: {sample}: {error}', flush=True)
        result[sample] = record
        checkpoint[sample] = record
        save_report(checkpoint)
        if index % 10 == 0:
            print(f'Checked: {index} / 100', flush=True)
    save_report(result)
    provenance.update(run_signature=signature, successful=created+reused, failed=failed,
                      source_words=sum(int(r['source_words']) for r in result.values() if r['status']=='ok'))
    METHOD.write_text(json.dumps(provenance, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'BERT files verified: {created+reused} / 100')
    print(f'Newly extracted: {created}; reused: {reused}; failed: {failed}')
    print('Total source words: ' + str(provenance['source_words']))
    print('Word feature dimension: 768')
    print('Output directory: ' + str(OUTPUT))
    print('Quality report: ' + str(REPORT))
    return int(failed > 0)


if __name__ == '__main__':
    sys.exit(main())
