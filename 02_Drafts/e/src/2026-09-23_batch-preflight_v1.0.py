from pathlib import Path
import json
import pandas as pd
from transformers import AutoTokenizer

project = Path(r'D:\01_Projects\HuaweiCup-2026')
draft = project / '02_Drafts' / 'e'

manifest = pd.read_csv(
    draft / '2026-09-23_mosei_sample-manifest_v1.0.csv',
    dtype=str, keep_default_na=False,
)
assert len(manifest) == 100
assert not manifest.duplicated(['video_id', 'clip_id']).any()
assert manifest['text'].str.strip().ne('').all()

labels = list((project / '01_Source' / 'E').rglob('label-100.xlsx'))
assert len(labels) == 1
source = labels[0].parent.resolve()

total_bytes = 0
for row in manifest.itertuples(index=False):
    expected = f'{row.video_id}/{row.clip_id}.mp4'
    assert row.video_relpath.replace('\\', '/') == expected
    video = (source / row.video_relpath).resolve()
    assert video.is_relative_to(source)
    assert video.is_file(), f'Missing video: {row.video_relpath}'
    total_bytes += video.stat().st_size

qc_path = draft / 'pilot-sample' / '2026-09-23_pilot-qc_v1.0.json'
qc = json.loads(qc_path.read_text(encoding='utf-8-sig'))
assert qc['sample_id'] == '-THoVjtIkeU/2'
assert qc['transcript'] == 'confirmed_by_listening'

tokenizer = AutoTokenizer.from_pretrained(
    r'D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594',
    local_files_only=True, use_fast=True,
)
lengths = [
    len(tokenizer(text, truncation=False)['input_ids'])
    for text in manifest['text']
]
assert max(lengths) <= 512

print('Samples checked:', len(manifest))
print('Source videos: 100 / 100')
print('Total video size:', f'{total_bytes / 1024**2:.2f} MiB')
print('Maximum words:', manifest['text'].str.split().str.len().max())
print('Maximum BERT tokens:', max(lengths))
print('Known transcript issues:',
      int((manifest['qc_status'] == 'transcript_review_needed').sum()))
print('Pilot review: recorded')
print('Preflight OK')
