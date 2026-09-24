from pathlib import Path
import hashlib
import shutil
import pandas as pd

project = Path(r'D:\01_Projects\HuaweiCup-2026')
draft = project / '02_Drafts' / 'e'
stage = draft / 'mosei-stage-v1.0'
mapping = draft / '2026-09-23_mosei-stage-map_v1.0.csv'

if mapping.exists():
    raise FileExistsError(mapping)

manifest = pd.read_csv(
    draft / '2026-09-23_mosei_sample-manifest_v1.0.csv',
    dtype=str, keep_default_na=False,
)
assert len(manifest) == 100
assert not manifest.duplicated(['video_id', 'clip_id']).any()

labels = list((project / '01_Source' / 'E').rglob('label-100.xlsx'))
assert len(labels) == 1
source_root = labels[0].parent.resolve()

def sha256(path):
    with path.open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()

plans = []
for i, row in enumerate(manifest.itertuples(index=False), start=1):
    assert row.video_relpath.replace('\\', '/') == (
        f'{row.video_id}/{row.clip_id}.mp4'
    )
    source = (source_root / row.video_relpath).resolve()
    assert source.is_relative_to(source_root)
    assert source.is_file(), str(source)
    target = stage / f'sample_{i:04d}.mp4'
    plans.append((source, target))

stage.mkdir(exist_ok=True)
source_hashes = []
staged_hashes = []
staged_paths = []

for i, (source, target) in enumerate(plans, start=1):
    original_hash = sha256(source)

    if not target.exists():
        with source.open('rb') as reader, target.open('xb') as writer:
            shutil.copyfileobj(reader, writer, length=1024 * 1024)

    copied_hash = sha256(target)
    assert copied_hash == original_hash, f'Hash mismatch: {target}'

    source_hashes.append(original_hash)
    staged_hashes.append(copied_hash)
    staged_paths.append(target.relative_to(draft).as_posix())

    if i % 10 == 0:
        print(f'Verified: {i}/{len(plans)}')

result = manifest.copy()
result['stage_relpath'] = staged_paths
result['source_sha256'] = source_hashes
result['staged_sha256'] = staged_hashes
result.to_csv(mapping, index=False, encoding='utf-8-sig', mode='x')

print('Staged videos:', len(result))
print('SHA-256 verified:', len(staged_hashes), '/', len(result))
print('Mapping:', mapping)
print('Destination:', stage)
