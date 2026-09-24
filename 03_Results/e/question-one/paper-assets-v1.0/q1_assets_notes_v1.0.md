# Question-one paper assets

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

Figure generation in this run: PNG and SVG generated.
