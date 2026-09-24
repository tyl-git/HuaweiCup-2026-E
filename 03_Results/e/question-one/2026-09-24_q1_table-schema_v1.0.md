# Question One: Table Schema

The TSV files are UTF-8, tab-delimited, one row per sample. Counts refer to original transcript words.
A false modality mask is a missing observation; the corresponding zero vector is not an observed neutral signal.
P1 means transcript discrepancy, no valid visual frame, or excluded speaker annotation. Other flagged samples are P2.
P1/P2 are a derived work order, not model confidence or measured error rates. All 65 queued rows remain pending human review.
Nine untimed source tokens are retained: six standalone punctuation tokens and three excluded speaker-name tokens in sample_0006. The latter drive its P1 review.

## Sample Table

| Field | Definition |
|---|---|
| `sample` | Stable staged sample ID in source spreadsheet order. |
| `video_id` | Original MOSEI video identifier. |
| `clip_id` | Original segment identifier within the video. |
| `source_excel_row` | 1-based row number in label-100.xlsx, including header. |
| `video_relpath` | Original relative video path below the label workbook directory. |
| `source_label` | Unmodified sentiment score from the source spreadsheet. |
| `source_annotation` | Unmodified source polarity annotation. |
| `transcript` | Unmodified source transcript; no ASR correction applied. |
| `source_qc_status` | Original manifest transcript review status. |
| `container_duration_s` | FFprobe container duration in seconds; not the audio duration. |
| `audio_duration_s` | Decoded 16 kHz mono WAV duration in seconds. |
| `video_frames` | Decoded frame count recorded by OpenFace QC. |
| `openface_valid_frames` | Frames with OpenFace success, confidence >= 0.8 and finite features. |
| `source_words` | Number of original whitespace-delimited transcript tokens. |
| `timed_words` | Tokens with an automatic CTC time interval. |
| `untimed_words` | Original tokens retained without a valid word interval. |
| `text_valid_words` | Tokens with a valid BERT word vector. |
| `audio_valid_words` | Tokens with positive overlap with eGeMAPS windows. |
| `vision_valid_words` | Tokens with positive overlap with valid OpenFace frames. |
| `all_modalities_valid_words` | Tokens where timing and all three modality masks are true. |
| `audio_min_coverage` | Lowest fraction of timed word covered by audio windows in the sample. |
| `vision_min_coverage` | Lowest fraction of timed word covered by valid video-frame intervals. |
| `vision_partial_words` | Visually valid words with less than full temporal coverage. |
| `text_shape` | Variable-length word matrix dimensions, original words x 768. |
| `audio_shape` | Variable-length word matrix dimensions, original words x 25. |
| `vision_shape` | Variable-length word matrix dimensions, original words x 49. |
| `alignment_review_flags` | Automatic CTC/transcript quality triage reasons; not an accuracy score. |
| `merge_review_flags` | Automatic audio/visual coverage or missingness reasons. |
| `needs_review` | True if any existing alignment or merge review flag is present. |
| `review_priority` | Derived review order, P1 before P2; blank for unflagged samples. |
| `human_review_status` | Pending for flagged samples; not inferred from automatic checks. |
| `source_video_sha256` | SHA-256 of the original video, independently recalculated. |
| `staged_video_sha256` | SHA-256 of the staged video, independently recalculated. |
| `audio_sha256` | SHA-256 of decoded WAV, verified against extraction QC. |
| `audio_feature_sha256` | SHA-256 of eGeMAPS frame CSV, verified against extraction QC. |
| `text_feature_sha256` | SHA-256 of BERT word NPZ, verified against extraction QC. |
| `merged_feature_sha256` | SHA-256 of the merged word-level NPZ, independently recalculated. |
| `stage_relpath` | Staged video path relative to the draft directory. |
| `merged_relpath` | Merged NPZ path relative to the draft directory. |

## Review Queue

Queue rows are a subset of the sample table. Additional fields:

| Field | Definition |
|---|---|
| `review_reason` | Semicolon-separated automatic/source flags driving triage. |
| `suggested_action` | Suggested listening/video check; no correction has been applied. |

Original source, staged media and NPZ hashes are independently checked during generation.
