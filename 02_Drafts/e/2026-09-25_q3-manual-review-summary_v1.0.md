# Question 3 Attachment 4 Manual Review Summary

Review record: `2026-09-25_q3-manual-review-record_v1.0.csv`

## Scope

Eight samples marked `review` by the automatic attachment-4 time-quality report were checked against the original videos. The review checked transcript content, candidate-word occurrence, approximate timing, and whether the speaker was visually identifiable.

## Results

| file | transcript | timing | visual observation | final |
|---|---|---|---|---|
| 03 | broadly present | fail: candidate times too early | speaker visible | review |
| 05 | pass | pass | speaker visible | pass |
| 11 | approximately present in the later remux checked by the reviewer | review: ffprobe reports 5.803971 s for the pipeline file, while a player/remux showed about 11 s | pending | review |
| 13 | broadly present | pass: approximately aligned | speaker distant/blurred; two people visible | review |
| 14 | review: word order and `Toronto` unclear | fail: `He` occurred around 4 s | not fully confirmed | review |
| 15 | fail: only `However, despite their` was heard | fail | not reported | fail |
| 16 | pass | fail: overall early shift | speaker visible | review |
| 20 | review: `wrap-up` sounded like `for you` | fail: overall shift | speaker visible | review |

Counts: 8 records; only file 05 has a clear manual pass for both transcript and approximate timing. File 11 remains under review because playback and ffprobe give different apparent durations; no precise time claim is made for it.

## Interpretation

The current Wav2Vec2 CTC output is an automatic candidate localization, not an exact human-verified word boundary. The mapping script aligns the provided transcript against the complete video audio and does not first estimate the interval in which that transcript occurs. When a clip contains extra leading speech or when the supplied wording differs from the audio, the forced path can assign an incorrect phrase or spread the transcript across an inappropriate interval. This is not evidence of a single global audio/video offset.

Therefore, the Q3 results may claim that occlusion identifies influential text positions and that the displayed times are candidate evidence. They must not claim exact word-level timestamps for the reviewed samples. Manual review is required before using any timestamp as a precise video explanation. The file-11 container/playback timeline discrepancy requires separate resolution; a text match in a remux does not validate the original automatic times.
