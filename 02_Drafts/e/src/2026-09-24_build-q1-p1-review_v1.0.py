"""Build read-only P1 evidence and sampled video frames, without changing review status."""
from pathlib import Path
import csv
import hashlib
import io
import json
import subprocess
import wave
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DRAFT = Path(__file__).resolve().parents[1]
FFMPEG = Path(r"D:\06_Apps\ffmpeg\bin\ffmpeg.exe")
REVIEW_VIDEO_DIR = DRAFT / "q1-manual-review-normalized-v1.0"
IDS = [f"sample_{i:04d}" for i in [1, 6, 10, 11, 42, 43, 88]]
OUT = DRAFT / "2026-09-24_q1-p1-review-evidence_v1.0.json"
SHEET = DRAFT / "2026-09-24_q1-p1-review-contact_v1.0.png"

def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))

font = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 18)
small = ImageFont.truetype(r"C:\Windows\Fonts\arial.ttf", 15)
sheet = Image.new("RGB", (1320, 1890), "#f5f7fa")
draw = ImageDraw.Draw(sheet)
draw.text((20, 12), "Review: six P1 samples + silent P2 sample 0043; all human judgments remain pending", fill="#15253b", font=font)
evidence = []
for row, sample in enumerate(IDS):
    detail = json.loads((DRAFT / "mosei-word-times-v1.0" / f"{sample}.json").read_text(encoding="utf-8"))
    face = rows(DRAFT / "mosei-openface-pilot-v1.0" / f"{sample}.csv")
    success = np.array([int(r["success"]) == 1 for r in face])
    confidence = np.array([float(r["confidence"]) for r in face])
    with np.load(DRAFT / "mosei-multimodal-aligned-v1.0" / f"{sample}.npz", allow_pickle=False) as npz:
        names = npz["vision_feature_names"].tolist()
        finite = np.array([all(np.isfinite(float(r[k])) for k in names) for r in face])
        valid = success & (confidence >= .8) & finite
        assert valid.sum() == npz["vision_source_valid_mask"].sum()
        word_evidence = [dict(word=str(w), start_s=None if not t else float(pair[0]), end_s=None if not t else float(pair[1]), visual_valid=bool(v), visual_coverage=float(c)) for w, pair, t, v, c in zip(npz["words"], npz["word_times_s"], npz["timing_mask"], npz["vision_mask"], npz["vision_coverage"])]
    with wave.open(str(DRAFT / "mosei-audio-v1.0" / f"{sample}.wav"), "rb") as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
    direct_check = None
    if not np.any(pcm):
        video_path = REVIEW_VIDEO_DIR / f"{sample}__normalized.mp4"
        decoded = subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video_path), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], check=True, capture_output=True)
        direct_pcm = np.frombuffer(decoded.stdout, dtype="<i2")
        native = subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-i", str(video_path), "-map", "0:a:0", "-f", "f32le", "-"], check=True, capture_output=True)
        native_pcm = np.frombuffer(native.stdout, dtype="<f4")
        direct_check = dict(samples=int(direct_pcm.size), nonzero_samples=int(np.count_nonzero(direct_pcm)), same_as_saved_wav=bool(np.array_equal(direct_pcm.astype(np.float64) / 32768.0, pcm)), native_no_downmix_float_values=int(native_pcm.size), native_no_downmix_nonzero_values=int(np.count_nonzero(native_pcm)), native_no_downmix_peak=float(np.max(np.abs(native_pcm))))
    duration = float(detail["audio_duration_s"])
    stats = dict(frames=len(face), success_frames=int(success.sum()), valid_frames=int(valid.sum()), confidence_min=float(confidence.min()), confidence_max=float(confidence.max()), confidence_positive_frames=int((confidence > 0).sum()))
    y = 46 + row * 262
    draw.text((20, y), f"{sample} | audio {duration:.3f}s | valid visual frames {stats['valid_frames']}/{stats['frames']}", fill="#15253b", font=font)
    times = [round(duration * ratio, 3) for ratio in [.1, .35, .65, .9]]
    for column, seconds in enumerate(times):
        cp = subprocess.run([str(FFMPEG), "-hide_banner", "-loglevel", "error", "-nostdin", "-ss", str(seconds), "-i", str(REVIEW_VIDEO_DIR / f"{sample}__normalized.mp4"), "-frames:v", "1", "-vf", "scale=318:180:force_original_aspect_ratio=decrease,pad=318:180:(ow-iw)/2:(oh-ih)/2", "-f", "image2pipe", "-vcodec", "png", "-"], check=True, capture_output=True)
        frame = Image.open(io.BytesIO(cp.stdout)).convert("RGB")
        x = 20 + column * 324
        sheet.paste(frame, (x, y + 32))
        draw.text((x, y + 216), f"seek {seconds:.3f} s", fill="#35465e", font=small)
    evidence.append(dict(sample=sample, review_status="pending", original_transcript=detail["original_text"], asr_hypothesis=detail["greedy_asr"], audio_duration_s=duration, visual_statistics=stats, pcm_rms=float(np.sqrt(np.mean(pcm ** 2))), pcm_peak=float(np.max(np.abs(pcm))), direct_audio_decode_check=direct_check, sampled_video_seek_seconds=times, sampled_frames_limit="Four sampled frames do not establish the content of the whole clip or speaker identity", word_evidence=word_evidence))
sheet.save(SHEET)
silence_rows = []
audio_paths = sorted((DRAFT / "mosei-audio-v1.0").glob("sample_*.wav"))
assert len(audio_paths) == 100, f"Expected 100 WAV files, got {len(audio_paths)}"
for path in audio_paths:
    with wave.open(str(path), "rb") as wav:
        assert wav.getsampwidth() == 2 and wav.getnchannels() == 1 and wav.getframerate() == 16000
        data = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    assert data.size > 0
    silence_rows.append(dict(sample=path.stem, wav_relpath=path.relative_to(DRAFT).as_posix(), wav_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), samples=int(data.size), nonzero_samples=int(np.count_nonzero(data)), all_samples_zero=bool(not np.any(data))))
silent_ids = [r["sample"] for r in silence_rows if r["all_samples_zero"]]
assert all(item["direct_audio_decode_check"]["nonzero_samples"] == 0 and item["direct_audio_decode_check"]["same_as_saved_wav"] for item in evidence if item["sample"] in silent_ids)
original_timed = 0
silent_timed = 0
original_complete = 0
silent_complete = 0
for path in sorted((DRAFT / "mosei-multimodal-aligned-v1.0").glob("sample_*.npz")):
    with np.load(path, allow_pickle=False) as data:
        count = int((data["timing_mask"] & data["audio_mask"]).sum())
        complete = int(data["all_modalities_mask"].sum())
        original_timed += count
        original_complete += complete
        if path.stem in silent_ids:
            silent_timed += count
            silent_complete += complete
silence_scan = dict(checked_wavs=len(silence_rows), all_zero_samples=silent_ids, historical_timing_and_audio_mask_words=original_timed, all_zero_sample_timed_words=silent_timed, candidate_timed_words_excluding_verified_digital_silence=original_timed - silent_timed, historical_all_modalities_mask_words=original_complete, all_zero_sample_all_modalities_mask_words=silent_complete, candidate_all_modalities_words_excluding_verified_digital_silence=original_complete - silent_complete, limitation="Nonzero samples do not establish speech presence, transcript correctness or word timing accuracy", records=silence_rows)
OUT.write_text(json.dumps(dict(review_status="all_pending", samples=evidence, all_audio_silence_scan=silence_scan), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(OUT)
print(SHEET)
for item in evidence:
    print(item["sample"], item["visual_statistics"], "pcm_rms", item["pcm_rms"])
print("All-zero WAV scan", len(silence_rows), silent_ids, "candidate timed words", original_timed - silent_timed)
