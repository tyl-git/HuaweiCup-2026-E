"""Map aligned attachment-4 token evidence to automatically estimated media times."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import subprocess
import unicodedata
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import BertTokenizerFast, Wav2Vec2ForCTC, Wav2Vec2Processor


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q3 = ROOT / "03_Results/e/question-three"
INPUT = Q3 / "q3-explanations-v1.0"
OUT = Q3 / "q3-evidence-time-v1.0"
BERT = Path(r"D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594")
CTC = Path(r"D:\06_Apps\huggingface-data\hub\models--facebook--wav2vec2-base-960h\snapshots\22aad52d435eb6dbaf354bdad9b0da84ce7d6156")
FFMPEG = Path(r"D:\06_Apps\ffmpeg\bin\ffmpeg.exe")
FFPROBE = Path(r"D:\06_Apps\ffmpeg\bin\ffprobe.exe")
NUMBERS = {"28": "TWENTY EIGHT"}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run(command):
    result = subprocess.run([str(x) for x in command], capture_output=True, check=True)
    return result.stdout


def media(video):
    info = json.loads(run([FFPROBE, "-v", "error", "-show_entries",
                           "format=duration:stream=codec_type,start_time", "-of", "json", video]))
    duration = float(info["format"]["duration"])
    frame_text = run([FFPROBE, "-v", "error", "-select_streams", "v:0",
                      "-show_entries", "frame=best_effort_timestamp_time",
                      "-of", "csv=p=0", video]).decode("utf-8")
    frames = np.asarray([float(line.strip().split(",")[0]) for line in frame_text.splitlines()
                         if line.strip() and line.strip().split(",")[0] != "N/A"], dtype=np.float64)
    if not len(frames) or np.any(np.diff(frames) < 0):
        raise RuntimeError(f"No ordered video frame times: {video}")
    audio = np.frombuffer(run([FFMPEG, "-nostdin", "-v", "error", "-i", video,
                               "-vn", "-ac", "1", "-ar", "16000", "-f", "f32le",
                               "-acodec", "pcm_f32le", "pipe:1"]), dtype="<f4").copy()
    if not len(audio) or not np.isfinite(audio).all():
        raise RuntimeError(f"No finite audio: {video}")
    return duration, frames, audio, info


def words_for(text):
    words = []
    for match in re.finditer(r"\S+", text):
        raw = match.group()
        normalized = unicodedata.normalize("NFKC", raw).upper()
        normalized = normalized.replace("\u2019", "'").replace("\u2018", "'")
        normalized = re.sub(r"[-\u2013\u2014]", " ", normalized)
        normalized = re.sub(r"[^A-Z0-9' ]", "", normalized)
        parts = []
        for part in normalized.split():
            parts.append(NUMBERS.get(part, "" if any(c.isdigit() for c in part) else part.strip("'")))
        normalized = " ".join(x for x in parts if x)
        words.append({"text": raw, "char_start": match.start(), "char_end": match.end(),
                      "ctc_text": normalized, "start_s": None, "end_s": None,
                      "mean_ctc_probability": None})
    return words


def align_words(audio, words, processor, model, device):
    vocab = processor.tokenizer.get_vocab()
    target, owners = [], []
    for i, word in enumerate(words):
        if not word["ctc_text"]:
            continue
        if target:
            target.append(vocab["|"])
            owners.append(None)
        for character in word["ctc_text"]:
            target.append(vocab["|" if character == " " else character])
            owners.append(None if character == " " else i)
    if not target:
        raise RuntimeError("Transcript has no alignable CTC words")
    with torch.inference_mode():
        batch = processor(audio, sampling_rate=16000, return_tensors="pt")
        logits = model(batch.input_values.to(device)).logits.float().cpu()
    step = math.prod(model.config.conv_stride) / 16000
    repetitions = sum(a == b for a, b in zip(target, target[1:]))
    if len(target) + repetitions > logits.shape[1]:
        raise RuntimeError("Transcript longer than available CTC bins")
    paths, scores = torchaudio.functional.forced_align(
        logits.log_softmax(-1), torch.tensor([target], dtype=torch.int64),
        blank=processor.tokenizer.pad_token_id)
    spans = torchaudio.functional.merge_tokens(
        paths[0], scores[0].exp(), blank=processor.tokenizer.pad_token_id)
    if [span.token for span in spans] != target:
        raise RuntimeError("CTC path differs from transcript")
    grouped = {i: [] for i, word in enumerate(words) if word["ctc_text"]}
    for owner, span in zip(owners, spans):
        if owner is not None:
            grouped[owner].append(span)
    probabilities = []
    for i, spans in grouped.items():
        start, end = spans[0].start * step, spans[-1].end * step
        if not 0 <= start < end <= len(audio) / 16000 + step:
            raise RuntimeError("Invalid CTC word time")
        probability = sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)
        words[i].update(start_s=start, end_s=end, mean_ctc_probability=probability)
        probabilities.append(probability)
    prediction = processor.batch_decode(logits.argmax(-1))[0]
    reference = " ".join(word["ctc_text"] for word in words if word["ctc_text"])
    return word_error_rate(reference, prediction), float(np.mean(probabilities)), prediction, step


def word_error_rate(reference, hypothesis):
    first, second = reference.split(), hypothesis.split()
    previous = list(range(len(second) + 1))
    for i, a in enumerate(first, 1):
        current = [i]
        for j, b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[j] + 1,
                               previous[j - 1] + (a != b)))
        previous = current
    return previous[-1] / max(1, len(first))


def matching_word(words, start, end):
    matches = [(min(end, w["char_end"]) - max(start, w["char_start"]), i)
               for i, w in enumerate(words)]
    overlap, index = max(matches, default=(0, -1))
    return index if overlap > 0 else None


def frame_near(frames, seconds):
    return int(np.argmin(np.abs(frames - seconds)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    explanations = read_csv(INPUT / "attachment4_explanations.csv")
    effects = read_csv(INPUT / "attachment4_position_effects.csv")
    if len(explanations) != 20 or {x["file_id"] for x in explanations} != {
            f"{i:02}" for i in range(1, 21)}:
        raise RuntimeError("Attachment-4 explanation set differs")
    if not all(x.is_file() for x in (BERT / "tokenizer.json", CTC / "config.json", FFMPEG, FFPROBE)):
        raise FileNotFoundError("Local model or media tool missing")
    for row in explanations:
        if not Path(row["video_file"]).is_file():
            raise FileNotFoundError(row["video_file"])
    print(f"Preflight OK: 20 videos, {len(effects)} aligned-position effects")
    if args.check:
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    temporary = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or temporary.exists():
        raise FileExistsError("Time-map output exists; refusing to overwrite")
    processor = Wav2Vec2Processor.from_pretrained(str(CTC), local_files_only=True)
    model = Wav2Vec2ForCTC.from_pretrained(str(CTC), local_files_only=True).to(args.device).eval()
    tokenizer = BertTokenizerFast.from_pretrained(str(BERT), local_files_only=True)
    by_file = {name: [] for name in (row["file_id"] for row in explanations)}
    for row in effects:
        by_file[row["file_id"]].append(row)
    word_rows, evidence_rows, quality_rows = [], [], []
    for explanation in explanations:
        file_id, text = explanation["file_id"], explanation["raw_text"]
        video = Path(explanation["video_file"])
        duration, frames, audio, stream_info = media(video)
        words = words_for(text)
        error = ""
        try:
            wer, mean_probability, asr, step = align_words(
                audio, words, processor, model, args.device)
            quality = ("review" if wer > .35 or mean_probability < .20
                       else "automatic_candidate")
        except (RuntimeError, ValueError, KeyError) as exc:
            wer, mean_probability, asr, step = None, None, "", None
            quality, error = "unmapped", str(exc)
        encoded = tokenizer(text, padding="max_length", truncation=True,
                            max_length=50, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        for i, word in enumerate(words):
            word_rows.append({"file_id": file_id, "word_1based": i + 1,
                              "word": word["text"], "char_start": word["char_start"],
                              "char_end": word["char_end"], "start_s": word["start_s"],
                              "end_s": word["end_s"],
                              "mean_ctc_probability": word["mean_ctc_probability"],
                              "alignment_status": quality if word["start_s"] is not None else "unmapped"})
        for effect in by_file[file_id]:
            pos = int(effect["position_0based"])
            start, end = offsets[pos]
            index = matching_word(words, start, end) if end > start else None
            word = words[index] if index is not None else None
            mapped = word is not None and word["start_s"] is not None
            midpoint = (word["start_s"] + word["end_s"]) / 2 if mapped else None
            frame_index = frame_near(frames, midpoint) if mapped else None
            evidence_rows.append({"file_id": file_id, "modality": effect["modality"],
                                  "position_0based": pos, "text_fragment": effect["text_fragment"],
                                  "class_probability_delta": effect["class_probability_delta"],
                                  "word_1based": index + 1 if index is not None else "",
                                  "word": word["text"] if word else "",
                                  "start_s": word["start_s"] if mapped else "",
                                  "end_s": word["end_s"] if mapped else "",
                                  "video_frame_0based": frame_index if mapped else "",
                                  "video_frame_time_s": float(frames[frame_index]) if mapped else "",
                                  "mapping_status": quality if mapped else "unmapped"})
        quality_rows.append({"file_id": file_id, "video_sha256": sha(video),
                             "video_duration_s": duration, "audio_duration_s": len(audio) / 16000,
                             "video_frames": len(frames), "transcript_words": len(words),
                             "timed_words": sum(w["start_s"] is not None for w in words),
                             "asr_wer": wer, "mean_ctc_probability": mean_probability,
                             "ctc_step_s": step, "alignment_status": quality,
                             "error": error, "greedy_asr": asr,
                             "stream_start_times_json": json.dumps([
                                 {"codec_type": s.get("codec_type"), "start_time": s.get("start_time")}
                                 for s in stream_info["streams"]])})
        print(f"Time map: {file_id}/20 status={quality}", flush=True)
    temporary.mkdir(parents=True)
    save_csv(temporary / "attachment4_word_times.csv", word_rows,
             ("file_id", "word_1based", "word", "char_start", "char_end",
              "start_s", "end_s", "mean_ctc_probability", "alignment_status"))
    save_csv(temporary / "attachment4_evidence_times.csv", evidence_rows,
             ("file_id", "modality", "position_0based", "text_fragment",
              "class_probability_delta", "word_1based", "word", "start_s", "end_s",
              "video_frame_0based", "video_frame_time_s", "mapping_status"))
    save_csv(temporary / "attachment4_time_quality.csv", quality_rows,
             ("file_id", "video_sha256", "video_duration_s", "audio_duration_s",
              "video_frames", "transcript_words", "timed_words", "asr_wer",
              "mean_ctc_probability", "ctc_step_s", "alignment_status", "error",
              "greedy_asr", "stream_start_times_json"))
    summary = {"schema": "q3_evidence_time_v1.0", "automatic_not_human_verified": True,
               "method": "Wav2Vec2 CTC forced transcript alignment, token-char overlap, nearest decoded frame timestamp",
               "ctc_model": str(CTC), "bert_tokenizer": str(BERT),
               "sample_count": len(quality_rows), "position_count": len(evidence_rows),
               "status_counts": {s: sum(row["alignment_status"] == s for row in quality_rows)
                                 for s in ("automatic_candidate", "review", "unmapped")},
               "input_sha256": {p.name: sha(p) for p in INPUT.glob("*.csv")},
               "output_sha256": {p.name: sha(p) for p in temporary.glob("*.csv")}}
    (temporary / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.rename(OUT)
    print(f"Time map saved: {OUT}")


if __name__ == "__main__":
    main()
