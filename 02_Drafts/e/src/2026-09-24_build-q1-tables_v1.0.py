"""Audit question-one artifacts and export reproducible sample/review tables.

The exports are derived reports. Original media, labels, features, and QC reports
are read only. Run with --check to compare existing exports without writing them.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
DRAFT = PROJECT / "02_Drafts" / "e"
RESULT = PROJECT / "03_Results" / "e" / "question-one"
STEM = "2026-09-24_q1"

SOURCES = {
    "manifest": "2026-09-23_mosei_sample-manifest_v1.0.csv",
    "stage": "2026-09-23_mosei-stage-map_v1.0.csv",
    "media": "2026-09-23_mosei_media-check_v1.0.csv",
    "openface": "2026-09-23_mosei-openface-qc_v1.0.csv",
    "audio": "2026-09-24_mosei-audio-check_v1.0.csv",
    "egemaps": "2026-09-24_mosei-audio-egemaps-check_v1.0.csv",
    "alignment": "2026-09-24_mosei-word-alignment-check_v1.0.csv",
    "bert": "2026-09-24_mosei-text-bert-check_v1.0.csv",
    "merged": "2026-09-24_mosei-multimodal-check_v1.0.csv",
}
SAMPLE_FIELDS = [
    "sample", "video_id", "clip_id", "source_excel_row", "video_relpath",
    "source_label", "source_annotation", "transcript", "source_qc_status",
    "container_duration_s", "audio_duration_s", "video_frames",
    "openface_valid_frames", "source_words", "timed_words", "untimed_words",
    "text_valid_words", "audio_valid_words", "vision_valid_words",
    "all_modalities_valid_words", "audio_min_coverage", "vision_min_coverage",
    "vision_partial_words", "text_shape", "audio_shape", "vision_shape",
    "alignment_review_flags", "merge_review_flags", "needs_review",
    "review_priority", "human_review_status", "source_video_sha256",
    "staged_video_sha256", "audio_sha256", "audio_feature_sha256",
    "text_feature_sha256", "merged_feature_sha256", "stage_relpath",
    "merged_relpath",
]
REVIEW_FIELDS = [
    "sample", "video_id", "clip_id", "review_priority", "review_reason",
    "suggested_action", "human_review_status", "source_words", "timed_words",
    "vision_valid_words", "all_modalities_valid_words", "source_qc_status",
    "alignment_review_flags", "merge_review_flags", "video_relpath",
    "merged_relpath",
]
SAMPLE_DESCRIPTIONS = {
    "sample": "Stable staged sample ID in source spreadsheet order.",
    "video_id": "Original MOSEI video identifier.",
    "clip_id": "Original segment identifier within the video.",
    "source_excel_row": "1-based row number in label-100.xlsx, including header.",
    "video_relpath": "Original relative video path below the label workbook directory.",
    "source_label": "Unmodified sentiment score from the source spreadsheet.",
    "source_annotation": "Unmodified source polarity annotation.",
    "transcript": "Unmodified source transcript; no ASR correction applied.",
    "source_qc_status": "Original manifest transcript review status.",
    "container_duration_s": "FFprobe container duration in seconds; not the audio duration.",
    "audio_duration_s": "Decoded 16 kHz mono WAV duration in seconds.",
    "video_frames": "Decoded frame count recorded by OpenFace QC.",
    "openface_valid_frames": "Frames with OpenFace success, confidence >= 0.8 and finite features.",
    "source_words": "Number of original whitespace-delimited transcript tokens.",
    "timed_words": "Tokens with an automatic CTC time interval.",
    "untimed_words": "Original tokens retained without a valid word interval.",
    "text_valid_words": "Tokens with a valid BERT word vector.",
    "audio_valid_words": "Tokens with positive overlap with eGeMAPS windows.",
    "vision_valid_words": "Tokens with positive overlap with valid OpenFace frames.",
    "all_modalities_valid_words": "Tokens where timing and all three modality masks are true.",
    "audio_min_coverage": "Lowest fraction of timed word covered by audio windows in the sample.",
    "vision_min_coverage": "Lowest fraction of timed word covered by valid video-frame intervals.",
    "vision_partial_words": "Visually valid words with less than full temporal coverage.",
    "text_shape": "Variable-length word matrix dimensions, original words x 768.",
    "audio_shape": "Variable-length word matrix dimensions, original words x 25.",
    "vision_shape": "Variable-length word matrix dimensions, original words x 49.",
    "alignment_review_flags": "Automatic CTC/transcript quality triage reasons; not an accuracy score.",
    "merge_review_flags": "Automatic audio/visual coverage or missingness reasons.",
    "needs_review": "True if any existing alignment or merge review flag is present.",
    "review_priority": "Derived review order, P1 before P2; blank for unflagged samples.",
    "human_review_status": "Pending for flagged samples; not inferred from automatic checks.",
    "source_video_sha256": "SHA-256 of the original video, independently recalculated.",
    "staged_video_sha256": "SHA-256 of the staged video, independently recalculated.",
    "audio_sha256": "SHA-256 of decoded WAV, verified against extraction QC.",
    "audio_feature_sha256": "SHA-256 of eGeMAPS frame CSV, verified against extraction QC.",
    "text_feature_sha256": "SHA-256 of BERT word NPZ, verified against extraction QC.",
    "merged_feature_sha256": "SHA-256 of the merged word-level NPZ, independently recalculated.",
    "stage_relpath": "Staged video path relative to the draft directory.",
    "merged_relpath": "Merged NPZ path relative to the draft directory.",
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    require(path.is_file(), f"Missing file: {path}")
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, skipinitialspace=True))


def keyed(rows: list[dict[str, str]], key: str, name: str) -> dict[str, dict[str, str]]:
    result = {row[key]: row for row in rows}
    require(len(rows) == len(result) == 100, f"{name}: expected 100 unique rows")
    return result


def verify_weights(z: np.lib.npyio.NpzFile, name: str, n: int) -> None:
    mask = z[f"{name}_mask"].astype(bool)
    count = z[f"{name}_frame_count"]
    weights = z[f"{name}_source_weights"]
    intervals = z[f"{name}_source_intervals_s"]
    coverage = z["audio_window_coverage" if name == "audio" else "vision_coverage"]
    require(mask.shape == count.shape == coverage.shape == (n,), f"{name}: word-axis schema")
    require(weights.shape == (n, len(intervals)) and intervals.shape[1:] == (2,), f"{name}: source-axis schema")
    require(np.isfinite(intervals).all() and np.all(intervals[:, 1] > intervals[:, 0]), f"{name}: invalid intervals")
    require(np.all(np.diff(intervals[:, 0]) > 0), f"{name}: unordered intervals")
    require(np.isfinite(weights).all() and np.all(weights >= 0), f"{name}: invalid weights")
    require(np.allclose(weights.sum(axis=1), mask.astype(float), atol=2e-6), f"{name}: weight rows do not sum to masks")
    require(np.array_equal((weights > 0).sum(axis=1), count), f"{name}: frame counts disagree with weights")
    require(np.all((coverage >= 0) & (coverage <= 1 + 1e-9)), f"{name}: coverage outside [0,1]")
    require(np.array_equal(mask, count > 0), f"{name}: mask/count mismatch")
    require(np.all(coverage[~mask] == 0), f"{name}: missing words have nonzero coverage")
    if name == "vision":
        valid = z["vision_source_valid_mask"]
        require(valid.shape == (len(intervals),), "vision: validity shape")
        require(np.all(weights[:, ~valid] == 0), "vision: weights on invalid frames")
    times = z["word_times_s"]
    overlap = np.maximum(0, np.minimum(times[:, None, 1], intervals[None, :, 1]) -
                         np.maximum(times[:, None, 0], intervals[None, :, 0]))
    require(np.all((weights > 0) <= (overlap > 0)), f"{name}: weight outside word interval")
    overlap[~z["timing_mask"]] = 0
    if name == "vision":
        overlap[:, ~z["vision_source_valid_mask"]] = 0
    overlap[overlap < 1e-12] = 0
    sums = overlap.sum(axis=1, keepdims=True)
    expected_weights = np.divide(overlap, sums, out=np.zeros_like(overlap), where=sums > 0)
    require(np.allclose(weights, expected_weights, atol=2e-7), f"{name}: normalized overlap weights")
    for j in range(n):
        if not mask[j]:
            continue
        start, end = times[j]
        right, union = start, 0.0
        for left, stop in intervals[overlap[j] > 0]:
            left, stop = max(start, left), min(end, stop)
            if stop > max(right, left):
                union += stop - max(right, left)
                right = stop
        require(abs(coverage[j] - union / (end - start)) < 1e-8, f"{name}: union coverage")


def verify_projection(z: np.lib.npyio.NpzFile, name: str, path: Path) -> None:
    records = read_csv(path)
    expected = z["audio_feature_names" if name == "audio" else "vision_feature_names"].tolist()
    values = np.array([[float(row[field]) for field in expected] for row in records], dtype=np.float64)
    weights = z[f"{name}_source_weights"].astype(np.float64)
    require(values.shape == (weights.shape[1], len(expected)), f"{name}: source dimension mismatch")
    if name == "vision":
        values[~z["vision_source_valid_mask"]] = 0
    require(np.isfinite(values).all(), f"{name}: nonfinite source feature")
    pooled = weights @ values
    require(np.allclose(pooled, z[name], atol=2e-4, rtol=2e-5), f"{name}: feature/weight reconstruction mismatch")


def priority_and_action(source: dict[str, str], align: dict[str, str], merged: dict[str, str]) -> tuple[str, str, str]:
    align_flags = set(filter(None, align["review_flags"].split(";")))
    merge_flags = set(filter(None, merged["merge_review_flags"].split(";")))
    reasons = sorted(align_flags | merge_flags | ({source["qc_status"]} if source["qc_status"] != "not_reviewed" else set()))
    priority = "P1" if ("no_valid_visual_frames" in merge_flags or
                        source["qc_status"] == "transcript_review_needed" or
                        "speaker_annotation_review" in align_flags) else "P2"
    if "no_valid_visual_frames" in merge_flags:
        action = "Check the video and face track; keep the visual mask false unless valid frames can be established."
    elif source["qc_status"] == "transcript_review_needed":
        action = "Listen against the original video and compare with the source transcript; preserve the source wording until confirmed."
    elif "speaker_annotation_review" in align_flags:
        action = "Listen and verify whether bracketed speaker text is metadata or spoken content."
    elif "vision_missing_words" in merge_flags or "vision_partial_coverage" in merge_flags:
        action = "Inspect the affected word intervals, video frame timestamps, and face detection coverage."
    else:
        action = "Listen to the flagged phrase and inspect the automatic word boundaries."
    return priority, ";".join(reasons), action


def generate() -> tuple[dict[str, str], dict]:
    tables = {name: read_csv(DRAFT / filename) for name, filename in SOURCES.items()}
    manifest, stage = tables["manifest"], tables["stage"]
    require(len(manifest) == len(stage) == 100, "Manifest/stage row count")
    require(len({(r["video_id"], r["clip_id"]) for r in manifest}) == 100, "Duplicate original identity")
    by = {name: keyed(tables[name], "sample", name) for name in
          ("openface", "audio", "egemaps", "alignment", "bert", "merged")}
    media = keyed(tables["media"], "video_relpath", "media")
    label_books = list((PROJECT / "01_Source" / "E").rglob("label-100.xlsx"))
    require(len(label_books) == 1, "Expected one original label-100.xlsx")
    source_root = label_books[0].parent
    label_table = pd.read_excel(label_books[0], dtype=str, keep_default_na=False)
    require(len(label_table) == 100, "Original label workbook row count")

    rows, queue, untimed_records = [], [], []
    flag_counts, alignment_counts, merge_counts = Counter(), Counter(), Counter()
    totals = Counter()
    pilot = json.loads((DRAFT / "pilot-sample" / "2026-09-23_pilot-qc_v1.0.json").read_text(encoding="utf-8-sig"))
    for i, (original, mapped, (_, excel)) in enumerate(zip(manifest, stage, label_table.iterrows()), 1):
        sample = f"sample_{i:04d}"
        require(mapped["stage_relpath"] == f"mosei-stage-v1.0/{sample}.mp4", f"{sample}: unexpected stage path")
        for field in ("source_excel_row", "video_id", "clip_id", "video_relpath", "text", "label", "annotation", "qc_status", "qc_note"):
            require(original[field] == mapped[field], f"{sample}: manifest/stage {field}")
        require(int(original["source_excel_row"]) == i + 1, f"{sample}: spreadsheet row")
        for field in ("video_id", "clip_id", "text", "annotation"):
            require(str(excel[field]) == original[field], f"{sample}: original Excel {field}")
        require(abs(float(excel["label"]) - float(original["label"])) < 1e-6, f"{sample}: original Excel label")
        require(original["video_relpath"] == f'{original["video_id"]}/{original["clip_id"]}.mp4', f"{sample}: original path")
        source_video = source_root / original["video_relpath"]
        staged_video = DRAFT / mapped["stage_relpath"]
        source_hash = digest(source_video)
        stage_hash = digest(staged_video)
        require(source_hash == stage_hash == mapped["source_sha256"] == mapped["staged_sha256"], f"{sample}: source/staged hash")
        m = media[original["video_relpath"]]
        require(m["read_status"] == "readable", f"{sample}: media unreadable")
        records = {key: by[key][sample] for key in by}
        o, a, eg, al, bert, mm = [records[key] for key in ("openface", "audio", "egemaps", "alignment", "bert", "merged")]
        for name, rec in records.items():
            require(rec.get("status", "ok") in ("ok", "needs_review", "auto_checked"), f"{sample}: {name} failed")
            if "video_id" in rec:
                require(rec["video_id"] == original["video_id"] and rec["clip_id"] == original["clip_id"], f"{sample}: {name} identity")
        require(a["source_sha256"] == source_hash and eg["audio_sha256"] == a["audio_sha256"], f"{sample}: audio provenance")
        require(int(m["video_nb_read_frames"]) == int(o["frames"]) == int(mm["vision_frame_count"]), f"{sample}: video frame count")
        require(int(o["valid_frames"]) == int(mm["vision_valid_frames"]), f"{sample}: valid frame count")
        for field in ("source_words", "timed_words", "untimed_words"):
            other = "aligned_words" if field == "timed_words" else field
            if field != "untimed_words":
                require(int(mm[field]) == int(al[other]), f"{sample}: word QC {field}")
        require(int(mm["source_words"]) == int(mm["timed_words"]) + int(mm["untimed_words"]), f"{sample}: word count")
        require(int(bert["source_words"]) == int(mm["source_words"]), f"{sample}: BERT count")
        require(bert["alignment_review_flags"] == al["review_flags"] == mm["alignment_review_flags"], f"{sample}: alignment flags")
        wav = DRAFT / a["audio_relpath"]
        audio_features = DRAFT / eg["feature_relpath"]
        text_features = DRAFT / "mosei-text-bert-v1.0" / f"{sample}.npz"
        merged_features = DRAFT / mm["output_relpath"]
        require(digest(wav) == a["audio_sha256"], f"{sample}: WAV hash")
        require(digest(audio_features) == eg["feature_sha256"], f"{sample}: audio feature hash")
        require(digest(text_features) == bert["output_sha256"], f"{sample}: BERT feature hash")
        merged_hash = digest(merged_features)
        require(merged_hash == mm["output_sha256"], f"{sample}: merged feature hash")

        with np.load(merged_features, allow_pickle=False) as z:
            n = int(mm["source_words"])
            require(z["text"].shape == (n, 768) and z["audio"].shape == (n, 25) and
                    z["vision"].shape == (n, 49), f"{sample}: feature shape")
            require(all(np.isfinite(z[k]).all() for k in ("text", "audio", "vision")), f"{sample}: nonfinite features")
            require(int(z["sequence_length"]) == n and str(z["sample_id"]) == sample, f"{sample}: NPZ identity")
            for key in ("video_id", "clip_id"):
                require(str(z[key]) == original[key], f"{sample}: NPZ {key}")
            require(str(z["source_video_relpath"]) == original["video_relpath"] and
                    str(z["transcript"]) == original["text"] and
                    str(z["label_original"]) == original["label"] and
                    str(z["annotation"]) == original["annotation"], f"{sample}: NPZ source values")
            require(float(z["label"]) == float(original["label"]), f"{sample}: NPZ numeric label")
            words = z["words"].tolist()
            matches = list(re.finditer(r"\S+", original["text"]))
            require(words == [x.group() for x in matches] and len(words) == n, f"{sample}: original words")
            require(np.array_equal(z["word_indices"], np.arange(1, n + 1)), f"{sample}: word indices")
            require(np.array_equal(z["word_char_spans"], [x.span() for x in matches]), f"{sample}: word character offsets")
            timed = z["timing_mask"]
            masks = {name: z[f"{name}_mask"] for name in ("text", "audio", "vision")}
            require(all(mask.shape == (n,) and mask.dtype.kind == "b" for mask in [timed, *masks.values()]),
                    f"{sample}: boolean word-axis masks")
            joint = timed & masks["text"] & masks["audio"] & masks["vision"]
            require(np.array_equal(joint, z["all_modalities_mask"]), f"{sample}: joint mask")
            require(np.isfinite(z["word_times_s"][timed]).all() and
                    np.isnan(z["word_times_s"][~timed]).all(), f"{sample}: missing time representation")
            require(np.all(z["word_times_s"][timed, 1] > z["word_times_s"][timed, 0]), f"{sample}: word time order")
            valid_times = z["word_times_s"][timed]
            require(np.all(valid_times[:, 0] >= 0) and
                    np.all(valid_times[:, 1] <= float(a["audio_duration_seconds"]) + 1 / 16000) and
                    np.all(valid_times[1:, 0] >= valid_times[:-1, 1] - 1e-9), f"{sample}: time range/overlap")
            word_rows = read_csv(DRAFT / al["words_relpath"])
            require([row["word"] for row in word_rows] == words, f"{sample}: word CSV identity")
            for word in word_rows:
                if not word["start_s"]:
                    untimed_records.append({"sample": sample, "word_index": int(word["word_index"]),
                                            "word": word["word"], "reason": word["review_flags"]})
            word_times = np.array([[float(row["start_s"]), float(row["end_s"])]
                                  if row["start_s"] else [np.nan, np.nan] for row in word_rows])
            require(np.array_equal(word_times, z["word_times_s"], equal_nan=True), f"{sample}: word CSV time mapping")
            with np.load(text_features, allow_pickle=False) as text_source:
                require(np.array_equal(text_source["word_features"], z["text"]) and
                        np.array_equal(text_source["text_valid_mask"], masks["text"]), f"{sample}: BERT features/mask")
            require(np.all(z["audio"][~masks["audio"]] == 0) and
                    np.all(z["vision"][~masks["vision"]] == 0), f"{sample}: invalid feature placeholder")
            for name in ("audio", "vision"):
                verify_weights(z, name, n)
            require(int(z["vision_source_valid_mask"].sum()) == int(mm["vision_valid_frames"]), f"{sample}: source visual mask")
            for name, key in (("timed", "timed_words"), ("text", "text_valid_words"),
                              ("audio", "audio_valid_words"), ("vision", "vision_valid_words"),
                              ("joint", "all_modalities_valid_words")):
                actual = int((timed if name == "timed" else joint if name == "joint" else masks[name]).sum())
                require(actual == int(mm[key]), f"{sample}: {key}")
            require(bool(z["needs_review"]) == (mm["needs_review"] == "true"), f"{sample}: review state")
            require(str(z["alignment_review_flags"]) == al["review_flags"] and
                    str(z["merge_review_flags"]) == mm["merge_review_flags"], f"{sample}: NPZ review flags")
            require(not bool(z["exact_word_boundaries_manually_verified"]), f"{sample}: manual boundary claim")
            provenance = json.loads(str(z["provenance_json"]))
            for relative, recorded in provenance.items():
                if relative == "script_sha256" or relative == "alignment_record":
                    continue
                require(digest(DRAFT / relative) == recorded, f"{sample}: NPZ provenance {relative}")
            require(provenance["alignment_record"] == al, f"{sample}: NPZ alignment record")
            verify_projection(z, "audio", audio_features)
            verify_projection(z, "vision", DRAFT / "mosei-openface-pilot-v1.0" / f"{sample}.csv")

        flagged = mm["needs_review"] == "true"
        require(flagged == (al["status"] == "needs_review" or bool(mm["merge_review_flags"])), f"{sample}: review gate")
        priority, reason, action = priority_and_action(original, al, mm) if flagged else ("", "", "")
        row = {
            "sample": sample, "video_id": original["video_id"], "clip_id": original["clip_id"],
            "source_excel_row": original["source_excel_row"], "video_relpath": original["video_relpath"],
            "source_label": original["label"], "source_annotation": original["annotation"],
            "transcript": original["text"], "source_qc_status": original["qc_status"],
            "container_duration_s": m["container_duration"], "audio_duration_s": a["audio_duration_seconds"],
            "video_frames": o["frames"], "openface_valid_frames": o["valid_frames"],
            **{key: mm[key] for key in ("source_words", "timed_words", "untimed_words", "text_valid_words",
                                          "audio_valid_words", "vision_valid_words", "all_modalities_valid_words",
                                          "audio_min_coverage", "vision_min_coverage", "vision_partial_words")},
            "text_shape": f"{mm['source_words']}x768", "audio_shape": f"{mm['source_words']}x25",
            "vision_shape": f"{mm['source_words']}x49", "alignment_review_flags": al["review_flags"],
            "merge_review_flags": mm["merge_review_flags"], "needs_review": str(flagged).lower(),
            "review_priority": priority, "human_review_status": "pending" if flagged else "not_requested",
            "source_video_sha256": source_hash, "staged_video_sha256": stage_hash,
            "audio_sha256": a["audio_sha256"], "audio_feature_sha256": eg["feature_sha256"],
            "text_feature_sha256": bert["output_sha256"], "merged_feature_sha256": merged_hash,
            "stage_relpath": mapped["stage_relpath"], "merged_relpath": mm["output_relpath"],
        }
        rows.append(row)
        for key in ("source_words", "timed_words", "untimed_words", "text_valid_words", "audio_valid_words",
                    "vision_valid_words", "all_modalities_valid_words"):
            totals[key] += int(mm[key])
        totals["needs_review_samples"] += flagged
        totals["no_valid_visual_samples"] += int(mm["vision_valid_frames"]) == 0
        totals["alignment_review_samples"] += al["status"] == "needs_review"
        totals["merge_review_samples"] += bool(mm["merge_review_flags"])
        for flag in filter(None, al["review_flags"].split(";")):
            alignment_counts[flag] += 1
            flag_counts[flag] += 1
        for flag in filter(None, mm["merge_review_flags"].split(";")):
            merge_counts[flag] += 1
            flag_counts[flag] += 1
        if flagged:
            queue.append({
                "sample": sample, "video_id": original["video_id"], "clip_id": original["clip_id"],
                "review_priority": priority, "review_reason": reason, "suggested_action": action,
                "human_review_status": "pending", "source_words": mm["source_words"],
                "timed_words": mm["timed_words"], "vision_valid_words": mm["vision_valid_words"],
                "all_modalities_valid_words": mm["all_modalities_valid_words"],
                "source_qc_status": original["qc_status"], "alignment_review_flags": al["review_flags"],
                "merge_review_flags": mm["merge_review_flags"], "video_relpath": original["video_relpath"],
                "merged_relpath": mm["output_relpath"],
            })
        if i % 10 == 0:
            print(f"Audited: {i}/100", flush=True)

    require(len(rows) == 100 and len(queue) == 65, "Expected 100 samples and 65 flagged rows")
    require(totals["source_words"] == 1932 and totals["timed_words"] == 1923 and
            totals["all_modalities_valid_words"] == 1828 and totals["no_valid_visual_samples"] == 4,
            "Headline totals disagree with previous QC")
    queue.sort(key=lambda row: (row["review_priority"], row["sample"]))
    overview = {
        "schema_version": "1.0", "audit_date": "2026-09-24",
        "scope": "Question one: three-modality feature extraction and word-level temporal alignment; no sentiment predictions",
        "source_label_workbook": label_books[0].relative_to(PROJECT).as_posix(),
        "source_label_workbook_sha256": digest(label_books[0]),
        "input_reports_sha256": {key: digest(DRAFT / filename) for key, filename in SOURCES.items()},
        "totals": {"samples": len(rows), **dict(totals)},
        "review_priority_counts": dict(Counter(row["review_priority"] for row in queue)),
        "review_priority_rule": {
            "P1": "Source transcript discrepancy, zero valid visual frames, or excluded speaker annotation requiring listening review.",
            "P2": "All other samples already carrying alignment or merge review flags.",
            "blank": "No automatic sample review flag. This does not imply manual verification.",
        },
        "untimed_token_reason_counts": dict(Counter(row["reason"] for row in untimed_records)),
        "untimed_word_records": untimed_records,
        "alignment_flag_sample_counts": dict(sorted(alignment_counts.items())),
        "merge_flag_sample_counts": dict(sorted(merge_counts.items())),
        "all_flag_sample_counts": dict(sorted(flag_counts.items())),
        "pilot_human_check": {
            "sample_id": pilot["sample_id"], "transcript": pilot["transcript"],
            "phrase_timing": pilot["phrase_timing"], "exact_word_boundaries": pilot["exact_word_boundaries"],
            "note": "Pilot phrase-level listening does not clear a flagged sample or verify exact word boundaries.",
        },
        "checks": ["original spreadsheet and manifest match by row", "all 100 original and staged video hashes match",
                   "all reported source and NPZ hashes match", "original words and character spans match NPZ",
                   "modality masks, dimensions, weights, and valid-frame counts reconcile",
                   "audio/vision word vectors reconstruct from recorded source weights"],
        "limitations": ["Automatic CTC timestamps are not manually verified word boundaries.",
                        "65 flagged samples await targeted human review.",
                        "OpenFace tracks one face; speaker identity is not verified.",
                        "A zero vector with a false mask is a missing-data placeholder, not a neutral measurement."],
    }

    def tsv_text(records: list[dict], fields: list[str]) -> str:
        from io import StringIO
        stream = StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=fields, dialect="excel-tab", lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
        return stream.getvalue()

    appendix = [
        "# 问题一：100 条样本结果附表",
        "",
        "本表展示自动提取的三模态特征与词级时序对齐结果。复核标记仅用于筛查，不代表人工确认。",
        "复核栏空白表示无自动复核标记，**不代表已经人工核实**。P1/P2 表示待复核优先级。",
        "",
        "| 样本 | 原视频 / 片段 | 原始词数 | 有时间词数 | 文本有效词 | 声音有效词 | 画面有效词 | 三模态均有效词 | 复核 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        appendix.append("| {sample} | {video_id} / {clip_id} | {source_words} | {timed_words} | "
                        "{text_valid_words} | {audio_valid_words} | {vision_valid_words} | "
                        "{all_modalities_valid_words} | {review_priority} |".format(**row))
    appendix.extend(["", "数据来源：同目录样本汇总 TSV 和审计 JSON。", ""])
    dictionary = [
        "# Question One: Table Schema",
        "",
        "The TSV files are UTF-8, tab-delimited, one row per sample. Counts refer to original transcript words.",
        "A false modality mask is a missing observation; the corresponding zero vector is not an observed neutral signal.",
        "P1 means transcript discrepancy, no valid visual frame, or excluded speaker annotation. Other flagged samples are P2.",
        "P1/P2 are a derived work order, not model confidence or measured error rates. All 65 queued rows remain pending human review.",
        "Nine untimed source tokens are retained: six standalone punctuation tokens and three excluded speaker-name tokens in sample_0006. The latter drive its P1 review.",
        "", "## Sample Table", "", "| Field | Definition |", "|---|---|",
    ]
    dictionary.extend(f"| `{field}` | {SAMPLE_DESCRIPTIONS[field]} |" for field in SAMPLE_FIELDS)
    dictionary.extend(["", "## Review Queue", "", "Queue rows are a subset of the sample table. Additional fields:", "",
                       "| Field | Definition |", "|---|---|",
                       "| `review_reason` | Semicolon-separated automatic/source flags driving triage. |",
                       "| `suggested_action` | Suggested listening/video check; no correction has been applied. |",
                       "", "Original source, staged media and NPZ hashes are independently checked during generation.", ""])
    files = {
        f"{STEM}_sample-summary_v1.0.tsv": tsv_text(rows, SAMPLE_FIELDS),
        f"{STEM}_review-queue_v1.0.tsv": tsv_text(queue, REVIEW_FIELDS),
        f"{STEM}_sample-appendix_v1.0.md": "\n".join(appendix),
        f"{STEM}_table-schema_v1.0.md": "\n".join(dictionary),
        f"{STEM}_audit_v1.0.json": json.dumps(overview, ensure_ascii=False, indent=2) + "\n",
    }
    return files, overview


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify exports without writing files")
    args = parser.parse_args()
    files, overview = generate()
    if not args.check:
        RESULT.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        path = RESULT / name
        if args.check:
            require(path.is_file(), f"Missing export: {path}")
            require(path.read_text(encoding="utf-8") == content, f"Existing export differs: {path}")
        elif not path.exists() or path.read_text(encoding="utf-8") != content:
            temporary = path.with_suffix(path.suffix + ".partial")
            temporary.write_text(content, encoding="utf-8", newline="")
            temporary.replace(path)
    print(json.dumps(overview["totals"], ensure_ascii=False, indent=2))
    print("Exports verified" if args.check else f"Exports written: {RESULT}")


if __name__ == "__main__":
    main()
