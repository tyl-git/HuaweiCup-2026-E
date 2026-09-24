"""Explain one frozen, validation-selected aligned temporal checkpoint.

No parameter fitting or checkpoint selection occurs here. Attention/gate scores
are candidates; actual influence is measured by masking observed inputs and
recomputing the same frozen predictor. Aligned token positions are not seconds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
OUT = ROOT / "03_Results" / "e" / "question-three" / "q3-explanations-v1.0"
MODEL_ROOT = Q2 / "q2-temporal-v1.0"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")
VALID_FIDELITY_N = 100


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                     allow_nan=False) + "\n", encoding="utf-8")


def save_csv(path, rows, fields):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def prepare_checkpoint(path, temporal, torch):
    path = Path(path).resolve()
    if path.name != "best.pt" or path.parent.parent.parent != MODEL_ROOT.resolve():
        raise ValueError("Checkpoint must be an existing q2-temporal-v1.0 variant/seed/best.pt")
    variant, seed = path.parent.parent.name, int(path.parent.name.removeprefix("seed_"))
    if variant not in ("temporal", "no_temporal", "no_gap_signal"):
        raise ValueError("Unexpected temporal variant")
    recorded = read_json(path.parent / "metrics.json")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if (sha(path) != recorded["checkpoint_sha256"] or saved["model"] != variant
            or saved["seed"] != seed or saved["epoch"] != recorded["best_epoch"]
            or saved["normalization_sha256"] != sha(STATS)
            or json.loads(json.dumps(saved["config"])) != recorded["config"]):
        raise RuntimeError("Checkpoint metadata, metrics or normalization fingerprint differs")
    config = temporal.TemporalConfig(**saved["config"])
    model = temporal.build_model(saved["train_priors"], config, torch,
                                 temporal=variant != "no_temporal",
                                 gap_signal=variant != "no_gap_signal")
    model.load_state_dict(saved["model_state"], strict=True)
    model.eval()
    return model, config, recorded, {"variant": variant, "seed": seed,
                                     "checkpoint": str(path), "checkpoint_sha256": sha(path),
                                     "best_epoch": saved["epoch"]}


def tensor_inputs(features, content, observed, torch, device):
    return ({m: torch.as_tensor(np.asarray(features[m]).copy(), device=device,
                                dtype=torch.float32) for m in MODALITIES},
            torch.as_tensor(np.asarray(content).copy(), device=device, dtype=torch.bool),
            {m: torch.as_tensor(np.asarray(observed[m]).copy(), device=device,
                                dtype=torch.bool) for m in MODALITIES})


def predict(model, features, content, observed, torch):
    with torch.inference_mode():
        logits, intensity, evidence = model(features, content, observed,
                                            return_evidence=True)
        probabilities = torch.softmax(logits, dim=-1)
    return probabilities.cpu().numpy(), intensity.cpu().numpy(), {
        key: value.cpu().numpy() for key, value in evidence.items()}


def occlude(model, features, content, observed, modality, positions, target,
            baseline_probability, baseline_intensity, torch):
    changed = {m: mask.clone() for m, mask in observed.items()}
    changed[modality][:, positions] = False
    probability, intensity, _ = predict(model, features, content, changed, torch)
    return (float(baseline_probability - probability[0, target]),
            float(baseline_intensity - intensity[0]))


def tokenize_with_offsets(batch, tokenizer):
    result = []
    for i, raw_text in enumerate(batch.raw_text):
        encoded = tokenizer(raw_text, padding="max_length", truncation=True,
                            max_length=50, return_offsets_mapping=True)
        for name in ("input_ids", "attention_mask", "token_type_ids"):
            if not np.array_equal(encoded[name], getattr(batch, name)[i]):
                raise RuntimeError(f"{batch.ids[i]}: official tokens differ from local tokenizer")
        words = {}
        for position, word_id in enumerate(encoded.word_ids()):
            if word_id is None or not batch.content_mask[i, position]:
                continue
            start, end = encoded["offset_mapping"][position]
            if word_id not in words:
                words[word_id] = {"positions": [], "start": start, "end": end}
            words[word_id]["positions"].append(position)
            words[word_id]["start"] = min(words[word_id]["start"], start)
            words[word_id]["end"] = max(words[word_id]["end"], end)
        position_to_word = {}
        for word_id, word in words.items():
            word["text"] = raw_text[word["start"]:word["end"]]
            for position in word["positions"]:
                position_to_word[position] = word_id
        result.append((words, position_to_word))
    return result


def candidate_positions(evidence, observed, modality, limit=1):
    index = MODALITIES.index(modality)
    score = evidence["modality_gate"][0, :, index] * evidence["time_weight"][0]
    indices = np.flatnonzero(observed[modality][0].cpu().numpy())
    ordered = sorted(indices, key=lambda pos: (-float(score[pos]), int(pos)))
    return ordered[:limit], score


def explain_one(model, features, content, observed, torch, word_map=None):
    probability, intensity, evidence = predict(model, features, content, observed, torch)
    cls = int(probability[0].argmax())
    base_p, base_intensity = float(probability[0, cls]), float(intensity[0])
    effects, position_rows = {}, []
    for modality in MODALITIES:
        valid_positions = np.flatnonzero(observed[modality][0].cpu().numpy())
        total_effect = occlude(model, features, content, observed, modality,
                               valid_positions, cls, base_p, base_intensity, torch) if len(valid_positions) else (0.0, 0.0)
        effects[modality] = {"class_probability_delta": total_effect[0],
                             "intensity_delta": total_effect[1],
                             "observed_positions": int(len(valid_positions))}
        candidate, candidate_score = candidate_positions(evidence, observed, modality)
        candidate_set = set(candidate)
        for pos in valid_positions:
            delta_p, delta_y = occlude(model, features, content, observed, modality,
                                       [int(pos)], cls, base_p, base_intensity, torch)
            word = None
            if word_map is not None:
                words, pos_to_word = word_map
                word = words.get(pos_to_word.get(int(pos)))
            position_rows.append({
                "modality": modality, "position_0based": int(pos),
                "candidate_gate_time": float(candidate_score[pos]),
                "candidate_top1": int(pos) in candidate_set,
                "class_probability_delta": delta_p, "intensity_delta": delta_y,
                "text_char_start": word["start"] if word else "",
                "text_char_end": word["end"] if word else "",
                "text_fragment": word["text"] if word else "",
            })
    total_abs = sum(abs(item["class_probability_delta"]) for item in effects.values())
    for modality in MODALITIES:
        effects[modality]["relative_influence"] = (abs(effects[modality]["class_probability_delta"])
                                                   / total_abs if total_abs else 0.0)
    primary = max(MODALITIES, key=lambda m: effects[m]["relative_influence"]) if total_abs else "none"
    position_rows.sort(key=lambda item: (-abs(item["class_probability_delta"]),
                                         MODALITIES.index(item["modality"]), item["position_0based"]))
    return {"predicted_class_id": cls, "predicted_class": CLASS_NAMES[cls],
            "predicted_intensity": base_intensity, "predicted_class_probability": base_p,
            "class_probabilities": probability[0].tolist(),
            "primary_modality": primary, "modality_effects": effects,
            "top_evidence": position_rows[:5]}, position_rows


def valid_fidelity(model, features, content, observed, batch, torch):
    order = sorted(range(len(batch)), key=lambda i: hashlib.sha256(batch.ids[i].encode()).digest())
    selected = order[:VALID_FIDELITY_N]
    top_effects, random_effects = [], []
    for row in selected:
        one_features = {m: features[m][row:row + 1] for m in MODALITIES}
        one_content = content[row:row + 1]
        one_observed = {m: observed[m][row:row + 1] for m in MODALITIES}
        p, y, evidence = predict(model, one_features, one_content, one_observed, torch)
        target = int(p[0].argmax())
        candidate, _ = candidate_positions(evidence, one_observed, "text")
        indices = np.flatnonzero(one_observed["text"][0].cpu().numpy())
        if not len(candidate) or not len(indices):
            continue
        rng = np.random.default_rng(int.from_bytes(hashlib.sha256(batch.ids[row].encode()).digest()[:8], "big"))
        random_pos = int(rng.choice(indices))
        for position, destination in ((candidate[0], top_effects), (random_pos, random_effects)):
            delta, _ = occlude(model, one_features, one_content, one_observed, "text",
                               [int(position)], target, float(p[0, target]), float(y[0]), torch)
            destination.append(delta)
    return {"sample_rule": "first_100_valid_ids_by_sha256", "sample_count": len(top_effects),
            "target": "predicted_class_probability", "comparison_modality": "text",
            "gate_time_top1_mean_drop": float(np.mean(top_effects)),
            "deterministic_random_mean_drop": float(np.mean(random_effects)),
            "gate_time_top1_above_random_fraction": float(np.mean(np.asarray(top_effects) > random_effects))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch
    import importlib.util
    import sys

    def local(name, filename):
        spec = importlib.util.spec_from_file_location(name, SRC / filename)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    temporal = local("q3_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    loader = local("q3_loader", "2026-09-24_q2_data_v1.0.py")
    cache = local("q3_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = local("q3_normalizer", "2026-09-24_q2-normalization_v1.0.py")
    robust = local("q3_robust", "2026-09-24_train-q2-robust_v1.0.py")
    metrics = local("q3_metrics", "2026-09-24_q2-evaluation_v1.0.py")
    model, config, recorded, selected = prepare_checkpoint(args.checkpoint, temporal, torch)
    official = loader.load_official()
    valid = official["valid"]
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[4], 4)
    if len(special) != 20 or len(valid) != 728:
        raise RuntimeError("Unexpected validation or attachment-4 sample count")
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {split: cache.verify_cache(official[split], model_info)
                  for split in ("train", "valid")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    source = scaler.metadata["source_metadata"]
    if (source["source_sha256"] != dict(valid.source_sha256)
            or source["text_feature_sha256"] != cache_meta["train"]["feature_sha256"]):
        raise RuntimeError("Normalizer and train source/cache differ")
    provenance_path = MODEL_ROOT / ("provenance.json" if selected["variant"] == "temporal"
                                     else f"provenance_{selected['variant']}.json")
    provenance = read_json(provenance_path)
    if (provenance["selection_split"] != "valid" or provenance["normalization_sha256"] != sha(STATS)
            or provenance["bert_cache_sha256"] != {split: cache_meta[split]["feature_sha256"]
                                                   for split in ("train", "valid")}):
        raise RuntimeError("Training provenance differs from verified train/valid inputs")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    print(f"Preflight OK: {selected['variant']} seed={selected['seed']} epoch={selected['best_epoch']}; "
          "728 valid, 20 attachment-4 aligned samples")
    if args.check:
        print("No BERT inference, task inference, or writing")
        return
    if OUT.exists() or OUT.with_name(OUT.name + ".partial").exists():
        raise FileExistsError("Q3 output already exists; refusing to overwrite")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import BertModel, BertTokenizerFast

    encoder = BertModel.from_pretrained(str(cache.MODEL), local_files_only=True,
                                        attn_implementation="eager").to(args.device).float().eval()
    encoder.requires_grad_(False)
    tokenizer = BertTokenizerFast.from_pretrained(str(cache.MODEL), local_files_only=True)
    word_maps = tokenize_with_offsets(special, tokenizer)
    special_text = np.empty((20, 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, 20, 10):
            stop = min(20, start + 10)
            inputs = {key: torch.as_tensor(np.asarray(getattr(special, key)[start:stop]).copy(),
                                           dtype=torch.long, device=args.device)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            special_text[start:stop] = encoder(**inputs).last_hidden_state.cpu().numpy()
    difference = np.abs(special_text[special.content_mask].astype(np.float64)
                        - special.reference_text[special.content_mask].astype(np.float64))
    if not np.isfinite(special_text).all() or float(difference.max()) > 0.01:
        raise RuntimeError("Specialist BERT encoding differs from supplied reference text")
    del encoder
    model = model.to(args.device).eval()
    valid_text = np.load(cache.OUT / "valid.npy", mmap_mode="r", allow_pickle=False)
    valid_arrays, valid_masks = robust.normalize_once(valid,
        {"text": valid_text, "audio": valid.audio, "vision": valid.vision}, scaler)
    special_arrays, special_masks = robust.normalize_once(special,
        {"text": special_text, "audio": special.audio, "vision": special.vision}, scaler)
    vf, vc, vm = tensor_inputs(valid_arrays, valid.content_mask, valid_masks, torch, args.device)
    sf, sc, sm = tensor_inputs(special_arrays, special.content_mask, special_masks, torch, args.device)
    valid_tensors = (vf, vc, vm,
                     torch.as_tensor(valid.classification_labels.copy(), dtype=torch.long, device=args.device),
                     torch.as_tensor(valid.regression_labels.copy(), dtype=torch.float32, device=args.device))
    valid_report = temporal.evaluate(model, valid_tensors, config, torch, metrics)
    if abs(valid_report["selection_loss"] - recorded["valid_complete"]["selection_loss"]) > 1e-5:
        raise RuntimeError("Frozen validation metric reproduction failed")
    fidelity = valid_fidelity(model, vf, vc, vm, valid, torch)
    prediction_rows, explanation_rows, local_rows = [], [], []
    video_dir = loader.DEFAULT_SPECIAL[4] / "videos"
    for i in range(len(special)):
        features = {m: sf[m][i:i + 1] for m in MODALITIES}
        masks = {m: sm[m][i:i + 1] for m in MODALITIES}
        result, positions = explain_one(model, features, sc[i:i + 1], masks,
                                        torch, word_maps[i])
        name = special.source_paths[i].stem
        video = video_dir / f"{name}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        prediction_rows.append({"file_id": name, "sample_id": special.ids[i],
                                "polarity": result["predicted_class"],
                                "intensity": result["predicted_intensity"]})
        explanation_rows.append({"file_id": name, "sample_id": special.ids[i],
                                 "polarity": result["predicted_class"],
                                 "intensity": result["predicted_intensity"],
                                 "predicted_class_probability": result["predicted_class_probability"],
                                 "primary_modality": result["primary_modality"],
                                 "modality_effects_json": json.dumps(result["modality_effects"], ensure_ascii=False),
                                 "top_evidence_json": json.dumps(result["top_evidence"], ensure_ascii=False),
                                 "raw_text": special.raw_text[i], "video_file": str(video),
                                 "time_mapping_status": "aligned_position_only_seconds_unverified"})
        for position in positions:
            local_rows.append({"file_id": name, "sample_id": special.ids[i], **position})
        print(f"Attachment 4: {i + 1}/20", flush=True)
    temporary = OUT.with_name(OUT.name + ".partial")
    temporary.mkdir(parents=True)
    save_csv(temporary / "attachment4_predictions.csv", prediction_rows,
             ("file_id", "sample_id", "polarity", "intensity"))
    save_csv(temporary / "attachment4_explanations.csv", explanation_rows,
             ("file_id", "sample_id", "polarity", "intensity", "predicted_class_probability",
              "primary_modality", "modality_effects_json", "top_evidence_json", "raw_text",
              "video_file", "time_mapping_status"))
    save_csv(temporary / "attachment4_position_effects.csv", local_rows,
             ("file_id", "sample_id", "modality", "position_0based", "candidate_gate_time",
              "candidate_top1", "class_probability_delta", "intensity_delta",
              "text_char_start", "text_char_end", "text_fragment"))
    summary = {"schema": "q3_explanations_v1.0", "checkpoint": selected,
               "aligned_only": True, "specialist_label_use": False,
               "normalization_sha256": sha(STATS), "bert_model": model_info,
               "attachment4_source_sha256": dict(special.source_sha256),
               "attachment4_bert_reference_max_absolute_error": float(difference.max()),
               "valid_report": valid_report, "valid_fidelity": fidelity,
               "explanation_rule": "absolute leave-one-modality-out class-probability change, normalized across modalities; signed deltas retained",
               "position_rule": "single observed-position occlusion after frozen BERT; gate times time attention is a candidate only",
               "time_mapping_status": "Token offsets and aligned indices are verified; exact seconds and video frames require separate audio/video alignment and review",
               "output_sha256": {path.name: sha(path) for path in temporary.glob("*.csv")}}
    save_json(temporary / "summary.json", summary)
    temporary.rename(OUT)
    print(f"Q3 explanation output saved: {OUT}")


if __name__ == "__main__":
    main()
