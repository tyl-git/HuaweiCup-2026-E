"""Evaluate frozen aligned temporal models on official test and attachment 3.

The evaluation set and checkpoint list are fixed before inference. Specialist
samples have no labels, so only predictions and observation audits are saved.
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
TRAINED = Q2 / "q2-temporal-v1.0"
OUT = Q2 / "q2-temporal-final-v1.0"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
VARIANTS = ("temporal", "no_temporal", "no_gap_signal")
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CLASSES = ("Negative", "Neutral", "Positive")


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save_csv(path, rows, columns):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    temporal = load("2026-09-24_train-q2-temporal_v1.0.py")
    loader = load("2026-09-24_q2_data_v1.0.py")
    cache = load("2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = load("2026-09-24_q2-normalization_v1.0.py")
    robust = load("2026-09-24_train-q2-robust_v1.0.py")
    metrics = load("2026-09-24_q2-evaluation_v1.0.py")
    official = loader.load_official()
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[3], 3)
    if len(official["test"]) != 727 or len(special) != 30:
        raise RuntimeError("Official test or attachment-3 count changed")
    model_info = cache.model_fingerprint(cache.MODEL)
    meta = {split: cache.verify_cache(official[split], model_info)
            for split in ("train", "valid", "test")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    source = scaler.metadata["source_metadata"]
    if (source["source_sha256"] != dict(official["test"].source_sha256)
            or source["text_feature_sha256"] != meta["train"]["feature_sha256"]):
        raise RuntimeError("Normalizer source differs from official train")

    jobs = []
    for variant in VARIANTS:
        provenance = read_json(TRAINED / ("provenance.json" if variant == "temporal"
                                          else f"provenance_{variant}.json"))
        if (provenance["selection_split"] != "valid"
                or provenance["normalization_sha256"] != sha(STATS)
                or provenance["aligned_source_sha256"] != dict(official["test"].source_sha256)
                or provenance["bert_cache_sha256"] != {split: meta[split]["feature_sha256"]
                                                       for split in ("train", "valid")}):
            raise RuntimeError(f"{variant}: train/valid provenance differs")
        for seed in SEEDS:
            folder = TRAINED / variant / f"seed_{seed}"
            recorded = read_json(folder / "metrics.json")
            checkpoint = folder / "best.pt"
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (sha(checkpoint) != recorded["checkpoint_sha256"]
                    or saved["model"] != variant or saved["seed"] != seed
                    or saved["epoch"] != recorded["best_epoch"]
                    or saved["normalization_sha256"] != sha(STATS)
                    or json.loads(json.dumps(saved["config"])) != recorded["config"]):
                raise RuntimeError(f"{variant}/{seed}: checkpoint metadata differs")
            config = temporal.TemporalConfig(**saved["config"])
            model = temporal.build_model(saved["train_priors"], config, torch,
                                         temporal=variant != "no_temporal",
                                         gap_signal=variant != "no_gap_signal")
            model.load_state_dict(saved["model_state"], strict=True)
            jobs.append((variant, seed, config, model, sha(checkpoint)))
    print(f"Preflight OK: official test=727, attachment 3=30, frozen checkpoints={len(jobs)}")
    if args.check:
        print("No BERT inference, task prediction, metrics or writing")
        return

    partial = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or partial.exists():
        raise FileExistsError("Final output exists; refusing to overwrite")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import BertModel

    encoder = BertModel.from_pretrained(str(cache.MODEL), local_files_only=True,
                                        attn_implementation="eager").to(args.device).float().eval()
    encoder.requires_grad_(False)
    special_text = np.empty((len(special), 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(special), 10):
            stop = min(start + 10, len(special))
            inputs = {key: torch.as_tensor(np.asarray(getattr(special, key)[start:stop]).copy(),
                                           dtype=torch.long, device=args.device)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            special_text[start:stop] = encoder(**inputs).last_hidden_state.cpu().numpy()
    if not np.isfinite(special_text).all():
        raise RuntimeError("Attachment-3 BERT features are nonfinite")
    del encoder

    test = official["test"]
    raw = {
        "test": {"text": np.load(cache.OUT / "test.npy", mmap_mode="r", allow_pickle=False),
                 "audio": test.audio, "vision": test.vision},
        "attachment3": {"text": special_text, "audio": special.audio, "vision": special.vision},
    }
    arrays, masks, tensors = {}, {}, {}
    for key, batch in (("test", test), ("attachment3", special)):
        arrays[key], masks[key] = robust.normalize_once(batch, raw[key], scaler)
        features = {m: torch.as_tensor(arrays[key][m].copy(), device=args.device)
                    for m in MODALITIES}
        content = torch.as_tensor(batch.content_mask.copy(), dtype=torch.bool,
                                  device=args.device)
        observed = {m: torch.as_tensor(masks[key][m].copy(), dtype=torch.bool,
                                       device=args.device) for m in MODALITIES}
        tensors[key] = (features, content, observed)

    partial.mkdir(parents=True)
    reports, specialist = [], []
    for variant, seed, config, model, checkpoint_hash in jobs:
        model = model.to(args.device).eval()
        tf, tc, tm = tensors["test"]
        test_tensors = (tf, tc, tm,
                        torch.as_tensor(test.classification_labels.copy(), dtype=torch.long,
                                        device=args.device),
                        torch.as_tensor(test.regression_labels.copy(), dtype=torch.float32,
                                        device=args.device))
        report, predictions = temporal.evaluate(model, test_tensors, config, torch,
                                                 metrics, predictions=True)
        output = partial / f"{variant}_seed_{seed}_test_predictions.npz"
        np.savez_compressed(output, ids=np.asarray(test.ids), **predictions)
        reports.append({"model": variant, "seed": seed, "checkpoint_sha256": checkpoint_hash,
                        "prediction_sha256": sha(output), "report": report})
        sf, sc, sm = tensors["attachment3"]
        with torch.inference_mode():
            logits, intensity = model(sf, sc, sm)
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            strength = intensity.cpu().numpy()
        for i, path in enumerate(special.source_paths):
            specialist.append({"file_id": path.stem, "model": variant, "seed": seed,
                               "polarity": CLASSES[int(probability[i].argmax())],
                               "intensity": float(strength[i]),
                               "negative_probability": float(probability[i, 0]),
                               "neutral_probability": float(probability[i, 1]),
                               "positive_probability": float(probability[i, 2])})
        print(f"{variant} seed={seed}: test and attachment 3 done", flush=True)

    columns = ("file_id", "model", "seed", "polarity", "intensity",
               "negative_probability", "neutral_probability", "positive_probability")
    save_csv(partial / "attachment3_all_models.csv", specialist, columns)
    chosen = [row for row in specialist if row["model"] == "temporal" and row["seed"] == 20260926]
    save_csv(partial / "attachment3_predictions.csv", chosen, columns)
    audit = []
    for i, path in enumerate(special.source_paths):
        row = {"file_id": path.stem, "content_positions": int(special.content_mask[i].sum())}
        for modality in MODALITIES:
            mask = masks["attachment3"][modality][i]
            row[f"{modality}_observed_positions"] = int(mask.sum())
            row[f"{modality}_zero_content_positions"] = int((special.content_mask[i] & ~mask).sum())
        audit.append(row)
    save_csv(partial / "attachment3_observation_audit.csv", audit, tuple(audit[0]))
    summary = {"schema": "q2_temporal_final_v1.0", "feature_version": "aligned_50",
               "selection_split": "valid", "official_test_n": len(test),
               "specialist_n": len(special), "specialist_labels_used": False,
               "specialist_primary": {"model": "temporal", "seed": 20260926,
                                      "rule": "lowest temporal validation selection loss"},
               "text_missing_scope": "post-BERT feature masking in train/valid; attachment 3 tokens have no internal attention gaps",
               "source_sha256": {"official": dict(test.source_sha256),
                                 "attachment3": dict(special.source_sha256)},
               "normalization_sha256": sha(STATS), "bert_model": model_info,
               "test_cache_sha256": meta["test"]["feature_sha256"],
               "results": reports,
               "output_sha256": {path.name: sha(path) for path in partial.glob("*.csv")}}
    (partial / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                       indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    partial.rename(OUT)
    print(f"Final Q2 output saved: {OUT}")


def load(filename):
    import importlib.util
    import sys

    path = SRC / filename
    name = "q2_final_" + path.stem.replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    main()
