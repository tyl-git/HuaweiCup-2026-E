"""Evaluate the frozen sqrt-balanced temporal model on official test and attachment 3."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results" / "e" / "question-two"
TRAINED = Q2 / "q2-temporal-balanced-sqrt-v1.0"
OUT = Q2 / "q2-temporal-balanced-sqrt-final-v1.0"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
PLAN = Q2 / "2026-09-25_q2-balanced-test-plan_v1.0.json"
SEEDS = (20260924, 20260925, 20260926)
MODALITIES = ("text", "audio", "vision")
CLASSES = ("Negative", "Neutral", "Positive")


def load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()

    import torch

    expected_plan = {
        "split": "official_test",
        "official_test_n": 727,
        "attachment3_n": 30,
        "variant": "temporal_balanced_sqrt",
        "weight_mode": "sqrt",
        "selection_split": "valid",
        "seeds": list(SEEDS),
        "test_tuning": False,
    }
    if read_json(PLAN) != expected_plan:
        raise RuntimeError("Balanced test plan differs from frozen protocol")

    temporal = load("balanced_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    loader = load("balanced_loader", "2026-09-24_q2_data_v1.0.py")
    cache = load("balanced_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = load("balanced_normalizer", "2026-09-24_q2-normalization_v1.0.py")
    robust = load("balanced_robust", "2026-09-24_train-q2-robust_v1.0.py")
    metrics = load("balanced_metrics", "2026-09-24_q2-evaluation_v1.0.py")

    official = loader.load_official()
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[3], 3)
    test = official["test"]
    if len(test) != 727 or len(special) != 30:
        raise RuntimeError("Official test or attachment-3 count changed")
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {split: cache.verify_cache(official[split], model_info)
                  for split in ("train", "valid", "test")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    source_meta = scaler.metadata["source_metadata"]
    if (source_meta["source_sha256"] != dict(test.source_sha256)
            or source_meta["text_feature_sha256"] != cache_meta["train"]["feature_sha256"]):
        raise RuntimeError("Normalizer source differs from verified official data")

    experiment = read_json(TRAINED / "balanced_experiment.json")
    if (experiment.get("weight_mode") != "sqrt"
            or experiment.get("weighted_scope") != "optimizer classification CE only"
            or experiment.get("validation_selection")
            != "ordinary validation cross_entropy + L1 on complete input"
            or experiment.get("test_evaluated") is not False):
        raise RuntimeError("Balanced experiment provenance is not frozen")
    provenance = read_json(TRAINED / "provenance.json")
    if (provenance["selection_split"] != "valid"
            or provenance["normalization_sha256"] != sha(STATS)
            or provenance["aligned_source_sha256"] != dict(test.source_sha256)
            or provenance["bert_cache_sha256"] != {
                split: cache_meta[split]["feature_sha256"] for split in ("train", "valid")
            }):
        raise RuntimeError("Balanced training provenance differs from official inputs")

    jobs = []
    for seed in SEEDS:
        folder = TRAINED / "temporal" / f"seed_{seed}"
        checkpoint = folder / "best.pt"
        recorded = read_json(folder / "metrics.json")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if (sha(checkpoint) != recorded["checkpoint_sha256"]
                or saved["model"] != "temporal" or saved["seed"] != seed
                or saved["epoch"] != recorded["best_epoch"]
                or json.loads(json.dumps(saved["config"])) != recorded["config"]
                or saved["normalization_sha256"] != sha(STATS)):
            raise RuntimeError(f"temporal-balanced-sqrt/{seed}: checkpoint metadata differs")
        config = temporal.TemporalConfig(**saved["config"])
        model = temporal.build_model(saved["train_priors"], config, torch)
        model.load_state_dict(saved["model_state"], strict=True)
        jobs.append((seed, config, model, sha(checkpoint)))

    print(f"Preflight OK: official test=727, attachment 3=30, frozen checkpoints={len(jobs)}")
    if args.check:
        print("No BERT inference, task prediction, metrics or writing")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if OUT.exists() or OUT.with_name(OUT.name + ".partial").exists():
        raise FileExistsError("Balanced final output exists; refusing to overwrite")

    import os
    os.environ.update({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"})
    from transformers import BertModel
    encoder = BertModel.from_pretrained(
        str(cache.MODEL), local_files_only=True, attn_implementation="eager"
    ).to(args.device).float().eval()
    encoder.requires_grad_(False)
    special_text = np.empty((len(special), 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(special), 10):
            stop = min(start + 10, len(special))
            inputs = {key: torch.as_tensor(
                np.asarray(getattr(special, key)[start:stop]).copy(),
                dtype=torch.long, device=args.device
            ) for key in ("input_ids", "attention_mask", "token_type_ids")}
            special_text[start:stop] = encoder(**inputs).last_hidden_state.cpu().numpy()
    if not np.isfinite(special_text).all():
        raise RuntimeError("Attachment-3 BERT features are nonfinite")
    del encoder

    raw = {
        "test": {"text": np.load(cache.OUT / "test.npy", mmap_mode="r", allow_pickle=False),
                 "audio": test.audio, "vision": test.vision},
        "attachment3": {"text": special_text, "audio": special.audio, "vision": special.vision},
    }
    tensors, masks = {}, {}
    for key, batch in (("test", test), ("attachment3", special)):
        arrays, masks[key] = robust.normalize_once(batch, raw[key], scaler)
        features = {m: torch.as_tensor(arrays[m].copy(), device=args.device)
                    for m in MODALITIES}
        content = torch.as_tensor(batch.content_mask.copy(), dtype=torch.bool,
                                  device=args.device)
        observed = {m: torch.as_tensor(masks[key][m].copy(), dtype=torch.bool,
                                       device=args.device) for m in MODALITIES}
        tensors[key] = (features, content, observed)

    partial = OUT.with_name(OUT.name + ".partial")
    partial.mkdir(parents=True)
    reports, specialist = [], []
    for seed, config, model, checkpoint_hash in jobs:
        model = model.to(args.device).eval()
        tf, tc, tm = tensors["test"]
        test_tensors = (tf, tc, tm,
                        torch.as_tensor(test.classification_labels.copy(), dtype=torch.long,
                                        device=args.device),
                        torch.as_tensor(test.regression_labels.copy(), dtype=torch.float32,
                                        device=args.device))
        report, predictions = temporal.evaluate(model, test_tensors, config, torch,
                                                 metrics, predictions=True)
        output = partial / f"temporal_balanced_sqrt_seed_{seed}_test_predictions.npz"
        np.savez_compressed(output, ids=np.asarray(test.ids), **predictions)
        reports.append({"model": "temporal_balanced_sqrt", "seed": seed,
                        "checkpoint_sha256": checkpoint_hash,
                        "prediction_sha256": sha(output), "report": report})
        sf, sc, sm = tensors["attachment3"]
        with torch.inference_mode():
            logits, intensity = model(sf, sc, sm)
            probability = torch.softmax(logits, dim=1).cpu().numpy()
            strength = intensity.cpu().numpy()
        for i, path in enumerate(special.source_paths):
            specialist.append({"file_id": path.stem, "model": "temporal_balanced_sqrt",
                               "seed": seed, "polarity": CLASSES[int(probability[i].argmax())],
                               "intensity": float(strength[i]),
                               "negative_probability": float(probability[i, 0]),
                               "neutral_probability": float(probability[i, 1]),
                               "positive_probability": float(probability[i, 2])})
        print(f"balanced_sqrt seed={seed}: test and attachment 3 done", flush=True)

    import csv
    columns = ("file_id", "model", "seed", "polarity", "intensity",
               "negative_probability", "neutral_probability", "positive_probability")
    for filename, rows in (("attachment3_all_models.csv", specialist),
                           ("attachment3_predictions.csv", [x for x in specialist
                            if x["seed"] == 20260926])):
        with (partial / filename).open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
    audit = []
    for i, path in enumerate(special.source_paths):
        row = {"file_id": path.stem, "content_positions": int(special.content_mask[i].sum())}
        for modality in MODALITIES:
            mask = masks["attachment3"][modality][i]
            row[f"{modality}_observed_positions"] = int(mask.sum())
            row[f"{modality}_zero_content_positions"] = int((special.content_mask[i] & ~mask).sum())
        audit.append(row)
    with (partial / "attachment3_observation_audit.csv").open("w", newline="",
                                                                  encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(audit[0]))
        writer.writeheader()
        writer.writerows(audit)

    summary = {"schema": "q2_temporal_balanced_sqrt_final_v1.0",
               "feature_version": "aligned_50", "selection_split": "valid",
               "official_test_n": len(test), "specialist_n": len(special),
               "specialist_labels_used": False, "variant": "temporal_balanced_sqrt",
               "weight_mode": "sqrt", "weight_source": "official train labels only",
               "test_tuning": False, "source_sha256": {"official": dict(test.source_sha256),
               "attachment3": dict(special.source_sha256)}, "normalization_sha256": sha(STATS),
               "bert_model": model_info, "test_cache_sha256": cache_meta["test"]["feature_sha256"],
               "results": reports, "output_sha256": {p.name: sha(p) for p in partial.iterdir()
               if p.suffix in (".csv", ".npz")}}
    (partial / "summary.json").write_text(json.dumps(summary, ensure_ascii=False,
                                                       indent=2, allow_nan=False) + "\n",
                                          encoding="utf-8")
    partial.rename(OUT)
    print(f"Balanced final Q2 output saved: {OUT}")


if __name__ == "__main__":
    main()
