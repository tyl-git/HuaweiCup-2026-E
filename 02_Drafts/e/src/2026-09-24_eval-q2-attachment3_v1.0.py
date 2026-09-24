"""Run the frozen aligned temporal model on all unlabeled attachment-3 files."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
Q2 = ROOT / "03_Results/e/question-two"
OUT = Q2 / "q2-attachment3-v1.0"
MODEL_ROOT = Q2 / "q2-temporal-v1.0"
CHECKPOINT = MODEL_ROOT / "temporal/seed_20260926/best.pt"
STATS = Q2 / "2026-09-24_q2-normalization_v1.0.npz"
MODALITIES = ("text", "audio", "vision")
CLASS_NAMES = ("Negative", "Neutral", "Positive")


def local(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def csv_write(path, rows, fields):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    import torch

    loader = local("q2_a3_loader", "2026-09-24_q2_data_v1.0.py")
    cache = local("q2_a3_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    normalizer = local("q2_a3_normalizer", "2026-09-24_q2-normalization_v1.0.py")
    robust = local("q2_a3_robust", "2026-09-24_train-q2-robust_v1.0.py")
    temporal = local("q2_a3_temporal", "2026-09-24_train-q2-temporal_v1.0.py")
    masks = local("q2_a3_masks", "2026-09-24_q2-mask-semantics_v1.0.py")

    official = loader.load_official()
    special = loader.load_special_directory(loader.DEFAULT_SPECIAL[3], 3)
    if len(special) != 30 or special.classification_labels is not None:
        raise RuntimeError("Attachment 3 count or label contract differs")
    model_info = cache.model_fingerprint(cache.MODEL)
    verified = {split: cache.verify_cache(official[split], model_info)
                for split in ("train", "valid")}
    scaler = normalizer.MultimodalStandardizer.load(STATS)
    if (scaler.metadata["source_metadata"]["source_sha256"]
            != dict(official["train"].source_sha256)
            or scaler.metadata["source_metadata"]["text_feature_sha256"]
            != verified["train"]["feature_sha256"]):
        raise RuntimeError("Normalizer source differs from verified training data")
    provenance = read_json(MODEL_ROOT / "provenance.json")
    if (provenance["selection_split"] != "valid"
            or provenance["normalization_sha256"] != sha(STATS)
            or provenance["bert_cache_sha256"]
            != {split: verified[split]["feature_sha256"] for split in verified}):
        raise RuntimeError("Training provenance differs")
    record = read_json(CHECKPOINT.parent / "metrics.json")
    saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    if (sha(CHECKPOINT) != record["checkpoint_sha256"]
            or saved["model"] != "temporal" or saved["seed"] != 20260926
            or saved["epoch"] != record["best_epoch"]
            or saved["normalization_sha256"] != sha(STATS)
            or json.loads(json.dumps(saved["config"])) != record["config"]):
        raise RuntimeError("Frozen checkpoint metadata differs")
    config = temporal.TemporalConfig(**saved["config"])
    model = temporal.build_model(saved["train_priors"], config, torch)
    model.load_state_dict(saved["model_state"], strict=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    print("Preflight OK: 30 aligned unlabeled files; frozen temporal seed=20260926 epoch=3")
    if args.check:
        print("No BERT encoding, task inference, or writing")
        return
    temporary = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or temporary.exists():
        raise FileExistsError("Output already exists; refusing to overwrite")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from transformers import BertModel

    encoder = BertModel.from_pretrained(str(cache.MODEL), local_files_only=True,
                                        attn_implementation="eager").to(args.device).float().eval()
    encoder.requires_grad_(False)
    text = np.empty((30, 50, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, 30, 10):
            stop = min(30, start + 10)
            inputs = {key: torch.as_tensor(np.asarray(getattr(special, key)[start:stop]).copy(),
                                           dtype=torch.long, device=args.device)
                      for key in ("input_ids", "attention_mask", "token_type_ids")}
            text[start:stop] = encoder(**inputs).last_hidden_state.cpu().numpy()
    del encoder
    arrays, observed = robust.normalize_once(
        special, {"text": text, "audio": special.audio, "vision": special.vision}, scaler)
    state = masks.build_mask_state(special)
    if any(not np.array_equal(observed[m], state.effective_observed[m]) for m in MODALITIES):
        raise RuntimeError("Normalization changed observation masks")
    model = model.to(args.device).eval()
    feature_tensors = {m: torch.as_tensor(arrays[m], dtype=torch.float32, device=args.device)
                       for m in MODALITIES}
    content = torch.as_tensor(special.content_mask.copy(), dtype=torch.bool, device=args.device)
    effective = {m: torch.as_tensor(observed[m].copy(), dtype=torch.bool, device=args.device)
                 for m in MODALITIES}
    with torch.inference_mode():
        logits, intensity = model(feature_tensors, content, effective)
        probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
        intensity = intensity.cpu().numpy()
    if not (np.isfinite(probabilities).all() and np.isfinite(intensity).all()):
        raise RuntimeError("Nonfinite predictions")
    rows, audit = [], []
    for i, source in enumerate(special.source_paths):
        cls = int(probabilities[i].argmax())
        rows.append({"file_id": source.stem, "polarity": CLASS_NAMES[cls],
                     "intensity": float(intensity[i])})
        audit.append({"file_id": source.stem, "sample_id": special.ids[i],
                      "source_sha256": special.source_sha256[str(source)],
                      "content_positions": int(state.content[i].sum()),
                      "source_zero_runs_json": json.dumps({m: masks.runs(state.source_zero[m][i])
                                                           for m in MODALITIES}),
                      "observed_positions_json": json.dumps({m: int(observed[m][i].sum())
                                                               for m in MODALITIES}),
                      "class_probabilities_json": json.dumps(probabilities[i].tolist())})
    temporary.mkdir(parents=True)
    csv_write(temporary / "attachment3_predictions.csv", rows,
              ("file_id", "polarity", "intensity"))
    csv_write(temporary / "attachment3_mask_audit.csv", audit,
              ("file_id", "sample_id", "source_sha256", "content_positions",
               "source_zero_runs_json", "observed_positions_json", "class_probabilities_json"))
    summary = {"schema": "q2_attachment3_v1.0", "aligned_only": True,
               "unlabeled": True, "n": 30, "selected_on": "valid_complete_selection_loss",
               "checkpoint_sha256": sha(CHECKPOINT), "normalization_sha256": sha(STATS),
               "bert_model": model_info, "source_sha256": dict(special.source_sha256),
               "mask_rule": "content excludes CLS/SEP; within-content all-zero source rows are unavailable; padding is separate",
               "text_missing_limit": "Attachment 3 has no explicit internal text-token gaps",
               "output_sha256": {path.name: sha(path) for path in temporary.glob("*.csv")}}
    (temporary / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.rename(OUT)
    print(f"Attachment 3 predictions saved: {OUT}")


if __name__ == "__main__":
    main()
