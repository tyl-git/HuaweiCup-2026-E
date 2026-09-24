"""Evaluate the frozen Q2 models on the official test split."""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
ROOT = SRC.parents[2]
RESULTS = ROOT / "03_Results/e/question-two"
PLAN = RESULTS / "2026-09-24_q2-final-test-plan_v1.0.json"
STATS = RESULTS / "2026-09-24_q2-normalization_v1.0.npz"
OUT = RESULTS / "q2-final-test-v1.0"
MODELS = ("gated_clean", "fusion", "gated_aug")


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check", action="store_true")
    action.add_argument("--run", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()

    plan = json.loads(PLAN.read_text(encoding="utf-8-sig"))
    if (plan["split"] != "official_test" or plan["sample_count"] != 727
            or plan["primary_model"] != "gated_clean"
            or plan["comparison_models"] != ["fusion", "gated_aug"]
            or plan["seeds"] != [20260924, 20260925, 20260926]):
        raise RuntimeError("Frozen test plan differs from the agreed protocol")

    loader = load("final_loader", "2026-09-24_q2_data_v1.0.py")
    cache = load("final_cache", "2026-09-24_prepare-q2-text_v1.0.py")
    norm = load("final_norm", "2026-09-24_q2-normalization_v1.0.py")
    baseline = load("final_baseline", "2026-09-24_train-q2-baselines_v1.0.py")
    robust = load("final_robust", "2026-09-24_train-q2-robust_v1.0.py")

    data = loader.load_official()
    test = data["test"]
    if len(test) != 727:
        raise RuntimeError("Unexpected official test size")
    model_info = cache.model_fingerprint(cache.MODEL)
    cache_meta = {split: cache.verify_cache(data[split], model_info)
                  for split in ("train", "valid", "test")}
    standardizer = norm.MultimodalStandardizer.load(STATS)
    source_hashes = dict(test.source_sha256)
    source_meta = standardizer.metadata["source_metadata"]
    if (source_meta["source_sha256"] != source_hashes
            or source_meta["text_feature_sha256"] != cache_meta["train"]["feature_sha256"]):
        raise RuntimeError("Normalizer source differs from the verified data")

    provenances = {}
    for kind, folder in (("baseline", "q2-baselines-v1.0"),
                         ("robust", "q2-robust-v1.0")):
        provenance = json.loads((RESULTS / folder / "provenance.json").read_text(encoding="utf-8"))
        expected_source = next(iter(source_hashes.values())) if kind == "baseline" else source_hashes
        if (provenance["official_source_sha256"] != expected_source
                or provenance["normalization_sha256"] != sha(STATS)
                or provenance["selection_split"] != "valid"):
            raise RuntimeError(f"{kind}: training provenance mismatch")
        for split in ("train", "valid"):
            if provenance["bert_cache_sha256"][split] != cache_meta[split]["feature_sha256"]:
                raise RuntimeError(f"{kind}: {split} cache mismatch")
        for filename, digest in provenance["dependency_sha256"].items():
            if sha(SRC / filename) != digest:
                raise RuntimeError(f"{kind}: training dependency changed: {filename}")
        provenances[kind] = provenance

    jobs = []
    for name in MODELS:
        kind = "baseline" if name == "fusion" else "robust"
        folder = RESULTS / ("q2-baselines-v1.0" if kind == "baseline" else "q2-robust-v1.0")
        for seed in plan["seeds"]:
            run = folder / name / f"seed_{seed}"
            checkpoint = run / "best.pt"
            recorded = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
            if sha(checkpoint) != recorded["checkpoint_sha256"]:
                raise RuntimeError(f"{name}/{seed}: checkpoint hash mismatch")
            saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if (saved["model"] != name or saved["seed"] != seed
                    or saved["epoch"] != recorded["best_epoch"]
                    or json.loads(json.dumps(saved["config"])) != recorded["config"]
                    or saved["train_priors"] != recorded["train_priors"]
                    or saved["normalization_sha256"] != sha(STATS)):
                raise RuntimeError(f"{name}/{seed}: checkpoint metadata mismatch")
            if kind == "baseline":
                config = baseline.TrainingConfig(**saved["config"])
                model = baseline.build_model(baseline.MODES[name], saved["train_priors"], config, torch)
            else:
                if saved["augmentation_enabled"] != (name == "gated_aug"):
                    raise RuntimeError(f"{name}/{seed}: augmentation flag mismatch")
                config = robust.RobustConfig(**saved["config"])
                model = robust.build_model(saved["train_priors"], config, torch)
            model.load_state_dict(saved["model_state"], strict=True)
            jobs.append((name, seed, config, model, recorded["checkpoint_sha256"]))

    print(f"Preflight OK: {len(test)} test samples, {len(jobs)} frozen checkpoints")
    if args.check:
        print("No test prediction, metric calculation, or writing")
        return
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    temporary = OUT.with_name(OUT.name + ".partial")
    if OUT.exists() or temporary.exists():
        raise FileExistsError("Final test output already exists; refusing to overwrite")

    text = np.load(cache.OUT / "test.npy", mmap_mode="r", allow_pickle=False)
    pooled, available = baseline.pool_normalized(
        test, {"text": text, "audio": test.audio, "vision": test.vision}, standardizer)
    temporary.mkdir(parents=True)
    rows = []
    for name, seed, config, model, checkpoint_hash in jobs:
        if name == "fusion":
            tensors = baseline.tensors_for(test, pooled, available, baseline.MODES[name], torch, args.device)
            report, predictions = baseline.evaluate(
                model.to(args.device), tensors, config, torch,
                load("final_metrics", "2026-09-24_q2-evaluation_v1.0.py"),
                return_predictions=True)
        else:
            tensors = robust.tensors_for(test, pooled, available, torch, args.device)
            report, predictions = robust.evaluate(
                model.to(args.device), tensors, config, torch,
                load("final_metrics", "2026-09-24_q2-evaluation_v1.0.py"),
                return_predictions=True)
        output = temporary / f"{name}_seed_{seed}_predictions.npz"
        np.savez_compressed(output, ids=np.asarray(test.ids),
                            available=np.column_stack([available[m] for m in robust.MODALITIES]),
                            **predictions)
        rows.append({"model": name, "seed": seed, "checkpoint_sha256": checkpoint_hash,
                     "predictions_sha256": sha(output), "report": report})
        print(f"{name} seed={seed}: test predictions saved", flush=True)

    summary = {"split": "official_test", "n": len(test), "plan_sha256": sha(PLAN),
               "test_cache_sha256": cache_meta["test"]["feature_sha256"],
               "normalization_sha256": sha(STATS), "results": rows}
    (temporary / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8")
    temporary.rename(OUT)
    print(f"Final test evaluation saved: {OUT}")


if __name__ == "__main__":
    main()