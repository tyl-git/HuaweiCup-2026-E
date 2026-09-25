"""Run an isolated class-balanced temporal Q2 experiment.

This wrapper reuses the frozen-data, mask, model, augmentation, checkpoint,
and audit protocol from ``2026-09-24_train-q2-temporal_v1.0.py``.  During
optimization only, the classification cross-entropy receives a class weight
computed from official train labels.  Validation remains ordinary
cross-entropy plus L1, so checkpoint selection is comparable with the existing
temporal result and no test data is used.

The wrapper never writes to q2-temporal-v1.0.  Each weight mode has its own
output directory: q2-temporal-balanced-sqrt-v1.0 or
q2-temporal-balanced-inverse-v1.0.
"""

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
TEMPORAL_FILE = SRC / "2026-09-24_train-q2-temporal_v1.0.py"
DATA_FILE = SRC / "2026-09-24_q2_data_v1.0.py"
WEIGHT_MODES = ("sqrt", "inverse")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module: {path}")
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def class_weights(labels: np.ndarray, mode: str) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0 or np.any((labels < 0) | (labels > 2)):
        raise ValueError("Expected non-empty labels in {0,1,2}")
    counts = np.bincount(labels, minlength=3).astype(np.float64)
    if np.any(counts <= 0):
        raise ValueError(f"Every class must occur in train labels: {counts.tolist()}")
    probabilities = counts / counts.sum()
    raw = 1.0 / (np.sqrt(probabilities) if mode == "sqrt" else probabilities)
    weights = raw / raw.mean()
    return counts, weights.astype(np.float32)


def run(args: argparse.Namespace) -> None:
    import torch

    temporal = load_module("q2_temporal_balanced_impl", TEMPORAL_FILE)
    data = load_module("q2_temporal_balanced_data", DATA_FILE)
    batches = data.load_official()
    train_labels = np.asarray(batches["train"].classification_labels)
    counts, weights = class_weights(train_labels, args.weight_mode)
    output = (ROOT / "03_Results" / "e" / "question-two" /
              f"q2-temporal-balanced-{args.weight_mode}-v1.0")
    temporal.OUT = output

    original_ce = torch.nn.functional.cross_entropy

    def training_cross_entropy(input, target, *ce_args, **ce_kwargs):
        # Original evaluate() runs under inference_mode, while optimizer steps
        # run with gradients enabled.  This keeps validation selection unweighted.
        if torch.is_grad_enabled() and "weight" not in ce_kwargs:
            ce_kwargs["weight"] = torch.as_tensor(
                weights, dtype=input.dtype, device=input.device)
        return original_ce(input, target, *ce_args, **ce_kwargs)

    torch.nn.functional.cross_entropy = training_cross_entropy
    try:
        metadata = {
            "schema": "q2_temporal_balanced_v1.0",
            "weight_mode": args.weight_mode,
            "weight_formula": "(1/sqrt(pi_c))/mean(1/sqrt(pi))" if args.weight_mode == "sqrt"
            else "(1/pi_c)/mean(1/pi)",
            "train_class_counts": counts.astype(int).tolist(),
            "train_class_probabilities": (counts / counts.sum()).tolist(),
            "classification_weights": weights.tolist(),
            "weighted_scope": "optimizer classification CE only",
            "validation_selection": "ordinary validation cross_entropy + L1 on complete input",
            "test_evaluated": False,
            "specialists_evaluated": False,
            "source_script": str(TEMPORAL_FILE),
            "source_script_sha256": sha256(TEMPORAL_FILE),
        }
        if args.action == "preflight":
            # The original preflight validates loaders, masks, forward shapes,
            # and checkpoint contracts without optimizer steps or output writes.
            sys.argv = [str(TEMPORAL_FILE), "--preflight", "--device", args.device,
                        "--model", "temporal"]
            temporal.main()
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return
        if args.action == "self-test":
            temporal.self_test()
            expected_counts = np.asarray([967, 758, 1670])
            if not np.array_equal(counts, expected_counts):
                raise AssertionError(f"Unexpected train counts: {counts.tolist()}")
            if not np.isfinite(weights).all() or abs(float(weights.mean()) - 1.0) > 1e-6:
                raise AssertionError("Invalid class weights")
            print("Balanced self-test OK")
            print(json.dumps(metadata, ensure_ascii=False, indent=2))
            return

        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Refusing to overwrite prior balanced run: {output}")
        output.mkdir(parents=True, exist_ok=True)
        (output / "balanced_experiment.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        sys.argv = [str(TEMPORAL_FILE), "--run", "--device", args.device,
                    "--model", "temporal"]
        temporal.main()
        print(f"Balanced temporal train/valid complete: {output}")
    finally:
        torch.nn.functional.cross_entropy = original_ce


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--self-test", dest="action", action="store_const", const="self-test")
    actions.add_argument("--preflight", dest="action", action="store_const", const="preflight")
    actions.add_argument("--run", dest="action", action="store_const", const="run")
    parser.add_argument("--weight-mode", choices=WEIGHT_MODES, default="sqrt")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
