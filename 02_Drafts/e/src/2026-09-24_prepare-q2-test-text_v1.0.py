"""Create the frozen BERT feature cache for the official test split only.

Train/valid caches are verified first.  This script never fits a model or
trains the task model; it only encodes test ``text_bert`` with the exact local
snapshot used for the committed train/valid cache.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SRC = Path(__file__).resolve().parent
OUT = ROOT / "03_Results/e/question-two/bert-cache-v1.0"
MODEL = Path(
    r"D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased"
    r"\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
)


def load_module(name: str, filename: str):
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


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def verify_test_cache(batch, model_info, cache_module):
    path = OUT / "test.npy"
    meta_path = OUT / "test.json"
    if not path.is_file() or not meta_path.is_file():
        raise FileNotFoundError("test.npy and test.json must both exist")
    expected = cache_module.input_contract(batch, model_info)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata["contract"] != expected:
        raise RuntimeError("test cache source/model/token contract changed")
    if sha(path) != metadata["feature_sha256"]:
        raise RuntimeError("test cache hash mismatch")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (len(batch), 50, 768) or values.dtype != np.float32:
        raise RuntimeError(f"invalid test feature cache shape/dtype: {values.shape}, {values.dtype}")
    if not np.isfinite(values).all():
        raise RuntimeError("test cache contains nonfinite values")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify train/valid/test caches without inference or writing")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    loader = load_module("q2_test_cache_loader", "2026-09-24_q2_data_v1.0.py")
    cache_module = load_module("q2_test_cache_base", "2026-09-24_prepare-q2-text_v1.0.py")
    data = loader.load_official()
    model_info = cache_module.model_fingerprint(MODEL)

    for split in ("train", "valid"):
        cache_module.verify_cache(data[split], model_info)
        print(f"{split}: verified cached {len(data[split])} samples", flush=True)

    if args.check:
        verify_test_cache(data["test"], model_info, cache_module)
        print("test: verified cached 727 samples", flush=True)
        print("Train/valid/test BERT caches verified; no inference or writing", flush=True)
        return

    test_path = OUT / "test.npy"
    test_meta_path = OUT / "test.json"
    partial = OUT / "test.partial.npy"
    if test_path.exists() or test_meta_path.exists() or partial.exists():
        if test_path.is_file() and test_meta_path.is_file() and not partial.exists():
            verify_test_cache(data["test"], model_info, cache_module)
            print("test: existing cache verified; no writing", flush=True)
            return
        raise RuntimeError("Incomplete or conflicting test cache; inspect before replacing")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import transformers
    from transformers import BertModel

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    torch.manual_seed(20260924)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    encoder = BertModel.from_pretrained(
        str(MODEL), local_files_only=True, attn_implementation="eager"
    ).to(args.device).float().eval()
    encoder.requires_grad_(False)
    print(f"Frozen local BERT ready: {args.device}, float32, eager attention", flush=True)

    batch = data["test"]
    OUT.mkdir(parents=True, exist_ok=True)
    values = np.lib.format.open_memmap(
        partial, mode="w+", dtype=np.float32, shape=(len(batch), 50, 768)
    )
    max_error = 0.0
    total_error = 0.0
    compared = 0
    with torch.inference_mode():
        for start in range(0, len(batch), args.batch_size):
            end = min(start + args.batch_size, len(batch))
            inputs = {
                key: torch.tensor(getattr(batch, key)[start:end], dtype=torch.long, device=args.device)
                for key in ("input_ids", "attention_mask", "token_type_ids")
            }
            result = encoder(**inputs).last_hidden_state.cpu().numpy()
            if result.shape != (end - start, 50, 768) or not np.isfinite(result).all():
                raise RuntimeError(f"test: invalid BERT output at {start}")
            values[start:end] = result
            active = batch.content_mask[start:end]
            delta = np.abs(result[active].astype(np.float64) - batch.reference_text[start:end][active])
            if delta.size:
                max_error = max(max_error, float(delta.max()))
                total_error += float(delta.sum())
                compared += delta.size
            if start == 0 or end % (args.batch_size * 25) == 0 or end == len(batch):
                print(f"test: encoded {end}/{len(batch)}", flush=True)
    values.flush()
    del values

    metadata = {
        "contract": cache_module.input_contract(batch, model_info),
        "feature_sha256": sha(partial),
        "runtime": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": args.device,
            "batch_size": args.batch_size,
            "seed": 20260924,
            "attention_implementation": "eager",
            "allow_tf32": False,
        },
        "reference_comparison": {
            "positions": int(batch.content_mask.sum()),
            "max_absolute_error": max_error,
            "mean_absolute_error": total_error / compared,
        },
    }
    partial.replace(test_path)
    test_meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    verify_test_cache(batch, model_info, cache_module)
    print("Train/valid/test BERT caches verified; no task training or prediction", flush=True)


if __name__ == "__main__":
    main()
