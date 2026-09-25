"""Prepare an input-level text-gap experiment for Q2.

This is an isolated supplementary experiment.  It never changes the official
BERT cache, aligned PKL files, normalization statistics, checkpoints, or test
outputs.  A contiguous text gap is inserted into ``input_ids`` before frozen
BERT encoding with ``[MASK]`` (103).  The resulting rows are then marked
unavailable by ``effective_mask`` before any pooling.  Only train/valid are
supported; test is deliberately rejected.
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
RESULTS = ROOT / "03_Results" / "e" / "question-two"
OUT = RESULTS / "q2-text-safe-v1.0"
TEXT_CACHE = RESULTS / "bert-cache-v1.0"
MODEL = Path(
    r"D:\06_Apps\huggingface-data\hub\models--google-bert--bert-base-uncased"
    r"\snapshots\86b5e0934494bd15c9632b12f734a8a67f723594"
)
SEQUENCE_LENGTH = 50
MASK_TOKEN_ID = 103
PLAN_SEED = 20260931
FRACTIONS = (0.2, 0.4)
POSITIONS = ("start", "middle", "end", "random")


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SRC / filename)
    if spec is None or spec.loader is None:
        raise ImportError(filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def array_sha(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def stable_start(sample_id: str, modality: str, choices: int) -> int:
    key = f"{PLAN_SEED}\0{sample_id}\0{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % choices


def make_removed(batch, fraction: float, position: str) -> np.ndarray:
    if fraction not in FRACTIONS or position not in POSITIONS:
        raise ValueError("unsupported text-gap condition")
    content = np.asarray(batch.content_mask, dtype=bool)
    removed = np.zeros_like(content, dtype=bool)
    for row, sample_id in enumerate(batch.ids):
        eligible = np.flatnonzero(content[row])
        if not len(eligible):
            continue
        target = min(len(eligible), max(1, int(np.ceil(fraction * len(eligible)))))
        choices = len(eligible) - target + 1
        if position == "start":
            offset = 0
        elif position == "middle":
            offset = (choices - 1) // 2
        elif position == "end":
            offset = choices - 1
        else:
            offset = stable_start(str(sample_id), "text", choices)
        begin = int(eligible[offset])
        end = int(eligible[offset + target - 1]) + 1
        removed[row, begin:end] = True
    return removed


def verify_inputs(batch, split: str) -> None:
    if batch.split != split or len(batch) < 1:
        raise RuntimeError(f"unexpected {split} batch")
    if batch.reference_text is None:
        raise RuntimeError(f"{split}: reference text is unavailable")
    if batch.input_ids.shape != (len(batch), SEQUENCE_LENGTH):
        raise RuntimeError(f"{split}: unexpected token shape")
    if not np.array_equal(batch.observation_masks["text"], batch.content_mask):
        raise RuntimeError(f"{split}: text observation is not the content mask")
    cache_meta = TEXT_CACHE / f"{split}.json"
    cache_values = TEXT_CACHE / f"{split}.npy"
    if not cache_meta.is_file() or not cache_values.is_file():
        raise FileNotFoundError(f"missing verified BERT cache for {split}")
    meta = json.loads(cache_meta.read_text(encoding="utf-8"))
    if meta.get("contract", {}).get("shape") != [len(batch), 50, 768]:
        raise RuntimeError(f"{split}: cache contract shape mismatch")
    if sha256(cache_values) != meta.get("feature_sha256"):
        raise RuntimeError(f"{split}: cache checksum mismatch")


def self_test(loader) -> None:
    batches = loader.load_official()
    train, valid = batches["train"], batches["valid"]
    for split, batch in (("train", train), ("valid", valid)):
        verify_inputs(batch, split)
        first = make_removed(batch, 0.2, "start")
        second = make_removed(batch, 0.2, "start")
        assert np.array_equal(first, second)
        assert not np.any(first & ~batch.content_mask)
        assert np.all(first.sum(axis=1) >= 1)
        for position in POSITIONS:
            mask = make_removed(batch, 0.4, position)
            assert not np.any(mask & ~batch.content_mask)
    assert np.all(train.input_ids[:, 0] == 101)
    assert np.all(valid.input_ids[:, 0] == 101)
    print("Text-safe self-test OK: deterministic spans, content bounds, cache contracts")


def encode_condition(batch, removed, encoder, torch, batch_size: int, device):
    token_ids = np.asarray(batch.input_ids, dtype=np.int64).copy()
    token_ids[removed] = MASK_TOKEN_ID
    attention = np.asarray(batch.attention_mask, dtype=np.int64).copy()
    token_types = np.asarray(batch.token_type_ids, dtype=np.int64).copy()
    features = np.empty((len(batch), SEQUENCE_LENGTH, 768), dtype=np.float32)
    with torch.inference_mode():
        for start in range(0, len(batch), batch_size):
            end = min(start + batch_size, len(batch))
            inputs = {
                "input_ids": torch.as_tensor(token_ids[start:end], dtype=torch.long, device=device),
                "attention_mask": torch.as_tensor(attention[start:end], dtype=torch.long, device=device),
                "token_type_ids": torch.as_tensor(token_types[start:end], dtype=torch.long, device=device),
            }
            output = encoder(**inputs).last_hidden_state.detach().cpu().numpy()
            if output.shape != (end - start, SEQUENCE_LENGTH, 768):
                raise RuntimeError("invalid BERT output shape")
            if not np.isfinite(output).all():
                raise RuntimeError("nonfinite BERT output")
            features[start:end] = output
    # Removed rows are unavailable to downstream pooling.  Their encoder output
    # is not used, but retaining the array shape makes the interface identical.
    features[removed] = 0.0
    effective = np.asarray(batch.content_mask, dtype=bool) & ~removed
    return features, effective, token_ids


def prepare(loader, *, batch_size: int, device: str) -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing experiment: {OUT}")
    batches = loader.load_official()
    del batches["test"]
    for split, batch in batches.items():
        verify_inputs(batch, split)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from transformers import BertModel

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    torch.manual_seed(20260924)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    encoder = BertModel.from_pretrained(
        str(MODEL), local_files_only=True, attn_implementation="eager"
    ).to(device).float().eval()
    encoder.requires_grad_(False)
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "q2_text_safe_v1.0",
        "source": "official aligned_50 train/valid only",
        "test_encoded": False,
        "bert_model_snapshot": MODEL.name,
        "mask_token_id": MASK_TOKEN_ID,
        "mask_scope": "contiguous content-token rows before frozen BERT; removed rows zeroed after encoding",
        "plan_seed": PLAN_SEED,
        "fractions": list(FRACTIONS),
        "positions": list(POSITIONS),
        "splits": {key: {"n": len(value), "ids_sha256": array_sha(np.asarray(value.ids, dtype="U"))}
                   for key, value in batches.items()},
    }
    for split, batch in batches.items():
        for fraction in FRACTIONS:
            for position in POSITIONS:
                condition = f"{int(fraction * 100):02d}pct_{position}"
                removed = make_removed(batch, fraction, position)
                features, effective, masked_ids = encode_condition(
                    batch, removed, encoder, torch, batch_size, device
                )
                path = OUT / f"{split}_textgap_{condition}.npz"
                np.savez_compressed(
                    path,
                    ids=np.asarray(batch.ids, dtype="U"),
                    features=features,
                    effective_mask=effective,
                    removed_mask=removed,
                    masked_input_ids=masked_ids,
                )
                print(f"Saved: {path.name}", flush=True)
    manifest["files"] = {path.name: sha256(path) for path in sorted(OUT.glob("*.npz"))}
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Text-safe preparation complete: {len(manifest['files'])} files; test untouched")


def check(loader) -> None:
    if not OUT.is_dir():
        raise FileNotFoundError(OUT)
    manifest = json.loads((OUT / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("test_encoded") is not False or manifest.get("mask_token_id") != MASK_TOKEN_ID:
        raise RuntimeError("invalid text-safe manifest")
    batches = loader.load_official()
    del batches["test"]
    expected = []
    for split, batch in batches.items():
        verify_inputs(batch, split)
        for fraction in FRACTIONS:
            for position in POSITIONS:
                condition = f"{int(fraction * 100):02d}pct_{position}"
                path = OUT / f"{split}_textgap_{condition}.npz"
                expected.append(path.name)
                if sha256(path) != manifest.get("files", {}).get(path.name):
                    raise RuntimeError(f"saved file checksum mismatch: {path.name}")
                with np.load(path, allow_pickle=False) as archive:
                    if set(archive.files) != {"ids", "features", "effective_mask", "removed_mask", "masked_input_ids"}:
                        raise RuntimeError(f"unexpected fields in {path.name}")
                    features = archive["features"]
                    effective = archive["effective_mask"]
                    removed = archive["removed_mask"]
                    ids = archive["ids"].astype(str)
                    masked = archive["masked_input_ids"]
                    if features.shape != (len(batch), 50, 768) or features.dtype != np.float32:
                        raise RuntimeError(f"invalid feature shape/dtype: {path.name}")
                    if effective.shape != (len(batch), 50) or removed.shape != effective.shape:
                        raise RuntimeError(f"invalid mask shape: {path.name}")
                    if not np.isfinite(features).all() or np.any(removed & ~batch.content_mask):
                        raise RuntimeError(f"invalid feature/mask values: {path.name}")
                    if not np.array_equal(effective, batch.content_mask & ~removed):
                        raise RuntimeError(f"effective mask mismatch: {path.name}")
                    if not np.all(features[removed] == 0):
                        raise RuntimeError(f"removed rows were not zeroed: {path.name}")
                    if not np.array_equal(ids, np.asarray(batch.ids, dtype=str)):
                        raise RuntimeError(f"ID order mismatch: {path.name}")
                    if not np.all(masked[removed] == MASK_TOKEN_ID):
                        raise RuntimeError(f"input mask mismatch: {path.name}")
                    if not np.array_equal(masked[~removed], batch.input_ids[~removed]):
                        raise RuntimeError(f"unmasked input token changed: {path.name}")
                    if not np.array_equal(removed, make_removed(batch, fraction, position)):
                        raise RuntimeError(f"removed span changed: {path.name}")
    actual = sorted(path.name for path in OUT.glob("*.npz"))
    if actual != sorted(expected):
        raise RuntimeError("text-safe file set mismatch")
    print(f"Text-safe check passed: {len(actual)} files; train/valid only; test untouched")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("preflight", "prepare", "check"))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch size must be positive")
    loader = load_module("q2_text_safe_data", "2026-09-24_q2_data_v1.0.py")
    if args.mode == "preflight":
        self_test(loader)
    elif args.mode == "prepare":
        prepare(loader, batch_size=args.batch_size, device=args.device)
    else:
        check(loader)


if __name__ == "__main__":
    main()
