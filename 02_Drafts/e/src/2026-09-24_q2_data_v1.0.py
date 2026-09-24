"""Read-only loading of the trusted official aligned PKL attachments.

Text features are deliberately named ``reference_text``: training and specialist
inference must use the same later frozen-BERT encoding of the returned tokens.
No model is downloaded or run, no normalization is fitted, and no file is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "01_Source" / "E" / "E题数据"
DEFAULT_OFFICIAL = SOURCE / "附件2-数据集特征文件" / "aligned_50.pkl"
DEFAULT_SPECIAL = {
    3: SOURCE / "附件3-模态缺失特征样本" / "对齐版本",
    4: SOURCE / "附件4-可解释专项视频样本与特征文件"
    / "附件4-可解释专项视频样本与特征文件" / "对齐版本",
}
SPLIT_COUNTS = {"train": 3395, "valid": 728, "test": 727}
MODALITY_DIMENSIONS = {"text": 768, "audio": 74, "vision": 35}
SEQUENCE_LENGTH = 50
VOCABULARY_SIZE = 30522


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass(frozen=True)
class AlignedBatch:
    """One native split or specialist collection, always with a batch axis.

    ``ids`` is the original ID when provided; attachment 3 has no original ID,
    so its key is ``attachment3:<file stem>`` and ``original_ids`` contains None.
    ``observation_masks`` is content AND nonzero observation for audio/vision;
    text uses official content-token availability before any simulated masking.
    ``nonzero_masks`` records the raw numeric property at all 50 positions and
    includes text only when a precomputed reference is actually present.
    """

    split: str
    ids: tuple[str, ...]
    original_ids: tuple[str | None, ...]
    raw_text: tuple[str | None, ...]
    input_ids: np.ndarray
    attention_mask: np.ndarray
    token_type_ids: np.ndarray
    content_mask: np.ndarray
    audio: np.ndarray
    vision: np.ndarray
    reference_text: np.ndarray | None
    observation_masks: Mapping[str, np.ndarray]
    nonzero_masks: Mapping[str, np.ndarray]
    classification_labels: np.ndarray | None
    regression_labels: np.ndarray | None
    source_paths: tuple[Path, ...]
    source_sha256: Mapping[str, str]

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def sequence_lengths(self) -> np.ndarray:
        return _readonly(self.attention_mask.sum(axis=1))

    @property
    def content_lengths(self) -> np.ndarray:
        return _readonly(self.content_mask.sum(axis=1))

    def summary(self) -> dict:
        return {
            "split": self.split,
            "samples": len(self),
            "sequence_length": SEQUENCE_LENGTH,
            "attention_positions": int(self.attention_mask.sum()),
            "content_positions": int(self.content_mask.sum()),
            "observed_content_positions": {
                key: int(value.sum()) for key, value in self.observation_masks.items()
            },
            "samples_without_observation": {
                key: int((~value.any(axis=1)).sum())
                for key, value in self.observation_masks.items()
            },
            "precomputed_text_available": self.reference_text is not None,
            "labels_available": self.classification_labels is not None,
            "source_files": len(self.source_sha256),
        }


def _numeric(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name}: expected shape {shape}, got {array.shape}")
    if array.dtype.kind not in "iuf":
        raise ValueError(f"{name}: expected real numeric dtype, got {array.dtype}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: nonfinite values")
    return array


def _continuous(value, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _numeric(value, shape, name)
    with np.errstate(over="ignore", under="ignore"):
        converted = array.astype(np.float32, copy=False)
    if not np.isfinite(converted).all():
        raise ValueError(f"{name}: values overflow float32")
    if not np.array_equal(np.any(array != 0, axis=-1), np.any(converted != 0, axis=-1)):
        raise ValueError(f"{name}: float32 conversion changes nonzero observation mask")
    return _readonly(converted)


def _tokens(value, n: int, name: str) -> tuple[np.ndarray, ...]:
    array = _numeric(value, (n, 3, SEQUENCE_LENGTH), name)
    if not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name}: token inputs must contain exact integers")
    if np.any(array < 0) or np.any(array[:, 0] >= VOCABULARY_SIZE):
        raise ValueError(f"{name}: token ID out of BERT vocabulary range")
    if not np.isin(array[:, 1:], (0, 1)).all():
        raise ValueError(f"{name}: attention and token types must be binary")
    converted = array.astype(np.int64, copy=False)
    ids, attention, types = converted[:, 0], converted[:, 1], converted[:, 2]
    active = attention.astype(bool)
    lengths = active.sum(axis=1)
    if np.any(lengths < 3):
        raise ValueError(f"{name}: needs CLS, at least one content token, and SEP")
    expected = np.arange(SEQUENCE_LENGTH)[None, :] < lengths[:, None]
    if not np.array_equal(active, expected):
        raise ValueError(f"{name}: attention must be a continuous prefix")
    if np.any(ids[:, 0] != 101) or np.any(ids[np.arange(n), lengths - 1] != 102):
        raise ValueError(f"{name}: expected CLS first and SEP last")
    if np.any(ids[~active] != 0) or np.any(types[~active] != 0) or np.any(ids[active] == 0):
        raise ValueError(f"{name}: PAD token or segment inconsistent with attention")
    special = active & np.isin(ids, (101, 102))
    if np.any(special.sum(axis=1) != 2):
        raise ValueError(f"{name}: unexpected internal CLS or SEP token")
    content = active & ~special
    return tuple(_readonly(x) for x in (ids, active, types, content))


def _batch(
    data: dict,
    *,
    split: str,
    n: int,
    ids: tuple[str, ...],
    original_ids: tuple[str | None, ...],
    raw_text: tuple[str | None, ...],
    paths: tuple[Path, ...],
    hashes: Mapping[str, str],
    labeled: bool,
) -> AlignedBatch:
    if not (n == len(ids) == len(original_ids) == len(raw_text) == len(paths)):
        raise ValueError(f"{split}: metadata lengths do not match")
    if n < 1 or len(set(ids)) != n or any(not x for x in ids):
        raise ValueError(f"{split}: empty or duplicate sample IDs")
    input_ids, attention, token_types, content = _tokens(data["text_bert"], n, split)
    audio = _continuous(data["audio"], (n, 50, 74), f"{split}/audio")
    vision = _continuous(data["vision"], (n, 50, 35), f"{split}/vision")
    reference = (
        _continuous(data["text"], (n, 50, 768), f"{split}/text")
        if "text" in data else None
    )
    nonzero = {"audio": np.any(audio != 0, axis=-1), "vision": np.any(vision != 0, axis=-1)}
    if reference is not None:
        nonzero["text"] = np.any(reference != 0, axis=-1)
    observations = {"text": content}
    observations.update({key: content & nonzero[key] for key in ("audio", "vision")})
    classification = regression = None
    if labeled:
        classes = _numeric(data["classification_labels"], (n,), f"{split}/classification")
        if not np.isin(classes, (0, 1, 2)).all():
            raise ValueError(f"{split}: expected classification labels 0, 1, 2")
        scores = _numeric(data["regression_labels"], (n,), f"{split}/regression")
        if np.any(np.abs(scores) > 3):
            raise ValueError(f"{split}: regression labels outside [-3, 3]")
        if not np.array_equal(classes, np.sign(scores) + 1):
            raise ValueError(f"{split}: classification labels contradict regression sign")
        classification = _readonly(classes.astype(np.int64, copy=False))
        regression = _readonly(scores.astype(np.float32, copy=False))
    return AlignedBatch(
        split=split, ids=ids, original_ids=original_ids, raw_text=raw_text,
        input_ids=input_ids, attention_mask=attention, token_type_ids=token_types,
        content_mask=content, audio=audio, vision=vision, reference_text=reference,
        observation_masks=MappingProxyType({k: _readonly(v) for k, v in observations.items()}),
        nonzero_masks=MappingProxyType({k: _readonly(v) for k, v in nonzero.items()}),
        classification_labels=classification, regression_labels=regression,
        source_paths=paths, source_sha256=MappingProxyType(dict(hashes)),
    )


def _load_pickle(path: Path) -> tuple[dict, str]:
    # Only load trusted competition material; pickle is an executable format.
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
        if handle.read(1):
            raise ValueError(f"Trailing data after pickle: {path}")
    if not isinstance(data, dict):
        raise ValueError(f"Expected a dictionary: {path}")
    return data, file_sha256(path)


def load_official(path: Path = DEFAULT_OFFICIAL) -> dict[str, AlignedBatch]:
    """Load all native splits; reject cross-split sample/video ID leakage."""
    path = Path(path).resolve()
    data, digest = _load_pickle(path)
    if set(data) != set(SPLIT_COUNTS):
        raise ValueError(f"Expected native splits {tuple(SPLIT_COUNTS)}, got {tuple(data)}")
    expected_keys = {"id", "raw_text", "text_bert", "text", "audio", "vision",
                     "classification_labels", "regression_labels"}
    result = {}
    for split, expected_n in SPLIT_COUNTS.items():
        raw = data[split]
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise ValueError(f"Unexpected official fields in {split}")
        raw_ids = np.asarray(raw["id"])
        raw_text = np.asarray(raw["raw_text"])
        if raw_ids.shape != (expected_n,) or raw_text.shape != (expected_n,):
            raise ValueError(f"{split}: unexpected sample count or text/ID shape")
        if raw_ids.dtype.kind not in "US" or raw_text.dtype.kind not in "US":
            raise ValueError(f"{split}: expected string IDs and raw text")
        ids = tuple(str(item) for item in raw_ids)
        result[split] = _batch(
            raw, split=split, n=expected_n, ids=ids, original_ids=ids,
            raw_text=tuple(str(item) for item in raw_text), paths=(path,) * expected_n,
            hashes={str(path): digest}, labeled=True,
        )
    seen_ids: set[str] = set()
    seen_videos: set[str] = set()
    for split, batch in result.items():
        if any("$_$" not in sample_id for sample_id in batch.ids):
            raise ValueError(f"{split}: official ID lacks video/clip delimiter")
        ids = set(batch.ids)
        videos = {sample_id.split("$_$", 1)[0] for sample_id in ids}
        if ids & seen_ids or videos & seen_videos:
            raise ValueError(f"{split}: sample or source-video ID overlaps another split")
        seen_ids.update(ids)
        seen_videos.update(videos)
    return result


def load_special(path: Path, attachment: int) -> AlignedBatch:
    """Normalize the exact attachment 3/4 aligned schemas without reshaping."""
    path = Path(path).resolve()
    if attachment not in (3, 4):
        raise ValueError("attachment must be 3 or 4")
    data, digest = _load_pickle(path)
    split = f"attachment{attachment}"
    if attachment == 3:
        if set(data) != {"test"} or not isinstance(data["test"], dict):
            raise ValueError(f"{path}: attachment 3 requires an outer test dictionary")
        raw = data["test"]
        if set(raw) != {"text_bert", "audio", "vision"}:
            raise ValueError(f"{path}: unexpected attachment 3 fields")
        ids = (f"attachment3:{path.stem}",)
        original_ids = (None,)
        texts = (None,)
    else:
        if set(data) != {"id", "raw_text", "text_bert", "text", "audio", "vision"}:
            raise ValueError(f"{path}: unexpected attachment 4 fields")
        if not isinstance(data["id"], (str, np.str_)) or not isinstance(data["raw_text"], (str, np.str_)):
            raise ValueError(f"{path}: attachment 4 ID/text must be scalar strings")
        raw = {}
        for key, shape in (("text_bert", (3, 50)), ("text", (50, 768)),
                           ("audio", (50, 74)), ("vision", (50, 35))):
            original = _numeric(data[key], shape, f"{path.name}/{key}")
            raw[key] = original[None, ...]
        ids = original_ids = (str(data["id"]),)
        texts = (str(data["raw_text"]),)
    return _batch(raw, split=split, n=1, ids=ids, original_ids=original_ids,
                  raw_text=texts, paths=(path,), hashes={str(path): digest}, labeled=False)


def load_special_directory(directory: Path, attachment: int) -> AlignedBatch:
    """Load the complete 30-file or 20-file aligned collection in file order."""
    if attachment not in (3, 4):
        raise ValueError("attachment must be 3 or 4")
    directory = Path(directory).resolve()
    expected_names = ({f"附件3_{i:02}.pkl" for i in range(1, 31)} if attachment == 3
                      else {f"{i:02}.pkl" for i in range(1, 21)})
    actual_names = {path.name for path in directory.glob("*.pkl")}
    if actual_names != expected_names:
        raise ValueError(f"{directory}: missing {expected_names - actual_names}, extra {actual_names - expected_names}")
    batches = [load_special(directory / name, attachment) for name in sorted(expected_names)]
    raw = {
        "text_bert": np.stack([
            np.stack((batch.input_ids[0], batch.attention_mask[0], batch.token_type_ids[0]))
            for batch in batches
        ]),
        "audio": np.concatenate([batch.audio for batch in batches]),
        "vision": np.concatenate([batch.vision for batch in batches]),
    }
    if attachment == 4:
        raw["text"] = np.concatenate([batch.reference_text for batch in batches])
    return _batch(
        raw, split=f"attachment{attachment}", n=len(batches),
        ids=tuple(batch.ids[0] for batch in batches),
        original_ids=tuple(batch.original_ids[0] for batch in batches),
        raw_text=tuple(batch.raw_text[0] for batch in batches),
        paths=tuple(batch.source_paths[0] for batch in batches),
        hashes={key: value for batch in batches for key, value in batch.source_sha256.items()},
        labeled=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--attachment3", type=Path, default=DEFAULT_SPECIAL[3])
    parser.add_argument("--attachment4", type=Path, default=DEFAULT_SPECIAL[4])
    args = parser.parse_args()
    batches = load_official(args.official)
    batches["attachment3"] = load_special_directory(args.attachment3, 3)
    batches["attachment4"] = load_special_directory(args.attachment4, 4)
    print(json.dumps({key: batch.summary() for key, batch in batches.items()}, ensure_ascii=False, indent=2))
    print("Read-only loader verified: 4850 official samples and 50 specialist files")


if __name__ == "__main__":
    main()
