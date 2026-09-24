"""Train-only, mask-aware normalization for official aligned inputs.

The caller supplies uniformly re-encoded frozen-BERT text features. This module
does not load PKL files, run BERT, or substitute ``reference_text`` for that input.
Content excludes CLS/SEP/PAD. A row contributes only when content, the caller's
observation mask, and its raw numeric nonzero mask are all true.

Standardized observations can legitimately be zero. Always pass the returned
observation masks to a model; never infer new masks from standardized values.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np


MODALITIES = ("text", "audio", "vision")
SEQUENCE_LENGTH = 50
SCHEMA = "huaweicup-q2-normalization-v1.0"


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      allow_nan=False, separators=(",", ":")).encode("utf-8")


def _file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _array_sha256(array: np.ndarray, chunk_rows: int = 2048) -> str:
    """Fingerprint shape, dtype, and C-order bytes without one large copy."""
    digest = hashlib.sha256(_canonical_json({"shape": list(array.shape),
                                             "dtype": array.dtype.str}))
    rows_per_chunk = max(1, min(chunk_rows, (4 * 1024 * 1024) // max(1, array[0].nbytes)))
    for start in range(0, len(array), rows_per_chunk):
        digest.update(memoryview(np.ascontiguousarray(array[start:start + rows_per_chunk])))
    return digest.hexdigest()


def _mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf":
        raise ValueError(f"{name}: expected binary mask with shape {shape}")
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name}: expected binary mask")
    return array.astype(bool, copy=False)


def _validate(
    features: Mapping[str, np.ndarray],
    content_mask: np.ndarray,
    observation_masks: Mapping[str, np.ndarray],
    *,
    chunk_rows: int,
    dimensions: Mapping[str, int] | None = None,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    if set(features) != set(MODALITIES) or set(observation_masks) != set(MODALITIES):
        raise ValueError("features and observation_masks must contain text/audio/vision")
    content = np.asarray(content_mask)
    if content.ndim != 2 or content.shape[0] < 1 or content.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("content_mask must have shape [N,50] with N >= 1")
    content = _mask(content, content.shape, "content_mask")
    arrays, effective, numeric_nonzero = {}, {}, {}
    for modality in MODALITIES:
        array = np.asarray(features[modality])
        if (array.ndim != 3 or array.shape[:2] != content.shape or array.shape[2] < 1
                or array.dtype.kind not in "iuf"):
            raise ValueError(f"{modality}: expected real features [N,50,D]")
        if dimensions is not None and array.shape[2] != dimensions[modality]:
            raise ValueError(f"{modality}: feature dimension differs from training")
        # Validate all values, including excluded positions, without a large bool temporary.
        nonzero = np.empty(content.shape, dtype=bool)
        for start in range(0, len(array), max(1, chunk_rows // SEQUENCE_LENGTH)):
            stop = start + max(1, chunk_rows // SEQUENCE_LENGTH)
            if not np.isfinite(array[start:stop]).all():
                raise ValueError(f"{modality}: nonfinite features")
            nonzero[start:stop] = np.any(array[start:stop] != 0, axis=-1)
        observed = _mask(observation_masks[modality], content.shape,
                         f"{modality}/observation_mask")
        arrays[modality] = array
        numeric_nonzero[modality] = nonzero
        effective[modality] = content & observed & nonzero
    return arrays, content, effective, numeric_nonzero


@dataclass(frozen=True)
class ModalityStatistics:
    count: int
    mean: np.ndarray
    std: np.ndarray
    scale: np.ndarray


@dataclass(frozen=True)
class NormalizedFeatures:
    features: Mapping[str, np.ndarray]
    content_mask: np.ndarray
    observation_masks: Mapping[str, np.ndarray]
    raw_nonzero_masks: Mapping[str, np.ndarray]


class MultimodalStandardizer:
    """Population z-score using only observed training content positions.

    ``source_metadata`` is required and must be a nonempty JSON-compatible dict.
    Record at least the source-file SHA-256 values and frozen-BERT model/cache
    fingerprints. The module additionally fingerprints the exact input arrays,
    content mask, supplied observation masks, and effective fitting masks.

    ``split='train'`` is required explicitly. The caller is responsible for its
    truth; this guard cannot establish provenance of arbitrary supplied arrays.
    A fitted instance cannot be fitted again. Create a new instance deliberately
    when changing the training definition.
    """

    def __init__(self, *, min_scale: float = 1e-6, chunk_rows: int = 2048):
        if not np.isfinite(min_scale) or min_scale <= 0:
            raise ValueError("min_scale must be finite and positive")
        if not isinstance(chunk_rows, int) or isinstance(chunk_rows, bool) or chunk_rows < 1:
            raise ValueError("chunk_rows must be a positive integer")
        self.min_scale = float(min_scale)
        self.chunk_rows = chunk_rows
        self._statistics: dict[str, ModalityStatistics] = {}
        self._metadata: dict = {}

    @property
    def statistics(self) -> Mapping[str, ModalityStatistics]:
        return MappingProxyType(self._statistics)

    @property
    def metadata(self) -> dict:
        return copy.deepcopy(self._metadata)

    def fit(
        self,
        features: Mapping[str, np.ndarray],
        content_mask: np.ndarray,
        observation_masks: Mapping[str, np.ndarray],
        *,
        split: str,
        source_metadata: Mapping,
    ) -> MultimodalStandardizer:
        if split != "train":
            raise ValueError("Normalization may only be fitted on the native train split")
        if self._statistics:
            raise RuntimeError("Already fitted; transform other splits without refitting")
        if not isinstance(source_metadata, Mapping) or not source_metadata:
            raise ValueError("Nonempty source_metadata is required")
        provenance = json.loads(_canonical_json(dict(source_metadata)))
        arrays, content, masks, _ = _validate(
            features, content_mask, observation_masks, chunk_rows=self.chunk_rows)
        statistics = {}
        fingerprints = {}
        for modality in MODALITIES:
            raw = arrays[modality]
            selected_mask = masks[modality].reshape(-1)
            dimension = raw.shape[2]
            flat = raw.reshape(-1, dimension)
            count = 0
            mean = np.zeros(dimension, dtype=np.float64)
            m2 = np.zeros(dimension, dtype=np.float64)
            for start in range(0, len(flat), self.chunk_rows):
                selected = flat[start:start + self.chunk_rows][
                    selected_mask[start:start + self.chunk_rows]].astype(np.float64)
                n = len(selected)
                if not n:
                    continue
                block_mean = selected.mean(axis=0, dtype=np.float64)
                centered = selected - block_mean
                block_m2 = np.einsum("ij,ij->j", centered, centered)
                total = count + n
                delta = block_mean - mean
                m2 += block_m2 + delta * delta * (count * n / total)
                mean += delta * (n / total)
                count = total
            if count == 0:
                raise ValueError(f"{modality}: no observed training content positions")
            std = np.sqrt(np.maximum(m2 / count, 0))
            scale = np.maximum(std, self.min_scale)
            if not all(np.isfinite(x).all() for x in (mean, std, scale)):
                raise ValueError(f"{modality}: statistics overflow float64")
            statistics[modality] = ModalityStatistics(
                count, _readonly(mean), _readonly(std), _readonly(scale))
            fingerprints[modality] = {
                "shape": list(raw.shape), "dtype": raw.dtype.str,
                "features_sha256": _array_sha256(raw),
                "supplied_observation_mask_sha256": _array_sha256(
                    np.asarray(observation_masks[modality], dtype=bool)),
                "effective_observation_mask_sha256": _array_sha256(masks[modality]),
            }
        self._statistics = statistics
        self._metadata = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "fit_split": split,
            "sample_count": int(content.shape[0]),
            "sequence_length": SEQUENCE_LENGTH,
            "algorithm": "population_zscore_float64_block_merge",
            "variance_ddof": 0,
            "min_scale": self.min_scale,
            "chunk_rows": self.chunk_rows,
            "fit_mask_rule": "content AND supplied_observed AND raw_row_nonzero",
            "masked_output": "exact_float32_zero",
            "source_metadata": provenance,
            "content_mask_sha256": _array_sha256(content),
            "training_inputs": fingerprints,
            "statistics": {
                key: {"count": value.count, "dimensions": len(value.mean),
                      "floored_scale_dimensions": int((value.std < self.min_scale).sum())}
                for key, value in statistics.items()
            },
        }
        return self

    def transform(
        self,
        features: Mapping[str, np.ndarray],
        content_mask: np.ndarray,
        observation_masks: Mapping[str, np.ndarray],
    ) -> NormalizedFeatures:
        if not self._statistics:
            raise RuntimeError("Fit on train or load saved training statistics first")
        arrays, content, masks, numeric_nonzero = _validate(
            features, content_mask, observation_masks, chunk_rows=self.chunk_rows,
            dimensions={key: len(value.mean) for key, value in self._statistics.items()})
        result = {}
        for modality in MODALITIES:
            array = arrays[modality]
            stats = self._statistics[modality]
            dimension = array.shape[2]
            flat = array.reshape(-1, dimension)
            selected_mask = masks[modality].reshape(-1)
            output = np.zeros(array.shape, dtype=np.float32)
            out_flat = output.reshape(-1, dimension)
            for start in range(0, len(flat), self.chunk_rows):
                mask = selected_mask[start:start + self.chunk_rows]
                selected = flat[start:start + self.chunk_rows][mask].astype(np.float64)
                with np.errstate(over="ignore", invalid="ignore"):
                    normalized = ((selected - stats.mean) / stats.scale).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError(f"{modality}: standardized values overflow float32")
                out_flat[start:start + self.chunk_rows][mask] = normalized
            result[modality] = _readonly(output)
        return NormalizedFeatures(
            features=MappingProxyType(result),
            content_mask=_readonly(content.copy()),
            observation_masks=MappingProxyType({k: _readonly(v) for k, v in masks.items()}),
            raw_nonzero_masks=MappingProxyType({k: _readonly(v) for k, v in numeric_nonzero.items()}),
        )

    def save(self, path: str | Path) -> tuple[Path, Path]:
        """Create a stats NPZ and adjacent JSON manifest; never overwrite files."""
        if not self._statistics:
            raise RuntimeError("No training statistics to save")
        path = Path(path).resolve()
        if path.suffix.lower() != ".npz":
            raise ValueError("Statistics path must end in .npz")
        if "01_source" in {part.lower() for part in path.parts}:
            raise ValueError("Source material is read-only; save under drafts/results")
        manifest_path = path.with_suffix(".json")
        if path.exists() or manifest_path.exists():
            raise FileExistsError("Statistics or manifest already exists; choose a new version")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {}
        for key, stats in self._statistics.items():
            payload.update({f"{key}_mean": stats.mean, f"{key}_std": stats.std,
                            f"{key}_scale": stats.scale})
        with path.open("xb") as handle:
            np.savez_compressed(handle, **payload)
        manifest = self.metadata
        manifest["statistics_file"] = path.name
        manifest["statistics_file_sha256"] = _file_sha256(path)
        manifest["manifest_sha256"] = hashlib.sha256(_canonical_json(manifest)).hexdigest()
        with manifest_path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        return path, manifest_path

    @classmethod
    def load(cls, path: str | Path) -> MultimodalStandardizer:
        """Verify both the NPZ checksum and canonical manifest checksum."""
        path = Path(path).resolve()
        manifest = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
        digest = manifest.pop("manifest_sha256", None)
        if digest != hashlib.sha256(_canonical_json(manifest)).hexdigest():
            raise ValueError("Manifest SHA-256 mismatch")
        if manifest.get("schema") != SCHEMA or manifest.get("fit_split") != "train":
            raise ValueError("Unsupported normalization schema or non-training statistics")
        if manifest.get("statistics_file") != path.name:
            raise ValueError("Statistics filename differs from manifest")
        if manifest.get("statistics_file_sha256") != _file_sha256(path):
            raise ValueError("Statistics file SHA-256 mismatch")
        result = cls(min_scale=manifest["min_scale"], chunk_rows=manifest["chunk_rows"])
        expected = {f"{modality}_{field}" for modality in MODALITIES
                    for field in ("mean", "std", "scale")}
        statistics = {}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected:
                raise ValueError("Unexpected statistics arrays")
            for modality in MODALITIES:
                info = manifest["statistics"][modality]
                count, dimension = info["count"], info["dimensions"]
                if (not isinstance(count, int) or isinstance(count, bool) or count < 1
                        or not isinstance(dimension, int) or dimension < 1):
                    raise ValueError(f"{modality}: invalid statistics count/dimension")
                values = [archive[f"{modality}_{field}"].copy()
                          for field in ("mean", "std", "scale")]
                if any(x.shape != (dimension,) or x.dtype != np.float64
                       or not np.isfinite(x).all() for x in values):
                    raise ValueError(f"{modality}: invalid statistics array")
                mean, std, scale = values
                if np.any(std < 0) or not np.array_equal(scale, np.maximum(std, result.min_scale)):
                    raise ValueError(f"{modality}: invalid standard deviation/scale")
                statistics[modality] = ModalityStatistics(
                    count, _readonly(mean), _readonly(std), _readonly(scale))
        manifest.pop("statistics_file")
        manifest.pop("statistics_file_sha256")
        result._statistics = statistics
        result._metadata = manifest
        return result


def self_test() -> None:
    """Exercise leakage, mask preservation, degeneracy, and artifact integrity."""
    def rejects(error, action):
        try:
            action()
        except error:
            return
        raise AssertionError(f"Expected {error.__name__}")

    rng = np.random.default_rng(20260924)
    content = np.zeros((3, 50), dtype=bool)
    content[:, 1:5] = True
    features = {}
    observations = {}
    for modality, dimension in zip(MODALITIES, (5, 3, 2)):
        data = rng.normal(3, 2, size=(3, 50, dimension)).astype(np.float32)
        data[:, 0] = 1e6  # Excluded special-token/padding outliers must not affect fit.
        data[:, -1] = -1e6
        data[:, :, -1] = 7  # Exactly constant observed dimension.
        data[0, 1] = 0  # Missing even if caller's observed mask is overly broad.
        data[0, 2, 0] = 0  # A zero component in a nonzero row is still observed.
        observed = content.copy()
        observed[1, 3] = False  # Explicitly unavailable despite numeric nonzero values.
        features[modality] = data
        observations[modality] = observed
    snapshots = {key: value.copy() for key, value in features.items()}
    source = {"test_fixture": "synthetic_only", "text_encoding": "synthetic_fixture"}
    normalizer = MultimodalStandardizer(chunk_rows=7)
    rejects(ValueError, lambda: normalizer.fit(features, content, observations,
                                              split="valid", source_metadata=source))
    normalizer.fit(features, content, observations, split="train", source_metadata=source)
    train = normalizer.transform(features, content, observations)
    for modality in MODALITIES:
        mask = content & observations[modality] & np.any(features[modality] != 0, axis=-1)
        selected = features[modality][mask].astype(np.float64)
        stats = normalizer.statistics[modality]
        assert stats.count == 10
        np.testing.assert_allclose(stats.mean, selected.mean(axis=0), rtol=1e-13, atol=1e-13)
        np.testing.assert_allclose(stats.std, selected.std(axis=0), rtol=1e-13, atol=1e-13)
        np.testing.assert_array_equal(train.features[modality][~mask], 0)
        np.testing.assert_array_equal(train.observation_masks[modality], mask)
        np.testing.assert_array_equal(features[modality], snapshots[modality])
        np.testing.assert_allclose(train.features[modality][mask].mean(axis=0), 0, atol=2e-7)
        np.testing.assert_allclose(train.features[modality][mask].std(axis=0)[:-1], 1, atol=2e-7)
        np.testing.assert_array_equal(train.features[modality][mask][:, -1], 0)
        assert stats.scale[-1] == normalizer.min_scale
        assert not train.features[modality].flags.writeable
    frozen_means = {key: value.mean.copy() for key, value in normalizer.statistics.items()}
    shifted = {key: value + 100 for key, value in features.items()}
    valid = normalizer.transform(shifted, content, observations)
    for key in MODALITIES:
        np.testing.assert_array_equal(normalizer.statistics[key].mean, frozen_means[key])
        mask = valid.observation_masks[key]
        expected = ((shifted[key][mask].astype(np.float64) - frozen_means[key])
                    / normalizer.statistics[key].scale).astype(np.float32)
        np.testing.assert_array_equal(valid.features[key][mask], expected)
    rejects(RuntimeError, lambda: normalizer.fit(features, content, observations,
                                                split="train", source_metadata=source))
    all_missing = {key: np.zeros_like(value) for key, value in features.items()}
    missing = normalizer.transform(all_missing, content, observations)
    assert all(not x.any() for x in missing.observation_masks.values())
    assert all(not x.any() for x in missing.features.values())
    rejects(ValueError, lambda: MultimodalStandardizer().fit(
        all_missing, content, observations, split="train", source_metadata=source))
    nonfinite = {key: value.copy() for key, value in features.items()}
    nonfinite["audio"][0, -1, 0] = np.nan
    rejects(ValueError, lambda: normalizer.transform(nonfinite, content, observations))
    wrong_shape = dict(features, vision=features["vision"][:, :, :1])
    rejects(ValueError, lambda: normalizer.transform(wrong_shape, content, observations))
    # A legitimate observed row equal to the training mean becomes zero, yet stays observed.
    constant = {key: np.ones((1, 50, 1), dtype=np.float32) for key in MODALITIES}
    one_mask = np.zeros((1, 50), dtype=bool)
    one_mask[0, 2] = True
    one_observed = {key: one_mask for key in MODALITIES}
    single = MultimodalStandardizer().fit(constant, one_mask, one_observed,
                                          split="train", source_metadata=source)
    single_out = single.transform(constant, one_mask, one_observed)
    assert all(not x.any() for x in single_out.features.values())
    assert all(x[0, 2] for x in single_out.observation_masks.values())
    with tempfile.TemporaryDirectory(prefix="q2-normalization-",
                                     dir=Path(__file__).resolve().parent) as folder:
        path, manifest_path = normalizer.save(Path(folder) / "synthetic.npz")
        loaded = MultimodalStandardizer.load(path)
        restored = loaded.transform(features, content, observations)
        assert loaded.metadata == normalizer.metadata
        for key in MODALITIES:
            np.testing.assert_array_equal(restored.features[key], train.features[key])
            np.testing.assert_array_equal(restored.observation_masks[key], train.observation_masks[key])
        rejects(FileExistsError, lambda: normalizer.save(path))
        manifest_bytes = manifest_path.read_bytes()
        changed = json.loads(manifest_bytes)
        changed["source_metadata"]["test_fixture"] = "changed"
        manifest_path.write_text(json.dumps(changed), encoding="utf-8")
        rejects(ValueError, lambda: MultimodalStandardizer.load(path))
        manifest_path.write_bytes(manifest_bytes)
        with path.open("ab") as handle:
            handle.write(b"corrupted")
        rejects(ValueError, lambda: MultimodalStandardizer.load(path))
    print("Self-test OK: train-only fit, masked statistics, unchanged inputs, zero fill,")
    print("constant dimensions, empty observations, validation transform, and SHA-256 round trip.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--self-test", action="store_true")
    group.add_argument("--inspect", type=Path, metavar="STATS_NPZ")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        normalizer = MultimodalStandardizer.load(args.inspect)
        print(json.dumps(normalizer.metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
