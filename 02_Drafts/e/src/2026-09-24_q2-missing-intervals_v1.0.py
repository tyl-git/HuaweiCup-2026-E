"""Deterministic contiguous missing-modality intervals for Q2 experiments.

Plans use content-token coordinates and the original observation masks. Save or
reuse one plan for paired model comparisons; never infer observation from
standardized feature values.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np


MODALITIES = ("text", "audio", "vision")
POSITIONS = ("start", "middle", "end", "random")
SEQUENCE_LENGTH = 50


def _binary_mask(value: np.ndarray, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype.kind not in "biuf" or not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name}: expected binary mask with shape {shape}")
    return array.astype(bool, copy=False)


def _inputs(
    content_mask: np.ndarray,
    observation_masks: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    content = np.asarray(content_mask)
    if content.ndim != 2 or content.shape[0] < 1 or content.shape[1] != SEQUENCE_LENGTH:
        raise ValueError("content_mask must have shape [N,50] with N >= 1")
    content = _binary_mask(content, content.shape, "content_mask")
    for row in content:
        positions = np.flatnonzero(row)
        if len(positions) and positions[-1] - positions[0] + 1 != len(positions):
            raise ValueError("content_mask must be contiguous in each sample")
    if set(observation_masks) != set(MODALITIES):
        raise ValueError("observation_masks must contain text, audio, and vision")
    observed = {}
    for modality in MODALITIES:
        mask = _binary_mask(observation_masks[modality], content.shape, modality)
        if np.any(mask & ~content):
            raise ValueError(f"{modality}: observation outside content")
        observed[modality] = mask
    return content, observed


def _stable_start(seed: int, sample_id: str, modality: str, choices: int) -> int:
    key = f"{seed}\0{sample_id}\0{modality}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % choices


@dataclass(frozen=True)
class IntervalPlan:
    sample_ids: tuple[str, ...]
    modalities: tuple[str, ...]
    fraction: float
    position: str
    seed: int
    removed_masks: Mapping[str, np.ndarray]
    intervals: Mapping[str, np.ndarray]
    original_masks: Mapping[str, np.ndarray]
    originally_observed: Mapping[str, int]
    removed_counts: Mapping[str, int]

    def summary(self) -> dict:
        """Report actual removal against the original observable denominator."""
        return {
            "samples": len(self.sample_ids),
            "modalities": list(self.modalities),
            "target_fraction": self.fraction,
            "position": self.position,
            "seed": self.seed,
            "per_modality": {
                modality: {
                    "originally_observed": self.originally_observed[modality],
                    "removed": self.removed_counts[modality],
                    "actual_fraction": (
                        self.removed_counts[modality] / self.originally_observed[modality]
                        if self.originally_observed[modality] else None
                    ),
                }
                for modality in self.modalities
            },
        }


@dataclass(frozen=True)
class MaskedFeatures:
    features: Mapping[str, np.ndarray]
    observation_masks: Mapping[str, np.ndarray]


def make_interval_plan(
    content_mask: np.ndarray,
    observation_masks: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
    *,
    modalities: Sequence[str],
    fraction: float,
    position: str,
    seed: int,
) -> IntervalPlan:
    """Select a contiguous content-coordinate interval for each selected modality.

    ``fraction`` targets original observable rows per sample, rounded up. If
    observations have gaps, the selected coordinate span includes those gaps;
    only originally observable rows count as removed. Empty modalities retain
    the sample with interval (-1, -1) and zero removed rows.
    """
    content, observed = _inputs(content_mask, observation_masks)
    ids = tuple(sample_ids)
    if len(ids) != len(content) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError("sample_ids must be one nonempty string per sample")
    if len(set(ids)) != len(ids):
        raise ValueError("sample_ids must be unique for repeatable masks")
    selected = tuple(modalities)
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= set(MODALITIES):
        raise ValueError("modalities must be unique members of text/audio/vision")
    if not np.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if position not in POSITIONS:
        raise ValueError(f"position must be one of {POSITIONS}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError("seed must be an integer")
    removed, intervals, originals, denominators, counts = {}, {}, {}, {}, {}
    for modality in selected:
        mask = np.zeros(content.shape, dtype=bool)
        spans = np.full((len(ids), 2), -1, dtype=np.int64)
        source = observed[modality]
        for row, sample_id in enumerate(ids):
            eligible = np.flatnonzero(source[row])
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
                offset = _stable_start(seed, sample_id, modality, choices)
            begin, end = int(eligible[offset]), int(eligible[offset + target - 1]) + 1
            mask[row, begin:end] = source[row, begin:end]
            spans[row] = (begin, end)
        mask.setflags(write=False)
        spans.setflags(write=False)
        removed[modality] = mask
        intervals[modality] = spans
        original = source.copy()
        original.setflags(write=False)
        originals[modality] = original
        denominators[modality] = int(source.sum())
        counts[modality] = int(mask.sum())
    return IntervalPlan(
        sample_ids=ids, modalities=selected, fraction=float(fraction), position=position,
        seed=seed, removed_masks=MappingProxyType(removed),
        intervals=MappingProxyType(intervals),
        original_masks=MappingProxyType(originals),
        originally_observed=MappingProxyType(denominators),
        removed_counts=MappingProxyType(counts),
    )


def apply_interval_plan(
    features: Mapping[str, np.ndarray],
    content_mask: np.ndarray,
    observation_masks: Mapping[str, np.ndarray],
    sample_ids: Sequence[str],
    plan: IntervalPlan,
) -> MaskedFeatures:
    """Return new feature and observation arrays; leave all caller inputs intact."""
    content, observed = _inputs(content_mask, observation_masks)
    if tuple(sample_ids) != plan.sample_ids or len(content) != len(plan.sample_ids):
        raise ValueError("plan sample IDs/order differ from supplied batch")
    if set(features) != set(MODALITIES):
        raise ValueError("features must contain text, audio, and vision")
    output_features, output_masks = {}, {}
    for modality in MODALITIES:
        array = np.asarray(features[modality])
        if (array.ndim != 3 or array.shape[:2] != content.shape
                or array.dtype.kind not in "iuf" or not np.isfinite(array).all()):
            raise ValueError(f"{modality}: expected finite real features [N,50,D]")
        removed = plan.removed_masks.get(modality)
        if removed is not None and (
            removed.shape != content.shape
            or not np.array_equal(plan.original_masks[modality], observed[modality])
        ):
            raise ValueError(f"{modality}: plan is incompatible with original observations")
        copied = array.copy()
        copied_mask = observed[modality].copy()
        if removed is not None:
            copied[removed] = 0
            copied_mask[removed] = False
        copied.setflags(write=False)
        copied_mask.setflags(write=False)
        output_features[modality] = copied
        output_masks[modality] = copied_mask
    return MaskedFeatures(MappingProxyType(output_features), MappingProxyType(output_masks))
