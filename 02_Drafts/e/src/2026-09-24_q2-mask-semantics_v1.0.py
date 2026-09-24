"""Keep content, source-zero, injected-missing and effective masks distinct.

The source-zero mask reports all-zero feature rows within content. In the
competition's attachment 3 these rows represent local missing intervals, but
the same numeric pattern in the official training data has no provenance label.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

import numpy as np


MODALITIES = ("text", "audio", "vision")
SRC = Path(__file__).resolve().parent


@dataclass(frozen=True)
class MaskState:
    content: np.ndarray
    source_observed: Mapping[str, np.ndarray]
    source_zero: Mapping[str, np.ndarray]
    injected_missing: Mapping[str, np.ndarray]
    effective_observed: Mapping[str, np.ndarray]


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def build_mask_state(batch, injected_missing: Mapping[str, np.ndarray] | None = None) -> MaskState:
    """Construct per-position masks without treating padding as missing data."""
    content = np.asarray(batch.content_mask, dtype=bool)
    if content.ndim != 2 or content.shape[1] != 50:
        raise ValueError("content mask must have shape [N,50]")
    if set(batch.observation_masks) != set(MODALITIES):
        raise ValueError("expected observation masks for text/audio/vision")
    injected_missing = {} if injected_missing is None else injected_missing
    if not set(injected_missing) <= set(MODALITIES):
        raise ValueError("unknown injected-missing modality")
    source_observed, source_zero, injected, effective = {}, {}, {}, {}
    for modality in MODALITIES:
        observed = np.asarray(batch.observation_masks[modality], dtype=bool)
        missing = np.asarray(injected_missing.get(modality, np.zeros_like(content)), dtype=bool)
        if observed.shape != content.shape or missing.shape != content.shape:
            raise ValueError(f"{modality}: mask shape mismatch")
        if np.any(observed & ~content):
            raise ValueError(f"{modality}: observation outside content")
        if np.any(missing & ~observed):
            raise ValueError(f"{modality}: injected mask must remove source-observed rows only")
        source_observed[modality] = _readonly(observed.copy())
        source_zero[modality] = _readonly((content & ~observed).copy())
        injected[modality] = _readonly(missing.copy())
        effective[modality] = _readonly((observed & ~missing).copy())
        assert np.array_equal(content, source_observed[modality] | source_zero[modality])
    return MaskState(_readonly(content.copy()), source_observed, source_zero, injected, effective)


def runs(mask: np.ndarray) -> list[list[int]]:
    """Return maximal true intervals as zero-based, half-open [start,end) pairs."""
    row = np.asarray(mask, dtype=bool)
    if row.shape != (50,):
        raise ValueError("run mask must have shape [50]")
    edges = np.diff(np.pad(row.astype(np.int8), (1, 1)))
    return [[int(start), int(end)] for start, end in zip(np.flatnonzero(edges == 1),
                                                          np.flatnonzero(edges == -1))]


def inspect_attachment3(directory: Path) -> dict:
    spec = importlib.util.spec_from_file_location("q2_mask_loader", SRC / "2026-09-24_q2_data_v1.0.py")
    loader = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loader
    spec.loader.exec_module(loader)
    batch = loader.load_special_directory(directory, 3)
    masks = build_mask_state(batch)
    items = []
    for index, sample_id in enumerate(batch.ids):
        item = {"sample_id": sample_id,
                "content_positions": int(masks.content[index].sum()), "modalities": {}}
        for modality in MODALITIES:
            zero = masks.source_zero[modality][index]
            item["modalities"][modality] = {
                "source_observed_positions": int(masks.source_observed[modality][index].sum()),
                "source_zero_positions": int(zero.sum()),
                "source_zero_runs": runs(zero),
                "injected_missing_positions": 0,
                "effective_observed_positions": int(masks.effective_observed[modality][index].sum()),
            }
        items.append(item)
    return {"split": "attachment3", "samples": len(items), "sequence_length": 50,
            "index_convention": "zero-based half-open [start,end)",
            "mask_semantics": {
                "content": "attention=1 excluding CLS and SEP",
                "source_observed": "content positions with a nonzero feature row; text uses content positions",
                "source_zero": "content positions with an all-zero source feature row",
                "injected_missing": "none in attachment 3; reserved for simulated test conditions",
                "effective_observed": "source_observed AND NOT injected_missing",
            }, "items": items}


def self_test() -> None:
    content = np.zeros((2, 50), dtype=bool)
    content[:, 1:7] = True
    observed = {modality: content.copy() for modality in MODALITIES}
    observed["audio"][0, 3:5] = False
    injected = {"vision": np.zeros_like(content)}
    injected["vision"][0, 4:6] = True
    batch = SimpleNamespace(content_mask=content, observation_masks=observed)
    state = build_mask_state(batch, injected)
    assert runs(state.source_zero["audio"][0]) == [[3, 5]]
    assert runs(state.injected_missing["vision"][0]) == [[4, 6]]
    assert state.effective_observed["audio"][0].sum() == 4
    assert state.effective_observed["vision"][0].sum() == 4
    assert not state.source_zero["audio"][0, 0] and not state.source_zero["audio"][0, 7]
    assert not state.source_zero["vision"].any()
    assert not state.injected_missing["audio"].any()
    assert not state.effective_observed["vision"][0, 4:6].any()
    assert state.source_observed["vision"][0, 4:6].all()
    print("Mask semantics self-test OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--self-test", action="store_true")
    action.add_argument("--attachment3", type=Path, help="Read-only mask inspection of the aligned directory")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        print(json.dumps(inspect_attachment3(args.attachment3), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
