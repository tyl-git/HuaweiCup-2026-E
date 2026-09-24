"""Focused tests for deterministic Q2 missing-modality interval ablation."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("2026-09-24_q2-missing-intervals_v1.0.py")
SPEC = importlib.util.spec_from_file_location("q2_missing_intervals", MODULE_PATH)
assert SPEC and SPEC.loader
missing = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = missing
SPEC.loader.exec_module(missing)


class MissingIntervalTests(unittest.TestCase):
    def setUp(self):
        self.ids = ("clip-a", "clip-b")
        self.content = np.zeros((2, 50), dtype=bool)
        self.content[0, 1:11] = True
        self.content[1, 1:8] = True
        self.observed = {
            "text": self.content.copy(),
            "audio": self.content.copy(),
            "vision": self.content.copy(),
        }
        # A naturally absent audio row must not become part of a removal count.
        self.observed["audio"][0, 5] = False
        self.features = {
            modality: np.full((2, 50, dimension), float(index + 1), dtype=np.float32)
            for index, (modality, dimension) in enumerate(
                (("text", 3), ("audio", 2), ("vision", 4))
            )
        }

    def test_contiguous_content_interval_and_actual_ratio(self):
        plan = missing.make_interval_plan(
            self.content, self.observed, self.ids,
            modalities=("audio",), fraction=0.4, position="middle", seed=11,
        )
        spans = plan.intervals["audio"]
        self.assertEqual(tuple(spans[0]), (3, 8))
        self.assertTrue(np.all(self.content[0, spans[0, 0]:spans[0, 1]]))
        self.assertEqual(plan.removed_counts["audio"], 7)  # ceil(9 * .4) + ceil(7 * .4)
        self.assertEqual(plan.originally_observed["audio"], 16)  # 9 + 7
        self.assertAlmostEqual(plan.summary()["per_modality"]["audio"]["actual_fraction"], 7 / 16)

    def test_apply_zeroes_features_and_updates_observation_masks(self):
        plan = missing.make_interval_plan(
            self.content, self.observed, self.ids,
            modalities=("text", "audio"), fraction=0.3, position="start", seed=1,
        )
        original_features = {key: value.copy() for key, value in self.features.items()}
        original_masks = {key: value.copy() for key, value in self.observed.items()}
        masked = missing.apply_interval_plan(
            self.features, self.content, self.observed, self.ids, plan,
        )
        for modality in ("text", "audio"):
            removed = plan.removed_masks[modality]
            self.assertTrue(np.all(masked.features[modality][removed] == 0))
            self.assertTrue(np.all(~masked.observation_masks[modality][removed]))
            self.assertTrue(np.array_equal(
                masked.observation_masks[modality], self.observed[modality] & ~removed,
            ))
        self.assertTrue(np.array_equal(masked.features["vision"], self.features["vision"]))
        self.assertTrue(np.array_equal(masked.observation_masks["vision"], self.observed["vision"]))
        for key in self.features:
            self.assertTrue(np.array_equal(self.features[key], original_features[key]))
            self.assertTrue(np.array_equal(self.observed[key], original_masks[key]))
        with self.assertRaises(ValueError):
            masked.features["audio"][0, 0, 0] = 8

    def test_random_plan_is_reproducible_and_batch_order_invariant(self):
        first = missing.make_interval_plan(
            self.content, self.observed, self.ids,
            modalities=("text", "audio", "vision"), fraction=0.5,
            position="random", seed=2026,
        )
        repeat = missing.make_interval_plan(
            self.content, self.observed, self.ids,
            modalities=("text", "audio", "vision"), fraction=0.5,
            position="random", seed=2026,
        )
        for modality in first.modalities:
            self.assertTrue(np.array_equal(first.removed_masks[modality], repeat.removed_masks[modality]))
        reverse = missing.make_interval_plan(
            self.content[::-1], {key: value[::-1] for key, value in self.observed.items()}, self.ids[::-1],
            modalities=("text", "audio", "vision"), fraction=0.5,
            position="random", seed=2026,
        )
        for modality in first.modalities:
            by_id = {sample_id: mask for sample_id, mask in zip(self.ids, first.removed_masks[modality])}
            for sample_id, mask in zip(self.ids[::-1], reverse.removed_masks[modality]):
                self.assertTrue(np.array_equal(by_id[sample_id], mask))

    def test_empty_original_observation_is_retained_without_fake_removal(self):
        observed = {key: value.copy() for key, value in self.observed.items()}
        observed["vision"][:] = False
        plan = missing.make_interval_plan(
            self.content, observed, self.ids,
            modalities=("vision",), fraction=0.8, position="end", seed=0,
        )
        self.assertEqual(plan.removed_counts["vision"], 0)
        self.assertEqual(plan.originally_observed["vision"], 0)
        self.assertTrue(np.all(plan.intervals["vision"] == -1))

    def test_rejects_observation_outside_content(self):
        observed = {key: value.copy() for key, value in self.observed.items()}
        observed["vision"][0, 20] = True
        with self.assertRaisesRegex(ValueError, "outside content"):
            missing.make_interval_plan(
                self.content, observed, self.ids,
                modalities=("vision",), fraction=0.2, position="start", seed=0,
            )

    def test_rejects_plan_applied_to_changed_original_observation(self):
        plan = missing.make_interval_plan(
            self.content, self.observed, self.ids,
            modalities=("audio",), fraction=0.2, position="start", seed=0,
        )
        changed = {key: value.copy() for key, value in self.observed.items()}
        changed["audio"][0, 5] = True
        with self.assertRaisesRegex(ValueError, "incompatible"):
            missing.apply_interval_plan(self.features, self.content, changed, self.ids, plan)

    def test_rejects_noncontiguous_content(self):
        content = self.content.copy()
        content[0, 5] = False
        with self.assertRaisesRegex(ValueError, "contiguous"):
            missing.make_interval_plan(
                content, self.observed, self.ids,
                modalities=("text",), fraction=0.2, position="middle", seed=0,
            )


if __name__ == "__main__":
    unittest.main()
