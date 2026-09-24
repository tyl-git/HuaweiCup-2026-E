"""Boundary tests for the aligned competition-data loader."""

from __future__ import annotations

import importlib.util
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


LOADER_PATH = Path(__file__).with_name("2026-09-24_q2_data_v1.0.py")
SPEC = importlib.util.spec_from_file_location("q2_data_v1", LOADER_PATH)
loader = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = loader
SPEC.loader.exec_module(loader)


def sample_tokens(dtype=np.float32):
    tokens = np.zeros((3, 50), dtype=dtype)
    tokens[0, :3] = [101, 999, 102]
    tokens[1, :3] = 1
    return tokens


class LoaderTests(unittest.TestCase):
    def make_batch(self, **changes):
        raw = {
            "text_bert": sample_tokens()[None],
            "audio": np.ones((1, 50, 74), dtype=np.float64),
            "vision": np.zeros((1, 50, 35), dtype=np.float64),
            "classification_labels": np.array([1.0]),
            "regression_labels": np.array([0.0]),
        }
        raw.update(changes)
        return loader._batch(raw, split="train", n=1, ids=("video$_$1",),
                             original_ids=("video$_$1",), raw_text=("test",),
                             paths=(Path("test.pkl"),), hashes={}, labeled=True)

    def test_special_three_preserves_zero_rows_and_has_no_reference_text(self):
        tokens = sample_tokens()
        audio = np.zeros((1, 50, 74), dtype=np.float32)
        audio[0, 1, 0] = 2
        vision = np.zeros((1, 50, 35), dtype=np.float32)
        with tempfile.TemporaryDirectory(dir=LOADER_PATH.parent) as folder:
            path = Path(folder) / "附件3_01.pkl"
            with path.open("wb") as handle:
                pickle.dump({"test": {"text_bert": tokens[None], "audio": audio,
                                      "vision": vision}}, handle)
            batch = loader.load_special(path, 3)
        self.assertIsNone(batch.reference_text)
        self.assertIsNone(batch.original_ids[0])
        self.assertEqual(batch.input_ids.dtype, np.int64)
        self.assertEqual(batch.content_mask[0].nonzero()[0].tolist(), [1])
        self.assertEqual(batch.observation_masks["audio"][0].nonzero()[0].tolist(), [1])
        self.assertFalse(batch.observation_masks["vision"].any())
        self.assertEqual(batch.audio[0, 1, 0], 2)

    def test_fractional_float_token_is_rejected_before_integer_conversion(self):
        tokens = sample_tokens()
        tokens[0, 1] = 999.5
        with tempfile.TemporaryDirectory(dir=LOADER_PATH.parent) as folder:
            path = Path(folder) / "附件3_01.pkl"
            with path.open("wb") as handle:
                pickle.dump({"test": {"text_bert": tokens[None],
                                      "audio": np.zeros((1, 50, 74)),
                                      "vision": np.zeros((1, 50, 35))}}, handle)
            with self.assertRaisesRegex(ValueError, "exact integers"):
                loader.load_special(path, 3)

    def test_special_four_flat_shape_and_padding_text(self):
        reference = np.zeros((50, 768), dtype=np.float32)
        reference[3, :] = 1
        with tempfile.TemporaryDirectory(dir=LOADER_PATH.parent) as folder:
            path = Path(folder) / "13.pkl"
            with path.open("wb") as handle:
                pickle.dump({"id": "13", "raw_text": "test", "text_bert": sample_tokens(np.int64),
                             "text": reference, "audio": np.zeros((50, 74), dtype=np.float64),
                             "vision": np.zeros((50, 35), dtype=np.float64)}, handle)
            batch = loader.load_special(path, 4)
        self.assertEqual(batch.audio.dtype, np.float32)
        self.assertEqual(batch.vision.dtype, np.float32)
        self.assertTrue(batch.nonzero_masks["text"][0, 3])
        self.assertFalse(batch.content_mask[0, 3])
        self.assertFalse(batch.observation_masks["vision"].any())
        self.assertIsNone(batch.classification_labels)

    def test_attention_hole_is_rejected_without_compressing_indices(self):
        tokens = sample_tokens(np.int64)
        tokens[0, :4] = [101, 999, 1000, 102]
        tokens[1, :4] = 1
        tokens[1, 1] = 0
        with self.assertRaisesRegex(ValueError, "continuous prefix"):
            loader._tokens(tokens[None], 1, "test")

    def test_shape_mismatch_is_not_silently_reshaped(self):
        with self.assertRaisesRegex(ValueError, "expected shape"):
            self.make_batch(audio=np.zeros((1, 74, 50)))

    def test_nonfinite_features_are_rejected(self):
        audio = np.zeros((1, 50, 74))
        audio[0, 1, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.make_batch(audio=audio)

    def test_invalid_or_inconsistent_labels_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "classification labels"):
            self.make_batch(classification_labels=np.array([3.0]))
        with self.assertRaisesRegex(ValueError, "contradict"):
            self.make_batch(classification_labels=np.array([2.0]))

    def test_arrays_and_mask_mapping_disallow_accidental_changes(self):
        batch = self.make_batch()
        with self.assertRaises(ValueError):
            batch.audio[0, 1, 0] = 9
        with self.assertRaises(ValueError):
            batch.content_mask[0, 1] = False
        with self.assertRaises(TypeError):
            batch.observation_masks["vision"] = np.ones((1, 50))
        self.assertEqual(batch.audio[0, 0, 0], 1)
        self.assertFalse(batch.observation_masks["audio"][0, 0])
        self.assertEqual(batch.observation_masks["audio"].sum(), 1)

    def test_float_conversion_cannot_hide_observation(self):
        audio = np.zeros((1, 50, 74), dtype=np.float64)
        audio[0, 1, 0] = 1e-300
        with self.assertRaisesRegex(ValueError, "changes nonzero observation"):
            self.make_batch(audio=audio)


if __name__ == "__main__":
    unittest.main()
