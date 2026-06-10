from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.normalized_time import assign_normalized_event_time, denormalize_event_window, normalize_event_window, resample_window


class NormalizedTimeTests(unittest.TestCase):
    def test_assigns_position_and_bucket(self) -> None:
        events = [{"event_id": "e1", "local_time_center": 0.55}]
        aligned = assign_normalized_event_time(events, 10)
        self.assertAlmostEqual(aligned[0]["universal_position"], 0.55)
        self.assertEqual(aligned[0]["universal_bucket"], 5)

    def test_resample_window_shape(self) -> None:
        values = [1.0, 2.0, 3.0]
        resampled = resample_window(values, 5)
        self.assertEqual(resampled.shape, (5,))
        multivariate = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        resampled_mv = resample_window(multivariate, 6)
        self.assertEqual(resampled_mv.shape, (6, 2))

    def test_normalize_and_denormalize(self) -> None:
        values = [1.0, 3.0, 2.0, 4.0]
        normalized = normalize_event_window(values, grid_size=8)
        restored = denormalize_event_window(normalized, target_length=4)
        self.assertEqual(normalized.shape, (8,))
        self.assertEqual(restored.shape, (4,))


if __name__ == "__main__":
    unittest.main()
