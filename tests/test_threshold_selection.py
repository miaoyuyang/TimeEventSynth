from __future__ import annotations

import unittest
import warnings
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.event_metrics import (
    compute_event_metrics,
    scores_to_events,
    tune_threshold_by_event_f1,
)
from src.evaluation.point_metrics import (
    precision_recall_f1_at_threshold,
    select_validation_threshold,
    tune_threshold_by_point_f1,
)
from src.evaluation.thresholding import calibrate_threshold_synthetic_separation


def _spiky_scores(length: int = 200) -> tuple[list[int], list[float]]:
    labels = [0] * length
    labels[80:100] = [1] * 20
    labels[150:165] = [1] * 15
    scores = [0.02] * length
    for idx in range(80, 100):
        scores[idx] = 0.62 + 0.03 * ((idx - 80) % 4)
    for idx in range(150, 165):
        scores[idx] = 0.64 + 0.02 * ((idx - 150) % 3)
    for idx in range(0, length, 3):
        if (idx < 80 or idx >= 100) and (idx < 150 or idx >= 165):
            scores[idx] = 0.55
    return labels, scores


class ThresholdSelectionTests(unittest.TestCase):
    def test_point_f1_threshold_creates_many_short_false_events(self) -> None:
        labels, scores = _spiky_scores()
        grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 201)]
        point_best = tune_threshold_by_point_f1(labels, scores, grid)
        low_threshold_events = compute_event_metrics(
            labels,
            y_score=scores,
            threshold=0.52,
            min_event_length=1,
            merge_gap=0,
        )
        point_events = compute_event_metrics(
            labels,
            y_score=scores,
            threshold=point_best["threshold"],
            min_event_length=1,
            merge_gap=0,
        )
        self.assertGreater(low_threshold_events["event_count_pred"], 10.0)
        self.assertLess(low_threshold_events["event_precision"], 0.5)
        self.assertLessEqual(point_best["threshold"], 0.62)
        self.assertGreaterEqual(point_events["event_count_pred"], 1.0)

    def test_event_f1_threshold_selects_cleaner_threshold(self) -> None:
        labels, scores = _spiky_scores()
        grid = [round(x, 3) for x in np.linspace(0.0, 1.0, 201)]
        point_best = tune_threshold_by_point_f1(labels, scores, grid)
        event_best = tune_threshold_by_event_f1(
            scores,
            labels,
            grid,
            iou_threshold=0.1,
            min_event_length=3,
            merge_gap=2,
        )
        point_events = compute_event_metrics(
            labels,
            y_score=scores,
            threshold=point_best["threshold"],
            min_event_length=3,
            merge_gap=2,
        )
        event_events = compute_event_metrics(
            labels,
            y_score=scores,
            threshold=event_best["threshold"],
            min_event_length=3,
            merge_gap=2,
        )
        self.assertGreater(event_best["threshold"], point_best["threshold"])
        self.assertGreaterEqual(event_events["event_precision"], point_events["event_precision"])
        self.assertLessEqual(event_events["event_count_pred"], point_events["event_count_pred"])

    def test_min_event_length_removes_short_false_positives(self) -> None:
        scores = [0.0, 0.9, 0.0, 0.9, 0.9, 0.9, 0.0]
        labels = [0, 0, 0, 1, 1, 1, 0]
        short_events = scores_to_events(scores, threshold=0.5, min_event_length=1, merge_gap=0)
        long_events = scores_to_events(scores, threshold=0.5, min_event_length=3, merge_gap=0)
        self.assertEqual(short_events, [(1, 2), (3, 6)])
        self.assertEqual(long_events, [(3, 6)])

    def test_merge_gap_merges_nearby_event_fragments(self) -> None:
        scores = [0.9, 0.9, 0.0, 0.9, 0.9, 0.0]
        labels = [1, 1, 0, 1, 1, 0]
        separate = scores_to_events(scores, threshold=0.5, min_event_length=1, merge_gap=0)
        merged = scores_to_events(scores, threshold=0.5, min_event_length=1, merge_gap=1)
        self.assertEqual(separate, [(0, 2), (3, 5)])
        self.assertEqual(merged, [(0, 5)])

    def test_select_validation_threshold_uses_event_mode(self) -> None:
        labels, scores = _spiky_scores()
        eval_cfg = {
            "threshold_selection": "event_f1",
            "event_iou_threshold": 0.1,
            "threshold_grid_size": 101,
            "prediction_smoothing": {
                "enabled": True,
                "min_event_length": 3,
                "merge_gap": 2,
            },
        }
        selected = select_validation_threshold(labels, scores, eval_cfg)
        self.assertEqual(selected["mode"], "event_f1")
        self.assertIn("validation_debug", selected)
        self.assertIn("best_point_f1", selected["validation_debug"])
        self.assertIn("best_event_f1", selected["validation_debug"])
        self.assertGreaterEqual(selected["event_f1"], selected["validation_debug"]["best_event_f1"]["event_f1"] - 1e-9)

    def test_point_metrics_do_not_warn_when_val_labels_are_all_normal(self) -> None:
        labels = [0] * 50
        scores = [0.1 + 0.01 * (idx % 5) for idx in range(50)]
        grid = [0.05, 0.1, 0.15, 0.2]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tune_threshold_by_point_f1(labels, scores, grid)
            precision_recall_f1_at_threshold(labels, scores, threshold=0.12)
            calibrate_threshold_synthetic_separation(
                normal_scores=scores,
                synthetic_anomaly_scores=[0.8, 0.85, 0.9],
            )
        sklearn_warnings = [
            item
            for item in caught
            if issubclass(item.category, UserWarning) and "classes not in y_true" in str(item.message)
        ]
        self.assertEqual(sklearn_warnings, [])


if __name__ == "__main__":
    unittest.main()
