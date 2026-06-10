from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.detectors.classical import RandomForestWindowDetector, build_point_features
from src.experiments.run_low_label import _mask_train_labels
from src.experiments.pipeline import evaluate_detector


class SupervisedDetectorTests(unittest.TestCase):
    def test_build_point_features_include_window_stats(self) -> None:
        values = np.asarray([1.0, 2.0, 3.0, 2.0, 1.0])
        features = build_point_features(values, window=3, num_lags=2)
        self.assertEqual(features.shape[0], len(values))
        self.assertGreater(features.shape[1], 5)

    def test_random_forest_sees_different_positive_counts(self) -> None:
        records = make_synthetic_records(seed=99, num_series=6)
        train = records[:4]
        low = _mask_train_labels(train, labeled_fraction=0.25, seed=99)
        high = _mask_train_labels(train, labeled_fraction=1.0, seed=99)

        detector_low = RandomForestWindowDetector(window_size=5, random_state=99)
        detector_high = RandomForestWindowDetector(window_size=5, random_state=99)
        detector_low.fit(low)
        detector_high.fit(high)

        self.assertLess(detector_low.train_positive_count, detector_high.train_positive_count)
        self.assertGreater(detector_low.train_positive_count, 0)

    def test_higher_label_fraction_changes_real_only_metrics(self) -> None:
        records = make_synthetic_records(seed=7, num_series=10)
        config = {
            "seed": 7,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "low_label": {"default_fraction": 0.05},
            "detector": {
                "name": "random_forest_window",
                "window_size": 7,
                "negative_sample_ratio": 3,
                "class_weight": "balanced",
                "params": {"n_estimators": 50, "max_depth": 6, "random_state": 7},
            },
            "evaluation": {"threshold_metric": "point_f1", "event_iou_threshold": 0.1},
        }
        from src.experiments.pipeline import split_records

        train, val, test, _ = split_records(records, config)
        low_train = _mask_train_labels(train, 0.25, 7)
        high_train = _mask_train_labels(train, 1.0, 7)
        low_result = evaluate_detector(config, low_train, val, test, labeled_fraction=0.25, real_train_count=len(low_train))
        high_result = evaluate_detector(config, high_train, val, test, labeled_fraction=1.0, real_train_count=len(high_train))
        self.assertLess(
            low_result["train_label_stats"]["num_real_positive_train_points"],
            high_result["train_label_stats"]["num_real_positive_train_points"],
        )
        self.assertLess(
            low_result["train_label_stats"]["num_positive_train_event_windows"],
            high_result["train_label_stats"]["num_positive_train_event_windows"],
        )


if __name__ == "__main__":
    unittest.main()
