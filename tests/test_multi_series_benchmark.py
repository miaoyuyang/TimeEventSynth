from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_stats import build_experiment_dataset_stats
from src.datasets.load_options import DatasetLoadOptions
from src.datasets.split_builder import build_series_split
from src.datasets.synthetic_data import make_synthetic_records
from src.datasets.tsb_loader import TimeSeriesRecord, filter_records
from src.experiments.config import load_config


class MultiSeriesBenchmarkTests(unittest.TestCase):
    def test_synthetic_split_has_multiple_test_series(self) -> None:
        records = make_synthetic_records(seed=7, num_series=30)
        split_ids = build_series_split(
            records,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=7,
            stratify_by_has_event=True,
            group_by_parent_folder=True,
        )
        self.assertGreaterEqual(len(split_ids["test"]), 5)
        stats = build_experiment_dataset_stats(records, split_ids)
        self.assertEqual(stats["num_series_after_filter"], 30)
        self.assertGreaterEqual(stats["num_test_series"], 5)
        self.assertGreaterEqual(stats["num_test_event_windows"], 20)
        self.assertGreaterEqual(len(stats["parent_folders"]), 2)

    def test_group_aware_split_keeps_parent_folders_together(self) -> None:
        records = [
            TimeSeriesRecord(
                series_id=f"ECG/series_{idx}",
                values=np.arange(10, dtype=float),
                labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
                timestamps=None,
                source_path="x",
                metadata={"parent_folder": "ECG"},
            )
            for idx in range(4)
        ] + [
            TimeSeriesRecord(
                series_id=f"NASA/series_{idx}",
                values=np.arange(10, dtype=float),
                labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
                timestamps=None,
                source_path="y",
                metadata={"parent_folder": "NASA"},
            )
            for idx in range(4)
        ]
        split_ids = build_series_split(
            records,
            train_ratio=0.5,
            val_ratio=0.25,
            test_ratio=0.25,
            seed=1,
            group_by_parent_folder=True,
        )
        for split_name in ("train", "val", "test"):
            parents = {record.metadata["parent_folder"] for record in records if record.series_id in split_ids[split_name]}
            for parent in parents:
                parent_series = {record.series_id for record in records if record.metadata["parent_folder"] == parent}
                split_series = set(split_ids[split_name]) & parent_series
                self.assertTrue(split_series == parent_series or not split_series)

    def test_filter_include_patterns(self) -> None:
        records = [
            TimeSeriesRecord("ECG/a", np.ones(10), np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]), None, "a", {}),
            TimeSeriesRecord("NASA/b", np.ones(10), np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]), None, "b", {}),
        ]
        filtered, summary = filter_records(records, DatasetLoadOptions(include_patterns=["ECG/*"], min_event_windows=1))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].series_id, "ECG/a")
        self.assertEqual(summary["dropped_pattern_mismatch"], 1)

    def test_filter_patterns_and_event_ratio(self) -> None:
        records = [
            TimeSeriesRecord("ECG/a", np.ones(10), np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]), None, "a", {}),
            TimeSeriesRecord("ECG/b", np.ones(10), np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1]), None, "b", {}),
            TimeSeriesRecord("ECG/c", np.ones(10), np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), None, "c", {}),
        ]
        filtered, summary = filter_records(
            records,
            DatasetLoadOptions(
                max_event_ratio=0.5,
                min_event_windows=1,
            ),
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0].series_id, "ECG/a")
        self.assertEqual(summary["dropped_high_event_ratio"], 1)
        self.assertEqual(summary["dropped_few_events"], 1)

    def test_config_normalizes_dataset_filters(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
        self.assertEqual(config["dataset"]["max_series"], 50)
        self.assertEqual(config["split"]["group_by_parent_folder"], True)
        self.assertIn("MSL/*", config["dataset"]["include_patterns"])

    def test_max_series_subsample_spans_parent_folders(self) -> None:
        records = [
            TimeSeriesRecord(
                series_id=f"ECG/series_{idx}",
                values=np.ones(10),
                labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
                timestamps=None,
                source_path="a",
                metadata={"parent_folder": "ECG"},
            )
            for idx in range(20)
        ] + [
            TimeSeriesRecord(
                series_id=f"NASA/series_{idx}",
                values=np.ones(10),
                labels=np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0]),
                timestamps=None,
                source_path="b",
                metadata={"parent_folder": "NASA"},
            )
            for idx in range(20)
        ]
        filtered, _ = filter_records(
            records,
            DatasetLoadOptions(max_series=10, min_event_windows=1, group_by_parent_folder=True, seed=1),
        )
        parents = {record.metadata["parent_folder"] for record in filtered}
        self.assertEqual(len(filtered), 10)
        self.assertEqual(parents, {"ECG", "NASA"})

    def test_tsb_real_smoke_config_inherits_public_v2_filters(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "experiment_cross_dataset_compatibility_tsb_real_smoke.yaml")
        include = config["dataset"]["include_patterns"]
        self.assertIn("MSL/*", include)
        self.assertIn("NAB/*", include)
        self.assertNotEqual(include, [])


if __name__ == "__main__":
    unittest.main()
