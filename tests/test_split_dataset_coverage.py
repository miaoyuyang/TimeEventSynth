from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.split_builder import _split_items_by_ratio, build_series_split
from src.datasets.tsb_loader import TimeSeriesRecord
from src.experiments.pipeline import (
    balance_records_by_dataset,
    split_records,
    summarize_split_dataset_test_coverage,
    validate_dataset_balanced_test_coverage,
)


def _record(dataset: str, idx: int, *, anomalous: bool) -> TimeSeriesRecord:
    labels = np.array([0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=int) if anomalous else np.zeros(10, dtype=int)
    return TimeSeriesRecord(
        series_id=f"{dataset}/series_{idx}",
        values=np.arange(10, dtype=float),
        labels=labels,
        timestamps=None,
        source_path="test",
        metadata={"dataset_name": dataset, "parent_folder": dataset},
    )


class SplitDatasetCoverageTests(unittest.TestCase):
    def test_stratify_by_dataset_keeps_each_dataset_in_test(self) -> None:
        records = []
        for dataset in ("MSL", "NAB", "TODS"):
            records.extend(
                [
                    _record(dataset, 0, anomalous=True),
                    _record(dataset, 1, anomalous=True),
                    _record(dataset, 2, anomalous=False),
                ]
            )
        split_ids = build_series_split(
            records,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=7,
            stratify_by_dataset=True,
            stratify_by_has_event=True,
        )
        summary = summarize_split_dataset_test_coverage(records, split_ids)
        self.assertEqual(summary["datasets_missing_from_test"], [])
        for dataset in ("MSL", "NAB", "TODS"):
            self.assertGreaterEqual(summary["per_dataset"][dataset]["num_test_timelines"], 1)

    def test_stratify_by_dataset_takes_precedence_over_parent_folder_grouping(self) -> None:
        records = []
        for dataset in ("MSL", "NAB", "TODS"):
            records.extend(
                [
                    _record(dataset, 0, anomalous=True),
                    _record(dataset, 1, anomalous=True),
                    _record(dataset, 2, anomalous=False),
                ]
            )
        split_ids = build_series_split(
            records,
            train_ratio=0.7,
            val_ratio=0.1,
            test_ratio=0.2,
            seed=9,
            stratify_by_dataset=True,
            stratify_by_has_event=True,
            group_by_parent_folder=True,
        )
        summary = summarize_split_dataset_test_coverage(records, split_ids)
        self.assertEqual(summary["datasets_missing_from_test"], [])

    def test_small_balanced_subgroups_still_reach_test(self) -> None:
        records = [
            _record("MSL", 0, anomalous=True),
            _record("MSL", 1, anomalous=False),
            _record("NAB", 0, anomalous=True),
            _record("NAB", 1, anomalous=False),
        ]
        balanced, _ = balance_records_by_dataset(records, max_timelines_per_dataset=2, seed=3)
        config = {
            "seed": 3,
            "split": {"seed": 3, "train_ratio": 0.5, "val_ratio": 0.0, "test_ratio": 0.5, "stratify_by_dataset": True},
            "data": {"train_ratio": 0.5, "dev_ratio": 0.0, "test_ratio": 0.5},
        }
        _, _, _, split_ids = split_records(balanced, config)
        summary = validate_dataset_balanced_test_coverage(balanced, split_ids)
        self.assertEqual(summary["datasets_missing_from_test"], [])

    def test_balance_enforces_min_timelines_for_dataset_balanced_splits(self) -> None:
        records = [_record("MSL", 0, anomalous=True), _record("NAB", 0, anomalous=True)]
        with self.assertRaisesRegex(ValueError, "at least 2 timelines per dataset"):
            balance_records_by_dataset(records, max_timelines_per_dataset=1, min_timelines_per_dataset=2, seed=1)

    def test_split_items_by_ratio_requires_two_series_for_test_holdout(self) -> None:
        import random

        with self.assertRaisesRegex(ValueError, "at least two timelines per dataset"):
            _split_items_by_ratio(
                [("MSL/series_0", True)],
                0.7,
                0.1,
                0.2,
                random.Random(0),
                require_test=True,
            )

    def test_split_records_validates_dataset_balanced_test_coverage(self) -> None:
        records = [
            _record("MSL", 0, anomalous=True),
            _record("MSL", 1, anomalous=False),
            _record("NAB", 0, anomalous=True),
            _record("NAB", 1, anomalous=False),
        ]
        config = {
            "seed": 11,
            "evaluation": {"dataset_balanced": True, "stratify_by_dataset": True},
            "split": {"seed": 11, "train_ratio": 0.5, "val_ratio": 0.0, "test_ratio": 0.5, "stratify_by_dataset": True},
            "data": {"train_ratio": 0.5, "dev_ratio": 0.0, "test_ratio": 0.5},
        }
        _, _, _, split_ids = split_records(records, config)
        summary = summarize_split_dataset_test_coverage(records, split_ids)
        self.assertEqual(summary["datasets_missing_from_test"], [])


if __name__ == "__main__":
    unittest.main()
