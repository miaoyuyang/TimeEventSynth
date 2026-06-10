from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.policies import build_augmentation_result
from src.datasets.tsb_loader import TimeSeriesRecord
from src.synthesis.augment_dataset import (
    DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER,
    records_from_oversampled_events,
    resolve_oversample_target_multiplier,
)


class OversamplingTests(unittest.TestCase):
    def _anomaly_record(self, series_id: str = "s1", length: int = 5) -> TimeSeriesRecord:
        return TimeSeriesRecord(
            series_id=series_id,
            values=np.ones(length),
            labels=np.array([0, 1, 1, 0, 0][:length]),
            timestamps=None,
            source_path="test",
            metadata={},
        )

    def test_only_synthetic_copies_are_materialized(self) -> None:
        train = [self._anomaly_record("a"), self._anomaly_record("b")]
        appended = records_from_oversampled_events(train, target_multiplier=2.0)
        self.assertTrue(appended)
        self.assertTrue(all(record.metadata.get("synthetic") for record in appended))
        self.assertTrue(all("__oversampled__" in record.series_id for record in appended))

    def test_oversampled_records_include_context_metadata(self) -> None:
        train = [self._anomaly_record("a")]
        appended = records_from_oversampled_events(train, target_multiplier=2.0)
        self.assertTrue(appended)
        metadata = appended[0].metadata
        self.assertEqual(metadata.get("target_series"), "a")
        self.assertEqual(metadata.get("target_event_interval"), [1, 3])
        self.assertEqual(len(metadata.get("target_series_values", [])), 5)
        self.assertEqual(len(metadata.get("target_series_labels", [])), 5)

    def test_resolve_multiplier_from_config(self) -> None:
        config = {"synthesis": {"random_oversample_target_multiplier": 3.5}}
        self.assertEqual(resolve_oversample_target_multiplier(config), 3.5)
        self.assertEqual(resolve_oversample_target_multiplier(None), DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER)

    def test_higher_multiplier_yields_at_least_as_many_copies(self) -> None:
        train = [self._anomaly_record("a"), self._anomaly_record("b")]
        low = records_from_oversampled_events(train, target_multiplier=1.0)
        high = records_from_oversampled_events(train, target_multiplier=3.0)
        self.assertGreaterEqual(len(high), len(low))

    def test_policy_path_matches_shared_helper(self) -> None:
        train = [self._anomaly_record("a"), self._anomaly_record("b")]
        config = {"synthesis": {"random_oversample_target_multiplier": 2.0}}
        direct = records_from_oversampled_events(train, config=config)
        policy = build_augmentation_result(
            train,
            donor_records=[],
            policy_name="random_event_oversampling",
            config=config,
            labeled_fraction=0.2,
        )
        self.assertEqual(len(policy.synthetic_windows), len(direct))
        self.assertEqual(
            [record.series_id for record in policy.synthetic_windows],
            [record.series_id for record in direct],
        )


if __name__ == "__main__":
    unittest.main()
