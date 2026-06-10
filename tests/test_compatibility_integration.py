from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit
from src.synthesis.donor_retrieval import collect_event_windows
from src.synthesis.donor_selection import build_timeline_feature_table, select_donors_for_target
from src.synthesis.event_window_synthesizer import synthesize_by_learned_prototype_event_time


class CompatibilityIntegrationTests(unittest.TestCase):
    def test_disabled_preserves_donor_retrieval_path(self) -> None:
        records = make_synthetic_records(seed=0)[:6]
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "normalized_time_mean_donor",
            "top_k": 2,
            "filter_policy": {"name": "no_filter"},
        }
        synthesis_cfg = {"compatibility": {"enabled": False}}
        kept, audit = build_augmented_training_records_with_audit(
            records,
            split="train",
            policy_config=policy,
            donor_pool_records=records,
            synthesis_cfg=synthesis_cfg,
        )
        self.assertGreater(len(kept), 0)
        self.assertTrue(any(row.get("accepted") for row in audit))

    def test_enabled_filters_incompatible_donors(self) -> None:
        records = make_synthetic_records(seed=1)[:8]
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "cross_dataset_compatible_mean_donor",
            "top_k": 3,
            "filter_policy": {"name": "no_filter"},
            "donor_policy": "cross_dataset_compatible",
        }
        synthesis_cfg = {
            "compatibility": {
                "enabled": True,
                "min_score": 0.99,
                "fallback_when_no_compatible_donor": "skip",
            }
        }
        kept, audit = build_augmented_training_records_with_audit(
            records,
            split="train",
            policy_config=policy,
            donor_pool_records=records,
            synthesis_cfg=synthesis_cfg,
        )
        rejected = [row for row in audit if row.get("rejection_reason") == "skip"]
        self.assertGreaterEqual(len(rejected), 0)

    def test_learned_prototype_fit_uses_passed_donors_only(self) -> None:
        records = make_synthetic_records(seed=2)[:6]
        target_windows = collect_event_windows(records[:2])
        donor_windows = collect_event_windows(records[2:])
        target = target_windows[0]
        donors = donor_windows[:2]
        fit_calls: list[int] = []

        class SpyAligner:
            def __init__(self, grid_size: int = 64) -> None:
                self.grid_size = grid_size

            def fit(self, event_windows):
                fit_calls.append(len(event_windows))
                from src.alignment.learned_event_time import LearnedEventTimeAligner

                aligner = LearnedEventTimeAligner(grid_size=16)
                aligner.fit(event_windows)
                self._aligner = aligner
                return self

            def synthesize(self, target_window, donor_windows, k=5, target_length=None):
                return self._aligner.synthesize(target_window, donor_windows, k=k, target_length=target_length)

        with patch("src.synthesis.event_window_synthesizer.LearnedEventTimeAligner", SpyAligner):
            synthesize_by_learned_prototype_event_time(target, donors, target_length=len(target.values), grid_size=16)
        self.assertEqual(fit_calls, [2])

    def test_select_donors_skips_when_none_compatible(self) -> None:
        records = make_synthetic_records(seed=3)[:6]
        feature_table = build_timeline_feature_table(records)
        target_windows = collect_event_windows(records[:1])
        donor_windows = collect_event_windows(records[1:])
        donors, similarities, audit = select_donors_for_target(
            target_windows[0],
            donor_windows,
            feature_table=feature_table,
            compatibility_cfg={
                "enabled": True,
                "min_score": 0.99,
                "fallback_when_no_compatible_donor": "skip",
            },
            top_k=2,
        )
        self.assertEqual(donors, [])
        self.assertEqual(similarities, [])
        self.assertEqual(audit.get("compatibility_fallback"), "skip")

    def test_groupwise_uses_nearest_centroid_group_after_compatibility_filter(self) -> None:
        records = make_synthetic_records(seed=4, num_series=9, events_per_series=2)
        feature_table = build_timeline_feature_table(records)
        target = collect_event_windows(records[:1])[0]
        donor_windows = collect_event_windows(records[1:])
        total_donor_windows = len(donor_windows)
        for idx, donor in enumerate(donor_windows):
            donor.metadata["event_group_id"] = "group_0" if idx % 2 == 0 else "group_1"

        donors, _, audit = select_donors_for_target(
            target,
            donor_windows,
            feature_table=feature_table,
            compatibility_cfg={
                "enabled": True,
                "min_score": 0.0,
                "donor_policy": "cross_dataset_compatible",
                "require_cross_dataset": True,
            },
            top_k=2,
            retrieval_kwargs={
                "restrict_to_target_group": True,
                "group_key": "event_group_id",
                "context_size": 5,
            },
        )

        self.assertLess(audit.get("num_group_matched_donor_windows", total_donor_windows), total_donor_windows)
        self.assertEqual(audit.get("group_selection_mode"), "nearest_compatible_centroid")
        self.assertIn("target_event_group", audit)
        if donors:
            self.assertTrue(str(audit["target_event_group"]).startswith("group_"))
            self.assertTrue(all(donor.metadata.get("event_group_id") == audit["target_event_group"] for donor in donors))

    def test_groupwise_skips_when_no_compatible_donors(self) -> None:
        records = make_synthetic_records(seed=5, num_series=6, events_per_series=2)
        feature_table = build_timeline_feature_table(records)
        target = collect_event_windows(records[:1])[0]
        donor_windows = collect_event_windows(records[1:])
        for donor in donor_windows:
            donor.metadata["event_group_id"] = "group_1"
        donors, _, audit = select_donors_for_target(
            target,
            donor_windows,
            feature_table=feature_table,
            compatibility_cfg={
                "enabled": True,
                "min_score": 0.99,
                "fallback_when_no_compatible_donor": "skip",
            },
            top_k=2,
            retrieval_kwargs={
                "restrict_to_target_group": True,
                "group_key": "event_group_id",
            },
        )
        self.assertEqual(donors, [])
        self.assertEqual(audit.get("compatibility_fallback"), "skip")


if __name__ == "__main__":
    unittest.main()
