from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.compatibility_summary import build_compatibility_summary
from src.synthesis.dataset_identity import count_dataset_pairs, resolve_dataset_name
from src.synthesis.donor_policy_sanity import DonorPolicySanityError, validate_donor_policy_pair_counts
from src.synthesis.donor_retrieval import EventWindow
from src.synthesis.donor_selection import (
    _apply_donor_policy_candidate_filter,
    _restrict_cross_dataset,
    build_timeline_feature_table,
    select_donors_for_target,
)
from src.synthesis.synthetic_audit import build_donor_pair_audit_rows, flatten_audit_row
from src.datasets.synthetic_data import make_synthetic_records
from src.datasets.tsb_loader import TimeSeriesRecord


def _feature_table_three_domains() -> dict[str, dict[str, Any]]:
    records = make_synthetic_records(seed=0, num_series=9, events_per_series=2)
    return build_timeline_feature_table(records, synthesis_cfg={})


class CrossDatasetDonorPolicyTests(unittest.TestCase):
    def test_resolve_dataset_name_per_domain(self) -> None:
        records = make_synthetic_records(seed=1, num_series=6, events_per_series=1)
        names = {resolve_dataset_name(r) for r in records}
        self.assertGreaterEqual(len(names), 3)

    def test_cross_dataset_filter_excludes_same_domain(self) -> None:
        table = _feature_table_three_domains()
        target = "domain_a/series_000"
        candidates = sorted(table.keys())
        filtered = _apply_donor_policy_candidate_filter(
            target, candidates, table, "cross_dataset_compatible"
        )
        self.assertNotIn("domain_a/series_003", filtered)
        self.assertTrue(any(sid.startswith("domain_b/") or sid.startswith("domain_c/") for sid in filtered))

    def test_same_dataset_only_keeps_same_domain(self) -> None:
        table = _feature_table_three_domains()
        target = "domain_b/series_001"
        candidates = sorted(table.keys())
        filtered = _apply_donor_policy_candidate_filter(target, candidates, table, "same_dataset_only")
        self.assertTrue(all(table[sid]["dataset_name"] == "domain_b" for sid in filtered))

    def test_cross_dataset_policy_raises_when_single_domain(self) -> None:
        single = [
            TimeSeriesRecord(
                series_id="only/series_0",
                values=np.sin(np.linspace(0, 1, 20)),
                labels=np.zeros(20, dtype=int),
                timestamps=None,
                source_path="synthetic",
                metadata={"parent_folder": "only"},
            )
        ]
        table = build_timeline_feature_table(single)
        counts = count_dataset_pairs("only/series_0", ["only/series_0"], table)
        with self.assertRaises(DonorPolicySanityError):
            validate_donor_policy_pair_counts(
                "cross_dataset_compatible",
                same_dataset_pairs_considered=counts["same_dataset_pairs_considered"],
                cross_dataset_pairs_considered=counts["cross_dataset_pairs_considered"],
            )

    def test_compatibility_summary_per_policy_cross_counts(self) -> None:
        audit_rows = [
            {
                "donor_policy": "cross_dataset_compatible",
                "record_type": "donor_pair",
                "target_series_id": "domain_a/s0",
                "source_series_id": "domain_b/s1",
                "target_dataset": "domain_a",
                "source_dataset": "domain_b",
                "same_dataset": False,
                "compatibility_enabled": True,
                "compatibility_score": 0.8,
                "accepted": True,
                "rejection_stage": "compatibility",
            },
            {
                "donor_policy": "cross_dataset_compatible",
                "record_type": "donor_pair",
                "target_series_id": "domain_a/s0",
                "source_series_id": "domain_c/s2",
                "target_dataset": "domain_a",
                "source_dataset": "domain_c",
                "same_dataset": False,
                "compatibility_enabled": True,
                "compatibility_score": 0.4,
                "accepted": False,
                "rejection_stage": "compatibility",
                "rejection_reason": "below_compatibility_threshold",
            },
        ]
        summary = build_compatibility_summary(audit_rows)
        policy = summary["policies"]["cross_dataset_compatible"]
        self.assertEqual(policy["cross_dataset_pairs_considered"], 2)
        self.assertEqual(policy["same_dataset_pairs_considered"], 0)
        self.assertEqual(policy["cross_dataset_kept"], 1)
        self.assertEqual(policy["cross_dataset_rejected"], 1)

    def test_donor_pair_audit_same_dataset_false_for_cross_policy(self) -> None:
        table = _feature_table_three_domains()
        window = EventWindow(
            series_id="domain_a/series_000",
            start=0,
            end=5,
            values=np.zeros(5),
            label=1,
            metadata={},
        )
        records = [
            {
                "source_series_id": "domain_b/series_003",
                "final_decision": "rejected",
                "rejection_reason": "below_min_score:shape_similarity",
                "compatibility_score": 0.2,
                "shape_similarity": 0.2,
                "amplitude_compatibility": 0.2,
                "duration_compatibility": 0.2,
                "context_similarity": 0.2,
                "frequency_similarity": 0.2,
                "trend_similarity": 0.2,
            }
        ]
        rows = build_donor_pair_audit_rows(
            window,
            method_name="test",
            synthesis_method="learned_prototype_event_time",
            labeled_fraction=0.2,
            donor_policy="cross_dataset_compatible",
            filter_method="none",
            feature_table=table,
            compatibility_records=records,
            compatibility_enabled=True,
        )
        self.assertEqual(len(rows), 1)
        self.assertFalse(rows[0]["same_dataset"])
        flat = flatten_audit_row(rows[0])
        self.assertFalse(flat["same_dataset"])

    def test_select_donors_cross_dataset_only(self) -> None:
        records = make_synthetic_records(seed=2, num_series=9, events_per_series=3)
        table = build_timeline_feature_table(records)
        target_windows = [
            w
            for w in __import__(
                "src.synthesis.donor_retrieval", fromlist=["collect_event_windows"]
            ).collect_event_windows(records)
            if w.series_id == "domain_a/series_000"
        ]
        donor_windows = __import__(
            "src.synthesis.donor_retrieval", fromlist=["collect_event_windows"]
        ).collect_event_windows(records)
        target = target_windows[0]
        cfg = {
            "enabled": True,
            "donor_policy": "cross_dataset_compatible",
            "min_score": 0.0,
            "top_quantile": 0.5,
            "require_cross_dataset": True,
        }
        donors, _, audit = select_donors_for_target(
            target,
            donor_windows,
            feature_table=table,
            compatibility_cfg=cfg,
            top_k=2,
        )
        self.assertGreater(audit["cross_dataset_pairs_considered"], 0)
        self.assertEqual(audit["same_dataset_pairs_considered"], 0)
        for donor in donors:
            self.assertNotEqual(
                table[target.series_id]["dataset_name"],
                table[donor.series_id]["dataset_name"],
            )


if __name__ == "__main__":
    unittest.main()
