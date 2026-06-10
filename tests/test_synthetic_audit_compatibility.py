from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.experiments.audit_sanity import validate_synthetic_audit_csv
from src.experiments.synthesis_runner import build_rejection_summary
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit
from src.synthesis.synthetic_audit import AUDIT_CSV_COLUMNS, build_donor_rejection_audit_rows, flatten_audit_row


def _train_records():
    records = make_synthetic_records(seed=7, num_series=8, events_per_series=3)
    for record in records:
        record.labels = record.labels.copy()
        record.labels[record.labels > 0] = 1
    return records


class SyntheticAuditCompatibilityTests(unittest.TestCase):
    REQUIRED_COMPAT_COLUMNS = {
        "compatibility_enabled",
        "compatibility_score",
        "compatibility_decision",
        "compatibility_rejection_reason",
        "rejection_stage",
        "target_series_id",
        "source_series_id",
        "donor_policy",
    }

    def test_flatten_includes_compatibility_columns_when_disabled(self) -> None:
        train = _train_records()
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "normalized_time_mean_donor",
            "labeled_fraction": 0.5,
            "top_k": 2,
            "grid_size": 16,
            "filter_policy": {"name": "no_filter"},
        }
        _, audit_rows = build_augmented_training_records_with_audit(
            train,
            split="train",
            policy_config=policy,
            donor_pool_records=train,
            synthesis_cfg={"compatibility": {"enabled": False}, "filter_policy": "none"},
        )
        flat = flatten_audit_row(audit_rows[0])
        for column in self.REQUIRED_COMPAT_COLUMNS:
            self.assertIn(column, flat)
        self.assertFalse(flat["compatibility_enabled"])
        self.assertEqual(flat["compatibility_decision"], "not_applied")

    def test_rejected_donors_appear_when_compatibility_enabled(self) -> None:
        train = _train_records()
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "compat_audit_test",
            "labeled_fraction": 0.5,
            "top_k": 2,
            "grid_size": 16,
            "filter_policy": {"name": "no_filter"},
            "donor_policy": "cross_dataset_compatible",
        }
        synthesis_cfg = {
            "compatibility": {
                "enabled": True,
                "min_score": 0.0,
                "top_quantile": 0.5,
            },
            "filter_policy": "none",
        }
        _, audit_rows = build_augmented_training_records_with_audit(
            train,
            split="train",
            policy_config=policy,
            donor_pool_records=train,
            synthesis_cfg=synthesis_cfg,
        )
        donor_rejections = [row for row in audit_rows if row.get("record_type") in {"donor_pair", "donor_rejection"} and not row.get("accepted")]
        self.assertGreater(len(donor_rejections), 0)
        self.assertTrue(all(not row["accepted"] for row in donor_rejections))
        self.assertTrue(all(row["rejection_stage"] == "compatibility" for row in donor_rejections))

        flat_rows = [flatten_audit_row(row) for row in audit_rows]
        validate_synthetic_audit_csv(flat_rows, config={"synthesis": synthesis_cfg}, synthesis_requested=True)
        for column in AUDIT_CSV_COLUMNS:
            self.assertIn(column, flat_rows[0])

    def test_rejection_summary_includes_compatibility_counts(self) -> None:
        train = _train_records()
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "compat_summary_test",
            "top_k": 2,
            "grid_size": 16,
            "filter_policy": {"name": "no_filter"},
            "donor_policy": "compatibility_strict",
        }
        _, audit_rows = build_augmented_training_records_with_audit(
            train,
            split="train",
            policy_config=policy,
            donor_pool_records=train,
            synthesis_cfg={
                "compatibility": {"enabled": True, "min_score": 0.0, "top_quantile": 0.5},
                "filter_policy": "none",
            },
        )
        summary = build_rejection_summary([flatten_audit_row(row) for row in audit_rows])
        entry = summary["compat_summary_test"]
        self.assertIn("counts", entry)
        counts = entry["counts"]
        rejection_total = sum(value for key, value in counts.items() if key != "kept")
        self.assertGreater(rejection_total, 0)
        self.assertIn("rejected_incompatible_donor", counts)

    def test_flatten_missing_optional_fields_does_not_crash(self) -> None:
        row = {
            "candidate_id": "a:b:c:test",
            "method": "test",
            "target_start": 0,
            "target_end": 5,
            "target_length": 5,
            "compatibility_enabled": False,
            "accepted": False,
            "rejection_reason": "synthesis_failed",
            "rejection_stage": "synthesis",
        }
        flat = flatten_audit_row(row)
        self.assertFalse(flat["accepted"])
        self.assertTrue(np.isnan(float(flat["compatibility_score"])))

    def test_build_donor_rejection_rows_standalone(self) -> None:
        from src.synthesis.donor_retrieval import collect_event_windows

        train = _train_records()
        windows = collect_event_windows(train)
        target = windows[0]
        records = [
            {
                "source_series_id": "donor_a",
                "final_decision": "rejected",
                "rejection_reason": "cross_dataset_disallowed",
                "compatibility_score": 0.0,
                "shape_similarity": 0.0,
                "amplitude_compatibility": 0.0,
                "duration_compatibility": 0.0,
                "context_similarity": 0.0,
                "frequency_similarity": 0.0,
                "trend_similarity": 0.0,
            }
        ]
        rows = build_donor_rejection_audit_rows(
            target,
            method_name="m",
            synthesis_method="normalized_time_mean_donor",
            labeled_fraction=0.2,
            donor_policy="cross_dataset_compatible",
            filter_method="no_filter",
            feature_table={target.series_id: {"dataset": "a"}, "donor_a": {"dataset": "b"}},
            compatibility_records=records,
            compatibility_enabled=True,
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["rejection_reason"], "cross_dataset_disallowed")


if __name__ == "__main__":
    unittest.main()
