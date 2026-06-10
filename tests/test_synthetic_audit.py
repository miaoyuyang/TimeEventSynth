from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.datasets.tsb_loader import TimeSeriesRecord
from src.experiments.synthesis_runner import build_rejection_summary
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit, flatten_audit_row
from src.synthesis.uncertainty_filter import apply_filter_policy, resolve_filter_policy


def _tiny_train_records() -> list[TimeSeriesRecord]:
    records = make_synthetic_records(seed=1, num_series=6, events_per_series=4)
    for record in records:
        record.labels = record.labels.copy()
        record.labels[record.labels > 0] = 1
    return records[:4]


class SyntheticAuditTests(unittest.TestCase):
    def test_audit_has_one_row_per_candidate(self) -> None:
        train = _tiny_train_records()
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
            synthesis_cfg={"filter_policy": "none"},
        )
        self.assertGreater(len(audit_rows), 0)
        synthesis_candidates = [row for row in audit_rows if row.get("record_type") != "donor_rejection"]
        self.assertEqual(
            len(synthesis_candidates),
            len({row["candidate_id"] for row in synthesis_candidates}),
        )
        csv_rows = [flatten_audit_row(row) for row in audit_rows]
        self.assertTrue(all(row["rejection_reason"] is None or isinstance(row["rejection_reason"], str) for row in csv_rows))

    def test_rejection_reason_populated_for_filtered_candidates(self) -> None:
        rows = [
            {
                "confidence_components": {"aggregate_confidence": score},
                "kept": True,
                "accepted": True,
                "rejection_reason": None,
            }
            for score in [0.1, 0.2, 0.3, 0.9, 0.95]
        ]
        filtered = apply_filter_policy(
            rows,
            resolve_filter_policy({"min_confidence": 0.5}, override="min_confidence"),
        )
        rejected = [row for row in filtered if not row["accepted"]]
        self.assertGreater(len(rejected), 0)
        self.assertTrue(all(row["rejection_reason"] for row in rejected))

    def test_top_quantile_accepts_requested_fraction(self) -> None:
        scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        rows = [{"confidence_components": {"aggregate_confidence": score}} for score in scores]
        filtered = apply_filter_policy(
            rows,
            resolve_filter_policy({"confidence_quantile": 0.5}, override="top_quantile"),
        )
        accepted = sum(1 for row in filtered if row["accepted"])
        self.assertGreaterEqual(accepted / len(rows), 0.45)
        self.assertLessEqual(accepted / len(rows), 0.55)

    def test_top_quantile_skips_donor_pair_rows_without_confidence(self) -> None:
        rows = [
            {
                "record_type": "synthesis_candidate",
                "confidence_components": {"aggregate_confidence": 0.9},
            },
            {
                "record_type": "donor_pair",
                "confidence_components": {},
                "accepted": True,
                "rejection_stage": "compatibility",
            },
            {
                "record_type": "synthesis_candidate",
                "confidence_components": {"aggregate_confidence": 0.2},
            },
        ]
        filtered = apply_filter_policy(rows, {"name": "top_quantile", "quantile": 0.5})
        donor_pair = next(row for row in filtered if row.get("record_type") == "donor_pair")
        self.assertTrue(donor_pair["accepted"])
        self.assertEqual(donor_pair["rejection_stage"], "compatibility")
        synth_accepted = [row for row in filtered if row.get("record_type") == "synthesis_candidate" and row["accepted"]]
        self.assertEqual(len(synth_accepted), 1)
        self.assertGreaterEqual(synth_accepted[0]["confidence_components"]["aggregate_confidence"], 0.5)

    def test_min_confidence_accepts_above_threshold(self) -> None:
        rows = [
            {"confidence_components": {"aggregate_confidence": 0.2}},
            {"confidence_components": {"aggregate_confidence": 0.8}},
        ]
        filtered = apply_filter_policy(
            rows,
            resolve_filter_policy({"min_confidence": 0.5}, override="min_confidence"),
        )
        accepted = [row for row in filtered if row["accepted"]]
        self.assertEqual(len(accepted), 1)
        self.assertGreaterEqual(accepted[0]["confidence_components"]["aggregate_confidence"], 0.5)

    def test_rejection_summary_and_csv_export(self) -> None:
        train = _tiny_train_records()
        policy = {
            "method": "normalized_time_mean_donor",
            "method_name": "normalized_time_strict_filter",
            "labeled_fraction": 0.2,
            "top_k": 2,
            "grid_size": 16,
            "filter_policy": resolve_filter_policy(
                {
                    "filter_policy": "strict",
                    "strict_thresholds": {
                        "donor_similarity": 0.99,
                        "donor_agreement": 0.99,
                        "amplitude_compatibility": 0.99,
                    },
                    "min_confidence": 0.99,
                }
            ),
        }
        _, audit_rows = build_augmented_training_records_with_audit(
            train,
            split="train",
            policy_config=policy,
            donor_pool_records=train,
            synthesis_cfg={
                "filter_policy": "strict",
                "strict_thresholds": {
                    "donor_similarity": 0.99,
                    "donor_agreement": 0.99,
                    "amplitude_compatibility": 0.99,
                },
                "min_confidence": 0.99,
            },
        )
        csv_rows = [flatten_audit_row(row) for row in audit_rows]
        summary = build_rejection_summary(csv_rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "synthetic_audit.csv"
            pd.DataFrame(csv_rows).to_csv(path, index=False)
            frame = pd.read_csv(path)
            self.assertGreater(len(frame), 0)
            self.assertIn("candidate_id", frame.columns)
            self.assertIn("rejection_reason", frame.columns)
        self.assertIn("counts", summary["normalized_time_strict_filter"])
        self.assertGreater(summary["normalized_time_strict_filter"]["num_candidates"], 0)


if __name__ == "__main__":
    unittest.main()
