from __future__ import annotations

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.compatibility_summary import build_compatibility_summary
from src.experiments.compatibility_transfer_specs import (
    build_compatibility_transfer_specs,
    experiment_compatibility_enabled,
)
from src.experiments.config import load_config


class CompatibilityTransferTests(unittest.TestCase):
    def test_config_loads_experiments(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "experiment_compatibility_transfer.yaml")
        specs = build_compatibility_transfer_specs(config, smoke=False)
        names = {spec["name"] for spec in specs}
        self.assertIn("real_only", names)
        self.assertIn("learned_prototype_event_time__cross_dataset_compatible", names)
        self.assertIn("dtw_aligned_donor__cross_dataset_compatible", names)

    def test_smoke_subset_smaller(self) -> None:
        config = load_config(PROJECT_ROOT / "configs" / "experiment_compatibility_transfer.yaml")
        full = build_compatibility_transfer_specs(config, smoke=False)
        smoke = build_compatibility_transfer_specs(config, smoke=True)
        self.assertLess(len(smoke), len(full))

    def test_compatibility_enabled_detection(self) -> None:
        enabled_spec = {
            "kind": "synthetic",
            "donor_policy": "cross_dataset_compatible",
        }
        disabled_spec = {
            "kind": "synthetic",
            "donor_policy": "all_donors_no_filter",
        }
        self.assertTrue(experiment_compatibility_enabled(enabled_spec))
        self.assertFalse(experiment_compatibility_enabled(disabled_spec))

    def test_build_compatibility_summary_counts(self) -> None:
        audit_rows = [
            {
                "donor_policy": "cross_dataset_compatible",
                "record_type": "donor_pair",
                "target_series_id": "t1",
                "source_series_id": "s1",
                "target_dataset": "a",
                "source_dataset": "b",
                "same_dataset": False,
                "compatibility_enabled": True,
                "rejection_stage": "compatibility",
                "accepted": False,
                "compatibility_score": 0.4,
            },
        ]
        summary = build_compatibility_summary(audit_rows)
        self.assertEqual(summary["policies"]["cross_dataset_compatible"]["donor_pairs_rejected"], 1)
        self.assertEqual(summary["policies"]["cross_dataset_compatible"]["cross_dataset_pairs_considered"], 1)

    def test_verify_compatibility_donor_invariant(self) -> None:
        from src.experiments.audit_sanity import verify_compatibility_donor_invariant

        audit_rows = [
            {
                "method": "m",
                "compatibility_enabled": True,
                "target_series_id": "t1",
                "source_series_id": "bad",
                "rejection_stage": "compatibility",
                "accepted": False,
            },
            {
                "method": "m",
                "compatibility_enabled": True,
                "target_series_id": "t1",
                "accepted": True,
                "rejection_stage": "kept",
                "donor_series_ids": "good",
                "compatibility_score": 0.9,
            },
        ]
        self.assertEqual(verify_compatibility_donor_invariant(audit_rows, method_names=["m"]), [])
        audit_rows[1]["donor_series_ids"] = "bad"
        self.assertGreater(len(verify_compatibility_donor_invariant(audit_rows, method_names=["m"])), 0)


if __name__ == "__main__":
    unittest.main()
