from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.datasets.tsb_loader import TimeSeriesRecord
from src.experiments.pipeline import donor_pool_records, prepare_low_label_train_and_donor_pool, split_records
from src.experiments.run_low_label import _mask_train_labels
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit
from src.synthesis.donor_retrieval import collect_event_windows


class LeakageTests(unittest.TestCase):
    def test_donor_pool_excludes_test_by_default(self) -> None:
        records = make_synthetic_records(seed=7)
        config = {"seed": 7, "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2}, "synthesis": {"donor_source": "train_only"}}
        train, val, test, split_ids = split_records(records, config)
        donors = donor_pool_records(train, val, config)
        donor_ids = {record.series_id for record in donors}
        test_ids = set(split_ids["test"])
        self.assertTrue(donor_ids.isdisjoint(test_ids))

    def test_synthesis_donors_never_from_test(self) -> None:
        records = make_synthetic_records(seed=11)
        config = {
            "seed": 11,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "synthesis": {"donor_source": "train_only", "top_k": 2},
        }
        train, val, test, split_ids = split_records(records, config)
        test_ids = set(split_ids["test"])
        masked = _mask_train_labels(train, labeled_fraction=1.0, seed=11)
        donors = donor_pool_records(masked, val, config)
        _, audit = build_augmented_training_records_with_audit(
            masked,
            split="train",
            policy_config={"method": "normalized_time_mean_donor", "top_k": 2, "filter_policy": {"name": "no_filter"}},
            donor_pool_records=donors,
        )
        for row in audit:
            for donor_id in row.get("donor_ids", []):
                donor_series = str(donor_id).split(":")[0]
                self.assertNotIn(donor_series, test_ids)

    def test_low_label_masking_only_zeros_train_segments(self) -> None:
        records = make_synthetic_records(seed=3)
        train = records[:4]
        masked = _mask_train_labels(train, labeled_fraction=0.5, seed=3)
        for original, masked_record in zip(train, masked):
            self.assertTrue(np.any(masked_record.labels <= original.labels))

    def test_low_label_donor_pool_uses_masked_train_labels_only(self) -> None:
        records = make_synthetic_records(seed=17, num_series=8, events_per_series=4)
        config = {
            "seed": 17,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "synthesis": {"donor_source": "train_only", "top_k": 2},
        }
        train, val, _, _ = split_records(records, config)
        labeled_fraction = 0.25
        masked, donors = prepare_low_label_train_and_donor_pool(
            train,
            val,
            config,
            labeled_fraction=labeled_fraction,
            seed=17,
        )
        leaked_donors = donor_pool_records(train, val, config)
        masked_windows = collect_event_windows(masked)
        donor_windows = collect_event_windows(donors)
        leaked_windows = collect_event_windows(leaked_donors)
        self.assertEqual(len(donor_windows), len(masked_windows))
        self.assertGreater(len(leaked_windows), len(donor_windows))

    def test_low_label_train_val_donor_source_requires_explicit_override(self) -> None:
        records = make_synthetic_records(seed=23, num_series=8, events_per_series=4)
        config = {
            "seed": 23,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "synthesis": {"donor_source": "train_val", "top_k": 2},
        }
        train, val, _, _ = split_records(records, config)
        with self.assertRaisesRegex(ValueError, "allow_validation_donors_for_low_label"):
            prepare_low_label_train_and_donor_pool(
                train,
                val,
                config,
                labeled_fraction=0.25,
                seed=23,
            )

    def test_synthesis_disabled_outside_train_split(self) -> None:
        records = make_synthetic_records(seed=5)
        kept, audit = build_augmented_training_records_with_audit(
            records,
            split="test",
            policy_config={"method": "normalized_time_mean_donor", "top_k": 2},
        )
        self.assertEqual(kept, [])
        self.assertEqual(audit, [])


if __name__ == "__main__":
    unittest.main()
