from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.evaluation.reconstruction_metrics import (
    build_masked_observation,
    compute_mask_interval,
    evaluate_masked_reconstruction,
)
from src.experiments.pipeline import split_records
from src.experiments.run_masked_completion import (
    _masked_completion_selection_settings,
    build_donor_windows,
    run_masked_completion,
    run_masked_completion_experiment,
)
from src.synthesis.donor_retrieval import EventWindow
from src.synthesis.event_window_synthesizer import reconstruct_masked_event_window


class MaskedCompletionTests(unittest.TestCase):
    def test_mask_interval_inside_event(self) -> None:
        for policy in ("middle_30", "suffix_30", "prefix_30", "random_contiguous"):
            start, end = compute_mask_interval(10, 0.3, policy, rng=np.random.default_rng(0))
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, 10)
            self.assertGreater(end, start)

    def test_donor_pool_excludes_test_series(self) -> None:
        records = make_synthetic_records(seed=21)
        config = {
            "seed": 21,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "masked_completion": {"donor_source": "train_val", "allow_test_series_donors": False},
        }
        train, val, _, split_ids = split_records(records, config)
        test_ids = set(split_ids["test"])
        donors = build_donor_windows(train, val, test_ids, donor_source="train_val", allow_test_series_donors=False)
        donor_series = {window.series_id for window in donors}
        self.assertTrue(donor_series.isdisjoint(test_ids))

    def test_reconstruction_metrics_are_finite(self) -> None:
        original = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 4.0], dtype=float)
        mask_start, mask_end = 2, 4
        _, partial_labels = build_masked_observation(original, mask_start, mask_end)
        target = EventWindow(
            series_id="s1",
            start=0,
            end=len(original),
            values=original,
            label="anomaly",
        )
        donor = EventWindow(
            series_id="s2",
            start=0,
            end=len(original),
            values=original + 0.5,
            label="anomaly",
        )
        reconstructed = reconstruct_masked_event_window(
            target,
            donors=[donor],
            method="normalized_time_mean_donor",
            partial_labels=partial_labels,
        )
        metrics = evaluate_masked_reconstruction(original, reconstructed, mask_start, mask_end)
        for key, value in metrics.items():
            self.assertTrue(np.isfinite(value), msg=key)

    def test_linear_interpolation_baseline(self) -> None:
        original = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
        mask_start, mask_end = 1, 4
        _, partial_labels = build_masked_observation(original, mask_start, mask_end)
        target = EventWindow(series_id="s1", start=0, end=len(original), values=original, label="anomaly")
        reconstructed = reconstruct_masked_event_window(
            target,
            donors=[target],
            method="linear_interpolation",
            partial_labels=partial_labels,
        )
        expected = np.interp(np.arange(len(original)), [0, 3, 4], [0.0, 3.0, 4.0])
        self.assertTrue(np.allclose(reconstructed[1:4], expected[1:4], atol=1e-6))
        self.assertEqual(reconstructed[0], original[0])
        self.assertEqual(reconstructed[-1], original[-1])

    def test_smoke_run_masked_completion(self) -> None:
        train_events = [
            {
                "event_id": "d1",
                "series_id": "s2",
                "values": [1.0, 2.0, 3.0, 2.0, 1.0],
                "labels": [1, 1, 1, 1, 1],
            },
            {
                "event_id": "d2",
                "series_id": "s3",
                "values": [1.1, 2.1, 3.1, 2.1, 1.1],
                "labels": [1, 1, 1, 1, 1],
            },
        ]
        query = {
            "event_id": "q1",
            "series_id": "s1",
            "values": [1.0, 2.0, 3.0, 2.0, 1.0],
            "labels": [1, 1, 1, 1, 1],
        }
        outputs = run_masked_completion(train_events, [query], method="linear_interpolation")
        self.assertEqual(len(outputs), 1)
        self.assertIn("metrics", outputs[0])

    def test_experiment_runner_produces_outputs(self) -> None:
        config = {
            "seed": 7,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "masked_completion": {
                "mask_fraction": 0.3,
                "mask_policies": ["middle_30"],
                "methods": ["linear_interpolation", "normalized_time_mean_donor"],
                "donor_source": "train_val",
                "allow_test_series_donors": False,
                "top_k_donors": 2,
                "max_qualitative_examples": 1,
            },
            "synthesis": {"top_k": 2, "alignment_grid_size": 16, "compatibility": {"enabled": False}},
        }
        payload = run_masked_completion_experiment(config, use_synthetic=True)
        self.assertGreater(payload["metrics"]["num_evaluations"], 0)
        self.assertIn("linear_interpolation", payload["metrics"]["aggregate_by_method"])
        self.assertEqual(payload["metrics"]["donor_policy"], "all_donors_no_filter")

    def test_experiment_respects_compatible_donor_policy(self) -> None:
        from unittest.mock import patch

        config = {
            "seed": 9,
            "data": {"train_ratio": 0.6, "dev_ratio": 0.2, "test_ratio": 0.2},
            "masked_completion": {
                "mask_fraction": 0.3,
                "mask_policies": ["middle_30"],
                "methods": ["normalized_time_mean_donor"],
                "donor_source": "train_val",
                "allow_test_series_donors": False,
                "top_k_donors": 2,
                "donor_policy": "cross_dataset_compatible",
            },
            "synthesis": {
                "top_k": 2,
                "alignment_grid_size": 16,
                "compatibility": {"enabled": True, "min_score": 0.99, "fallback_when_no_compatible_donor": "skip"},
            },
        }
        with patch("src.experiments.run_masked_completion.select_donors_for_target", return_value=([], [], {})) as mock_select:
            payload = run_masked_completion_experiment(config, use_synthetic=True)
        self.assertEqual(payload["metrics"]["num_evaluations"], 0)
        self.assertTrue(mock_select.called)
        compatibility_cfg = mock_select.call_args.kwargs["compatibility_cfg"]
        self.assertTrue(compatibility_cfg.get("enabled"))

    def test_groupwise_masked_completion_uses_centroid_group_selection(self) -> None:
        compatibility_cfg, retrieval_kwargs = _masked_completion_selection_settings(
            {
                "donor_policy": "cross_dataset_compatible",
                "groupwise_matching": True,
                "group_key": "event_group_id",
                "context_size": 7,
            },
            {"compatibility": {"enabled": True}},
        )
        self.assertTrue(compatibility_cfg.get("enabled"))
        self.assertTrue(retrieval_kwargs["restrict_to_target_group"])
        self.assertEqual(retrieval_kwargs["group_key"], "event_group_id")
        self.assertEqual(retrieval_kwargs["group_selection_mode"], "nearest_compatible_centroid")
        self.assertEqual(retrieval_kwargs["context_size"], 7)


if __name__ == "__main__":
    unittest.main()
