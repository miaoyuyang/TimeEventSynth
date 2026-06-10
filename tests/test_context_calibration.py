from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.normalized_time import (
    apply_partial_target_synthesis,
    calibrate_amplitude_to_context,
    robust_std,
)
from src.synthesis.donor_retrieval import EventWindow, compute_context_features
from src.synthesis.event_window_synthesizer import (
    synthetic_training_labels_for_window,
    synthesize_context_aware,
)


class ContextCalibrationTests(unittest.TestCase):
    def _window_with_context(self) -> EventWindow:
        series = np.linspace(0.0, 1.0, 20)
        start, end = 8, 13
        return EventWindow(
            series_id="target",
            start=start,
            end=end,
            values=series[start:end].copy(),
            label="anomaly",
            metadata={
                "series_values": series,
                "series_labels": np.zeros(len(series), dtype=int),
            },
        )

    def _donor(self, scale: float = 10.0) -> EventWindow:
        return EventWindow(
            series_id="donor",
            start=0,
            end=5,
            values=np.asarray([scale, scale + 1, scale + 2, scale + 1, scale], dtype=float),
            label="anomaly",
        )

    def test_baseline_shift_matches_pre_event_mean(self) -> None:
        window = self._window_with_context()
        donor = self._donor(scale=50.0)
        synthetic, audit = synthesize_context_aware(
            window,
            [donor],
            base_method="normalized_time_mean_donor",
            target_length=len(window.values),
            amplitude_calibration="baseline_shift",
            context_size=5,
        )
        pre_mean = float(np.mean(window.metadata["series_values"][window.start - 5 : window.start]))
        self.assertAlmostEqual(float(np.mean(synthetic)), pre_mean, places=5)
        self.assertAlmostEqual(audit["shift"], pre_mean - audit["donor_original_mean"], places=5)
        self.assertEqual(audit["amplitude_calibration"], "baseline_shift")

    def test_robust_scale_avoids_division_by_zero(self) -> None:
        constant = np.full(5, 7.0)
        scaled, audit = calibrate_amplitude_to_context(
            constant,
            donor_values=constant,
            target_event_values=constant,
            series_values=np.concatenate([np.full(5, 1.0), constant, np.full(5, 1.0)]),
            start=5,
            end=10,
            mode="robust_scale_shift",
            context_size=5,
        )
        self.assertTrue(np.all(np.isfinite(scaled)))
        self.assertGreater(audit["scale"], 0.0)

    def test_output_length_preserved(self) -> None:
        window = self._window_with_context()
        donor = self._donor()
        for mode in ("none", "baseline_shift", "robust_scale_shift"):
            synthetic, _ = synthesize_context_aware(
                window,
                [donor],
                base_method="normalized_time_mean_donor",
                target_length=len(window.values),
                amplitude_calibration=mode,
            )
            self.assertEqual(len(synthetic), len(window.values))

    def test_no_nan_or_inf_in_synthetic_data(self) -> None:
        window = self._window_with_context()
        donors = [self._donor(scale=20.0), self._donor(scale=30.0)]
        for base_method in (
            "normalized_time_mean_donor",
            "dtw_aligned_donor",
            "learned_prototype_event_time",
        ):
            synthetic, audit = synthesize_context_aware(
                window,
                donors,
                base_method=base_method,
                target_length=len(window.values),
                amplitude_calibration="robust_scale_shift",
                partial_target_mode="residual_baseline",
            )
            self.assertTrue(np.all(np.isfinite(synthetic)), msg=base_method)
            for key in ("scale", "shift", "donor_original_mean", "target_context_mean"):
                self.assertTrue(np.isfinite(audit[key]), msg=f"{base_method}:{key}")

    def test_synthetic_training_labels_follow_partial_target_mode(self) -> None:
        partial = np.asarray([1, 0, 1, 0, 0], dtype=int)
        full = synthetic_training_labels_for_window(5, partial, partial_target_mode="full_replacement")
        anchor = synthetic_training_labels_for_window(5, partial, partial_target_mode="anchor_interpolate")
        residual = synthetic_training_labels_for_window(5, partial, partial_target_mode="residual_baseline")
        self.assertTrue(np.all(full == 1))
        np.testing.assert_array_equal(anchor, partial)
        np.testing.assert_array_equal(residual, partial)

    def test_synthesize_context_aware_returns_partial_labels_in_audit(self) -> None:
        series = np.linspace(0.0, 1.0, 10)
        window = EventWindow(
            series_id="target",
            start=2,
            end=7,
            values=series[2:7].copy(),
            label="anomaly",
            metadata={
                "series_values": series,
                "series_labels": np.asarray([0, 0, 1, 0, 1, 0, 0, 0, 0, 0], dtype=int),
            },
        )
        donor = EventWindow(
            series_id="donor",
            start=0,
            end=5,
            values=np.asarray([5.0, 6.0, 7.0, 6.0, 5.0]),
            label="anomaly",
        )
        _, audit = synthesize_context_aware(
            window,
            [donor],
            base_method="normalized_time_mean_donor",
            target_length=len(window.values),
            partial_target_mode="anchor_interpolate",
        )
        np.testing.assert_array_equal(
            audit["synthetic_point_labels"],
            [1, 0, 1, 0, 0],
        )

    def test_partial_target_anchor_interpolate(self) -> None:
        donor = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0])
        target = np.asarray([0.0, 0.0, 0.0, 0.0, 0.0])
        labels = np.asarray([1, 0, 1, 0, 0], dtype=int)
        blended = apply_partial_target_synthesis(
            donor,
            target,
            labels,
            mode="anchor_interpolate",
        )
        self.assertEqual(blended[0], target[0])
        self.assertEqual(blended[2], target[2])
        self.assertTrue(np.all(np.isfinite(blended)))

    def test_context_features_length(self) -> None:
        window = self._window_with_context()
        features = compute_context_features(window, context_size=5)
        self.assertEqual(features.shape, (8,))

    def test_robust_std_constant_series(self) -> None:
        self.assertGreater(robust_std(np.full(4, 3.0)), 0.0)


if __name__ == "__main__":
    unittest.main()
