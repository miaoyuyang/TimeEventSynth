from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporting import (
    REAL_ONLY_INVARIANT_WARNING,
    UNSUPPORTED_LOW_LABEL_WARNING,
    build_method_diagnostics,
    check_real_only_metrics_invariant,
    check_real_only_train_positives_monotonic,
    collect_low_label_warnings,
    detector_is_supervised,
)


class LowLabelDiagnosticsTests(unittest.TestCase):
    def test_unsupervised_detector_triggers_warning(self) -> None:
        warnings = collect_low_label_warnings(
            detector_cfg={"name": "isolation_forest", "model_type": "isolation_forest"},
            diagnostics_by_method={
                "real_only": build_method_diagnostics(
                    labeled_fraction=0.2,
                    method="real_only",
                    train_label_stats={"num_real_positive_train_points": 10, "num_positive_train_event_windows": 2},
                    num_synthetic_windows=0,
                    detector_cfg={"name": "isolation_forest", "model_type": "isolation_forest"},
                )
            },
        )
        self.assertIn(UNSUPPORTED_LOW_LABEL_WARNING, warnings)

    def test_supervised_detector_is_supervised_true(self) -> None:
        detector_cfg = {"name": "random_forest_window", "model_type": "random_forest_window"}
        self.assertTrue(detector_is_supervised(detector_cfg))
        diagnostic = build_method_diagnostics(
            labeled_fraction=0.2,
            method="real_only",
            train_label_stats={
                "num_real_positive_train_points": 12,
                "num_positive_train_event_windows": 3,
                "detector_train_negative_count": 40,
            },
            num_synthetic_windows=0,
            detector_cfg=detector_cfg,
        )
        self.assertTrue(diagnostic["detector_is_supervised"])
        self.assertEqual(diagnostic["train_positive_points"], 12)
        self.assertEqual(diagnostic["train_positive_windows"], 3)
        self.assertEqual(diagnostic["train_negative_points_sampled"], 40)

    def test_invariant_real_only_metrics_trigger_warning(self) -> None:
        rows = [
            {"method": "real_only", "labeled_fraction": 0.01, "auroc": 0.5, "auprc": 0.5, "event_f1": 0.5},
            {"method": "real_only", "labeled_fraction": 0.2, "auroc": 0.5, "auprc": 0.5, "event_f1": 0.5},
        ]
        self.assertEqual(check_real_only_metrics_invariant(rows), REAL_ONLY_INVARIANT_WARNING)
        warnings = collect_low_label_warnings(
            detector_cfg={"name": "random_forest_window"},
            comparison_rows=rows,
            diagnostics_rows=[
                {
                    "method": "real_only",
                    "labeled_fraction": 0.01,
                    "train_positive_points": 5,
                    "detector_is_supervised": True,
                },
                {
                    "method": "real_only",
                    "labeled_fraction": 0.2,
                    "train_positive_points": 20,
                    "detector_is_supervised": True,
                },
            ],
        )
        self.assertIn(REAL_ONLY_INVARIANT_WARNING, warnings)

    def test_non_monotonic_real_only_train_points_trigger_warning(self) -> None:
        rows = [
            {
                "method": "real_only",
                "labeled_fraction": 0.05,
                "train_positive_points": 20,
                "detector_is_supervised": True,
            },
            {
                "method": "real_only",
                "labeled_fraction": 0.2,
                "train_positive_points": 10,
                "detector_is_supervised": True,
            },
        ]
        warning = check_real_only_train_positives_monotonic(rows)
        self.assertIsNotNone(warning)
        self.assertIn("decreased", warning or "")


if __name__ == "__main__":
    unittest.main()
