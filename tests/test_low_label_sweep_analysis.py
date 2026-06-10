from __future__ import annotations

import json
import unittest
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporting import (
    REAL_ONLY_INVARIANT_WARNING,
    STRICT_FILTER_REJECT_ALL_WARNING,
    analyze_low_label_sweep,
    format_low_label_sweep_markdown,
)


class LowLabelSweepAnalysisTests(unittest.TestCase):
    def _sample_comparison(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "labeled_fraction": 0.01,
                    "method": "real_only",
                    "auroc": 0.5,
                    "auprc": 0.5,
                    "best_point_f1": 0.6,
                    "event_precision": 0.0,
                    "event_recall": 0.0,
                    "event_f1": 0.0,
                    "false_positive_events": 1.0,
                    "num_synthetic_windows": 0,
                    "num_synthetic_points": 0,
                    "num_rejected_synthetic_windows": 0,
                    "num_real_positive_train_points": 0,
                },
                {
                    "labeled_fraction": 0.2,
                    "method": "real_only",
                    "auroc": 0.9,
                    "auprc": 0.8,
                    "best_point_f1": 0.7,
                    "event_precision": 1.0,
                    "event_recall": 0.5,
                    "event_f1": 0.67,
                    "false_positive_events": 0.0,
                    "num_synthetic_windows": 0,
                    "num_synthetic_points": 0,
                    "num_rejected_synthetic_windows": 0,
                    "num_real_positive_train_points": 20,
                },
                {
                    "labeled_fraction": 0.2,
                    "method": "normalized_time_mean_donor",
                    "auroc": 0.91,
                    "auprc": 0.85,
                    "best_point_f1": 0.75,
                    "event_precision": 1.0,
                    "event_recall": 0.6,
                    "event_f1": 0.75,
                    "false_positive_events": 0.0,
                    "num_synthetic_windows": 5,
                    "num_synthetic_points": 40,
                    "num_rejected_synthetic_windows": 0,
                    "num_real_positive_train_points": 20,
                },
                {
                    "labeled_fraction": 0.2,
                    "method": "normalized_time_strict_filter",
                    "auroc": 0.9,
                    "auprc": 0.8,
                    "best_point_f1": 0.7,
                    "event_precision": 1.0,
                    "event_recall": 0.5,
                    "event_f1": 0.67,
                    "false_positive_events": 0.0,
                    "num_synthetic_windows": 0,
                    "num_synthetic_points": 0,
                    "num_rejected_synthetic_windows": 5,
                    "num_real_positive_train_points": 20,
                },
            ]
        )

    def test_best_method_selection(self) -> None:
        comparison = self._sample_comparison()
        per_series = pd.DataFrame({"series_id": ["s1"], "method": ["real_only"], "labeled_fraction": [0.2]})
        analysis = analyze_low_label_sweep(comparison, per_series)
        self.assertEqual(analysis["best_by_auprc"]["0.2"]["method"], "normalized_time_mean_donor")
        self.assertEqual(analysis["best_by_event_f1"]["0.2"]["method"], "normalized_time_mean_donor")

    def test_strict_filter_warning(self) -> None:
        comparison = self._sample_comparison()
        per_series = pd.DataFrame({"series_id": ["s1"], "method": ["real_only"], "labeled_fraction": [0.2]})
        analysis = analyze_low_label_sweep(comparison, per_series)
        self.assertTrue(any(STRICT_FILTER_REJECT_ALL_WARNING in warning for warning in analysis["warnings"]))

    def test_markdown_contains_sections(self) -> None:
        comparison = self._sample_comparison()
        per_series = pd.DataFrame({"series_id": ["s1"], "method": ["real_only"], "labeled_fraction": [0.2]})
        analysis = analyze_low_label_sweep(comparison, per_series)
        markdown = format_low_label_sweep_markdown(analysis)
        self.assertIn("Best Method by Labeled Fraction", markdown)
        self.assertIn("False-Positive Event Comparison", markdown)
        self.assertIn("Pipeline Sanity", markdown)

    def test_real_workspace_sweep_if_present(self) -> None:
        sweep_dir = PROJECT_ROOT / "outputs" / "low_label_sweep" / "synthetic" / "20260530_133750"
        comparison_path = sweep_dir / "low_label_sweep_comparison.csv"
        if not comparison_path.exists():
            self.skipTest("workspace sweep outputs not present")
        comparison = pd.read_csv(comparison_path)
        per_series = pd.read_csv(sweep_dir / "low_label_sweep_per_series.csv")
        summary = json.loads((sweep_dir / "low_label_sweep_summary.json").read_text(encoding="utf-8"))
        analysis = analyze_low_label_sweep(comparison, per_series, summary)
        self.assertGreater(len(analysis["fractions"]), 0)
        self.assertIn("interpretation", analysis)


if __name__ == "__main__":
    unittest.main()
