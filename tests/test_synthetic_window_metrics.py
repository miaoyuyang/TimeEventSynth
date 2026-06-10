from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.tsb_loader import TimeSeriesRecord
from src.experiments.run_low_label import synthetic_window_metrics


class SyntheticWindowMetricsTests(unittest.TestCase):
    def _record(self, series_id: str, length: int = 5) -> TimeSeriesRecord:
        return TimeSeriesRecord(
            series_id=series_id,
            values=np.ones(length),
            labels=np.array([0, 1, 1, 0, 0][:length]),
            timestamps=None,
            source_path="test",
            metadata={},
        )

    def test_oversample_path_counts_appended_windows(self) -> None:
        masked = [self._record("a"), self._record("b")]
        appended = [self._record("a__oversampled__0"), self._record("b__oversampled__0")]
        method_train = masked + appended
        metrics = synthetic_window_metrics(masked, method_train, audit_rows=None)
        self.assertEqual(metrics["num_synthetic_windows"], 2)
        self.assertEqual(metrics["num_synthetic_points"], 10)
        self.assertEqual(metrics["num_rejected_synthetic_windows"], 0)

    def test_synthesis_path_uses_audit_kept_counts(self) -> None:
        masked = [self._record("a")]
        method_train = masked + [self._record("syn_1")]
        audit_rows = [
            {"kept": True},
            {"kept": False},
        ]
        metrics = synthetic_window_metrics(masked, method_train, audit_rows=audit_rows)
        self.assertEqual(metrics["num_synthetic_windows"], 1)
        self.assertEqual(metrics["num_rejected_synthetic_windows"], 1)


if __name__ == "__main__":
    unittest.main()
