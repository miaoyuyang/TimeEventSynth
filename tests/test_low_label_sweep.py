from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.run_low_label_sweep import _aggregate_rows, _fraction_tag, _sweep_fractions


class LowLabelSweepTests(unittest.TestCase):
    def test_fraction_tag(self) -> None:
        self.assertEqual(_fraction_tag(0.2), "0p2")
        self.assertEqual(_fraction_tag(0.05), "0p05")

    def test_sweep_fractions_from_config(self) -> None:
        config = {"low_label": {"fractions": [0.01, 0.05, 0.1, 0.2]}}
        self.assertEqual(_sweep_fractions(config, None), [0.01, 0.05, 0.1, 0.2])

    def test_aggregate_rows(self) -> None:
        payloads = [
            (
                0.1,
                {
                    "comparison_rows": [
                        {"method": "real_only", "event_f1": 0.5, "auroc": 0.9},
                    ]
                },
            ),
            (
                0.2,
                {
                    "comparison_rows": [
                        {"method": "real_only", "event_f1": 0.6, "auroc": 0.91},
                    ]
                },
            ),
        ]
        rows = _aggregate_rows(payloads)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["labeled_fraction"], 0.1)
        self.assertEqual(rows[1]["event_f1"], 0.6)


if __name__ == "__main__":
    unittest.main()
