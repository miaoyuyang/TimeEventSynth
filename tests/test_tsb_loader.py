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

from src.datasets.load_options import DatasetLoadOptions
from src.datasets.tsb_loader import filter_records, load_tsb_records, normalize_binary_labels


class TsbLoaderTests(unittest.TestCase):
    def test_normalize_binary_labels(self) -> None:
        self.assertTrue(np.array_equal(normalize_binary_labels(pd.Series([0, 1, 0])), np.array([0, 1, 0])))
        self.assertTrue(np.array_equal(normalize_binary_labels(pd.Series([True, False])), np.array([1, 0])))
        self.assertTrue(
            np.array_equal(
                normalize_binary_labels(pd.Series(["normal", "anomaly", "Normal", "ANOMALY"])),
                np.array([0, 1, 0, 1]),
            )
        )

    def test_load_nested_out_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "ECG"
            nested.mkdir()
            (nested / "series_a.out").write_text("1.0,0\n2.0,0\n3.0,1\n4.0,1\n", encoding="utf-8")
            (nested / "series_b.out").write_text("0.5,0\n1.5,1\n2.5,1\n", encoding="utf-8")
            records = load_tsb_records(root)
            self.assertEqual(len(records), 2)
            ids = {record.series_id for record in records}
            self.assertIn("ECG/series_a", ids)
            self.assertIn("ECG/series_b", ids)

    def test_load_csv_with_named_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            pd.DataFrame(
                {
                    "timestamp": [0, 1, 2, 3],
                    "signal": [1.0, 2.0, 3.0, 4.0],
                    "is_anomaly": ["normal", "normal", "anomaly", "anomaly"],
                }
            ).to_csv(path, index=False)
            records = load_tsb_records(path)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].labels.tolist(), [0, 0, 1, 1])

    def test_filter_max_series_and_min_events(self) -> None:
        records = [
            type("R", (), {"series_id": "a", "labels": np.array([0, 1, 1, 0]), "source_path": "a"})(),
            type("R", (), {"series_id": "b", "labels": np.array([0, 0, 0, 0]), "source_path": "b"})(),
            type("R", (), {"series_id": "c", "labels": np.array([0, 1, 0, 0]), "source_path": "c"})(),
        ]
        filtered, summary = filter_records(records, DatasetLoadOptions(min_event_windows=1, max_series=1))
        self.assertEqual(len(filtered), 1)
        self.assertEqual(summary["dropped_few_events"], 1)


if __name__ == "__main__":
    unittest.main()
