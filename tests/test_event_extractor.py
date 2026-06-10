from __future__ import annotations

import unittest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.event_extractor import event_to_mask, labels_to_events
from src.datasets.split_builder import build_series_split


class EventExtractorTests(unittest.TestCase):
    def test_labels_to_events_basic_half_open(self) -> None:
        labels = [0, 0, 1, 1, 0, 1, 1, 1, 0]
        self.assertEqual(labels_to_events(labels), [(2, 4), (5, 8)])

    def test_labels_to_events_edge_cases(self) -> None:
        self.assertEqual(labels_to_events([1, 1, 0]), [(0, 2)])
        self.assertEqual(labels_to_events([0, 1, 1]), [(1, 3)])
        self.assertEqual(labels_to_events([0, 0, 0]), [])
        self.assertEqual(labels_to_events([1, 1, 1]), [(0, 3)])

    def test_merge_gap_behavior(self) -> None:
        labels = [1, 1, 0, 1, 1, 0, 0, 1]
        self.assertEqual(labels_to_events(labels, merge_gap=0), [(0, 2), (3, 5), (7, 8)])
        self.assertEqual(labels_to_events(labels, merge_gap=1), [(0, 5), (7, 8)])

    def test_event_to_mask(self) -> None:
        mask = event_to_mask(8, [(1, 3), (5, 8)])
        self.assertEqual(mask.tolist(), [0, 1, 1, 0, 0, 1, 1, 1])

    def test_split_coverage(self) -> None:
        records = [
            {"series_id": "a", "labels": [0, 0, 0]},
            {"series_id": "b", "labels": [1, 1, 0]},
            {"series_id": "c", "labels": [0, 1, 0]},
            {"series_id": "d", "labels": [0, 0, 0]},
            {"series_id": "e", "labels": [1, 1, 1]},
        ]
        splits = build_series_split(records, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)
        all_ids = set(splits["train"]) | set(splits["val"]) | set(splits["test"])
        self.assertEqual(all_ids, {"a", "b", "c", "d", "e"})
        self.assertTrue(set(splits["train"]).isdisjoint(splits["val"]))
        self.assertTrue(set(splits["train"]).isdisjoint(splits["test"]))
        self.assertTrue(set(splits["val"]).isdisjoint(splits["test"]))


if __name__ == "__main__":
    unittest.main()
