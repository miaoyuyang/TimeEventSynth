from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.synthesis.compatibility import (
    compute_pairwise_compatibility,
    compute_timeline_features,
    filter_compatible_donors,
    rank_compatible_donors,
)


def _sine_series(length: int = 200, freq: float = 0.05, noise: float = 0.02) -> np.ndarray:
    t = np.arange(length, dtype=float)
    return np.sin(2 * np.pi * freq * t) + noise * np.random.default_rng(0).standard_normal(length)


class CompatibilityTests(unittest.TestCase):
    def test_score_in_unit_interval(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "t"})
        source = compute_timeline_features(_sine_series(), config={"series_id": "s"})
        result = compute_pairwise_compatibility(source, target)
        self.assertGreaterEqual(result["compatibility_score"], 0.0)
        self.assertLessEqual(result["compatibility_score"], 1.0)
        for key in (
            "shape_similarity",
            "amplitude_compatibility",
            "duration_compatibility",
            "context_similarity",
            "frequency_similarity",
            "trend_similarity",
        ):
            self.assertGreaterEqual(result[key], 0.0)
            self.assertLessEqual(result[key], 1.0)

    def test_similar_timelines_score_higher_than_incompatible(self) -> None:
        base = _sine_series(length=300, noise=0.01)
        similar = base + 0.01 * np.random.default_rng(1).standard_normal(len(base))
        incompatible = np.random.default_rng(2).standard_normal(len(base)) * 50.0 + 100.0

        target = compute_timeline_features(base, config={"series_id": "target"})
        similar_features = compute_timeline_features(similar, config={"series_id": "similar"})
        incompatible_features = compute_timeline_features(incompatible, config={"series_id": "bad"})

        similar_score = compute_pairwise_compatibility(similar_features, target)["compatibility_score"]
        incompatible_score = compute_pairwise_compatibility(incompatible_features, target)["compatibility_score"]
        self.assertGreater(similar_score, incompatible_score)

    def test_filter_rejects_low_score_donors(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target"})
        good = compute_timeline_features(_sine_series(length=200), config={"series_id": "good"})
        bad = compute_timeline_features(np.random.default_rng(3).standard_normal(200) * 100, config={"series_id": "bad"})
        table = {"target": target, "good": good, "bad": bad}
        good_score = compute_pairwise_compatibility(good, target)["compatibility_score"]
        bad_score = compute_pairwise_compatibility(bad, target)["compatibility_score"]
        self.assertGreater(good_score, bad_score)
        min_score = bad_score + 0.5 * (good_score - bad_score)
        kept, records = filter_compatible_donors("target", ["good", "bad"], table, config={"min_score": min_score})
        self.assertIn("good", kept)
        self.assertNotIn("bad", kept)
        self.assertTrue(any(row["source_series_id"] == "bad" and row["final_decision"] == "rejected" for row in records))

    def test_top_k_limits_kept_donors(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target"})
        table = {"target": target}
        donors = []
        for idx in range(5):
            series_id = f"donor_{idx}"
            donors.append(series_id)
            table[series_id] = compute_timeline_features(
                _sine_series(length=200, noise=0.01 + 0.01 * idx),
                config={"series_id": series_id},
            )
        kept, _ = filter_compatible_donors("target", donors, table, config={"min_score": 0.0, "top_k": 2})
        self.assertEqual(len(kept), 2)

    def test_top_quantile_limits_kept_donors(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target"})
        table = {"target": target}
        donors = []
        for idx in range(4):
            series_id = f"donor_{idx}"
            donors.append(series_id)
            table[series_id] = compute_timeline_features(
                _sine_series(length=200, noise=0.02 * (idx + 1)),
                config={"series_id": series_id},
            )
        kept, _ = filter_compatible_donors(
            "target",
            donors,
            table,
            config={"min_score": 0.0, "top_quantile": 0.5},
        )
        self.assertEqual(len(kept), 2)

    def test_allow_cross_dataset_false_rejects(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target", "dataset": "A"})
        source = compute_timeline_features(_sine_series(), config={"series_id": "source", "dataset": "B"})
        result = compute_pairwise_compatibility(
            source,
            target,
            config={"allow_cross_dataset": False, "min_score": 0.0},
        )
        self.assertEqual(result["final_decision"], "rejected")
        self.assertEqual(result["rejection_reason"], "cross_dataset_disallowed")

    def test_missing_labels_do_not_crash(self) -> None:
        series = _sine_series()
        features = compute_timeline_features(series, labels=None)
        self.assertEqual(features["anomaly_segment_count"], 0.0)
        self.assertIn("local_context_summary", features)
        target = compute_timeline_features(series, config={"series_id": "t"})
        result = compute_pairwise_compatibility(features, target)
        self.assertTrue(np.isfinite(result["compatibility_score"]))

    def test_nan_values_do_not_crash(self) -> None:
        series = _sine_series()
        series[10:15] = np.nan
        labels = np.zeros(len(series), dtype=int)
        labels[50:60] = 1
        features = compute_timeline_features(series, labels=labels)
        self.assertTrue(np.isfinite(features["mean"]))
        target = compute_timeline_features(_sine_series(), config={"series_id": "t"})
        result = compute_pairwise_compatibility(features, target)
        self.assertTrue(np.isfinite(result["compatibility_score"]))

    def test_no_silent_keep_when_none_pass(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target"})
        bad = compute_timeline_features(np.random.default_rng(9).standard_normal(200) * 1000, config={"series_id": "bad"})
        kept, records = filter_compatible_donors(
            "target",
            ["bad"],
            {"target": target, "bad": bad},
            config={"min_score": 0.99},
        )
        self.assertEqual(kept, [])
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["final_decision"], "rejected")

    def test_rank_compatible_donors_sorted(self) -> None:
        target = compute_timeline_features(_sine_series(), config={"series_id": "target"})
        d1 = compute_timeline_features(_sine_series(noise=0.01), config={"series_id": "d1"})
        d2 = compute_timeline_features(np.random.default_rng(4).standard_normal(200) * 20, config={"series_id": "d2"})
        ranked = rank_compatible_donors("target", ["d1", "d2"], {"target": target, "d1": d1, "d2": d2})
        self.assertEqual(len(ranked), 2)
        self.assertGreaterEqual(ranked[0]["compatibility_score"], ranked[1]["compatibility_score"])


if __name__ == "__main__":
    unittest.main()
