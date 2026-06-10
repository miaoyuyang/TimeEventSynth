from __future__ import annotations

import unittest
from pathlib import Path
import sys
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.dtw_alignment import dtw_path, warp_source_to_target
from src.alignment.event_grouping import assign_event_groups
from src.alignment.learned_event_time import LearnedEventTimeAligner
from src.alignment.normalized_time import normalize_event_window
from src.synthesis.donor_retrieval import EventWindow, retrieve_donors, retrieve_topk_donors
from src.synthesis.event_window_synthesizer import (
    synthesize_by_dtw_donor,
    synthesize_by_learned_prototype_event_time,
    synthesize_by_mean_donor,
    synthesize_event_window,
)


class SynthesisTests(unittest.TestCase):
    def test_retrieval_and_synthesis(self) -> None:
        query = {"event_id": "q", "series_id": "s", "values": [1.0, 2.0, 3.0], "labels": [1, 1, 1], "local_time": [0.0, 0.5, 1.0]}
        donors = [
            {"event_id": "d1", "series_id": "s2", "values": [1.1, 2.1, 3.1], "labels": [1, 1, 1], "local_time": [0.0, 0.5, 1.0]},
            {"event_id": "d2", "series_id": "s3", "values": [4.0, 5.0, 6.0], "labels": [1, 1, 1], "local_time": [0.0, 0.5, 1.0]},
        ]
        matches = retrieve_donors(query, donors, top_k=1, method="normalized_time")
        self.assertEqual(len(matches), 1)
        synthetic = synthesize_event_window(query, matches)
        self.assertIsNotNone(synthetic)
        self.assertEqual(synthetic["source_donor_event_ids"], ["d1"])

    def test_retrieve_topk_excludes_same_series(self) -> None:
        target = EventWindow(series_id="s1", start=0, end=3, values=np.asarray([1.0, 2.0, 3.0]), label="anomaly")
        donors = [
            EventWindow(series_id="s1", start=5, end=8, values=np.asarray([1.0, 2.0, 3.0]), label="anomaly"),
            EventWindow(series_id="s2", start=0, end=3, values=np.asarray([1.1, 2.1, 3.1]), label="anomaly"),
        ]
        matches = retrieve_topk_donors(target, donors, k=5, exclude_same_series=True)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].series_id, "s2")

    def test_retrieve_topk_can_restrict_to_target_group(self) -> None:
        target = EventWindow(
            series_id="s1",
            start=0,
            end=3,
            values=np.asarray([1.0, 2.0, 3.0]),
            label="anomaly",
            metadata={"event_group_id": "group_0"},
        )
        donors = [
            EventWindow(
                series_id="s2",
                start=0,
                end=3,
                values=np.asarray([1.0, 2.0, 3.0]),
                label="anomaly",
                metadata={"event_group_id": "group_1"},
            ),
            EventWindow(
                series_id="s3",
                start=0,
                end=3,
                values=np.asarray([1.1, 2.1, 3.1]),
                label="anomaly",
                metadata={"event_group_id": "group_0"},
            ),
        ]
        matches = retrieve_topk_donors(
            target,
            donors,
            k=5,
            exclude_same_series=True,
            group_key="event_group_id",
            restrict_to_target_group=True,
        )
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].series_id, "s3")

    def test_assign_event_groups_annotates_windows(self) -> None:
        donors = [
            EventWindow(series_id="d1", start=0, end=4, values=np.asarray([0.0, 0.2, 0.4, 0.6]), label="anomaly"),
            EventWindow(series_id="d2", start=0, end=4, values=np.asarray([3.0, 2.5, 2.0, 1.5]), label="anomaly"),
        ]
        targets = [
            EventWindow(series_id="t1", start=0, end=4, values=np.asarray([0.1, 0.2, 0.5, 0.7]), label="anomaly"),
        ]
        result = assign_event_groups(donors, targets, config={"num_groups": 2, "grid_size": 8})
        self.assertEqual(result.num_groups, 2)
        self.assertIn("event_group_id", donors[0].metadata)
        self.assertIn("event_group_id", targets[0].metadata)

    def test_synthesized_window_matches_target_length(self) -> None:
        target = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.5, 2.0]), label="anomaly")
        donors = [
            EventWindow(series_id="s2", start=0, end=3, values=np.asarray([1.0, 2.0, 3.0]), label="anomaly"),
            EventWindow(series_id="s3", start=0, end=4, values=np.asarray([1.5, 2.5, 2.0, 3.0]), label="anomaly"),
        ]
        synthetic = synthesize_by_mean_donor(target, donors, target_length=len(target.values))
        self.assertEqual(len(synthetic), len(target.values))

    def test_dtw_path_monotonicity(self) -> None:
        path = dtw_path([1.0, 2.0, 3.0], [1.0, 1.5, 2.5, 3.0])
        self.assertTrue(len(path) > 0)
        self.assertTrue(all(path[idx][0] <= path[idx + 1][0] and path[idx][1] <= path[idx + 1][1] for idx in range(len(path) - 1)))

    def test_dtw_warp_output_length(self) -> None:
        warped = warp_source_to_target([1.0, 2.0, 3.0], 5)
        self.assertEqual(len(warped), 5)

    def test_dtw_no_crash_short_windows(self) -> None:
        warped = warp_source_to_target([1.0], [1.0, 2.0])
        self.assertEqual(len(warped), 2)

    def test_dtw_multivariate_shape_preservation(self) -> None:
        source = np.asarray([[1.0, 0.0], [2.0, 1.0], [3.0, 0.0]])
        target = np.asarray([[1.0, 0.0], [2.5, 1.0], [3.0, 0.5], [4.0, 0.0]])
        warped = warp_source_to_target(source, target)
        self.assertEqual(warped.shape, target.shape)

    def test_dtw_synthesis_matches_target_length(self) -> None:
        target = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.5, 2.0]), label="anomaly")
        donors = [
            EventWindow(series_id="s2", start=0, end=3, values=np.asarray([1.0, 2.0, 3.0]), label="anomaly"),
            EventWindow(series_id="s3", start=0, end=4, values=np.asarray([1.5, 2.5, 2.0, 3.0]), label="anomaly"),
        ]
        synthetic = synthesize_by_dtw_donor(target, donors, target_length=len(target.values))
        self.assertEqual(len(synthetic), len(target.values))

    def test_learned_prototype_synthesis_matches_target_length(self) -> None:
        target = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.5, 2.0]), label="anomaly")
        donors = [
            EventWindow(series_id="s2", start=0, end=3, values=np.asarray([1.0, 2.0, 3.0]), label="anomaly"),
            EventWindow(series_id="s3", start=0, end=4, values=np.asarray([1.5, 2.5, 2.0, 3.0]), label="anomaly"),
        ]
        synthetic = synthesize_by_learned_prototype_event_time(target, donors, target_length=len(target.values))
        self.assertEqual(len(synthetic), len(target.values))

    def test_learned_aligner_transform_inverse_no_crash(self) -> None:
        windows = [
            EventWindow(series_id="s1", start=0, end=4, values=np.asarray([1.0, 2.0, 1.5, 1.0]), label="anomaly"),
            EventWindow(series_id="s2", start=0, end=5, values=np.asarray([0.5, 1.5, 2.5, 1.5, 0.5]), label="anomaly"),
        ]
        aligner = LearnedEventTimeAligner(grid_size=8).fit(windows)
        transformed = aligner.transform_window(windows[0], grid_size=8)
        restored = aligner.inverse_transform(transformed, windows[1], target_length=len(windows[1].values))
        self.assertEqual(transformed.shape, (8, 1))
        self.assertEqual(restored.shape, (len(windows[1].values), 1))

    def test_learned_transform_applies_dtw_warp_not_linear_resample(self) -> None:
        """DTW warp to prototype must differ from plain normalized-time resampling."""
        slow_rise = EventWindow(
            series_id="slow",
            start=0,
            end=12,
            values=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            label="anomaly",
        )
        fast_spike = EventWindow(
            series_id="fast",
            start=0,
            end=6,
            values=np.asarray([0.0, 0.0, 6.0, 5.0, 1.0, 0.0]),
            label="anomaly",
        )
        aligner = LearnedEventTimeAligner(grid_size=8).fit([slow_rise, fast_spike])
        warped = aligner.transform_window(fast_spike, grid_size=8).reshape(-1)
        resampled = np.asarray(normalize_event_window(fast_spike.values, grid_size=8), dtype=float).reshape(-1)
        self.assertFalse(np.allclose(warped, resampled, atol=1e-6))

    def test_learned_prototype_differs_from_normalized_mean(self) -> None:
        target = EventWindow(series_id="t", start=0, end=8, values=np.linspace(0.0, 1.0, 8), label="anomaly")
        donors = [
            EventWindow(
                series_id="d1",
                start=0,
                end=12,
                values=np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
                label="anomaly",
            ),
            EventWindow(
                series_id="d2",
                start=0,
                end=6,
                values=np.asarray([0.0, 0.0, 6.0, 5.0, 1.0, 0.0]),
                label="anomaly",
            ),
        ]
        learned = synthesize_by_learned_prototype_event_time(target, donors, target_length=len(target.values), grid_size=8)
        mean = synthesize_by_mean_donor(target, donors, target_length=len(target.values))
        self.assertEqual(len(learned), len(target.values))
        self.assertFalse(np.allclose(learned, mean, atol=1e-6))

    def test_fit_stores_bidirectional_warp_maps(self) -> None:
        window = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]), label="anomaly")
        other = EventWindow(series_id="s2", start=0, end=4, values=np.asarray([0.5, 1.5, 2.5, 1.5]), label="anomaly")
        aligner = LearnedEventTimeAligner(grid_size=8).fit([window, other])
        warp = aligner.window_warps_[("s1", 0, 5)]
        self.assertEqual(warp.local_length, 5)
        self.assertGreater(len(warp.path_to_prototype), 0)
        self.assertGreater(len(warp.path_from_prototype), 0)

    def test_inverse_uses_stored_psi_at_matching_length(self) -> None:
        window = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]), label="anomaly")
        other = EventWindow(series_id="s2", start=0, end=4, values=np.asarray([0.5, 1.5, 2.5, 1.5]), label="anomaly")
        aligner = LearnedEventTimeAligner(grid_size=8).fit([window, other])
        proto = aligner.transform_window(window, grid_size=8)
        with patch("src.alignment.learned_event_time.dtw_path", side_effect=AssertionError("dtw_path should not run")):
            aligner.inverse_transform(proto, window, target_length=len(window.values))

    def test_inverse_reestimates_psi_when_target_length_differs(self) -> None:
        window = EventWindow(series_id="s1", start=0, end=5, values=np.asarray([1.0, 2.0, 3.0, 2.0, 1.0]), label="anomaly")
        other = EventWindow(series_id="s2", start=0, end=4, values=np.asarray([0.5, 1.5, 2.5, 1.5]), label="anomaly")
        aligner = LearnedEventTimeAligner(grid_size=8).fit([window, other])
        proto = aligner.transform_window(window, grid_size=8)
        with patch("src.alignment.learned_event_time.dtw_path", wraps=dtw_path) as mock_dtw:
            aligner.inverse_transform(proto, window, target_length=10)
        self.assertGreater(mock_dtw.call_count, 0)


if __name__ == "__main__":
    unittest.main()
