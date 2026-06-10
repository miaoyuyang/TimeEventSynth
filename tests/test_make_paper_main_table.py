from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

spec = importlib.util.spec_from_file_location(
    "make_paper_main_table",
    PROJECT_ROOT / "scripts" / "make_paper_main_table.py",
)
assert spec is not None and spec.loader is not None
make_paper_main_table = importlib.util.module_from_spec(spec)
spec.loader.exec_module(make_paper_main_table)
build_threshold_tradeoff_table = make_paper_main_table.build_threshold_tradeoff_table
build_timesynth_vs_random_table = make_paper_main_table.build_timesynth_vs_random_table


def test_build_timesynth_vs_random_table_formats_detector_wins(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "detector_backbone": "timesnet",
                "num_seeds": 5,
                "event_f1_delta_mean": 0.02,
                "event_f1_wins_timeeventsynth": 3,
                "event_f1_win_rate_timeeventsynth": 0.6,
                "event_precision_delta_mean": 0.01,
                "event_recall_delta_mean": -0.03,
                "false_positive_events_reduction_mean": 4.0,
                "false_positive_events_lower_fp_seeds": 4,
                "false_positive_events_lower_fp_rate": 0.8,
            }
        ]
    ).to_csv(tmp_path / "timeeventsynth_vs_random_win_table.csv", index=False)

    table = build_timesynth_vs_random_table(tmp_path)

    assert table.iloc[0]["Detector"] == "TimesNet"
    assert table.iloc[0]["Event-F1 Wins"] == "3/5 (60.0%)"
    assert table.iloc[0]["FP Reduction"] == "+4.0000"
    assert table.iloc[0]["Lower-FP Wins"] == "4/5 (80.0%)"


def test_build_threshold_tradeoff_table_filters_to_random_and_timeeventsynth(tmp_path: Path) -> None:
    pd.DataFrame(
        [
            {
                "split_seed": 0,
                "detector_backbone": "ocsvm",
                "augmentation_policy": "adaptive_groupwise_transfer",
                "selected_policy_name": "groupwise_cross_dataset_compatible",
                "best_event_f1_threshold": 0.1,
                "best_event_f1": 0.4,
                "best_false_positive_events": 20,
                "fp_limited_threshold": 0.8,
                "fp_limited_event_f1": 0.3,
                "fp_limited_event_precision": 1.0,
                "fp_limited_event_recall": 0.2,
                "fp_limited_false_positive_events": 0,
            },
            {
                "split_seed": 0,
                "detector_backbone": "ocsvm",
                "augmentation_policy": "real_only",
                "selected_policy_name": "real_only",
                "best_event_f1_threshold": 0.1,
                "best_event_f1": 0.4,
                "best_false_positive_events": 20,
                "fp_limited_threshold": 0.8,
                "fp_limited_event_f1": 0.3,
                "fp_limited_event_precision": 1.0,
                "fp_limited_event_recall": 0.2,
                "fp_limited_false_positive_events": 0,
            },
        ]
    ).to_csv(tmp_path / "threshold_tradeoff_summary.csv", index=False)

    table = build_threshold_tradeoff_table(tmp_path)

    assert len(table) == 1
    assert table.iloc[0]["Detector"] == "OCSVM"
    assert table.iloc[0]["Method"] == "TimeEventSynth"
    assert table.iloc[0]["FP-Limited FP"] == "0"
