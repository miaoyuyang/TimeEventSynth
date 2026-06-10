from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.run_backbone_augmentation import (
    _paired_policy_stats,
    _safe_wilcoxon_pvalue,
    _valid_seed_rows,
    aggregate_seed_metrics,
    build_analysis_report,
    build_detector_policy_gain_summary,
    build_detector_policy_pivot,
    build_timesynth_vs_random_win_table,
    build_multi_seed_analysis,
    build_paper_summary,
    reportable_aggregate_metrics,
    summarize_threshold_tradeoff,
    summarize_negative_cases,
)


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"detector_backbone": "iforest", "augmentation_policy": "real_only", "event_f1": 0.2, "best_point_f1": 0.6, "false_positive_events": 2, "compatibility_enabled": False, "metrics_valid": True, "auprc": 0.4, "auroc": 0.5, "synthetic_normal_score_gap": np.nan},
            {"detector_backbone": "iforest", "augmentation_policy": "cross_dataset_all", "event_f1": 0.1, "best_point_f1": 0.55, "false_positive_events": 3, "compatibility_enabled": False, "metrics_valid": True, "auprc": 0.35, "auroc": 0.45, "synthetic_normal_score_gap": -0.1},
            {"detector_backbone": "iforest", "augmentation_policy": "cross_dataset_compatible", "event_f1": 0.15, "best_point_f1": 0.58, "false_positive_events": 1, "compatibility_enabled": True, "metrics_valid": True, "auprc": 0.38, "auroc": 0.48, "synthetic_normal_score_gap": 0.05},
            {"detector_backbone": "iforest", "augmentation_policy": "compatibility_strict", "event_f1": 0.25, "best_point_f1": 0.61, "false_positive_events": 1, "compatibility_enabled": True, "metrics_valid": True, "auprc": 0.41, "auroc": 0.52, "synthetic_normal_score_gap": 0.07},
            {"detector_backbone": "ocsvm", "augmentation_policy": "real_only", "event_f1": 0.0, "best_point_f1": 0.7, "false_positive_events": 2, "compatibility_enabled": False, "metrics_valid": False, "metrics_invalid_reason": "y_true_single_class", "auprc": np.nan, "auroc": np.nan, "synthetic_normal_score_gap": np.nan},
            {"detector_backbone": "ocsvm", "augmentation_policy": "compatibility_strict", "event_f1": 0.0, "best_point_f1": 0.65, "false_positive_events": 3, "compatibility_enabled": True, "metrics_valid": False, "metrics_invalid_reason": "y_true_single_class", "auprc": np.nan, "auroc": np.nan, "synthetic_normal_score_gap": -0.01},
        ]
    )


def test_pivot_generation() -> None:
    pivot = build_detector_policy_pivot(_frame(), metric="event_f1")
    assert "detector_backbone" in pivot.columns
    assert "real_only" in pivot.columns
    assert "compatibility_strict" in pivot.columns


def test_gain_calculations_with_missing_policies() -> None:
    summary = build_detector_policy_gain_summary(_frame())
    row = summary[summary["detector_backbone"] == "iforest"].iloc[0]
    assert np.isfinite(row["gain_compatibility_strict_over_real_only"])


def test_invalid_metrics_are_ignored_in_pivots() -> None:
    pivot = build_detector_policy_pivot(_frame(), metric="event_f1")
    assert "ocsvm" not in set(pivot["detector_backbone"])


def test_analysis_report_contains_required_keys() -> None:
    report = build_analysis_report(_frame())
    report["main_detector_policy_table"] = build_detector_policy_gain_summary(_frame()).to_dict(orient="records")
    report["negative_cases"] = summarize_negative_cases(_frame())
    report["paper_ready_summary"] = build_paper_summary(report, num_detectors=1)
    for key in (
        "best_augmentation_policy_per_detector",
        "average_gain_compatibility_strict_over_real_only",
        "synthetic_augmentation_hurts",
        "paper_ready_summary",
    ):
        assert key in report
    assert "ocsvm" not in report["best_augmentation_policy_per_detector"]


def test_groupwise_policies_are_used_in_gain_summary_and_multi_seed_analysis() -> None:
    frame = pd.DataFrame(
        [
            {"split_seed": 0, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "real_only", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "event_f1": 0.2, "best_point_f1": 0.2, "auroc": 0.2, "auprc": 0.2, "event_precision": 0.2, "event_recall": 0.2, "false_positive_events": 2, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
            {"split_seed": 0, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "groupwise_cross_dataset_all", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": True, "labeled_fraction": 0.2, "event_f1": 0.25, "best_point_f1": 0.25, "auroc": 0.25, "auprc": 0.25, "event_precision": 0.25, "event_recall": 0.25, "false_positive_events": 3, "num_synthetic_windows": 5, "num_synthetic_points": 50, "num_rejected_synthetic_windows": 1, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
            {"split_seed": 0, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "groupwise_cross_dataset_compatible", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": True, "labeled_fraction": 0.2, "event_f1": 0.3, "best_point_f1": 0.3, "auroc": 0.3, "auprc": 0.3, "event_precision": 0.3, "event_recall": 0.3, "false_positive_events": 1, "num_synthetic_windows": 5, "num_synthetic_points": 50, "num_rejected_synthetic_windows": 1, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
            {"split_seed": 0, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "groupwise_compatibility_strict", "donor_policy": "", "synthesis_method": "none", "filter_policy": "strict", "compatibility_enabled": True, "labeled_fraction": 0.2, "event_f1": 0.35, "best_point_f1": 0.35, "auroc": 0.35, "auprc": 0.35, "event_precision": 0.35, "event_recall": 0.35, "false_positive_events": 1, "num_synthetic_windows": 4, "num_synthetic_points": 40, "num_rejected_synthetic_windows": 2, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
        ]
    )
    gain = build_detector_policy_gain_summary(frame)
    row = gain.iloc[0]
    assert row["cross_dataset_compatible_policy"] == "groupwise_cross_dataset_compatible"
    assert row["compatibility_strict_policy"] == "groupwise_compatibility_strict"
    aggregate = aggregate_seed_metrics(frame)
    report = build_multi_seed_analysis(frame, aggregate)
    stats = report["statistical_comparison"]
    assert any(item["policy_a"] == "groupwise_compatibility_strict" for item in stats)
    assert any(item["policy_a"] == "groupwise_cross_dataset_compatible" for item in stats)


def test_aggregate_seed_metrics_ignores_invalid_seed_rows() -> None:
    seed_frame = pd.DataFrame(
        [
            {"split_seed": 0, "dataset": "d", "detector_backbone": "ocsvm", "augmentation_policy": "real_only", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "event_f1": 0.9, "best_point_f1": 0.9, "auroc": 0.9, "auprc": 0.9, "event_precision": 0.9, "event_recall": 0.9, "false_positive_events": 0, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": False},
            {"split_seed": 1, "dataset": "d", "detector_backbone": "ocsvm", "augmentation_policy": "real_only", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "event_f1": 0.1, "best_point_f1": 0.1, "auroc": 0.1, "auprc": 0.1, "event_precision": 0.1, "event_recall": 0.1, "false_positive_events": 5, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
            {"split_seed": 0, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "real_only", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "event_f1": 0.2, "best_point_f1": 0.2, "auroc": 0.2, "auprc": 0.2, "event_precision": 0.2, "event_recall": 0.2, "false_positive_events": 2, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
            {"split_seed": 1, "dataset": "d", "detector_backbone": "iforest", "augmentation_policy": "real_only", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "event_f1": 0.4, "best_point_f1": 0.4, "auroc": 0.4, "auprc": 0.4, "event_precision": 0.4, "event_recall": 0.4, "false_positive_events": 1, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile", "metrics_valid": True},
        ]
    )
    aggregate = aggregate_seed_metrics(seed_frame)
    ocsvm = aggregate[aggregate["detector_backbone"] == "ocsvm"].iloc[0]
    iforest = aggregate[aggregate["detector_backbone"] == "iforest"].iloc[0]
    assert ocsvm["event_f1_mean"] == pytest.approx(0.1)
    assert ocsvm["num_seeds_total"] == 2
    assert ocsvm["num_seeds_valid"] == 1
    assert not bool(ocsvm["metrics_valid_all"])
    assert iforest["event_f1_mean"] == pytest.approx(0.3)
    assert bool(iforest["metrics_valid_all"])


def test_multi_seed_best_policy_skips_invalid_aggregate_groups() -> None:
    seed_frame = pd.DataFrame(
        [
            {"split_seed": 0, "detector_backbone": "ocsvm", "augmentation_policy": "real_only", "event_f1": 0.0, "best_point_f1": 0.7, "false_positive_events": 2, "metrics_valid": False},
            {"split_seed": 0, "detector_backbone": "ocsvm", "augmentation_policy": "compatibility_strict", "event_f1": 0.99, "best_point_f1": 0.99, "false_positive_events": 1, "metrics_valid": False},
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "real_only", "event_f1": 0.2, "best_point_f1": 0.6, "false_positive_events": 2, "metrics_valid": True},
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "compatibility_strict", "event_f1": 0.25, "best_point_f1": 0.61, "false_positive_events": 1, "metrics_valid": True},
        ]
    )
    aggregate = aggregate_seed_metrics(
        pd.DataFrame(
            [
                {**row, "dataset": "d", "donor_policy": "", "synthesis_method": "none", "filter_policy": "no_filter", "compatibility_enabled": False, "labeled_fraction": 0.2, "auroc": 0.5, "auprc": 0.5, "event_precision": 0.5, "event_recall": 0.5, "false_positive_events": 1, "num_synthetic_windows": 0, "num_synthetic_points": 0, "num_rejected_synthetic_windows": 0, "threshold_mode": "quantile", "threshold_calibration_method": "quantile"}
                for row in seed_frame.to_dict(orient="records")
            ]
        )
    )
    report = build_multi_seed_analysis(seed_frame, aggregate)
    assert "ocsvm" not in report["best_policy_per_detector"]
    assert report["best_policy_per_detector"]["iforest"]["augmentation_policy"] == "compatibility_strict"


def test_safe_wilcoxon_handles_identical_deltas_without_runtime_warning() -> None:
    import warnings

    deltas = np.array([0.0, 0.0, 0.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pvalue = _safe_wilcoxon_pvalue(deltas)
    assert pvalue == 1.0
    assert not any(issubclass(item.category, RuntimeWarning) for item in caught)


def test_valid_seed_rows_treat_missing_metrics_valid_as_invalid() -> None:
    frame = pd.DataFrame(
        [
            {"detector_backbone": "iforest", "metrics_valid": True, "event_f1": 0.2},
            {"detector_backbone": "iforest", "metrics_valid": np.nan, "event_f1": 0.99},
            {"detector_backbone": "iforest", "metrics_valid": False, "event_f1": 0.95},
        ]
    )
    valid = _valid_seed_rows(frame)
    assert len(valid) == 1
    assert valid.iloc[0]["event_f1"] == pytest.approx(0.2)


def test_reportable_aggregate_metrics_excludes_partial_or_invalid_groups() -> None:
    aggregate = pd.DataFrame(
        [
            {
                "detector_backbone": "ocsvm",
                "augmentation_policy": "compatibility_strict",
                "event_f1_mean": 0.99,
                "metrics_valid_all": False,
                "num_seeds_valid": 0,
                "num_seeds_invalid": 1,
            },
            {
                "detector_backbone": "iforest",
                "augmentation_policy": "compatibility_strict",
                "event_f1_mean": 0.25,
                "metrics_valid_all": True,
                "num_seeds_valid": 1,
                "num_seeds_invalid": 0,
            },
        ]
    )
    reportable = reportable_aggregate_metrics(aggregate)
    assert set(reportable["detector_backbone"]) == {"iforest"}


def test_paired_policy_stats_wilcoxon_on_identical_seed_deltas() -> None:
    frame = pd.DataFrame(
        [
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "real_only", "event_f1": 0.1, "metrics_valid": True},
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "cross_dataset_all", "event_f1": 0.1, "metrics_valid": True},
            {"split_seed": 1, "detector_backbone": "iforest", "augmentation_policy": "real_only", "event_f1": 0.2, "metrics_valid": True},
            {"split_seed": 1, "detector_backbone": "iforest", "augmentation_policy": "cross_dataset_all", "event_f1": 0.2, "metrics_valid": True},
            {"split_seed": 2, "detector_backbone": "iforest", "augmentation_policy": "real_only", "event_f1": 0.0, "metrics_valid": True},
            {"split_seed": 2, "detector_backbone": "iforest", "augmentation_policy": "cross_dataset_all", "event_f1": 0.0, "metrics_valid": True},
        ]
    )
    stats = _paired_policy_stats(frame, "iforest", "real_only", "cross_dataset_all")
    assert stats["wilcoxon_pvalue"] == 1.0


def test_timesynth_vs_random_win_table_counts_seed_wins_and_fp_reductions() -> None:
    frame = pd.DataFrame(
        [
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "random_event_oversampling", "event_f1": 0.1, "event_precision": 0.1, "event_recall": 0.1, "false_positive_events": 10, "metrics_valid": True},
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "adaptive_groupwise_transfer", "event_f1": 0.2, "event_precision": 0.2, "event_recall": 0.2, "false_positive_events": 8, "metrics_valid": True},
            {"split_seed": 1, "detector_backbone": "iforest", "augmentation_policy": "random_event_oversampling", "event_f1": 0.3, "event_precision": 0.3, "event_recall": 0.3, "false_positive_events": 5, "metrics_valid": True},
            {"split_seed": 1, "detector_backbone": "iforest", "augmentation_policy": "adaptive_groupwise_transfer", "event_f1": 0.1, "event_precision": 0.1, "event_recall": 0.1, "false_positive_events": 7, "metrics_valid": True},
        ]
    )
    table = build_timesynth_vs_random_win_table(frame)
    row = table.iloc[0]
    assert row["event_f1_wins_timeeventsynth"] == 1
    assert row["false_positive_events_lower_fp_seeds"] == 1


def test_threshold_tradeoff_summary_reports_best_and_fp_limited_operating_points() -> None:
    curve = pd.DataFrame(
        [
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "adaptive_groupwise_transfer", "selected_policy_name": "adaptive_groupwise_transfer", "threshold": 0.1, "event_f1": 0.4, "event_precision": 0.2, "event_recall": 1.0, "false_positive_events": 20, "event_count_true": 2},
            {"split_seed": 0, "detector_backbone": "iforest", "augmentation_policy": "adaptive_groupwise_transfer", "selected_policy_name": "adaptive_groupwise_transfer", "threshold": 0.8, "event_f1": 0.3, "event_precision": 1.0, "event_recall": 0.2, "false_positive_events": 0, "event_count_true": 2},
        ]
    )
    summary = summarize_threshold_tradeoff(curve)
    row = summary.iloc[0]
    assert row["best_event_f1_threshold"] == pytest.approx(0.1)
    assert row["fp_limited_threshold"] == pytest.approx(0.8)
