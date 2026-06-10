"""Compare detector backbones against augmentation policies."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from copy import deepcopy
from pathlib import Path
import sys
import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.policies import AugmentationResult, build_augmentation_result, is_adaptive_policy
from src.datasets.load_options import add_dataset_load_arguments, apply_dataset_cli_to_config
from src.detectors.autoencoder import AutoEncoderBackbone
from src.detectors.cnn import CNNBackbone
from src.detectors.base import DetectorBackbone, flatten_label_records, flatten_score_dict, merge_backbone_threshold_config
from src.detectors.iforest import IForestBackbone
from src.detectors.internal_classifier import InternalClassifierBackbone
from src.detectors.lof import LOFBackbone
from src.detectors.ocsvm import OCSVMBackbone
from src.detectors.timesnet import TimesNetBackbone
from src.experiments.config import load_config
from src.experiments.pipeline import (
    balance_records_by_dataset,
    build_detector_config,
    prepare_low_label_train_and_donor_pool,
    load_options_from_config,
    load_records_for_experiment,
    split_records,
    summarize_dataset_distribution,
    validate_dataset_balanced_test_coverage,
    warn_if_small_test_benchmark,
)
from src.experiments.audit_sanity import annotate_invalid_point_metrics
from src.evaluation.event_metrics import compute_event_metrics
from src.evaluation.point_metrics import compute_point_metrics, parse_prediction_smoothing
from src.utils.seeds import set_global_seed


def _flatten_labels_and_scores(records: list[Any], scores: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    y_true: list[int] = []
    y_score: list[float] = []
    for record in records:
        series_scores = np.asarray(scores[str(record.series_id)], dtype=float).reshape(-1)
        y_true.extend(int(x) for x in np.asarray(record.labels, dtype=int).reshape(-1).tolist())
        y_score.extend(float(x) for x in series_scores.tolist())
    return np.asarray(y_true, dtype=int), np.asarray(y_score, dtype=float)


def _backbone_params(config: dict[str, Any], backbone_name: str) -> dict[str, Any]:
    return dict(config.get("backbones", {}).get(backbone_name, {}))


def _resolve_backbone_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("detector_backbones") or config.get("experiment", {}).get("detector_backbones", [])
    specs: list[dict[str, Any]] = []
    for entry in configured:
        if isinstance(entry, str):
            specs.append({"name": entry, "config": _backbone_params(config, entry)})
        elif isinstance(entry, dict):
            name = str(entry.get("name"))
            merged = dict(_backbone_params(config, name))
            merged.update(dict(entry.get("config", {})))
            specs.append({"name": name, "config": merged})
        else:
            raise ValueError(f"Unsupported detector_backbones entry: {entry!r}")
    return specs


def _resolve_policy_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    configured = config.get("augmentation_policies") or config.get("experiment", {}).get("augmentation_policies", [])
    specs: list[dict[str, Any]] = []
    for entry in configured:
        if isinstance(entry, str):
            specs.append({"name": entry})
        elif isinstance(entry, dict):
            specs.append(dict(entry))
        else:
            raise ValueError(f"Unsupported augmentation_policies entry: {entry!r}")
    return specs


def build_backbone(config: dict[str, Any], backbone_name: str) -> DetectorBackbone:
    name = backbone_name.lower()
    if name in {"current_internal_classifier", "internal_classifier"}:
        params = _backbone_params(config, backbone_name)
        detector_cfg = build_detector_config(config)
        detector_cfg.update(params)
        backbone = InternalClassifierBackbone(detector_config=detector_cfg)
        for field in ("threshold_mode", "quantile", "threshold_quantile", "grid_size", "metric"):
            if params.get(field) is not None:
                setattr(backbone, field, params[field])
        return backbone
    if name == "iforest":
        params = _backbone_params(config, backbone_name)
        return IForestBackbone(
            window_size=int(params.get("window_size", config.get("detector", {}).get("window_size", 15))),
            stride=int(params.get("stride", 1)),
            contamination=float(params.get("contamination", 0.1)),
            random_state=int(params.get("random_state", config.get("seed", 42))),
            n_estimators=int(params.get("n_estimators", 200)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            max_train_windows=params.get("max_train_windows"),
        )
    if name == "ocsvm":
        params = _backbone_params(config, backbone_name)
        return OCSVMBackbone(
            window_size=int(params.get("window_size", config.get("detector", {}).get("window_size", 15))),
            stride=int(params.get("stride", 1)),
            kernel=str(params.get("kernel", "rbf")),
            gamma=params.get("gamma", "scale"),
            nu=float(params.get("nu", 0.1)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            max_train_windows=params.get("max_train_windows"),
        )
    if name == "lof":
        params = _backbone_params(config, backbone_name)
        return LOFBackbone(
            window_size=int(params.get("window_size", config.get("detector", {}).get("window_size", 15))),
            stride=int(params.get("stride", 1)),
            contamination=float(params.get("contamination", 0.1)),
            n_neighbors=int(params.get("n_neighbors", 20)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            max_train_windows=params.get("max_train_windows"),
        )
    if name == "autoencoder":
        params = _backbone_params(config, backbone_name)
        return AutoEncoderBackbone(
            window_size=int(params.get("window_size", config.get("detector", {}).get("window_size", 15))),
            num_lags=int(params.get("num_lags", config.get("detector", {}).get("num_lags", 2))),
            hidden_layer_sizes=tuple(params.get("hidden_layer_sizes", (64, 32, 64))),
            random_state=int(params.get("random_state", config.get("seed", 42))),
            max_iter=int(params.get("max_iter", 200)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            train_with_synthetic=bool(params.get("train_with_synthetic", False)),
            max_train_points=params.get("max_train_points"),
        )
    if name == "cnn":
        params = _backbone_params(config, backbone_name)
        return CNNBackbone(
            window_size=int(params.get("window_size", 32)),
            stride=int(params.get("stride", 1)),
            horizon=int(params.get("horizon", 1)),
            hidden_channels=int(params.get("hidden_channels", 32)),
            kernel_size=int(params.get("kernel_size", 3)),
            random_state=int(params.get("random_state", config.get("seed", 42))),
            epochs=int(params.get("epochs", 10)),
            batch_size=int(params.get("batch_size", 128)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            train_with_synthetic=bool(params.get("train_with_synthetic", False)),
            max_train_windows=params.get("max_train_windows"),
            device=str(params.get("device", "cpu")),
        )
    if name == "timesnet":
        params = _backbone_params(config, backbone_name)
        return TimesNetBackbone(
            window_size=int(params.get("window_size", 48)),
            stride=int(params.get("stride", 1)),
            horizon=int(params.get("horizon", 1)),
            d_model=int(params.get("d_model", 32)),
            top_k_periods=int(params.get("top_k_periods", 3)),
            num_blocks=int(params.get("num_blocks", 2)),
            random_state=int(params.get("random_state", config.get("seed", 42))),
            epochs=int(params.get("epochs", 10)),
            batch_size=int(params.get("batch_size", 128)),
            learning_rate=float(params.get("learning_rate", 1e-3)),
            threshold_mode=str(params.get("threshold_mode", "synthetic_separation")),
            quantile=float(params.get("quantile", 0.95)),
            train_with_synthetic=bool(params.get("train_with_synthetic", False)),
            max_train_windows=params.get("max_train_windows"),
            device=str(params.get("device", "cpu")),
        )
    raise ValueError(f"Unsupported detector backbone: {backbone_name}")


def build_backbone_from_spec(config: dict[str, Any], spec: dict[str, Any]) -> DetectorBackbone:
    name = str(spec["name"])
    cfg = dict(config)
    cfg.setdefault("backbones", {})
    cfg["backbones"] = dict(cfg["backbones"])
    cfg["backbones"][name] = dict(spec.get("config", {}))
    backbone = build_backbone(cfg, name)
    backbone.name = name
    return backbone


def _num_synthetic_windows(augmentation: AugmentationResult) -> int:
    if augmentation.audit_records:
        return sum(
            1
            for row in augmentation.audit_records
            if str(row.get("record_type")) == "synthesis_candidate" and bool(row.get("accepted", row.get("kept", False)))
        )
    return len(augmentation.synthetic_windows)


def _num_rejected_synthetic_windows(augmentation: AugmentationResult) -> int:
    if not augmentation.audit_records:
        return 0
    return sum(
        1
        for row in augmentation.audit_records
        if str(row.get("record_type")) == "synthesis_candidate" and not bool(row.get("accepted", row.get("kept", False)))
    )


def _num_synthetic_points(augmentation: AugmentationResult) -> int:
    return int(sum(len(np.asarray(window.labels, dtype=int)) for window in augmentation.synthetic_windows))


def _augmentation_summary_key(policy_name: str, detector_backbone: str | None = None) -> str:
    if detector_backbone:
        return f"{policy_name}__{detector_backbone}"
    return policy_name


def _annotate_augmentation_records(
    augmentation: AugmentationResult,
    *,
    detector_backbone: str,
) -> None:
    for row in augmentation.audit_records:
        row["detector_backbone"] = detector_backbone
        row["selected_policy_name"] = augmentation.selected_policy_name or augmentation.policy_name
        row["selection_reason"] = augmentation.selection_reason or ""
        row["fallback_used"] = bool(augmentation.fallback_used)


def build_detector_policy_pivot(frame: pd.DataFrame, *, metric: str) -> pd.DataFrame:
    """Pivot detector x policy performance for one metric."""
    if frame.empty:
        return pd.DataFrame(columns=["detector_backbone"])
    working = _analysis_ready_frame(frame)
    if working.empty:
        return pd.DataFrame(columns=["detector_backbone"])
    pivot = working.pivot_table(
        index="detector_backbone",
        columns="augmentation_policy",
        values=metric,
        aggfunc="mean",
    ).reset_index()
    pivot.columns.name = None
    return pivot


def _safe_policy_value(group: pd.DataFrame, policy: str, metric: str) -> float:
    subset = group[group["augmentation_policy"] == policy]
    if subset.empty:
        return float("nan")
    values = pd.to_numeric(subset[metric], errors="coerce").dropna()
    return float(values.mean()) if not values.empty else float("nan")


def build_detector_policy_gain_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Detector-level summary table centered on paper-facing policy comparisons."""
    rows: list[dict[str, Any]] = []
    working = _analysis_ready_frame(frame)
    if working.empty:
        return pd.DataFrame(columns=["detector_backbone"])
    for detector, group in working.groupby("detector_backbone", dropna=False):
        compat_policy = _preferred_policy(group, ["groupwise_compatibility_strict", "compatibility_strict"])
        cross_compat_policy = _preferred_policy(group, ["groupwise_cross_dataset_compatible", "cross_dataset_compatible"])
        cross_all_policy = _preferred_policy(group, ["groupwise_cross_dataset_all", "cross_dataset_all"])
        adaptive_policy = _preferred_policy(group, ["adaptive_groupwise_transfer"])
        real = _safe_policy_value(group, "real_only", "event_f1")
        random_os = _safe_policy_value(group, "random_event_oversampling", "event_f1")
        cross_all = _safe_policy_value(group, cross_all_policy or "cross_dataset_all", "event_f1")
        cross_compat = _safe_policy_value(group, cross_compat_policy or "cross_dataset_compatible", "event_f1")
        compat_strict = _safe_policy_value(group, compat_policy or "compatibility_strict", "event_f1")
        adaptive = _safe_policy_value(group, adaptive_policy or "adaptive_groupwise_transfer", "event_f1")
        false_pos_real = _safe_policy_value(group, "real_only", "false_positive_events")
        false_pos_strict = _safe_policy_value(group, compat_policy or "compatibility_strict", "false_positive_events")
        false_pos_adaptive = _safe_policy_value(group, adaptive_policy or "adaptive_groupwise_transfer", "false_positive_events")
        rows.append(
            {
                "detector_backbone": str(detector),
                "real_only_event_f1": real,
                "random_event_oversampling_event_f1": random_os,
                "cross_dataset_all_event_f1": cross_all,
                "cross_dataset_compatible_event_f1": cross_compat,
                "compatibility_strict_event_f1": compat_strict,
                "adaptive_groupwise_transfer_event_f1": adaptive,
                "cross_dataset_all_policy": cross_all_policy or "",
                "cross_dataset_compatible_policy": cross_compat_policy or "",
                "compatibility_strict_policy": compat_policy or "",
                "adaptive_groupwise_transfer_policy": adaptive_policy or "",
                "gain_compatibility_strict_over_real_only": compat_strict - real if np.isfinite(compat_strict) and np.isfinite(real) else float("nan"),
                "gain_compatibility_strict_over_cross_dataset_all": compat_strict - cross_all if np.isfinite(compat_strict) and np.isfinite(cross_all) else float("nan"),
                "false_positive_change_vs_real_only": false_pos_strict - false_pos_real if np.isfinite(false_pos_strict) and np.isfinite(false_pos_real) else float("nan"),
                "gain_adaptive_groupwise_transfer_over_real_only": adaptive - real if np.isfinite(adaptive) and np.isfinite(real) else float("nan"),
                "adaptive_false_positive_change_vs_real_only": false_pos_adaptive - false_pos_real if np.isfinite(false_pos_adaptive) and np.isfinite(false_pos_real) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def build_threshold_calibration_summary(frame: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "detector_backbone",
        "augmentation_policy",
        "selected_policy_name",
        "selection_reason",
        "fallback_used",
        "threshold",
        "threshold_mode",
        "threshold_calibration_method",
        "calibration_f1",
        "synthetic_score_mean",
        "normal_score_mean",
        "synthetic_normal_score_gap",
    ]
    present = [col for col in cols if col in frame.columns]
    return frame[present].copy()


def summarize_compatibility_mechanism(
    audit_rows: list[dict[str, Any]],
    rejection_summary: dict[str, Any],
    compatibility_summary: dict[str, Any],
) -> dict[str, Any]:
    donor_rows = [row for row in audit_rows if str(row.get("record_type")) == "donor_pair"]
    kept = [row for row in donor_rows if bool(row.get("accepted", False))]
    rejected = [row for row in donor_rows if not bool(row.get("accepted", False))]
    cross_rows = [row for row in donor_rows if bool(row.get("source_dataset")) and bool(row.get("target_dataset")) and str(row.get("source_dataset")) != str(row.get("target_dataset"))]
    same_rows = [row for row in donor_rows if bool(row.get("source_dataset")) and str(row.get("source_dataset")) == str(row.get("target_dataset"))]
    kept_scores = [float(row["compatibility_score"]) for row in kept if pd.notna(row.get("compatibility_score"))]
    rejected_scores = [float(row["compatibility_score"]) for row in rejected if pd.notna(row.get("compatibility_score"))]

    reason_counts: dict[str, int] = {}
    for per_policy in rejection_summary.values():
        if not isinstance(per_policy, dict):
            continue
        for payload in per_policy.values():
            if not isinstance(payload, dict):
                continue
            for reason, count in (payload.get("reasons") or {}).items():
                reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + int(count)
    top_reasons = sorted(reason_counts.items(), key=lambda item: (-item[1], item[0]))[:10]

    return {
        "cross_dataset_pairs_considered": int(len(cross_rows)),
        "cross_dataset_kept": int(sum(1 for row in cross_rows if bool(row.get("accepted", False)))),
        "cross_dataset_rejected": int(sum(1 for row in cross_rows if not bool(row.get("accepted", False)))),
        "same_dataset_kept": int(sum(1 for row in same_rows if bool(row.get("accepted", False)))),
        "same_dataset_rejected": int(sum(1 for row in same_rows if not bool(row.get("accepted", False)))),
        "mean_compatibility_score_kept": float(np.mean(kept_scores)) if kept_scores else float("nan"),
        "mean_compatibility_score_rejected": float(np.mean(rejected_scores)) if rejected_scores else float("nan"),
        "top_rejection_reasons": [{"reason": reason, "count": count} for reason, count in top_reasons],
        "per_policy_compatibility_summary": compatibility_summary,
    }


def summarize_threshold_calibration_effects(frame: pd.DataFrame) -> dict[str, Any]:
    working = _valid_seed_rows(frame)
    rows: list[dict[str, Any]] = []
    larger_gap_count = 0
    comparable = 0
    for detector, group in working.groupby("detector_backbone", dropna=False):
        naive = group[group["augmentation_policy"] == "cross_dataset_all"]
        compat = group[group["augmentation_policy"] == "cross_dataset_compatible"]
        if naive.empty or compat.empty:
            continue
        comparable += 1
        naive_gap = float(pd.to_numeric(naive["synthetic_normal_score_gap"], errors="coerce").mean())
        compat_gap = float(pd.to_numeric(compat["synthetic_normal_score_gap"], errors="coerce").mean())
        if np.isfinite(naive_gap) and np.isfinite(compat_gap) and compat_gap > naive_gap:
            larger_gap_count += 1
        rows.append(
            {
                "detector_backbone": str(detector),
                "naive_gap": naive_gap,
                "compatible_gap": compat_gap,
                "compatible_gap_larger": bool(np.isfinite(naive_gap) and np.isfinite(compat_gap) and compat_gap > naive_gap),
            }
        )
    return {
        "threshold_mode_by_detector_policy": build_threshold_calibration_summary(frame).to_dict(orient="records"),
        "compatibility_aware_larger_score_gap_count": int(larger_gap_count),
        "compatibility_aware_larger_score_gap_comparable_detectors": int(comparable),
        "gap_comparisons": rows,
    }


def summarize_negative_cases(frame: pd.DataFrame) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    working = _valid_seed_rows(frame)
    for detector, group in working.groupby("detector_backbone", dropna=False):
        real = group[group["augmentation_policy"] == "real_only"]
        if real.empty:
            continue
        real_row = real.iloc[0]
        for _, row in group.iterrows():
            if row["augmentation_policy"] == "real_only":
                continue
            reasons: list[str] = []
            if str(row["augmentation_policy"]) == "compatibility_strict" and float(row["event_f1"]) < float(real_row["event_f1"]):
                reasons.append("compatibility_strict_hurts_event_f1")
            if float(row["false_positive_events"]) > float(real_row["false_positive_events"]):
                reasons.append("synthetic_augmentation_increases_false_positives")
            gap = pd.to_numeric(pd.Series([row.get("synthetic_normal_score_gap")]), errors="coerce").iloc[0]
            if pd.notna(gap) and float(gap) <= 0.01:
                reasons.append("synthetic_normal_score_gap_nonpositive_or_small")
            if int(row.get("num_synthetic_windows", 0)) < 5 and str(row["augmentation_policy"]) != "real_only":
                reasons.append("too_few_synthetic_windows")
            if reasons:
                cases.append(
                    {
                        "detector_backbone": str(detector),
                        "augmentation_policy": str(row["augmentation_policy"]),
                        "reasons": reasons,
                    }
                )
    return cases


def build_paper_summary(report: dict[str, Any], num_detectors: int) -> str:
    improved = int(report.get("num_backbones_improved_by_compatibility_strict", 0))
    fp_reduction = int(
        sum(
            1
            for row in report.get("improvement_characterization", [])
            if float(row.get("deltas", {}).get("false_positive_reduction", 0.0)) > 0
        )
    )
    avg_gain = float(report.get("average_gain_compatibility_strict_over_real_only", 0.0))
    avg_cross = float(report.get("average_gain_cross_dataset_compatible_over_cross_dataset_all", 0.0))
    if improved == 0 and avg_gain <= 0:
        return (
            f"Across {num_detectors} detector backbones, compatibility-aware augmentation does not show a consistent "
            f"event-F1 improvement over real-only training in this run. Compared with naive cross-dataset transfer, "
            f"compatibility-aware transfer changes event F1 by {avg_cross:.4f} on average, indicating mixed or weak gains."
        )
    return (
        f"Across {num_detectors} detector backbones, compatibility-aware augmentation improves event F1 for "
        f"{improved} detectors and reduces false-positive events for {fp_reduction} detectors. Compared with naive "
        f"cross-dataset transfer, compatibility-aware transfer changes event F1 by {avg_cross:.4f} on average, "
        f"while compatibility_strict changes event F1 by {avg_gain:.4f} over real_only."
    )


def evaluate_backbone_with_augmentation(
    backbone: DetectorBackbone,
    augmentation: AugmentationResult,
    *,
    train_records: list[Any],
    val_records: list[Any],
    test_records: list[Any],
    config: dict[str, Any],
    labeled_fraction: float,
    split_seed: int,
    backbone_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    train_labels = [np.asarray(record.labels, dtype=int).reshape(-1) for record in train_records]
    backbone.fit(
        train_records,
        train_labels=train_labels,
        synthetic_windows=augmentation.synthetic_windows if backbone.supports_training_augmentation else None,
        config=config,
    )

    val_scores = backbone.score(val_records, config=config)
    test_scores = backbone.score(test_records, config=config)
    val_y_true, val_y_score = _flatten_labels_and_scores(val_records, val_scores)
    test_y_true, test_y_score = _flatten_labels_and_scores(test_records, test_scores)

    synthetic_scores = None
    if augmentation.synthetic_windows:
        synthetic_scores = flatten_score_dict(backbone.score(augmentation.synthetic_windows, config=config))

    threshold_cfg = merge_backbone_threshold_config(
        backbone,
        config.get("evaluation", {}),
        backbone_config=backbone_config,
    )
    calibration = backbone.calibrate_threshold(
        val_y_score,
        labels=val_y_true,
        synthetic_scores=synthetic_scores,
        config=threshold_cfg,
    )
    threshold = float(calibration.threshold)

    smoothing = parse_prediction_smoothing(config.get("evaluation", {}))
    min_event_length = smoothing["min_event_length"] if smoothing["enabled"] else 1
    merge_gap = smoothing["merge_gap"] if smoothing["enabled"] else 0
    iou_threshold = float(config.get("evaluation", {}).get("event_iou_threshold", 0.1))

    point_metrics = compute_point_metrics(test_y_true.tolist(), test_y_score.tolist(), threshold=threshold)
    point_metrics = annotate_invalid_point_metrics(test_y_true, test_y_score, point_metrics)
    event_metrics = compute_event_metrics(
        test_y_true.tolist(),
        y_score=test_y_score.tolist(),
        threshold=threshold,
        iou_threshold=iou_threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )
    metrics_valid = point_metrics.get("metrics_valid")
    return {
        "split_seed": int(split_seed),
        "dataset": str(config.get("dataset", {}).get("name", "dataset")),
        "detector_backbone": backbone.name,
        "augmentation_policy": augmentation.policy_name,
        "selected_policy_name": augmentation.selected_policy_name or augmentation.policy_name,
        "selection_reason": augmentation.selection_reason or "",
        "fallback_used": bool(augmentation.fallback_used),
        "donor_policy": augmentation.donor_policy or "",
        "synthesis_method": augmentation.synthesis_method,
        "filter_policy": augmentation.filter_policy or "",
        "compatibility_enabled": augmentation.compatibility_enabled,
        "labeled_fraction": float(labeled_fraction),
        "auroc": float(point_metrics["point_auroc"]),
        "auprc": float(point_metrics["point_auprc"]),
        "best_point_f1": float(point_metrics["point_f1"]),
        "event_precision": float(event_metrics["event_precision"]),
        "event_recall": float(event_metrics["event_recall"]),
        "event_f1": float(event_metrics["event_f1"]),
        "false_positive_events": int(event_metrics["false_positive_event_count"]),
        "threshold": threshold,
        "threshold_mode": str(calibration.details.get("threshold_mode", calibration.method)),
        "threshold_calibration_method": calibration.method,
        "calibration_f1": float(calibration.details.get("best_calibration_f1", calibration.details.get("best_metric_value", np.nan))),
        "synthetic_score_mean": float(calibration.details.get("synthetic_score_mean", np.nan)),
        "normal_score_mean": float(calibration.details.get("normal_score_mean", np.nan)),
        "synthetic_normal_score_gap": float(calibration.details.get("score_gap", np.nan)),
        "num_synthetic_windows": _num_synthetic_windows(augmentation),
        "num_synthetic_points": _num_synthetic_points(augmentation),
        "num_rejected_synthetic_windows": _num_rejected_synthetic_windows(augmentation),
        "metrics_valid": bool(metrics_valid is True),
        "metrics_invalid_reason": str(point_metrics.get("metrics_invalid_reason", "" if metrics_valid is True else "metrics_valid_missing")),
    }


def build_analysis_report(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}

    frame = _analysis_ready_frame(frame)
    if frame.empty:
        return {}

    summary: dict[str, Any] = {}
    best_per_detector: dict[str, Any] = {}
    for detector, group in frame.groupby("detector_backbone", dropna=False):
        best = group.sort_values(["event_f1", "auprc", "auroc"], ascending=[False, False, False]).iloc[0]
        best_per_detector[str(detector)] = {
            "augmentation_policy": str(best["augmentation_policy"]),
            "event_f1": float(best["event_f1"]),
            "auprc": float(best["auprc"]),
            "auroc": float(best["auroc"]),
        }
    summary["best_augmentation_policy_per_detector"] = best_per_detector

    strict_vs_real = frame.pivot_table(
        index="detector_backbone",
        columns="augmentation_policy",
        values="event_f1",
        aggfunc="mean",
    )
    compat_col = "groupwise_compatibility_strict" if "groupwise_compatibility_strict" in strict_vs_real.columns else "compatibility_strict"
    cross_compat_col = "groupwise_cross_dataset_compatible" if "groupwise_cross_dataset_compatible" in strict_vs_real.columns else "cross_dataset_compatible"
    cross_all_col = "groupwise_cross_dataset_all" if "groupwise_cross_dataset_all" in strict_vs_real.columns else "cross_dataset_all"
    if {compat_col, "real_only"}.issubset(strict_vs_real.columns):
        gains = strict_vs_real[compat_col] - strict_vs_real["real_only"]
        summary["average_gain_compatibility_strict_over_real_only"] = float(gains.mean())
        summary["num_backbones_improved_by_compatibility_strict"] = int((gains > 0).sum())
        summary["compatibility_strict_policy_name"] = compat_col
    if {cross_compat_col, cross_all_col}.issubset(strict_vs_real.columns):
        gains = strict_vs_real[cross_compat_col] - strict_vs_real[cross_all_col]
        summary["average_gain_cross_dataset_compatible_over_cross_dataset_all"] = float(gains.mean())
        summary["num_backbones_improved_by_cross_dataset_compatible"] = int((gains > 0).sum())
        summary["cross_dataset_compatible_policy_name"] = cross_compat_col
        summary["cross_dataset_all_policy_name"] = cross_all_col

    improvement_rows = []
    has_event_recall = "event_recall" in frame.columns
    has_event_precision = "event_precision" in frame.columns
    has_false_positive = "false_positive_events" in frame.columns
    for detector, group in frame.groupby("detector_backbone", dropna=False):
        real = group[group["augmentation_policy"] == "real_only"]
        compat = group[group["augmentation_policy"].isin([
            "compatibility_strict",
            "cross_dataset_compatible",
            "groupwise_compatibility_strict",
            "groupwise_cross_dataset_compatible",
        ])]
        if real.empty or compat.empty:
            continue
        best_compat = compat.sort_values(["event_f1", "auprc"], ascending=[False, False]).iloc[0]
        base = real.iloc[0]
        deltas = {
            "event_recall": float(best_compat["event_recall"] - base["event_recall"]) if has_event_recall else 0.0,
            "event_precision": float(best_compat["event_precision"] - base["event_precision"]) if has_event_precision else 0.0,
            "false_positive_reduction": float(base["false_positive_events"] - best_compat["false_positive_events"]) if has_false_positive else 0.0,
        }
        dominant = max(deltas, key=lambda key: abs(deltas[key]))
        improvement_rows.append(
            {
                "detector_backbone": str(detector),
                "best_compatible_policy": str(best_compat["augmentation_policy"]),
                "dominant_change": dominant,
                "deltas": deltas,
            }
        )
    summary["improvement_characterization"] = improvement_rows
    summary["num_detector_backbones_improved_by_compatibility_aware_augmentation"] = int(
        sum(1 for row in improvement_rows if row["deltas"]["event_recall"] > 0 or row["deltas"]["event_precision"] > 0 or row["deltas"]["false_positive_reduction"] > 0)
    )
    hurts: list[dict[str, Any]] = []
    for detector, group in frame.groupby("detector_backbone", dropna=False):
        real = group[group["augmentation_policy"] == "real_only"]
        if real.empty:
            continue
        base = real.iloc[0]
        for _, row in group.iterrows():
            if row["augmentation_policy"] == "real_only":
                continue
            if float(row["event_f1"]) < float(base["event_f1"]):
                hurts.append(
                    {
                        "detector_backbone": str(detector),
                        "augmentation_policy": str(row["augmentation_policy"]),
                        "event_f1_drop": float(row["event_f1"] - base["event_f1"]),
                    }
                )
    summary["synthetic_augmentation_hurts"] = hurts
    summary["average_false_positive_reduction"] = float(
        np.mean([row["deltas"]["false_positive_reduction"] for row in improvement_rows]) if improvement_rows else 0.0
    )
    return summary


def analysis_report_markdown(report: dict[str, Any]) -> str:
    if not report:
        return "# Analysis Report\n\nNo results available.\n"
    lines = ["# Analysis Report", ""]
    lines.append("## Best Augmentation Policy Per Detector")
    for detector, payload in report.get("best_augmentation_policy_per_detector", {}).items():
        lines.append(
            f"- `{detector}`: `{payload['augmentation_policy']}` "
            f"(event_f1={payload['event_f1']:.4f}, auprc={payload['auprc']:.4f})"
        )
    lines.append("")
    if "average_gain_compatibility_strict_over_real_only" in report:
        lines.append(
            f"- Average event-F1 gain of `compatibility_strict` over `real_only`: "
            f"{report['average_gain_compatibility_strict_over_real_only']:.4f}"
        )
    if "average_gain_cross_dataset_compatible_over_cross_dataset_all" in report:
        lines.append(
            f"- Average event-F1 gain of `cross_dataset_compatible` over `cross_dataset_all`: "
            f"{report['average_gain_cross_dataset_compatible_over_cross_dataset_all']:.4f}"
        )
    lines.append(
        f"- Detector backbones improved by compatibility-aware augmentation: "
        f"{report.get('num_detector_backbones_improved_by_compatibility_aware_augmentation', 0)}"
    )
    lines.append(
        f"- Average false-positive reduction: {report.get('average_false_positive_reduction', 0.0):.4f}"
    )
    lines.append("")
    lines.append("## Improvement Characterization")
    for row in report.get("improvement_characterization", []):
        lines.append(
            f"- `{row['detector_backbone']}` with `{row['best_compatible_policy']}`: "
            f"mainly `{row['dominant_change']}` {row['deltas']}"
        )
    if report.get("synthetic_augmentation_hurts"):
        lines.append("")
        lines.append("## Cases Where Synthetic Augmentation Hurts")
        for row in report["synthetic_augmentation_hurts"]:
            lines.append(
                f"- `{row['detector_backbone']}` with `{row['augmentation_policy']}`: "
                f"event_f1_drop={row['event_f1_drop']:.4f}"
            )
    lines.append("")
    return "\n".join(lines) + "\n"


AGGREGATE_METRICS = [
    "auroc",
    "auprc",
    "best_point_f1",
    "event_precision",
    "event_recall",
    "event_f1",
    "false_positive_events",
]


def _preferred_policy(group: pd.DataFrame, candidates: list[str]) -> str | None:
    for policy in candidates:
        if not group[group["augmentation_policy"] == policy].empty:
            return policy
    return None


def _valid_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep seed-level rows with explicitly valid evaluation metrics."""
    if frame.empty:
        return frame.copy()
    if "metrics_valid" not in frame.columns:
        return frame.copy()
    return frame[frame["metrics_valid"].eq(True)].copy()


def _reportable_aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep aggregate rows that are safe for paper-facing summaries and policy ranking."""
    if frame.empty:
        return frame.copy()
    if "metrics_valid_all" not in frame.columns:
        working = _valid_seed_rows(frame)
    else:
        mask = frame["metrics_valid_all"].eq(True)
        if "num_seeds_valid" in frame.columns:
            mask &= frame["num_seeds_valid"].fillna(0).astype(int) > 0
        working = frame[mask].copy()
    if "event_f1_mean" in working.columns:
        event_f1 = pd.to_numeric(working["event_f1_mean"], errors="coerce")
        working = working[event_f1.map(np.isfinite)].copy()
    return working.reset_index(drop=True)


def _analysis_ready_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Filter seed-level or aggregate frames before reporting tables."""
    if frame.empty:
        return frame.copy()
    if "metrics_valid_all" in frame.columns:
        return _reportable_aggregate_rows(frame)
    return _valid_seed_rows(frame)


def _safe_wilcoxon_pvalue(deltas: np.ndarray | pd.Series) -> float:
    """Wilcoxon signed-rank test with a guard for zero-variance paired deltas."""
    values = np.asarray(deltas, dtype=float).reshape(-1)
    if values.size < 2:
        return float("nan")
    if np.allclose(values, values[0]):
        return 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(wilcoxon(values).pvalue)


def _safe_nanmean(values: list[Any] | np.ndarray) -> float:
    """Mean over finite values; returns NaN without RuntimeWarning when empty or all-NaN."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmean(arr))


def _safe_nanmedian(values: list[Any] | np.ndarray) -> float:
    """Median over finite values; returns NaN without RuntimeWarning when empty or all-NaN."""
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return float("nan")
    return float(np.nanmedian(arr))


def _paired_policy_stats(frame: pd.DataFrame, detector: str, policy_a: str, policy_b: str, metric: str = "event_f1") -> dict[str, Any]:
    subset = frame[
        (frame["detector_backbone"] == detector)
        & (frame["augmentation_policy"].isin([policy_a, policy_b]))
        & (frame["metrics_valid"].eq(True))
    ].copy()
    if subset.empty:
        return {
            "detector_backbone": detector,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "metric": metric,
            "mean_delta": float("nan"),
            "std_delta": float("nan"),
            "wins_policy_a": 0,
            "num_seeds": 0,
            "win_rate_policy_a": float("nan"),
        }
    pivot = subset.pivot_table(index="split_seed", columns="augmentation_policy", values=metric, aggfunc="mean")
    if not {policy_a, policy_b}.issubset(set(pivot.columns)):
        return {
            "detector_backbone": detector,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "metric": metric,
            "mean_delta": float("nan"),
            "std_delta": float("nan"),
            "wins_policy_a": 0,
            "num_seeds": 0,
            "win_rate_policy_a": float("nan"),
        }
    deltas = (pivot[policy_a] - pivot[policy_b]).dropna()
    if deltas.empty:
        return {
            "detector_backbone": detector,
            "policy_a": policy_a,
            "policy_b": policy_b,
            "metric": metric,
            "mean_delta": float("nan"),
            "std_delta": float("nan"),
            "wins_policy_a": 0,
            "num_seeds": 0,
            "win_rate_policy_a": float("nan"),
        }
    result: dict[str, Any] = {
        "detector_backbone": detector,
        "policy_a": policy_a,
        "policy_b": policy_b,
        "metric": metric,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std(ddof=0)),
        "wins_policy_a": int((deltas > 0).sum()),
        "num_seeds": int(deltas.shape[0]),
        "win_rate_policy_a": float((deltas > 0).mean()),
    }
    if deltas.shape[0] >= 2:
        try:
            result["paired_t_pvalue"] = float(ttest_rel(pivot.loc[deltas.index, policy_a], pivot.loc[deltas.index, policy_b]).pvalue)
        except Exception:
            result["paired_t_pvalue"] = float("nan")
        try:
            result["wilcoxon_pvalue"] = _safe_wilcoxon_pvalue(deltas)
        except Exception:
            result["wilcoxon_pvalue"] = float("nan")
    return result


def aggregate_seed_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    for col, default in (
        ("selected_policy_name", ""),
        ("selection_reason", ""),
        ("fallback_used", False),
    ):
        if col not in working.columns:
            working[col] = default
    group_cols = [
        "dataset",
        "detector_backbone",
        "augmentation_policy",
        "selected_policy_name",
        "selection_reason",
        "fallback_used",
        "donor_policy",
        "synthesis_method",
        "filter_policy",
        "compatibility_enabled",
        "labeled_fraction",
    ]
    rows: list[dict[str, Any]] = []
    for keys, group in working.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        valid_group = _valid_seed_rows(group)
        row = {col: value for col, value in zip(group_cols, keys)}
        row["num_seeds_total"] = int(group["split_seed"].nunique())
        row["num_seeds_valid"] = int(valid_group["split_seed"].nunique()) if not valid_group.empty else 0
        row["num_seeds_invalid"] = int(row["num_seeds_total"] - row["num_seeds_valid"])
        row["num_seeds"] = int(row["num_seeds_valid"])
        threshold_modes = valid_group["threshold_mode"].dropna().astype(str)
        calibration_methods = valid_group["threshold_calibration_method"].dropna().astype(str)
        row["threshold_mode"] = str(threshold_modes.mode().iloc[0]) if not threshold_modes.empty else ""
        row["threshold_calibration_method"] = str(calibration_methods.mode().iloc[0]) if not calibration_methods.empty else ""
        for metric in AGGREGATE_METRICS:
            values = pd.to_numeric(valid_group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else float("nan")
            row[f"{metric}_std"] = float(values.std(ddof=0)) if not values.empty else float("nan")
            row[f"{metric}_median"] = float(values.median()) if not values.empty else float("nan")
            row[f"{metric}_min"] = float(values.min()) if not values.empty else float("nan")
            row[f"{metric}_max"] = float(values.max()) if not values.empty else float("nan")
        if valid_group.empty:
            row["num_synthetic_windows_mean"] = float("nan")
            row["num_synthetic_points_mean"] = float("nan")
            row["num_rejected_synthetic_windows_mean"] = float("nan")
        else:
            row["num_synthetic_windows_mean"] = float(pd.to_numeric(valid_group["num_synthetic_windows"], errors="coerce").mean())
            row["num_synthetic_points_mean"] = float(pd.to_numeric(valid_group["num_synthetic_points"], errors="coerce").mean())
            row["num_rejected_synthetic_windows_mean"] = float(pd.to_numeric(valid_group["num_rejected_synthetic_windows"], errors="coerce").mean())
        row["metrics_valid_partial"] = bool(
            row["num_seeds_valid"] > 0 and row["num_seeds_invalid"] > 0
        )
        row["metrics_valid_all"] = bool(row["num_seeds_invalid"] == 0 and row["num_seeds_valid"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)


def reportable_aggregate_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate metrics restricted to groups with valid metrics on every seed."""
    return _reportable_aggregate_rows(frame)


def build_multi_seed_analysis(seed_frame: pd.DataFrame, aggregate_frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {}
    valid_seed = _valid_seed_rows(seed_frame)
    aggregate_frame = _reportable_aggregate_rows(aggregate_frame)
    best_policy_per_detector: dict[str, Any] = {}
    if not aggregate_frame.empty:
        for detector, group in aggregate_frame.groupby("detector_backbone", dropna=False):
            best = group.sort_values(["event_f1_mean", "best_point_f1_mean"], ascending=[False, False]).iloc[0]
            best_policy_per_detector[str(detector)] = {
                "augmentation_policy": str(best["augmentation_policy"]),
                "event_f1_mean": float(best["event_f1_mean"]),
                "event_f1_std": float(best["event_f1_std"]),
                "best_point_f1_mean": float(best["best_point_f1_mean"]),
            }
    report["best_policy_per_detector"] = best_policy_per_detector
    report["best_augmentation_policy_per_detector"] = {
        detector: {
            "augmentation_policy": payload["augmentation_policy"],
            "event_f1": payload["event_f1_mean"],
            "auprc": float("nan"),
            "auroc": float("nan"),
        }
        for detector, payload in best_policy_per_detector.items()
    }

    detector_names = sorted(valid_seed["detector_backbone"].dropna().unique().tolist()) if not valid_seed.empty else []
    comparisons: list[dict[str, Any]] = []
    for detector in detector_names:
        detector_rows = valid_seed[valid_seed["detector_backbone"] == detector]
        compat_policy = _preferred_policy(detector_rows, ["groupwise_compatibility_strict", "compatibility_strict"])
        cross_compat_policy = _preferred_policy(detector_rows, ["groupwise_cross_dataset_compatible", "cross_dataset_compatible"])
        cross_all_policy = _preferred_policy(detector_rows, ["groupwise_cross_dataset_all", "cross_dataset_all"])
        comparisons.extend(
            [
                _paired_policy_stats(valid_seed, detector, compat_policy or "compatibility_strict", "real_only"),
                _paired_policy_stats(valid_seed, detector, cross_compat_policy or "cross_dataset_compatible", cross_all_policy or "cross_dataset_all"),
                _paired_policy_stats(valid_seed, detector, compat_policy or "compatibility_strict", "random_event_oversampling"),
            ]
        )
    report["statistical_comparison"] = comparisons

    strict_real = [row for row in comparisons if row["policy_a"] in {"compatibility_strict", "groupwise_compatibility_strict"} and row["policy_b"] == "real_only"]
    compat_cross = [row for row in comparisons if row["policy_a"] in {"cross_dataset_compatible", "groupwise_cross_dataset_compatible"} and row["policy_b"] in {"cross_dataset_all", "groupwise_cross_dataset_all"}]
    report["aggregate_improvement"] = {
        "number_of_detectors_improved_by_compatibility_strict_over_real_only": int(sum(1 for row in strict_real if float(row.get("mean_delta", float("nan"))) > 0)),
        "number_of_detectors_improved_by_cross_dataset_compatible_over_cross_dataset_all": int(sum(1 for row in compat_cross if float(row.get("mean_delta", float("nan"))) > 0)),
        "average_event_f1_gain_compatibility_strict_over_real_only": _safe_nanmean([row.get("mean_delta", np.nan) for row in strict_real]),
        "median_event_f1_gain_compatibility_strict_over_real_only": _safe_nanmedian([row.get("mean_delta", np.nan) for row in strict_real]),
        "average_event_f1_gain_cross_dataset_compatible_over_cross_dataset_all": _safe_nanmean([row.get("mean_delta", np.nan) for row in compat_cross]),
        "median_event_f1_gain_cross_dataset_compatible_over_cross_dataset_all": _safe_nanmedian([row.get("mean_delta", np.nan) for row in compat_cross]),
        "average_false_positive_reduction_compatibility_strict_over_real_only": _safe_nanmean([
            float(real.loc[seed, "false_positive_events"]) - float(strict.loc[seed, "false_positive_events"])
            for detector in detector_names
            for real, strict in [(
                valid_seed[(valid_seed["detector_backbone"] == detector) & (valid_seed["augmentation_policy"] == "real_only")].set_index("split_seed"),
                valid_seed[
                    (valid_seed["detector_backbone"] == detector)
                    & (
                        valid_seed["augmentation_policy"]
                        == (_preferred_policy(valid_seed[valid_seed["detector_backbone"] == detector], ["groupwise_compatibility_strict", "compatibility_strict"]) or "compatibility_strict")
                    )
                ].set_index("split_seed"),
            )]
            for seed in real.index.intersection(strict.index)
        ]),
    }
    report["average_gain_compatibility_strict_over_real_only"] = report["aggregate_improvement"]["average_event_f1_gain_compatibility_strict_over_real_only"]
    report["average_gain_cross_dataset_compatible_over_cross_dataset_all"] = report["aggregate_improvement"]["average_event_f1_gain_cross_dataset_compatible_over_cross_dataset_all"]
    report["num_backbones_improved_by_compatibility_strict"] = report["aggregate_improvement"]["number_of_detectors_improved_by_compatibility_strict_over_real_only"]
    report["num_backbones_improved_by_cross_dataset_compatible"] = report["aggregate_improvement"]["number_of_detectors_improved_by_cross_dataset_compatible_over_cross_dataset_all"]
    hurts: list[dict[str, Any]] = []
    for detector in detector_names:
        detector_rows = valid_seed[valid_seed["detector_backbone"] == detector]
        real = detector_rows[detector_rows["augmentation_policy"] == "real_only"].set_index("split_seed")
        for policy in sorted(detector_rows["augmentation_policy"].dropna().unique()):
            if policy == "real_only":
                continue
            other = detector_rows[detector_rows["augmentation_policy"] == policy].set_index("split_seed")
            common = real.index.intersection(other.index)
            if common.empty:
                continue
            delta = pd.to_numeric(other.loc[common, "event_f1"], errors="coerce") - pd.to_numeric(real.loc[common, "event_f1"], errors="coerce")
            if delta.mean() < 0:
                hurts.append(
                    {
                        "detector_backbone": detector,
                        "augmentation_policy": policy,
                        "mean_event_f1_delta_vs_real_only": float(delta.mean()),
                        "num_seeds": int(delta.shape[0]),
                    }
                )
    report["negative_cases"] = hurts
    num_detectors = max(len(detector_names), 1)
    strict_improved = report["aggregate_improvement"]["number_of_detectors_improved_by_compatibility_strict_over_real_only"]
    cross_improved = report["aggregate_improvement"]["number_of_detectors_improved_by_cross_dataset_compatible_over_cross_dataset_all"]
    strict_gain = report["aggregate_improvement"]["average_event_f1_gain_compatibility_strict_over_real_only"]
    cross_gain = report["aggregate_improvement"]["average_event_f1_gain_cross_dataset_compatible_over_cross_dataset_all"]
    report["paper_ready_summary"] = (
        f"Across {num_detectors} detector backbones, compatibility-aware augmentation improves event F1 for "
        f"{strict_improved} detectors against real_only and for {cross_improved} detectors against naive cross-dataset transfer. "
        f"The mean event-F1 delta is {strict_gain:.4f} for compatibility_strict over real_only and {cross_gain:.4f} "
        f"for cross_dataset_compatible over cross_dataset_all. These results should be interpreted as stability evidence, "
        f"not as a final detector-level claim without broader multi-seed, dataset-balanced runs."
    )
    return report


def run_backbone_augmentation_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
    records_override: list[Any] | None = None,
    load_summary_override: dict[str, Any] | None = None,
    data_path_override: str | None = None,
    seed_override: int | None = None,
) -> dict[str, Any]:
    run_seed = int(seed_override if seed_override is not None else config.get("seed", 42))
    set_global_seed(run_seed)
    if records_override is None:
        records, load_summary, data_path = load_records_for_experiment(
            config,
            project_root=PROJECT_ROOT,
            use_synthetic=use_synthetic,
            cli_data=cli_data,
            load_options=load_options,
        )
    else:
        records = list(records_override)
        load_summary = dict(load_summary_override or {})
        data_path = str(data_path_override or "synthetic")

    local_config = deepcopy(config)
    local_config["seed"] = run_seed
    local_config.setdefault("split", {})
    local_config["split"] = dict(local_config["split"])
    local_config["split"]["seed"] = run_seed
    if "stratify_by_dataset" in local_config.get("evaluation", {}):
        local_config["split"]["stratify_by_dataset"] = bool(local_config["evaluation"]["stratify_by_dataset"])
    train_records, val_records, test_records, split_ids = split_records(records, local_config)
    dataset_balanced = bool(local_config.get("evaluation", {}).get("dataset_balanced", False))
    split_dataset_coverage = None
    if dataset_balanced and bool(local_config.get("split", {}).get("stratify_by_dataset", False)):
        split_dataset_coverage = validate_dataset_balanced_test_coverage(records, split_ids)
    if int(sum(int(np.asarray(record.labels, dtype=int).sum()) for record in test_records)) <= 0:
        raise ValueError(f"Seed {run_seed} produced zero test anomalies; cannot evaluate reliably.")
    warn_if_small_test_benchmark(test_records, all_records=records)
    labeled_fraction = float(local_config.get("experiment", {}).get("labeled_fraction", local_config.get("low_label", {}).get("default_fraction", 0.2)))
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        local_config,
        labeled_fraction=labeled_fraction,
        seed=run_seed,
    )

    policy_specs = _resolve_policy_specs(local_config)
    backbone_specs = _resolve_backbone_specs(local_config)
    if not policy_specs or not backbone_specs:
        raise ValueError("experiment.detector_backbones and experiment.augmentation_policies must be configured.")

    augmentation_results: dict[tuple[str, str | None], AugmentationResult] = {}
    audit_rows: list[dict[str, Any]] = []
    compatibility_summary: dict[str, Any] = {}
    rejection_summary: dict[str, Any] = {}
    for policy_spec in policy_specs:
        policy_name = str(policy_spec["name"])
        if is_adaptive_policy(policy_name):
            continue
        augmentation = build_augmentation_result(
            masked_train,
            donor_records=donor_records,
            policy_name=policy_name,
            config=local_config,
            labeled_fraction=labeled_fraction,
            policy_config=policy_spec,
        )
        if policy_name.startswith("cross_dataset") or policy_name == "all_donors_no_filter":
            donor_rows = [row for row in augmentation.audit_records if str(row.get("record_type")) == "donor_pair"]
            cross_rows = [
                row for row in donor_rows
                if bool(row.get("source_dataset")) and bool(row.get("target_dataset"))
                and str(row.get("source_dataset")) != str(row.get("target_dataset"))
            ]
            if policy_name.startswith("cross_dataset") and not cross_rows:
                raise ValueError(f"Cross-dataset policy {policy_name} produced zero cross-dataset donor pairs.")
        if augmentation.compatibility_enabled:
            if augmentation.compatibility_summary.get("num_donor_pairs", 0) <= 0:
                raise ValueError(f"compatibility-enabled policy {policy_name} produced zero donor pairs")
            if not np.isfinite(float(augmentation.compatibility_summary.get("mean_compatibility_score", np.nan))):
                raise ValueError(f"compatibility-enabled policy {policy_name} produced no compatibility scores")
        if augmentation.policy_name != "real_only" and len(augmentation.synthetic_windows) == 0:
            print(f"WARNING: augmentation policy {policy_name} produced zero synthetic windows.")
        augmentation_results[(str(policy_name), None)] = augmentation
        audit_rows.extend(augmentation.audit_records)
        compatibility_summary[_augmentation_summary_key(str(policy_name))] = augmentation.compatibility_summary
        rejection_summary[_augmentation_summary_key(str(policy_name))] = augmentation.rejection_summary

    metric_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for backbone_spec in backbone_specs:
        backbone_name = str(backbone_spec["name"])
        for policy_spec in policy_specs:
            policy_name = str(policy_spec["name"])
            backbone = build_backbone_from_spec(local_config, backbone_spec)
            cache_key = (policy_name, backbone_name if is_adaptive_policy(policy_name) else None)
            augmentation = augmentation_results.get(cache_key)
            if augmentation is None:
                augmentation = build_augmentation_result(
                    masked_train,
                    donor_records=donor_records,
                    policy_name=policy_name,
                    config=local_config,
                    labeled_fraction=labeled_fraction,
                    policy_config=policy_spec,
                    detector_backbone=backbone_name,
                )
                if policy_name.startswith("cross_dataset") or policy_name == "all_donors_no_filter" or is_adaptive_policy(policy_name):
                    donor_rows = [row for row in augmentation.audit_records if str(row.get("record_type")) == "donor_pair"]
                    cross_rows = [
                        row for row in donor_rows
                        if bool(row.get("source_dataset")) and bool(row.get("target_dataset"))
                        and str(row.get("source_dataset")) != str(row.get("target_dataset"))
                    ]
                    selected_policy = str(augmentation.selected_policy_name or policy_name)
                    if selected_policy.startswith("cross_dataset") and not cross_rows:
                        raise ValueError(
                            f"Cross-dataset policy {policy_name} (selected={selected_policy}) produced zero cross-dataset donor pairs."
                        )
                if augmentation.compatibility_enabled:
                    if augmentation.compatibility_summary.get("num_donor_pairs", 0) <= 0:
                        raise ValueError(f"compatibility-enabled policy {policy_name} produced zero donor pairs")
                    if not np.isfinite(float(augmentation.compatibility_summary.get("mean_compatibility_score", np.nan))):
                        raise ValueError(f"compatibility-enabled policy {policy_name} produced no compatibility scores")
                if augmentation.policy_name != "real_only" and len(augmentation.synthetic_windows) == 0:
                    print(f"WARNING: augmentation policy {policy_name} produced zero synthetic windows.")
                _annotate_augmentation_records(augmentation, detector_backbone=backbone_name)
                augmentation_results[cache_key] = augmentation
                audit_rows.extend(augmentation.audit_records)
                compatibility_summary[_augmentation_summary_key(policy_name, backbone_name if is_adaptive_policy(policy_name) else None)] = augmentation.compatibility_summary
                rejection_summary[_augmentation_summary_key(policy_name, backbone_name if is_adaptive_policy(policy_name) else None)] = augmentation.rejection_summary
            row = evaluate_backbone_with_augmentation(
                backbone,
                augmentation,
                train_records=masked_train,
                val_records=val_records,
                test_records=test_records,
                config=local_config,
                labeled_fraction=labeled_fraction,
                split_seed=run_seed,
                backbone_config=dict(backbone_spec.get("config", {})),
            )
            metric_rows.append(row)
            threshold_rows.append(
                {
                    "detector_backbone": row["detector_backbone"],
                    "augmentation_policy": row["augmentation_policy"],
                    "selected_policy_name": row.get("selected_policy_name", row["augmentation_policy"]),
                    "selection_reason": row.get("selection_reason", ""),
                    "fallback_used": bool(row.get("fallback_used", False)),
                    "threshold": row["threshold"],
                    "threshold_mode": row["threshold_mode"],
                    "calibration_f1": row["calibration_f1"],
                    "synthetic_score_mean": row["synthetic_score_mean"],
                    "normal_score_mean": row["normal_score_mean"],
                    "synthetic_normal_score_gap": row["synthetic_normal_score_gap"],
                }
            )

    frame = pd.DataFrame(metric_rows)
    valid = _valid_seed_rows(frame)
    report = build_analysis_report(frame)
    report["main_detector_policy_table"] = build_detector_policy_gain_summary(frame).to_dict(orient="records")
    best_by_point: dict[str, Any] = {}
    best_by_event: dict[str, Any] = {}
    for detector, group in valid.groupby("detector_backbone", dropna=False):
        best_event_row = group.sort_values(["event_f1", "auprc"], ascending=[False, False]).iloc[0]
        best_point_row = group.sort_values(["best_point_f1", "auprc"], ascending=[False, False]).iloc[0]
        real_row = group[group["augmentation_policy"] == "real_only"]
        real_event = float(real_row["event_f1"].iloc[0]) if not real_row.empty else float("nan")
        real_point = float(real_row["best_point_f1"].iloc[0]) if not real_row.empty else float("nan")
        best_by_event[str(detector)] = {
            "best_augmentation_policy": str(best_event_row["augmentation_policy"]),
            "compatibility_aware": bool(best_event_row["compatibility_enabled"]),
            "event_f1": float(best_event_row["event_f1"]),
            "gain_over_real_only": float(best_event_row["event_f1"] - real_event) if np.isfinite(real_event) else float("nan"),
        }
        best_by_point[str(detector)] = {
            "best_augmentation_policy": str(best_point_row["augmentation_policy"]),
            "compatibility_aware": bool(best_point_row["compatibility_enabled"]),
            "best_point_f1": float(best_point_row["best_point_f1"]),
            "gain_over_real_only": float(best_point_row["best_point_f1"] - real_point) if np.isfinite(real_point) else float("nan"),
        }
    report["best_policy_per_detector_by_event_f1"] = best_by_event
    report["best_policy_per_detector_by_best_point_f1"] = best_by_point
    report["aggregate_improvement"] = {
        "number_of_detectors_improved_by_compatibility_strict_over_real_only": int(report.get("num_backbones_improved_by_compatibility_strict", 0)),
        "number_of_detectors_improved_by_cross_dataset_compatible_over_cross_dataset_all": int(report.get("num_backbones_improved_by_cross_dataset_compatible", 0)),
        "average_event_f1_gain": float(report.get("average_gain_compatibility_strict_over_real_only", 0.0)),
        "average_best_point_f1_gain": _safe_nanmean(
            (valid[valid["augmentation_policy"] == "compatibility_strict"].set_index("detector_backbone")["best_point_f1"]
                - valid[valid["augmentation_policy"] == "real_only"].set_index("detector_backbone")["best_point_f1"]).reindex(
                    valid["detector_backbone"].drop_duplicates()
                ).values
        ) if {"compatibility_strict", "real_only"}.issubset(set(valid["augmentation_policy"])) else float("nan"),
        "average_false_positive_reduction": float(report.get("average_false_positive_reduction", 0.0)),
        "median_event_f1_gain": _safe_nanmedian(
            [row["gain_compatibility_strict_over_real_only"] for row in report.get("main_detector_policy_table", [])]
        ),
    }
    report["compatibility_mechanism_diagnostics"] = summarize_compatibility_mechanism(audit_rows, rejection_summary, compatibility_summary)
    report["threshold_calibration_diagnostics"] = summarize_threshold_calibration_effects(frame)
    report["negative_cases"] = summarize_negative_cases(frame)
    report["paper_ready_summary"] = build_paper_summary(report, num_detectors=len(valid["detector_backbone"].drop_duplicates()))
    return {
        "records": records,
        "load_summary": load_summary,
        "data_path": data_path,
        "seed": run_seed,
        "split_ids": split_ids,
        "split_dataset_coverage": split_dataset_coverage,
        "metrics_frame": frame,
        "synthetic_audit": audit_rows,
        "rejection_summary": rejection_summary,
        "compatibility_summary": compatibility_summary,
        "threshold_diagnostics": pd.DataFrame(threshold_rows),
        "detector_policy_pivot_event_f1": build_detector_policy_pivot(frame, metric="event_f1"),
        "detector_policy_pivot_point_f1": build_detector_policy_pivot(frame, metric="best_point_f1"),
        "detector_policy_gain_summary": build_detector_policy_gain_summary(frame),
        "analysis_report": report,
    }


def run_backbone_augmentation_multiseed(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
) -> dict[str, Any]:
    base_records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    evaluation_cfg = config.get("evaluation", {})
    seeds = list(evaluation_cfg.get("seeds", [int(config.get("seed", 42))]))
    if not seeds:
        seeds = [int(config.get("seed", 42))]

    dataset_balanced = bool(evaluation_cfg.get("dataset_balanced", False))
    max_timelines_per_dataset = evaluation_cfg.get("max_timelines_per_dataset")
    max_points_per_dataset = evaluation_cfg.get("max_points_per_dataset")
    min_timelines_per_dataset = evaluation_cfg.get("min_timelines_per_dataset")
    if min_timelines_per_dataset is None and dataset_balanced:
        split_cfg = config.get("split", {})
        data_cfg = config.get("data", {})
        val_ratio = float(data_cfg.get("dev_ratio", data_cfg.get("val_ratio", split_cfg.get("val_ratio", 0.1))))
        min_timelines_per_dataset = 3 if val_ratio > 0 else 2
    per_seed_payloads: list[dict[str, Any]] = []
    dataset_balance_summary: dict[str, Any] = {
        "dataset_balanced": dataset_balanced,
        "base_before": summarize_dataset_distribution(base_records),
        "per_seed": {},
    }

    for seed in seeds:
        seed_records = list(base_records)
        if dataset_balanced:
            seed_records, balance_info = balance_records_by_dataset(
                seed_records,
                max_timelines_per_dataset=max_timelines_per_dataset,
                max_points_per_dataset=max_points_per_dataset,
                min_timelines_per_dataset=min_timelines_per_dataset,
                seed=int(seed),
                preserve_anomaly_timelines=True,
            )
        else:
            balance_info = {
                "enabled": False,
                "seed": int(seed),
                "before": summarize_dataset_distribution(seed_records),
                "after": summarize_dataset_distribution(seed_records),
            }
        after = balance_info["after"]["per_dataset"]
        stratify_by_dataset = bool(
            config.get("split", {}).get("stratify_by_dataset", evaluation_cfg.get("stratify_by_dataset", False))
        )
        for dataset_name, stats in after.items():
            num_timelines = int(stats.get("num_timelines", 0))
            if num_timelines <= 0:
                raise ValueError(f"Balanced sampling removed all timelines for dataset {dataset_name!r} at seed {seed}.")
            if dataset_balanced and stratify_by_dataset and min_timelines_per_dataset is not None:
                if num_timelines < int(min_timelines_per_dataset):
                    raise ValueError(
                        f"Dataset {dataset_name!r} has {num_timelines} timelines after balancing at seed {seed}, "
                        f"but at least {int(min_timelines_per_dataset)} are required for per-dataset test coverage."
                    )
        if dataset_balanced and stratify_by_dataset:
            preflight_config = deepcopy(config)
            preflight_config["seed"] = int(seed)
            preflight_config.setdefault("split", {})
            preflight_config["split"] = dict(preflight_config["split"])
            preflight_config["split"]["seed"] = int(seed)
            preflight_config["split"]["stratify_by_dataset"] = True
            _, _, _, preview_split_ids = split_records(seed_records, preflight_config)
            balance_info["split_dataset_coverage"] = validate_dataset_balanced_test_coverage(
                seed_records,
                preview_split_ids,
            )
        dataset_balance_summary["per_seed"][str(seed)] = balance_info
        seed_payload = run_backbone_augmentation_experiment(
            config,
            use_synthetic=use_synthetic,
            cli_data=cli_data,
            load_options=load_options,
            records_override=seed_records,
            load_summary_override=load_summary,
            data_path_override=data_path,
            seed_override=int(seed),
        )
        balance_info["split_dataset_coverage"] = seed_payload.get("split_dataset_coverage")
        per_seed_payloads.append(seed_payload)

    seed_metrics = pd.concat([payload["metrics_frame"] for payload in per_seed_payloads], ignore_index=True) if per_seed_payloads else pd.DataFrame()
    seed_audit = pd.concat(
        [pd.DataFrame(payload["synthetic_audit"]).assign(split_seed=int(payload["seed"])) for payload in per_seed_payloads],
        ignore_index=True,
    ) if per_seed_payloads else pd.DataFrame()
    threshold_diagnostics = pd.concat(
        [payload["threshold_diagnostics"].assign(split_seed=int(payload["seed"])) for payload in per_seed_payloads],
        ignore_index=True,
    ) if per_seed_payloads else pd.DataFrame()
    aggregate_metrics = aggregate_seed_metrics(seed_metrics)
    aggregate_metrics_reportable = reportable_aggregate_metrics(aggregate_metrics)
    analysis_report = build_multi_seed_analysis(seed_metrics, aggregate_metrics)

    return {
        "seeds": [int(seed) for seed in seeds],
        "seed_payloads": per_seed_payloads,
        "seed_metrics": seed_metrics,
        "seed_metrics_json": seed_metrics.to_dict(orient="records"),
        "aggregate_metrics": aggregate_metrics,
        "aggregate_metrics_reportable": aggregate_metrics_reportable,
        "aggregate_metrics_json": aggregate_metrics.to_dict(orient="records"),
        "aggregate_metrics_reportable_json": aggregate_metrics_reportable.to_dict(orient="records"),
        "synthetic_audit": seed_audit,
        "threshold_diagnostics": threshold_diagnostics,
        "dataset_balance_summary": dataset_balance_summary,
        "multi_seed_analysis_report": analysis_report,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run detector-backbone x augmentation-policy experiments.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_backbone_augmentation_smoke.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    load_options = apply_dataset_cli_to_config(config, args)
    payload = run_backbone_augmentation_multiseed(
        config,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
    )

    dataset_name = config.get("dataset", {}).get("name", "synthetic" if args.use_synthetic else "dataset")
    output_dir = args.output_dir
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = PROJECT_ROOT / "outputs" / "backbone_augmentation" / str(dataset_name) / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_metrics = payload["seed_metrics"]
    aggregate_metrics = payload["aggregate_metrics"]
    aggregate_metrics_reportable = payload.get(
        "aggregate_metrics_reportable",
        reportable_aggregate_metrics(aggregate_metrics),
    )
    seed_metrics.to_csv(output_dir / "seed_level_backbone_metrics.csv", index=False)
    (output_dir / "seed_metrics.json").write_text(json.dumps(payload["seed_metrics_json"], indent=2), encoding="utf-8")
    aggregate_metrics.to_csv(output_dir / "aggregate_backbone_comparison_metrics.csv", index=False)
    aggregate_metrics_reportable.to_csv(
        output_dir / "aggregate_backbone_comparison_metrics_reportable.csv",
        index=False,
    )
    (output_dir / "aggregate_backbone_comparison_metrics.json").write_text(
        json.dumps(payload["aggregate_metrics_json"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "aggregate_backbone_comparison_metrics_reportable.json").write_text(
        json.dumps(
            payload.get(
                "aggregate_metrics_reportable_json",
                aggregate_metrics_reportable.to_dict(orient="records"),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    aggregate_metrics.to_csv(output_dir / "aggregate_metrics.csv", index=False)
    aggregate_metrics_reportable.to_csv(output_dir / "aggregate_metrics_reportable.csv", index=False)
    (output_dir / "aggregate_metrics.json").write_text(
        json.dumps(payload["aggregate_metrics_json"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "aggregate_metrics_reportable.json").write_text(
        json.dumps(
            payload.get(
                "aggregate_metrics_reportable_json",
                aggregate_metrics_reportable.to_dict(orient="records"),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    # Backward-compatible aliases.
    seed_metrics.to_csv(output_dir / "backbone_comparison_metrics.csv", index=False)
    (output_dir / "backbone_comparison_metrics.json").write_text(
        seed_metrics.to_json(orient="records", indent=2),
        encoding="utf-8",
    )

    synthetic_audit = payload["synthetic_audit"]
    synthetic_audit.to_csv(output_dir / "synthetic_audit.csv", index=False)
    compatibility_summary = {str(item["seed"]): item["compatibility_summary"] for item in payload["seed_payloads"]}
    rejection_summary = {str(item["seed"]): item["rejection_summary"] for item in payload["seed_payloads"]}
    (output_dir / "rejection_summary.json").write_text(json.dumps(rejection_summary, indent=2), encoding="utf-8")
    (output_dir / "compatibility_summary.json").write_text(json.dumps(compatibility_summary, indent=2), encoding="utf-8")
    payload["threshold_diagnostics"].to_csv(output_dir / "threshold_diagnostics.csv", index=False)
    (output_dir / "dataset_balance_summary.json").write_text(json.dumps(payload["dataset_balance_summary"], indent=2), encoding="utf-8")

    comparison_aggregate = aggregate_metrics_reportable.rename(
        columns={
            "event_f1_mean": "event_f1",
            "best_point_f1_mean": "best_point_f1",
            "false_positive_events_mean": "false_positive_events",
        }
    )
    event_pivot = build_detector_policy_pivot(comparison_aggregate, metric="event_f1")
    point_pivot = build_detector_policy_pivot(comparison_aggregate, metric="best_point_f1")
    gain_summary = build_detector_policy_gain_summary(comparison_aggregate)
    event_pivot.to_csv(output_dir / "detector_policy_pivot_event_f1.csv", index=False)
    point_pivot.to_csv(output_dir / "detector_policy_pivot_point_f1.csv", index=False)
    gain_summary.to_csv(output_dir / "detector_policy_gain_summary.csv", index=False)

    analysis_report = payload["multi_seed_analysis_report"]
    (output_dir / "analysis_report.json").write_text(json.dumps(analysis_report, indent=2), encoding="utf-8")
    (output_dir / "analysis_report.md").write_text(analysis_report_markdown(analysis_report), encoding="utf-8")
    (output_dir / "multi_seed_analysis_report.json").write_text(json.dumps(analysis_report, indent=2), encoding="utf-8")
    (output_dir / "multi_seed_analysis_report.md").write_text(analysis_report_markdown(analysis_report), encoding="utf-8")
    (output_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    print(f"Saved backbone augmentation results to {output_dir}")


if __name__ == "__main__":
    main()
