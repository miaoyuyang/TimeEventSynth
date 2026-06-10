"""Reusable threshold calibration for detector-agnostic anomaly scoring.

This module separates threshold selection from detector training so synthetic
anomalous windows can be used as positive examples for calibration without
being injected into unsupervised detector fitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

from .point_metrics import best_f1_threshold


def _safe_array(values: np.ndarray | list[float] | None) -> np.ndarray:
    if values is None:
        return np.zeros(0, dtype=float)
    return np.asarray(values, dtype=float).reshape(-1)


def _threshold_grid(scores: np.ndarray, grid_size: int = 200) -> np.ndarray:
    if scores.size == 0:
        return np.asarray([0.5], dtype=float)
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if abs(hi - lo) < 1e-12:
        return np.asarray([lo], dtype=float)
    quantiles = np.linspace(0.0, 1.0, max(grid_size, 2))
    return np.unique(np.quantile(scores, quantiles))


def _balanced_accuracy_binary(labels: np.ndarray, predictions: np.ndarray) -> float:
    """Balanced accuracy for binary labels without sklearn single-class warnings."""
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    per_class: list[float] = []
    for idx in range(2):
        denom = float(cm[idx, :].sum())
        per_class.append(float(cm[idx, idx] / denom) if denom > 0 else 0.0)
    return float(np.mean(per_class))


def calibrate_threshold_quantile(scores: np.ndarray | list[float], quantile: float = 0.95) -> float:
    """Choose a threshold by score quantile, assuming higher means more anomalous."""
    values = _safe_array(scores)
    if values.size == 0:
        return 0.5
    return float(np.quantile(values, quantile))


def calibrate_threshold_oracle(
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int],
    metric: str = "f1",
) -> tuple[float, float]:
    """Choose the best threshold using validation labels.

    This is an oracle/upper-bound calibration mode and should not be used for
    test-time threshold selection.
    """
    score_array = _safe_array(scores)
    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if score_array.size == 0 or label_array.size != score_array.size:
        return 0.5, float("nan")
    if metric != "f1":
        raise ValueError(f"Unsupported oracle threshold metric: {metric}")
    best = best_f1_threshold(label_array.tolist(), score_array.tolist())
    threshold = float(best["threshold"]) if np.isfinite(best["threshold"]) else 0.5
    return threshold, float(best.get("f1", float("nan")))


def calibrate_threshold_synthetic_separation(
    normal_scores: np.ndarray | list[float],
    synthetic_anomaly_scores: np.ndarray | list[float],
    metric: str = "f1",
    *,
    fallback_quantile: float = 0.95,
    grid_size: int = 200,
    max_false_positive_rate: float | None = None,
    min_precision: float | None = None,
    false_positive_penalty: float = 0.0,
    fallback_when_inverted_gap: bool = False,
) -> tuple[float, dict[str, Any]]:
    """Calibrate a threshold separating mostly-normal and synthetic-anomalous scores."""
    normal = _safe_array(normal_scores)
    synthetic = _safe_array(synthetic_anomaly_scores)
    if synthetic.size == 0 or normal.size == 0:
        threshold = calibrate_threshold_quantile(normal, quantile=fallback_quantile)
        return threshold, {
            "synthetic_score_mean": float("nan"),
            "normal_score_mean": float(np.mean(normal)) if normal.size else float("nan"),
            "score_gap": float("nan"),
            "best_calibration_f1": float("nan"),
            "best_calibration_balanced_accuracy": float("nan"),
            "num_normal_scores": int(normal.size),
            "num_synthetic_scores": int(synthetic.size),
            "fallback_used": True,
            "fallback_reason": "empty_synthetic_scores" if synthetic.size == 0 else "empty_normal_scores",
        }
    score_gap = float(np.mean(synthetic) - np.mean(normal))
    if fallback_when_inverted_gap and score_gap <= 0.0:
        threshold = calibrate_threshold_quantile(normal, quantile=fallback_quantile)
        return threshold, {
            "synthetic_score_mean": float(np.mean(synthetic)),
            "normal_score_mean": float(np.mean(normal)),
            "score_gap": score_gap,
            "best_calibration_f1": float("nan"),
            "best_calibration_balanced_accuracy": float("nan"),
            "best_calibration_precision": float("nan"),
            "best_calibration_recall": float("nan"),
            "best_calibration_false_positive_rate": float("nan"),
            "num_normal_scores": int(normal.size),
            "num_synthetic_scores": int(synthetic.size),
            "fallback_used": True,
            "fallback_reason": "inverted_synthetic_normal_gap",
        }

    labels = np.concatenate([np.zeros(normal.size, dtype=int), np.ones(synthetic.size, dtype=int)], axis=0)
    scores = np.concatenate([normal, synthetic], axis=0)
    grid = _threshold_grid(scores, grid_size=grid_size)

    best_threshold = float(grid[0])
    best_value = -1.0
    best_f1 = float("nan")
    best_bal_acc = float("nan")
    best_precision = float("nan")
    best_recall = float("nan")
    best_fp_rate = float("nan")
    best_constraint_violation = float("inf")
    score_kwargs = {"labels": [0, 1], "zero_division": 0}
    for threshold in grid:
        predictions = (scores >= float(threshold)).astype(int)
        current_f1 = float(f1_score(labels, predictions, **score_kwargs))
        current_bal_acc = _balanced_accuracy_binary(labels, predictions)
        tp = float(np.sum((labels == 1) & (predictions == 1)))
        fp = float(np.sum((labels == 0) & (predictions == 1)))
        fn = float(np.sum((labels == 1) & (predictions == 0)))
        current_precision = float(tp / (tp + fp)) if tp + fp > 0 else 0.0
        current_recall = float(tp / (tp + fn)) if tp + fn > 0 else 0.0
        current_fp_rate = float(fp / normal.size) if normal.size else 0.0
        precision_violation = max(0.0, float(min_precision) - current_precision) if min_precision is not None else 0.0
        fp_rate_violation = max(0.0, current_fp_rate - float(max_false_positive_rate)) if max_false_positive_rate is not None else 0.0
        constraint_violation = precision_violation + fp_rate_violation
        current_value = current_f1 if metric == "f1" else current_bal_acc
        current_value -= float(false_positive_penalty) * current_fp_rate
        is_better = False
        if constraint_violation < best_constraint_violation - 1e-12:
            is_better = True
        elif abs(constraint_violation - best_constraint_violation) <= 1e-12 and current_value > best_value:
            is_better = True
        if is_better:
            best_value = current_value
            best_threshold = float(threshold)
            best_f1 = current_f1
            best_bal_acc = current_bal_acc
            best_precision = current_precision
            best_recall = current_recall
            best_fp_rate = current_fp_rate
            best_constraint_violation = constraint_violation

    diagnostics = {
        "synthetic_score_mean": float(np.mean(synthetic)),
        "normal_score_mean": float(np.mean(normal)) if normal.size else float("nan"),
        "score_gap": score_gap if normal.size else float("nan"),
        "best_calibration_f1": best_f1,
        "best_calibration_balanced_accuracy": best_bal_acc,
        "best_calibration_precision": best_precision,
        "best_calibration_recall": best_recall,
        "best_calibration_false_positive_rate": best_fp_rate,
        "constraint_violation": best_constraint_violation,
        "num_normal_scores": int(normal.size),
        "num_synthetic_scores": int(synthetic.size),
        "fallback_used": False,
    }
    return best_threshold, diagnostics


def calibrate_threshold(
    *,
    mode: str,
    scores: np.ndarray | list[float],
    labels: np.ndarray | list[int] | None = None,
    synthetic_scores: np.ndarray | list[float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Unified threshold-calibration dispatcher."""
    cfg = config or {}
    score_array = _safe_array(scores)
    label_array = None if labels is None else np.asarray(labels, dtype=int).reshape(-1)
    synthetic_array = _safe_array(synthetic_scores)
    threshold_mode = str(mode)

    if threshold_mode == "quantile":
        quantile = float(cfg.get("quantile", cfg.get("threshold_quantile", 0.95)))
        threshold = calibrate_threshold_quantile(score_array, quantile=quantile)
        diagnostics = {
            "quantile": quantile,
            "num_scores": int(score_array.size),
        }
    elif threshold_mode == "oracle_val":
        threshold, best_value = calibrate_threshold_oracle(score_array, label_array if label_array is not None else np.zeros_like(score_array), metric=str(cfg.get("metric", "f1")))
        diagnostics = {
            "best_metric_value": best_value,
            "num_scores": int(score_array.size),
        }
    elif threshold_mode == "synthetic_separation":
        metric = str(cfg.get("metric", "f1"))
        if label_array is not None and label_array.size == score_array.size and np.any(label_array == 0):
            normal_scores = score_array[label_array == 0]
        elif label_array is not None and label_array.size == score_array.size:
            normal_scores = np.asarray([], dtype=float)
        else:
            normal_scores = score_array
        threshold, diagnostics = calibrate_threshold_synthetic_separation(
            normal_scores,
            synthetic_array,
            metric=metric,
            fallback_quantile=float(cfg.get("quantile", cfg.get("threshold_quantile", 0.95))),
            grid_size=int(cfg.get("grid_size", 200)),
        )
    elif threshold_mode in {"fp_aware_synthetic_separation", "synthetic_separation_fp_aware"}:
        metric = str(cfg.get("metric", "f1"))
        if label_array is not None and label_array.size == score_array.size and np.any(label_array == 0):
            normal_scores = score_array[label_array == 0]
        elif label_array is not None and label_array.size == score_array.size:
            normal_scores = np.asarray([], dtype=float)
        else:
            normal_scores = score_array
        threshold, diagnostics = calibrate_threshold_synthetic_separation(
            normal_scores,
            synthetic_array,
            metric=metric,
            fallback_quantile=float(cfg.get("quantile", cfg.get("threshold_quantile", 0.99))),
            grid_size=int(cfg.get("grid_size", 200)),
            max_false_positive_rate=cfg.get("max_false_positive_rate"),
            min_precision=cfg.get("min_calibration_precision"),
            false_positive_penalty=float(cfg.get("false_positive_penalty", 1.0)),
            fallback_when_inverted_gap=bool(cfg.get("fallback_when_inverted_gap", True)),
        )
    else:
        raise ValueError(f"Unsupported threshold calibration mode: {threshold_mode}")

    return {
        "threshold": float(threshold),
        "threshold_mode": threshold_mode,
        "diagnostics": diagnostics,
    }
