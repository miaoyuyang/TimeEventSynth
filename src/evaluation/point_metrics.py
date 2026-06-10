"""Pointwise anomaly metrics with threshold tuning helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import warnings

from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score

from .event_metrics import (
    compute_event_metrics,
    tune_threshold_by_event_f1,
    tune_threshold_by_event_fbeta,
)

_BINARY_LABELS = [0, 1]


def auroc(y_true: list[int], y_score: list[float]) -> float:
    values = np.asarray(y_true, dtype=int)
    if len(np.unique(values)) < 2:
        warnings.warn("AUROC undefined because labels contain only one class.", RuntimeWarning)
        return float("nan")
    return float(roc_auc_score(values, np.asarray(y_score, dtype=float)))


def auprc(y_true: list[int], y_score: list[float]) -> float:
    values = np.asarray(y_true, dtype=int)
    if len(np.unique(values)) < 2:
        warnings.warn("AUPRC undefined because labels contain only one class.", RuntimeWarning)
        return float("nan")
    return float(average_precision_score(values, np.asarray(y_score, dtype=float)))


def precision_recall_f1_at_threshold(y_true: list[int], y_score: list[float], threshold: float) -> dict[str, float]:
    values = np.asarray(y_true, dtype=int)
    predictions = (np.asarray(y_score, dtype=float) >= threshold).astype(int)
    kwargs = {"labels": _BINARY_LABELS, "zero_division": 0}
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(values, predictions, **kwargs)),
        "recall": float(recall_score(values, predictions, **kwargs)),
        "f1": float(f1_score(values, predictions, **kwargs)),
    }


def best_f1_threshold(y_true: list[int], y_score: list[float]) -> dict[str, float]:
    values = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    if len(np.unique(values)) < 2:
        warnings.warn("Best F1 threshold undefined because labels contain only one class.", RuntimeWarning)
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    precision, recall, thresholds = precision_recall_curve(values, scores)
    if len(thresholds) == 0:
        warnings.warn("No thresholds available for best F1 tuning.", RuntimeWarning)
        return {"threshold": float("nan"), "precision": float("nan"), "recall": float("nan"), "f1": float("nan")}
    best = {"threshold": float(thresholds[0]), "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for idx, threshold in enumerate(thresholds):
        p = float(precision[idx + 1])
        r = float(recall[idx + 1])
        f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
        if f1 > best["f1"]:
            best = {"threshold": float(threshold), "precision": p, "recall": r, "f1": f1}
    return best


def build_threshold_grid(y_score: list[float], grid_size: int = 200) -> list[float]:
    scores = np.asarray(y_score, dtype=float)
    if scores.size == 0:
        return [0.5]
    lo = float(np.min(scores))
    hi = float(np.max(scores))
    if lo == hi:
        return [lo]
    return [float(x) for x in np.linspace(lo, hi, grid_size)]


def tune_threshold_by_point_f1(
    val_labels: list[int],
    val_scores: list[float],
    grid: Iterable[float],
) -> dict[str, float]:
    best = {"threshold": float("nan"), "precision": 0.0, "recall": 0.0, "f1": -1.0}
    for threshold in grid:
        metrics = precision_recall_f1_at_threshold(val_labels, val_scores, float(threshold))
        if metrics["f1"] > best["f1"]:
            best = metrics
    return best


def tune_threshold_by_fixed_quantile(
    val_scores: list[float],
    quantile: float = 0.95,
) -> dict[str, float]:
    scores = np.asarray(val_scores, dtype=float)
    if scores.size == 0:
        return {"threshold": 0.5, "quantile": float(quantile)}
    threshold = float(np.quantile(scores, quantile))
    return {"threshold": threshold, "quantile": float(quantile)}


def parse_prediction_smoothing(eval_cfg: dict[str, Any]) -> dict[str, Any]:
    smoothing = eval_cfg.get("prediction_smoothing", {})
    enabled = bool(smoothing.get("enabled", True))
    return {
        "enabled": enabled,
        "min_event_length": int(smoothing.get("min_event_length", 1)),
        "merge_gap": int(smoothing.get("merge_gap", 0)),
    }


def select_validation_threshold(
    val_labels: list[int],
    val_scores: list[float],
    eval_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Select a threshold using validation labels/scores only."""
    mode = str(eval_cfg.get("threshold_selection", eval_cfg.get("threshold_metric", "point_f1")))
    iou_threshold = float(eval_cfg.get("event_iou_threshold", 0.1))
    event_beta = float(eval_cfg.get("event_beta", 1.0))
    grid_size = int(eval_cfg.get("threshold_grid_size", 200))
    fixed_quantile = float(eval_cfg.get("fixed_quantile", 0.95))
    smoothing = parse_prediction_smoothing(eval_cfg)
    min_event_length = smoothing["min_event_length"] if smoothing["enabled"] else 1
    merge_gap = smoothing["merge_gap"] if smoothing["enabled"] else 0

    grid = build_threshold_grid(val_scores, grid_size)
    val_best_point = tune_threshold_by_point_f1(val_labels, val_scores, grid)
    val_best_event = tune_threshold_by_event_f1(
        val_scores,
        val_labels,
        grid,
        iou_threshold=iou_threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )
    val_best_event_fbeta = tune_threshold_by_event_fbeta(
        val_scores,
        val_labels,
        grid,
        iou_threshold=iou_threshold,
        beta=event_beta,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )

    if mode == "point_f1":
        selected = dict(val_best_point)
    elif mode == "event_f1":
        selected = dict(val_best_event)
    elif mode == "event_fbeta":
        selected = dict(val_best_event_fbeta)
    elif mode == "fixed_quantile":
        quantile_info = tune_threshold_by_fixed_quantile(val_scores, quantile=fixed_quantile)
        threshold = float(quantile_info["threshold"])
        point = precision_recall_f1_at_threshold(val_labels, val_scores, threshold)
        event = compute_event_metrics(
            val_labels,
            y_score=val_scores,
            threshold=threshold,
            iou_threshold=iou_threshold,
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )
        selected = {
            "threshold": threshold,
            "quantile": fixed_quantile,
            "precision": point["precision"],
            "recall": point["recall"],
            "f1": point["f1"],
            "event_precision": event["event_precision"],
            "event_recall": event["event_recall"],
            "event_f1": event["event_f1"],
        }
    else:
        raise ValueError(f"Unsupported evaluation.threshold_selection mode: {mode}")

    return {
        "mode": mode,
        "threshold": float(selected["threshold"]),
        "precision": float(selected.get("precision", float("nan"))),
        "recall": float(selected.get("recall", float("nan"))),
        "f1": float(selected.get("f1", float("nan"))),
        "event_precision": float(selected.get("event_precision", float("nan"))),
        "event_recall": float(selected.get("event_recall", float("nan"))),
        "event_f1": float(selected.get("event_f1", float("nan"))),
        "event_fbeta": float(selected.get("event_fbeta", float("nan"))),
        "event_iou_threshold": iou_threshold,
        "event_beta": event_beta,
        "prediction_smoothing": smoothing,
        "validation_debug": {
            "best_point_f1": val_best_point,
            "best_event_f1": val_best_event,
            "best_event_fbeta": val_best_event_fbeta,
        },
    }


def compute_point_metrics(y_true: list[int], y_score: list[float], threshold: float = 0.5) -> dict[str, float]:
    prf = precision_recall_f1_at_threshold(y_true, y_score, threshold)
    return {
        "point_precision": prf["precision"],
        "point_recall": prf["recall"],
        "point_f1": prf["f1"],
        "point_auroc": auroc(y_true, y_score),
        "point_auprc": auprc(y_true, y_score),
        "point_threshold": float(threshold),
    }
