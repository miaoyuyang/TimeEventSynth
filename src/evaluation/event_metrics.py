"""Event-level anomaly metrics using half-open event windows."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..datasets.event_extractor import labels_to_events


EventWindow = tuple[int, int]


def scores_to_events(
    scores: list[float] | np.ndarray,
    threshold: float,
    *,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> list[EventWindow]:
    """Convert anomaly scores to post-processed predicted event windows."""
    labels = (np.asarray(scores, dtype=float) >= threshold).astype(int).tolist()
    return labels_to_events(labels, min_length=min_event_length, merge_gap=merge_gap)


def predicted_events_from_scores(
    scores: list[float],
    threshold: float,
    *,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> list[EventWindow]:
    return scores_to_events(
        scores,
        threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )


def event_iou(left: EventWindow, right: EventWindow) -> float:
    inter_left = max(left[0], right[0])
    inter_right = min(left[1], right[1])
    intersection = max(0, inter_right - inter_left)
    union = max(left[1], right[1]) - min(left[0], right[0])
    return 0.0 if union <= 0 else float(intersection / union)


def precision_recall_f1_events(
    true_events: list[EventWindow],
    pred_events: list[EventWindow],
    iou_threshold: float = 0.1,
) -> dict[str, float]:
    matched_true: set[int] = set()
    matched_pred = 0
    for pred in pred_events:
        best_idx = None
        best_iou = 0.0
        for idx, truth in enumerate(true_events):
            if idx in matched_true:
                continue
            overlap = event_iou(pred, truth)
            if overlap > best_iou:
                best_iou = overlap
                best_idx = idx
        if best_idx is not None and best_iou >= iou_threshold:
            matched_true.add(best_idx)
            matched_pred += 1
    precision = matched_pred / len(pred_events) if pred_events else 0.0
    recall = matched_pred / len(true_events) if true_events else 0.0
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_positive_event_count": float(max(len(pred_events) - matched_pred, 0)),
    }


def event_fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0:
        return 0.0
    beta_sq = beta * beta
    return float((1.0 + beta_sq) * precision * recall / (beta_sq * precision + recall))


def tune_threshold_by_event_f1(
    val_scores: list[float],
    val_labels: list[int],
    grid: Iterable[float],
    iou_threshold: float = 0.1,
    *,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, float]:
    true_events = labels_to_events(val_labels)
    best = {
        "threshold": float("nan"),
        "event_precision": 0.0,
        "event_recall": 0.0,
        "event_f1": -1.0,
        "false_positive_event_count": float("inf"),
    }
    for threshold in grid:
        pred_events = scores_to_events(
            val_scores,
            float(threshold),
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )
        metrics = precision_recall_f1_events(true_events, pred_events, iou_threshold=iou_threshold)
        candidate = {
            "threshold": float(threshold),
            "event_precision": float(metrics["event_precision"]),
            "event_recall": float(metrics["event_recall"]),
            "event_f1": float(metrics["event_f1"]),
            "false_positive_event_count": float(metrics["false_positive_event_count"]),
        }
        if candidate["event_f1"] > best["event_f1"] + 1e-12:
            best = candidate
        elif abs(candidate["event_f1"] - best["event_f1"]) <= 1e-12:
            if candidate["false_positive_event_count"] < best["false_positive_event_count"] - 1e-12:
                best = candidate
            elif (
                abs(candidate["false_positive_event_count"] - best["false_positive_event_count"]) <= 1e-12
                and candidate["threshold"] > best["threshold"]
            ):
                best = candidate
    return best


def tune_threshold_by_event_fbeta(
    val_scores: list[float],
    val_labels: list[int],
    grid: Iterable[float],
    iou_threshold: float = 0.1,
    beta: float = 1.0,
    *,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, float]:
    true_events = labels_to_events(val_labels)
    best = {
        "threshold": float("nan"),
        "event_precision": 0.0,
        "event_recall": 0.0,
        "event_f1": -1.0,
        "event_fbeta": -1.0,
        "false_positive_event_count": float("inf"),
    }
    for threshold in grid:
        pred_events = scores_to_events(
            val_scores,
            float(threshold),
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )
        metrics = precision_recall_f1_events(true_events, pred_events, iou_threshold=iou_threshold)
        score = event_fbeta(metrics["event_precision"], metrics["event_recall"], beta)
        candidate = {
            "threshold": float(threshold),
            "event_precision": float(metrics["event_precision"]),
            "event_recall": float(metrics["event_recall"]),
            "event_f1": float(metrics["event_f1"]),
            "event_fbeta": float(score),
            "false_positive_event_count": float(metrics["false_positive_event_count"]),
        }
        if candidate["event_fbeta"] > best["event_fbeta"] + 1e-12:
            best = candidate
        elif abs(candidate["event_fbeta"] - best["event_fbeta"]) <= 1e-12:
            if candidate["false_positive_event_count"] < best["false_positive_event_count"] - 1e-12:
                best = candidate
            elif (
                abs(candidate["false_positive_event_count"] - best["false_positive_event_count"]) <= 1e-12
                and candidate["threshold"] > best["threshold"]
            ):
                best = candidate
    return best


def compute_event_metrics(
    y_true: list[int],
    y_pred: list[int] | None = None,
    *,
    y_score: list[float] | None = None,
    threshold: float = 0.5,
    iou_threshold: float = 0.1,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, float]:
    """Compute event metrics from predicted labels or scores."""
    true_events = labels_to_events(y_true)
    if y_pred is None and y_score is None:
        raise ValueError("Either y_pred or y_score must be provided.")
    if y_pred is not None:
        pred_events = labels_to_events(y_pred, min_length=min_event_length, merge_gap=merge_gap)
    else:
        pred_events = scores_to_events(
            y_score or [],
            threshold,
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )
    metrics = precision_recall_f1_events(true_events, pred_events, iou_threshold=iou_threshold)
    metrics["event_count_true"] = float(len(true_events))
    metrics["event_count_pred"] = float(len(pred_events))
    metrics["event_iou_threshold"] = float(iou_threshold)
    return metrics


def best_event_f1_threshold(
    y_true: list[int],
    y_score: list[float],
    *,
    candidate_thresholds: list[float] | None = None,
    iou_threshold: float = 0.1,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> dict[str, float]:
    thresholds = candidate_thresholds or [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    return tune_threshold_by_event_f1(
        y_score,
        y_true,
        thresholds,
        iou_threshold=iou_threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )
