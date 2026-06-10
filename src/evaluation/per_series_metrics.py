"""Per-series evaluation metrics for paper-ready reporting."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..datasets.tsb_loader import TimeSeriesRecord
from .event_metrics import compute_event_metrics
from .point_metrics import compute_point_metrics


def compute_per_series_metrics(
    records: list[TimeSeriesRecord],
    scores: dict[str, np.ndarray],
    *,
    threshold: float,
    event_iou_threshold: float = 0.1,
    min_event_length: int = 1,
    merge_gap: int = 0,
) -> list[dict[str, Any]]:
    """Compute pointwise and event metrics independently for each series."""
    rows: list[dict[str, Any]] = []
    for record in records:
        series_scores = scores[record.series_id]
        y_true = [int(x) for x in record.labels.tolist()]
        y_score = [float(x) for x in series_scores.tolist()]
        point = compute_point_metrics(y_true, y_score, threshold=threshold)
        event = compute_event_metrics(
            y_true,
            y_score=y_score,
            threshold=threshold,
            iou_threshold=event_iou_threshold,
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        )
        rows.append(
            {
                "series_id": record.series_id,
                "num_points": len(y_true),
                "num_anomaly_points": int(sum(y_true)),
                "num_true_event_windows": int(event.get("event_count_true", 0.0)),
                "num_pred_event_windows": int(event.get("event_count_pred", 0.0)),
                "auprc": point["point_auprc"],
                "point_precision": point["point_precision"],
                "point_recall": point["point_recall"],
                "point_f1": point["point_f1"],
                "point_auroc": point["point_auroc"],
                "point_auprc": point["point_auprc"],
                "event_precision": event["event_precision"],
                "event_recall": event["event_recall"],
                "event_f1": event["event_f1"],
                "false_positive_event_count": event["false_positive_event_count"],
                "event_count_true": event.get("event_count_true", 0.0),
                "event_count_pred": event.get("event_count_pred", 0.0),
            }
        )
    return rows
