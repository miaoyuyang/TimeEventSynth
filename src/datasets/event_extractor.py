"""Event extraction for time-series anomaly segments.

In this project, a contiguous anomaly segment is treated as an event timeline.
This module converts pointwise binary labels into half-open event windows [start, end).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


EventWindow = tuple[int, int]


@dataclass
class EventSegment:
    """A contiguous event/anomaly segment within one time series."""

    event_id: str
    series_id: str
    start_idx: int
    end_idx: int
    values: list[float] | list[list[float]]
    labels: list[int]
    local_time: list[float]
    context_values: list[float] | list[list[float]]


def labels_to_events(labels: Iterable[int], min_length: int = 1, merge_gap: int = 0) -> list[EventWindow]:
    """Convert pointwise labels into half-open anomaly windows [start, end)."""
    label_list = [1 if int(value) > 0 else 0 for value in labels]
    if min_length < 1:
        raise ValueError("min_length must be at least 1")
    if merge_gap < 0:
        raise ValueError("merge_gap must be non-negative")
    raw_events: list[EventWindow] = []
    start: int | None = None
    for idx, label in enumerate(label_list):
        if label == 1 and start is None:
            start = idx
        elif label == 0 and start is not None:
            raw_events.append((start, idx))
            start = None
    if start is not None:
        raw_events.append((start, len(label_list)))

    merged: list[EventWindow] = []
    for start_idx, end_idx in raw_events:
        if not merged:
            merged.append((start_idx, end_idx))
            continue
        prev_start, prev_end = merged[-1]
        if start_idx - prev_end <= merge_gap:
            merged[-1] = (prev_start, end_idx)
        else:
            merged.append((start_idx, end_idx))

    return [(start_idx, end_idx) for start_idx, end_idx in merged if (end_idx - start_idx) >= min_length]


def event_to_mask(length: int, events: Iterable[EventWindow]) -> np.ndarray:
    """Convert half-open event windows into a binary pointwise mask."""
    if length < 0:
        raise ValueError("length must be non-negative")
    mask = np.zeros(length, dtype=int)
    for start_idx, end_idx in events:
        if start_idx < 0 or end_idx < start_idx or end_idx > length:
            raise ValueError(f"Invalid event window [{start_idx}, {end_idx}) for length={length}")
        mask[start_idx:end_idx] = 1
    return mask


def compute_event_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate event statistics across time-series records."""
    num_series = len(records)
    lengths = [len(record["labels"]) for record in records]
    events = [event for record in records for event in labels_to_events(record["labels"])]
    event_lengths = [end_idx - start_idx for start_idx, end_idx in events]
    total_points = sum(lengths)
    anomaly_points = sum(int(np.sum(np.asarray(record["labels"], dtype=int) > 0)) for record in records)
    return {
        "num_series": num_series,
        "num_points": total_points,
        "num_events": len(events),
        "length_min": min(lengths) if lengths else 0,
        "length_mean": float(np.mean(lengths)) if lengths else 0.0,
        "length_max": max(lengths) if lengths else 0,
        "event_length_min": min(event_lengths) if event_lengths else 0,
        "event_length_mean": float(np.mean(event_lengths)) if event_lengths else 0.0,
        "event_length_max": max(event_lengths) if event_lengths else 0,
        "anomaly_point_ratio": (anomaly_points / total_points) if total_points else 0.0,
    }


def _normalize_local_time(length: int) -> list[float]:
    if length <= 1:
        return [0.0]
    return np.linspace(0.0, 1.0, length, dtype=float).tolist()


def extract_event_segments(
    timeline: dict[str, Any],
    *,
    min_event_length: int = 1,
    merge_gap: int = 0,
    context_padding: int = 0,
) -> list[EventSegment]:
    """Extract contiguous positive segments from a labeled series."""
    series_id = str(timeline["series_id"])
    labels = [int(x) for x in timeline["labels"]]
    values_array = np.asarray(timeline["values"], dtype=float)
    windows = labels_to_events(labels, min_length=min_event_length, merge_gap=merge_gap)

    segments: list[EventSegment] = []
    for event_idx, (event_start, event_end) in enumerate(windows):
        left = max(0, event_start - context_padding)
        right = min(len(values_array), event_end + context_padding)
        event_values = values_array[event_start:event_end]
        context_values = values_array[left:right]
        segments.append(
            EventSegment(
                event_id=f"{series_id}_event_{event_idx}",
                series_id=series_id,
                start_idx=event_start,
                end_idx=event_end,
                values=event_values.tolist(),
                labels=labels[event_start:event_end],
                local_time=_normalize_local_time(len(event_values)),
                context_values=context_values.tolist(),
            )
        )
    return segments
