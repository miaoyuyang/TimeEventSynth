"""Donor retrieval for normalized-time event-window synthesis."""

from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from typing import Any

import numpy as np

from ..alignment.normalized_time import local_context_stats, normalize_event_window
from ..datasets.event_extractor import extract_event_segments


@dataclass
class EventWindow:
    """An extracted anomaly window used for donor retrieval and synthesis."""

    series_id: str
    start: int
    end: int
    values: np.ndarray
    label: str
    metadata: dict[str, Any] = field(default_factory=dict)


def collect_event_windows(records: list[Any]) -> list[EventWindow]:
    """Extract event windows from full series records."""
    windows: list[EventWindow] = []
    for record in records:
        if hasattr(record, "series_id"):
            series_id = record.series_id
            values = np.asarray(record.values, dtype=float)
            labels = np.asarray(record.labels, dtype=int)
        else:
            series_id = record["series_id"]
            values = np.asarray(record["values"], dtype=float)
            labels = np.asarray(record["labels"], dtype=int)
        timeline = {
            "series_id": series_id,
            "values": values,
            "labels": labels,
        }
        segments = extract_event_segments(timeline, min_event_length=1, merge_gap=0, context_padding=0)
        for segment in segments:
            windows.append(
                EventWindow(
                    series_id=segment.series_id,
                    start=segment.start_idx,
                    end=segment.end_idx,
                    values=np.asarray(segment.values, dtype=float),
                    label="anomaly",
                    metadata={
                        "event_id": segment.event_id,
                        "length": segment.end_idx - segment.start_idx,
                        "series_values": values,
                        "series_labels": labels,
                    },
                )
            )
    return windows


def compute_context_features(window: EventWindow, *, context_size: int = 5) -> np.ndarray:
    """Contextual features for donor similarity beyond normalized shape."""
    series_values = window.metadata.get("series_values")
    if series_values is None:
        event = np.asarray(window.values, dtype=float).reshape(-1)
        return np.asarray(
            [
                float(window.end - window.start),
                float(np.mean(event)),
                float(np.std(event)),
                float(np.mean(event)),
                float(np.std(event)),
                float(np.max(event) - np.min(event)) if event.size else 0.0,
                0.0,
                0.0,
            ],
            dtype=float,
        )
    stats = local_context_stats(np.asarray(series_values, dtype=float), window.start, window.end, context_size=context_size)
    return np.asarray(
        [
            float(window.end - window.start),
            stats["pre_mean"],
            stats["pre_std"],
            stats["post_mean"],
            stats["post_std"],
            stats["event_amplitude_range"],
            stats["pre_slope"],
            stats["post_slope"],
        ],
        dtype=float,
    )


def compute_window_embedding(window: EventWindow, grid_size: int = 64, *, context_size: int = 5) -> np.ndarray:
    """Embed a window by normalized shape plus local context statistics."""
    normalized = normalize_event_window(window.values, grid_size=grid_size)
    normalized_2d = normalized.reshape(grid_size, -1) if np.asarray(normalized).ndim == 1 else np.asarray(normalized, dtype=float)
    flattened = normalized_2d.reshape(-1)
    shape_stats = np.asarray(
        [
            float(np.mean(normalized_2d)),
            float(np.std(normalized_2d)),
            float(np.max(normalized_2d)),
            float(np.min(normalized_2d)),
        ],
        dtype=float,
    )
    context = compute_context_features(window, context_size=context_size)
    return np.concatenate([flattened.astype(float), shape_stats, context], axis=0)


def retrieve_topk_donors(
    target_window: EventWindow,
    donor_windows: list[EventWindow],
    k: int = 5,
    exclude_same_series: bool = True,
    max_donors_per_source_series: int | None = None,
    avoid_single_series_dominance: bool = True,
    group_key: str | None = None,
    max_donors_per_group: int | None = None,
    restrict_to_target_group: bool = False,
    *,
    context_size: int = 5,
) -> list[tuple[EventWindow, float]]:
    """Retrieve top-k donor windows by shape + context embedding similarity."""
    target_embedding = compute_window_embedding(target_window, context_size=context_size)
    target_group = target_window.metadata.get(group_key) if group_key else None
    scored: list[tuple[EventWindow, float]] = []
    for donor in donor_windows:
        if exclude_same_series and donor.series_id == target_window.series_id:
            continue
        if restrict_to_target_group and group_key and target_group is not None:
            if donor.metadata.get(group_key) != target_group:
                continue
        similarity = _cosine(target_embedding, compute_window_embedding(donor, context_size=context_size))
        scored.append((donor, float(similarity)))
    scored.sort(key=lambda item: (-item[1], item[0].series_id, item[0].start, item[0].end))

    selected: list[tuple[EventWindow, float]] = []
    series_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    blocked_single_series: set[str] = set()
    for donor, score in scored:
        donor_series = str(donor.series_id)
        donor_group = str(donor.metadata.get(group_key, donor_series)) if group_key else donor_series
        if max_donors_per_source_series is not None and series_counts[donor_series] >= max_donors_per_source_series:
            continue
        if max_donors_per_group is not None and group_counts[donor_group] >= max_donors_per_group:
            continue
        if avoid_single_series_dominance and len(selected) > 0 and len({item[0].series_id for item in selected}) == 1:
            first_series = str(selected[0][0].series_id)
            if donor_series == first_series and donor_series not in blocked_single_series:
                blocked_single_series.add(donor_series)
                continue
        selected.append((donor, score))
        series_counts[donor_series] += 1
        group_counts[donor_group] += 1
        if len(selected) >= k:
            break
    return selected


def _cosine(x: np.ndarray, y: np.ndarray) -> float:
    denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-8)
    return float(np.dot(x, y) / denom)


def donor_similarity_stats(similarities: list[float]) -> dict[str, float]:
    """Summarize raw donor retrieval similarities for audit logging."""
    if not similarities:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    values = np.asarray(similarities, dtype=float)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def retrieve_donors(
    query_event: dict[str, Any],
    donor_pool: list[dict[str, Any]],
    *,
    top_k: int,
    method: str = "normalized_time",
) -> list[dict[str, Any]]:
    """Backward-compatible wrapper for the smoke path."""
    query = EventWindow(
        series_id=str(query_event["series_id"]),
        start=int(query_event.get("start_idx", 0)),
        end=int(query_event.get("end_idx", len(query_event["values"]))),
        values=np.asarray(query_event["values"], dtype=float),
        label=str(query_event.get("event_label", "anomaly")),
        metadata={"event_id": query_event.get("event_id")},
    )
    donors = [
        EventWindow(
            series_id=str(item["series_id"]),
            start=int(item.get("start_idx", 0)),
            end=int(item.get("end_idx", len(item["values"]))),
            values=np.asarray(item["values"], dtype=float),
            label=str(item.get("event_label", "anomaly")),
            metadata={"event_id": item.get("event_id")},
        )
        for item in donor_pool
    ]
    results = retrieve_topk_donors(query, donors, k=top_k, exclude_same_series=True)
    return [
        {
            "donor": {
                "event_id": donor.metadata.get("event_id"),
                "series_id": donor.series_id,
                "values": donor.values.tolist(),
                "label": donor.label,
            },
            "score": score,
        }
        for donor, score in results
    ]
