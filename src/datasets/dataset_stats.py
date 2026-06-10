"""Machine-readable dataset inspection statistics."""

from __future__ import annotations

from typing import Any

import numpy as np

from .event_extractor import compute_event_stats, labels_to_events
from .load_options import DatasetLoadOptions
from .split_builder import series_parent_folder
from .tsb_loader import TimeSeriesRecord


def _length_stats(lengths: list[int]) -> dict[str, float | int]:
    if not lengths:
        return {"min": 0, "mean": 0.0, "median": 0.0, "max": 0}
    return {
        "min": min(lengths),
        "mean": float(np.mean(lengths)),
        "median": float(np.median(lengths)),
        "max": max(lengths),
    }


def _records_subset(records: list[TimeSeriesRecord], split_ids: set[str]) -> list[TimeSeriesRecord]:
    return [record for record in records if record.series_id in split_ids]


def build_dataset_stats(
    records: list[TimeSeriesRecord],
    *,
    data_path: str,
    load_options: DatasetLoadOptions,
    load_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable inspection summary for a loaded dataset."""
    stats = compute_event_stats([{"series_id": record.series_id, "labels": record.labels} for record in records])
    lengths = [int(len(record.labels)) for record in records]
    series_events = [(record.series_id, labels_to_events(record.labels)) for record in records]
    event_lengths = [end_idx - start_idx for _, events in series_events for start_idx, end_idx in events]

    longest = sorted(((record.series_id, len(record.labels)) for record in records), key=lambda item: (-item[1], item[0]))[:10]
    top_event_count = sorted(((series_id, len(events)) for series_id, events in series_events), key=lambda item: (-item[1], item[0]))[
        :10
    ]
    parent_folders = sorted({series_parent_folder(record) for record in records})

    return {
        "dataset_name": load_options.dataset_name,
        "data_path": data_path,
        "load_options": load_options.to_dict(),
        "load_summary": load_summary or {},
        "num_series": len(records),
        "num_parent_folders": len(parent_folders),
        "parent_folders": parent_folders,
        "length_statistics": _length_stats(lengths),
        "length_stats": _length_stats(lengths),
        "num_event_windows": stats["num_events"],
        "total_event_windows": stats["num_events"],
        "event_length_statistics": _length_stats(event_lengths),
        "event_window_length_stats": _length_stats(event_lengths),
        "anomaly_point_ratio": stats["anomaly_point_ratio"],
        "event_ratio": stats["anomaly_point_ratio"],
        "num_points": stats["num_points"],
        "total_points": stats["num_points"],
        "total_event_points": int(sum(int(np.sum(record.labels > 0)) for record in records)),
        "top_longest_series": [{"series_id": series_id, "length": length} for series_id, length in longest],
        "top_series_by_event_count": [{"series_id": series_id, "num_events": count} for series_id, count in top_event_count],
    }


def build_experiment_dataset_stats(
    records: list[TimeSeriesRecord],
    split_ids: dict[str, list[str]],
    *,
    load_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build split-aware dataset stats for experiment artifacts."""
    train_ids = set(split_ids.get("train", []))
    val_ids = set(split_ids.get("val", []))
    test_ids = set(split_ids.get("test", []))
    train_records = _records_subset(records, train_ids)
    val_records = _records_subset(records, val_ids)
    test_records = _records_subset(records, test_ids)

    all_stats = compute_event_stats([{"series_id": record.series_id, "labels": record.labels} for record in records])
    test_stats = compute_event_stats([{"series_id": record.series_id, "labels": record.labels} for record in test_records])
    total_event_points = int(sum(int(np.sum(record.labels > 0)) for record in records))
    lengths = [int(len(record.labels)) for record in records]
    event_lengths = [
        end_idx - start_idx
        for record in records
        for start_idx, end_idx in labels_to_events(record.labels)
    ]
    parent_folders = sorted({series_parent_folder(record) for record in records})
    test_parent_folders = sorted({series_parent_folder(record) for record in test_records})

    return {
        "num_series_total": int((load_summary or {}).get("num_series_loaded", len(records))),
        "num_series_after_filter": len(records),
        "num_train_series": len(train_records),
        "num_val_series": len(val_records),
        "num_test_series": len(test_records),
        "total_points": all_stats["num_points"],
        "total_event_points": total_event_points,
        "total_event_windows": all_stats["num_events"],
        "event_ratio": all_stats["anomaly_point_ratio"],
        "length_stats": _length_stats(lengths),
        "event_window_length_stats": _length_stats(event_lengths),
        "num_parent_folders": len(parent_folders),
        "parent_folders": parent_folders,
        "test_parent_folders": test_parent_folders,
        "num_test_event_windows": test_stats["num_events"],
        "load_summary": load_summary or {},
    }
