"""Low-label train masking shared across experiment runners."""

from __future__ import annotations

import random
from typing import Any

import numpy as np

from src.datasets.event_extractor import extract_event_segments
from src.datasets.tsb_loader import TimeSeriesRecord


def mask_train_labels_for_low_label(
    records: list[TimeSeriesRecord],
    labeled_fraction: float,
    seed: int,
) -> list[TimeSeriesRecord]:
    """Hide a random subset of train anomaly segments to simulate low label budgets."""
    rng = random.Random(seed)
    masked: list[TimeSeriesRecord] = []
    for record in records:
        timeline = {"series_id": record.series_id, "values": record.values, "labels": record.labels}
        segments = extract_event_segments(timeline, min_event_length=1, merge_gap=0, context_padding=0)
        if not segments:
            keep_indices: set[int] = set()
        elif len(segments) == 1:
            keep_indices = {0} if labeled_fraction > 0 else set()
        else:
            keep_count = int(round(len(segments) * labeled_fraction))
            keep_count = max(0, min(keep_count, len(segments)))
            keep_indices = set(rng.sample(range(len(segments)), keep_count)) if keep_count > 0 else set()
        new_labels = record.labels.copy()
        for idx, segment in enumerate(segments):
            if idx in keep_indices:
                continue
            new_labels[segment.start_idx : segment.end_idx] = 0
        masked.append(
            TimeSeriesRecord(
                series_id=record.series_id,
                values=record.values.copy(),
                labels=new_labels,
                timestamps=record.timestamps,
                source_path=record.source_path,
                metadata={**record.metadata, "masked_fraction": 1.0 - labeled_fraction},
            )
        )
    return masked


def mask_train_labels(records: list[Any], labeled_fraction: float, seed: int) -> list[TimeSeriesRecord]:
    """Backward-compatible alias."""
    return mask_train_labels_for_low_label(records, labeled_fraction, seed)
