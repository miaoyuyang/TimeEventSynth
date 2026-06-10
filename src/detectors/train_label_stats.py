"""Training-label diagnostics for low-label supervised experiments."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..datasets.event_extractor import labels_to_events


def _is_synthetic_record(record: Any) -> bool:
    """Return True only for augmented synthetic/oversampled training windows."""
    series_id = str(getattr(record, "series_id", ""))
    source_path = str(getattr(record, "source_path", "") or "")
    if "__synthetic__" in series_id or "__oversampled__" in series_id:
        return True
    if source_path == "oversampled":
        return True
    metadata = getattr(record, "metadata", {}) or {}
    return bool(metadata.get("synthetic_copy", False))


def compute_train_label_stats(
    train_records: list[Any],
    *,
    labeled_fraction: float,
    real_train_count: int | None = None,
) -> dict[str, Any]:
    """Summarize label availability in the training set used by a detector."""
    real_pos_points = 0
    real_neg_points = 0
    synthetic_pos_points = 0
    synthetic_neg_points = 0
    real_pos_event_windows = 0
    synthetic_pos_event_windows = 0

    for record in train_records:
        labels = np.asarray(record.labels, dtype=int).reshape(-1)
        pos = int(np.sum(labels > 0))
        neg = int(len(labels) - pos)
        if _is_synthetic_record(record):
            synthetic_pos_points += pos
            synthetic_neg_points += neg
            synthetic_pos_event_windows += len(labels_to_events(labels))
        else:
            real_pos_points += pos
            real_neg_points += neg
            real_pos_event_windows += len(labels_to_events(labels))

    return {
        "labeled_fraction": float(labeled_fraction),
        "num_real_train_records": int(real_train_count if real_train_count is not None else sum(not _is_synthetic_record(r) for r in train_records)),
        "num_synthetic_train_records": int(sum(_is_synthetic_record(r) for r in train_records)),
        "num_positive_train_points": int(real_pos_points + synthetic_pos_points),
        "num_negative_train_points": int(real_neg_points + synthetic_neg_points),
        "num_real_positive_train_points": int(real_pos_points),
        "num_real_negative_train_points": int(real_neg_points),
        "num_positive_train_event_windows": int(real_pos_event_windows),
        "num_synthetic_positive_windows": int(synthetic_pos_event_windows),
        "num_synthetic_positive_points": int(synthetic_pos_points),
    }
