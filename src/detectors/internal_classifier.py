"""Wrapper around the existing internal supervised pointwise classifier."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .base import DetectorBackbone
from .classical import build_detector
from ..datasets.tsb_loader import TimeSeriesRecord


def _expand_synthetic_with_context(record: Any) -> Any:
    metadata = getattr(record, "metadata", {}) or {}
    if not bool(metadata.get("synthetic", False)):
        return record
    target_series_values = metadata.get("target_series_values")
    target_series_labels = metadata.get("target_series_labels")
    interval = metadata.get("target_event_interval")
    if target_series_values is None or target_series_labels is None or interval is None:
        return record

    full_values = np.asarray(target_series_values, dtype=float).copy()
    full_labels = np.asarray(target_series_labels, dtype=int).copy()
    if full_values.ndim == 1:
        full_values_2d = full_values.reshape(-1, 1)
    else:
        full_values_2d = full_values

    synthetic_values = np.asarray(record.values, dtype=float)
    if synthetic_values.ndim == 1:
        synthetic_values_2d = synthetic_values.reshape(-1, 1)
    else:
        synthetic_values_2d = synthetic_values
    synthetic_labels = np.asarray(record.labels, dtype=int).reshape(-1)

    start, end = int(interval[0]), int(interval[1])
    target_length = max(end - start, 0)
    if target_length <= 0 or synthetic_values_2d.shape[0] != target_length or synthetic_labels.shape[0] != target_length:
        return record
    if end > len(full_values_2d) or end > len(full_labels):
        return record

    full_values_2d[start:end] = synthetic_values_2d
    full_labels[start:end] = synthetic_labels
    expanded_values = full_values_2d[:, 0] if full_values.ndim == 1 else full_values_2d
    expanded_metadata = dict(metadata)
    expanded_metadata["synthetic_context_embedded"] = True
    return TimeSeriesRecord(
        series_id=str(getattr(record, "series_id", metadata.get("target_series", "synthetic"))) + "__context_embedded",
        values=np.asarray(expanded_values, dtype=float),
        labels=np.asarray(full_labels, dtype=int),
        timestamps=getattr(record, "timestamps", None),
        source_path="synthetic_context_embedded",
        metadata=expanded_metadata,
    )


@dataclass
class InternalClassifierBackbone(DetectorBackbone):
    """Use the project's existing supervised window classifier as a backbone."""

    detector_config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = "current_internal_classifier"
        self.supports_training_augmentation = True
        self.is_supervised = True
        self.model = build_detector(self.detector_config)

    def fit(
        self,
        train_series: list[Any],
        train_labels: list[np.ndarray] | None = None,
        synthetic_windows: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "InternalClassifierBackbone":
        del config
        embed_synthetic_context = bool(self.detector_config.get("embed_synthetic_context", True))
        prepared_synthetic = list(synthetic_windows or [])
        if embed_synthetic_context and prepared_synthetic:
            prepared_synthetic = [_expand_synthetic_with_context(window) for window in prepared_synthetic]
        combined = list(train_series) + prepared_synthetic
        combined_labels = None
        if train_labels is not None:
            combined_labels = list(train_labels) + [
                np.asarray(window.labels, dtype=int).reshape(-1) for window in prepared_synthetic
            ]
        self.model.fit(combined, combined_labels)
        return self

    def score(
        self,
        test_series: list[Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        del config
        return self.model.score(test_series)
