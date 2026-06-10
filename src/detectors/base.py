"""Minimal detector-backbone abstraction for augmentation experiments.

This layer treats anomaly detectors as pluggable backbones and keeps the
augmentation logic detector-agnostic. Supervised backbones may learn directly
from synthetic anomalous windows, while unsupervised backbones typically use
synthetic windows only for threshold calibration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..evaluation.thresholding import calibrate_threshold


def _as_2d(values: list[float] | list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D time-series values, got shape {array.shape}")


def make_windows(
    series: list[float] | list[list[float]] | np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Extract fixed-length windows and their half-open index spans."""
    values = _as_2d(series)
    n_points = len(values)
    if n_points == 0:
        return np.zeros((0, max(window_size, 1), values.shape[1]), dtype=float), []
    size = max(int(window_size), 1)
    step = max(int(stride), 1)
    if n_points < size:
        pad_len = size - n_points
        pad_value = values[-1:, :]
        pad = np.repeat(pad_value, pad_len, axis=0)
        padded = np.concatenate([values, pad], axis=0)
        return padded[np.newaxis, :, :].astype(float), [(0, n_points)]
    if n_points <= size:
        return values[np.newaxis, :, :].astype(float), [(0, n_points)]

    windows: list[np.ndarray] = []
    spans: list[tuple[int, int]] = []
    start = 0
    while start + size <= n_points:
        end = start + size
        windows.append(values[start:end].astype(float))
        spans.append((start, end))
        start += step
    if spans and spans[-1][1] < n_points:
        start = n_points - size
        windows.append(values[start:n_points].astype(float))
        spans.append((start, n_points))
    return np.asarray(windows, dtype=float), spans


def make_forecast_windows(
    series: list[float] | list[list[float]] | np.ndarray,
    window_size: int,
    stride: int = 1,
    *,
    horizon: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Extract forecasting windows, next-step targets, and target indices.

    Returns windows with shape ``[N, W, D]``, targets with shape ``[N, H, D]``,
    and the point indices where each forecast target begins.
    """
    values = _as_2d(series)
    n_points, n_dims = values.shape
    size = max(int(window_size), 1)
    step = max(int(stride), 1)
    horizon = max(int(horizon), 1)
    if n_points <= horizon:
        return (
            np.zeros((0, size, n_dims), dtype=float),
            np.zeros((0, horizon, n_dims), dtype=float),
            [],
        )

    windows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    indices: list[int] = []

    if n_points <= size:
        prefix = values[: max(n_points - horizon, 1)]
        if len(prefix) < size:
            pad_value = prefix[0:1] if len(prefix) else values[0:1]
            pad = np.repeat(pad_value, size - len(prefix), axis=0)
            prefix = np.concatenate([pad, prefix], axis=0)
        target = values[n_points - horizon : n_points]
        windows.append(prefix[-size:].astype(float))
        targets.append(target.astype(float))
        indices.append(n_points - horizon)
    else:
        start = 0
        while start + size + horizon <= n_points:
            end = start + size
            windows.append(values[start:end].astype(float))
            targets.append(values[end : end + horizon].astype(float))
            indices.append(end)
            start += step
        if not indices or indices[-1] != n_points - horizon:
            end = n_points - horizon
            start = max(0, end - size)
            window = values[start:end]
            if len(window) < size:
                pad = np.repeat(window[0:1], size - len(window), axis=0)
                window = np.concatenate([pad, window], axis=0)
            windows.append(window.astype(float))
            targets.append(values[end : end + horizon].astype(float))
            indices.append(end)
    return np.asarray(windows, dtype=float), np.asarray(targets, dtype=float), indices


def map_forecast_scores_to_points(
    n_points: int,
    target_indices: list[int],
    forecast_scores: np.ndarray,
    *,
    horizon: int = 1,
) -> np.ndarray:
    """Project forecast errors back to point-level anomaly scores."""
    if n_points <= 0:
        return np.zeros(0, dtype=float)
    scores = np.asarray(forecast_scores, dtype=float).reshape(-1)
    point_scores = np.zeros(n_points, dtype=float)
    counts = np.zeros(n_points, dtype=float)
    for start, score in zip(target_indices, scores):
        end = min(n_points, int(start) + max(int(horizon), 1))
        point_scores[start:end] += float(score)
        counts[start:end] += 1.0
    if target_indices:
        first_idx = max(0, min(int(target_indices[0]), n_points - 1))
        point_scores[:first_idx] = point_scores[first_idx]
        counts[:first_idx] = np.where(counts[:first_idx] <= 0, 1.0, counts[:first_idx])
    counts = np.where(counts <= 0, 1.0, counts)
    return (point_scores / counts).astype(float)


def window_features(
    windows: np.ndarray,
    *,
    include_stats: bool = True,
) -> np.ndarray:
    """Turn raw windows into tabular features for sklearn detectors."""
    array = np.asarray(windows, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected windows with shape [N, W, D], got {array.shape}")
    flattened = array.reshape(array.shape[0], -1)
    if not include_stats or array.shape[0] == 0:
        return flattened

    mean = np.mean(array, axis=1)
    std = np.std(array, axis=1)
    min_values = np.min(array, axis=1)
    max_values = np.max(array, axis=1)
    slope = array[:, -1, :] - array[:, 0, :]
    stats = np.concatenate([mean, std, min_values, max_values, slope], axis=1)
    return np.concatenate([flattened, stats], axis=1)


def map_window_scores_to_points(
    n_points: int,
    spans: list[tuple[int, int]],
    window_scores: np.ndarray,
    *,
    reduction: str = "max",
) -> np.ndarray:
    """Project window-level anomaly scores back to point-level scores."""
    if n_points <= 0:
        return np.zeros(0, dtype=float)
    scores = np.asarray(window_scores, dtype=float).reshape(-1)
    point_scores = np.zeros(n_points, dtype=float)
    counts = np.zeros(n_points, dtype=float)
    for (start, end), score in zip(spans, scores):
        if reduction == "max":
            point_scores[start:end] = np.maximum(point_scores[start:end], score)
        elif reduction == "mean":
            point_scores[start:end] += score
            counts[start:end] += 1.0
        else:
            raise ValueError(f"Unsupported window-score reduction: {reduction}")
    if reduction == "mean":
        counts = np.where(counts <= 0, 1.0, counts)
        point_scores = point_scores / counts
    return point_scores.astype(float)


def flatten_score_dict(scores: dict[str, np.ndarray]) -> np.ndarray:
    """Flatten a per-series score mapping into a single score vector."""
    if not scores:
        return np.zeros(0, dtype=float)
    return np.concatenate([np.asarray(values, dtype=float).reshape(-1) for values in scores.values()], axis=0)


def flatten_label_records(records: list[Any]) -> np.ndarray:
    """Flatten labels from TimeSeriesRecord-like objects."""
    if not records:
        return np.zeros(0, dtype=int)
    return np.concatenate([np.asarray(record.labels, dtype=int).reshape(-1) for record in records], axis=0)


def score_normalization_mode(config: dict[str, Any] | None) -> str:
    """Resolve how backbone ``score`` outputs are scaled before thresholding."""
    if not config:
        return "none"
    evaluation = config.get("evaluation", {})
    if not isinstance(evaluation, dict):
        return "none"
    return str(evaluation.get("score_normalization", "none")).lower()


def normalize_scores_per_record(raw_scores: np.ndarray) -> np.ndarray:
    """Min-max scale a single series/window score vector to [0, 1]."""
    values = np.asarray(raw_scores, dtype=float).reshape(-1)
    if values.size == 0:
        return values
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo < 1e-8:
        return np.zeros_like(values, dtype=float)
    return ((values - lo) / (hi - lo)).astype(float)


def apply_score_normalization(raw_scores: np.ndarray, mode: str | None) -> np.ndarray:
    """Apply configured score scaling.

    Default ``none`` keeps detector scores on a shared raw scale so validation and
    synthetic windows can be pooled for ``synthetic_separation`` calibration.
    Per-record min-max (``per_record``) destroys cross-series score gaps.
    """
    normalized_mode = str(mode or "none").lower()
    values = np.asarray(raw_scores, dtype=float).reshape(-1)
    if normalized_mode in {"none", "off", "raw"}:
        return values.astype(float)
    if normalized_mode in {"per_record", "minmax", "min_max"}:
        return normalize_scores_per_record(values)
    raise ValueError(f"Unsupported score_normalization mode: {normalized_mode}")


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    """Backward-compatible alias for per-record min-max normalization."""
    return normalize_scores_per_record(raw_scores)


@dataclass
class ThresholdCalibrationResult:
    """Threshold selection outcome for a detector backbone."""

    threshold: float
    method: str
    details: dict[str, Any]


_BACKBONE_THRESHOLD_FIELDS = (
    "threshold_mode",
    "quantile",
    "threshold_quantile",
    "grid_size",
    "metric",
    "max_false_positive_rate",
    "min_calibration_precision",
    "false_positive_penalty",
    "fallback_when_inverted_gap",
)


def _backbone_threshold_field_value(
    backbone: DetectorBackbone,
    field: str,
    *,
    backbone_config: dict[str, Any] | None = None,
) -> Any:
    """Read a threshold field from the wrapper, its YAML spec, or nested detector config."""
    value = getattr(backbone, field, None)
    if value is not None:
        return value
    spec_cfg = backbone_config or {}
    if spec_cfg.get(field) is not None:
        return spec_cfg[field]
    detector_cfg = getattr(backbone, "detector_config", None)
    if isinstance(detector_cfg, dict) and detector_cfg.get(field) is not None:
        return detector_cfg[field]
    return None


def merge_backbone_threshold_config(
    backbone: DetectorBackbone,
    calibration_config: dict[str, Any] | None = None,
    *,
    backbone_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge shared evaluation settings with per-backbone threshold calibration.

    ``calibration_config`` (typically ``evaluation``) supplies run-wide keys such as
    ``event_iou_threshold`` and oracle ``grid_size``. Per-backbone threshold fields
    (``threshold_mode``, ``quantile``, etc.) use precedence:

    1. attribute on the backbone wrapper (from ``detector_backbones[].config`` at build)
    2. ``backbone_config`` (same YAML block, passed explicitly by the runner)
    3. ``calibration_config`` fallback when neither source defines the field

    This prevents a global ``evaluation.threshold_mode`` from homogenizing IForest,
    OCSVM, LOF, and other detectors that declare their own calibration mode.
    """
    merged = dict(calibration_config or {})
    for field in _BACKBONE_THRESHOLD_FIELDS:
        value = _backbone_threshold_field_value(backbone, field, backbone_config=backbone_config)
        if value is None:
            continue
        merged[field] = value
    if merged.get("quantile") is not None:
        merged.setdefault("threshold_quantile", merged["quantile"])
    elif merged.get("threshold_quantile") is not None:
        merged.setdefault("quantile", merged["threshold_quantile"])
    return merged


class DetectorBackbone(ABC):
    """Simple detector API used by backbone-vs-augmentation experiments."""

    name: str
    supports_training_augmentation: bool = False
    is_supervised: bool = False

    @abstractmethod
    def fit(
        self,
        train_series: list[Any],
        train_labels: list[np.ndarray] | None = None,
        synthetic_windows: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "DetectorBackbone":
        """Fit the detector on train series."""

    @abstractmethod
    def score(
        self,
        test_series: list[Any],
        config: dict[str, Any] | None = None,
    ) -> dict[str, np.ndarray]:
        """Return per-series anomaly scores."""

    def calibrate_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray | None = None,
        synthetic_scores: np.ndarray | None = None,
        config: dict[str, Any] | None = None,
    ) -> ThresholdCalibrationResult:
        """Calibrate a decision threshold.

        Supervised backbones default to validation-label F1. Unsupervised
        backbones can instead use synthetic anomalous windows to separate normal
        validation scores from synthesized anomalies.
        """

        cfg = merge_backbone_threshold_config(self, config)
        labels_array = None if labels is None else np.asarray(labels, dtype=int).reshape(-1)
        score_array = np.asarray(scores, dtype=float).reshape(-1)
        synth_array = None if synthetic_scores is None else np.asarray(synthetic_scores, dtype=float).reshape(-1)

        mode = str(
            cfg.get(
                "threshold_mode",
                "synthetic_separation" if (not self.is_supervised and synth_array is not None and synth_array.size > 0) else (
                    "oracle_val" if labels_array is not None and labels_array.size == score_array.size else "quantile"
                ),
            )
        )
        result = calibrate_threshold(
            mode=mode,
            scores=score_array,
            labels=labels_array,
            synthetic_scores=synth_array,
            config=cfg,
        )
        method_name = {
            "synthetic_separation": "synthetic_positive_separation",
            "fp_aware_synthetic_separation": "fp_aware_synthetic_positive_separation",
            "synthetic_separation_fp_aware": "fp_aware_synthetic_positive_separation",
            "oracle_val": "validation_best_f1",
            "quantile": "quantile",
        }.get(result["threshold_mode"], str(result["threshold_mode"]))
        return ThresholdCalibrationResult(
            threshold=float(result["threshold"]),
            method=method_name,
            details={**dict(result.get("diagnostics", {})), "threshold_mode": str(result.get("threshold_mode", mode))},
        )

    def predict(self, scores: dict[str, np.ndarray] | np.ndarray, threshold: float) -> dict[str, np.ndarray] | np.ndarray:
        """Threshold anomaly scores into binary predictions."""
        if isinstance(scores, dict):
            return {
                str(series_id): (np.asarray(values, dtype=float) >= threshold).astype(int)
                for series_id, values in scores.items()
            }
        return (np.asarray(scores, dtype=float) >= threshold).astype(int)


__all__ = [
    "DetectorBackbone",
    "ThresholdCalibrationResult",
    "make_windows",
    "map_window_scores_to_points",
    "flatten_label_records",
    "flatten_score_dict",
    "window_features",
    "_normalize_scores",
]
