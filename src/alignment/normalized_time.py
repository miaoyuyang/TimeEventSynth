"""Normalized-time event-window alignment, interpolation, and amplitude calibration."""

from __future__ import annotations

from typing import Any

import numpy as np


def _as_2d(values: list[float] | list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D event window, got shape {array.shape}")


def _restore_shape(values: np.ndarray) -> np.ndarray:
    return values[:, 0] if values.ndim == 2 and values.shape[1] == 1 else values


def _to_1d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    return array.reshape(-1) if array.ndim == 1 else array[:, 0]


def resample_window(values: list[float] | list[list[float]] | np.ndarray, target_length: int) -> np.ndarray:
    """Linearly resample an event window to a target length."""
    if target_length < 1:
        raise ValueError("target_length must be at least 1")
    array = _as_2d(values)
    source_length = len(array)
    if source_length == target_length:
        return _restore_shape(array.copy())
    if source_length == 1:
        repeated = np.repeat(array, target_length, axis=0)
        return _restore_shape(repeated)
    source_x = np.linspace(0.0, 1.0, source_length, dtype=float)
    target_x = np.linspace(0.0, 1.0, target_length, dtype=float)
    resampled_columns = [np.interp(target_x, source_x, array[:, dim]) for dim in range(array.shape[1])]
    resampled = np.stack(resampled_columns, axis=1)
    return _restore_shape(resampled)


def normalize_event_window(values: list[float] | list[list[float]] | np.ndarray, grid_size: int = 64) -> np.ndarray:
    """Resample an event window onto a fixed normalized-time grid."""
    return resample_window(values, grid_size)


def denormalize_event_window(normalized_values: list[float] | list[list[float]] | np.ndarray, target_length: int) -> np.ndarray:
    """Resample a normalized event representation back to a target event length."""
    return resample_window(normalized_values, target_length)


def assign_normalized_event_time(events: list[dict[str, Any]], num_buckets: int) -> list[dict[str, Any]]:
    aligned: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        local_time = float(row.get("local_time_center", 0.0))
        row["universal_position"] = min(max(local_time, 0.0), 1.0)
        row["universal_bucket"] = min(int(row["universal_position"] * num_buckets), num_buckets - 1)
        aligned.append(row)
    return aligned


def robust_std(values: np.ndarray, *, eps: float = 1e-6) -> float:
    """Robust scale estimate from IQR (Gaussian-compatible)."""
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        return eps
    q75, q25 = np.percentile(array, [75, 25])
    iqr = float(q75 - q25)
    scale = iqr / 1.349 if iqr > 0 else float(np.std(array))
    return max(scale, eps)


def local_context_stats(
    series_values: np.ndarray,
    start: int,
    end: int,
    *,
    context_size: int = 5,
) -> dict[str, float]:
    """Compute pre/post event context statistics from a full series."""
    values = _to_1d(series_values)
    pre = values[max(0, start - context_size) : start]
    post = values[end : min(len(values), end + context_size)]
    event = values[start:end]
    pre_slope = float(pre[-1] - pre[0]) / max(len(pre) - 1, 1) if len(pre) >= 2 else 0.0
    post_slope = float(post[-1] - post[0]) / max(len(post) - 1, 1) if len(post) >= 2 else 0.0
    return {
        "pre_mean": float(np.mean(pre)) if pre.size else float(np.mean(values)) if values.size else 0.0,
        "pre_std": float(np.std(pre)) if pre.size else 0.0,
        "post_mean": float(np.mean(post)) if post.size else float(np.mean(values)) if values.size else 0.0,
        "post_std": float(np.std(post)) if post.size else 0.0,
        "event_amplitude_range": float(np.max(event) - np.min(event)) if event.size else 0.0,
        "pre_slope": pre_slope,
        "post_slope": post_slope,
    }


def apply_partial_target_synthesis(
    donor_shape: np.ndarray,
    target_event_values: np.ndarray,
    partial_labels: np.ndarray | None,
    *,
    mode: str = "full_replacement",
) -> np.ndarray:
    """Blend donor shape with observed target anchors for partially labeled windows."""
    donor = _to_1d(donor_shape)
    target = _to_1d(target_event_values)
    if len(donor) != len(target):
        donor = _to_1d(resample_window(donor, len(target)))

    if mode == "full_replacement" or partial_labels is None:
        return donor.copy()

    labels = np.asarray(partial_labels, dtype=int).reshape(-1)
    if labels.size != len(target):
        return donor.copy()
    if not np.any(labels > 0):
        return donor.copy()

    if mode == "residual_baseline":
        baseline = float(np.mean(target[labels <= 0])) if np.any(labels <= 0) else float(np.mean(target))
        residual = donor - float(np.mean(donor))
        blended = baseline + residual
        blended[labels > 0] = target[labels > 0]
        return blended

    if mode == "anchor_interpolate":
        anchor_mask = labels > 0
        blended = donor.copy()
        blended[anchor_mask] = target[anchor_mask]
        missing = ~anchor_mask
        if np.any(missing):
            anchor_indices = np.flatnonzero(anchor_mask)
            if anchor_indices.size >= 2:
                full_indices = np.arange(len(target))
                blended[missing] = np.interp(full_indices[missing], anchor_indices, target[anchor_indices])
            elif anchor_indices.size == 1:
                blended[missing] = target[anchor_indices[0]]
        return blended

    raise ValueError(f"Unsupported partial_target mode: {mode}")


def fill_masked_from_donor(
    donor_shape: np.ndarray,
    target_event_values: np.ndarray,
    partial_labels: np.ndarray,
) -> np.ndarray:
    """Keep observed target anchors and fill masked positions from donor shape."""
    donor = _to_1d(donor_shape)
    target = _to_1d(target_event_values)
    if len(donor) != len(target):
        donor = _to_1d(resample_window(donor, len(target)))
    labels = np.asarray(partial_labels, dtype=int).reshape(-1)
    if labels.size != len(target):
        raise ValueError("partial_labels length must match target event length")
    reconstructed = donor.copy()
    observed = labels > 0
    reconstructed[observed] = target[observed]
    return reconstructed


def calibrate_amplitude_to_context(
    synthetic_values: np.ndarray,
    *,
    donor_values: np.ndarray,
    target_event_values: np.ndarray,
    series_values: np.ndarray | None,
    start: int,
    end: int,
    mode: str = "none",
    context_size: int = 5,
    partial_labels: np.ndarray | None = None,
    match_partial_amplitude: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Adjust synthetic event amplitudes to match target local context."""
    synthetic = _to_1d(synthetic_values).copy()
    donor = _to_1d(donor_values)
    target = _to_1d(target_event_values)
    audit = {
        "donor_original_mean": float(np.mean(donor)) if donor.size else 0.0,
        "donor_original_std": float(np.std(donor)) if donor.size else 0.0,
        "target_context_mean": float(np.mean(target)) if target.size else 0.0,
        "target_context_std": float(np.std(target)) if target.size else 0.0,
        "scale": 1.0,
        "shift": 0.0,
        "amplitude_calibration": mode,
    }

    if mode == "none" or synthetic.size == 0:
        return synthetic, audit

    if series_values is not None:
        context = local_context_stats(series_values, start, end, context_size=context_size)
        audit["target_context_mean"] = float(context["pre_mean"])
        audit["target_context_std"] = float(context["pre_std"])

    target_mean = audit["target_context_mean"]
    donor_mean = float(np.mean(synthetic))
    shift = target_mean - donor_mean
    calibrated = synthetic + shift
    audit["shift"] = float(shift)

    if mode == "robust_scale_shift":
        donor_scale = robust_std(synthetic)
        target_scale = audit["target_context_std"] if audit["target_context_std"] > 0 else robust_std(target)
        if partial_labels is not None and match_partial_amplitude and np.any(np.asarray(partial_labels) > 0):
            observed = target[np.asarray(partial_labels, dtype=int) > 0]
            if observed.size:
                target_scale = max(robust_std(observed), robust_std(target))
        scale = target_scale / donor_scale
        calibrated = (synthetic - donor_mean) * scale + target_mean
        audit["scale"] = float(scale)
        audit["shift"] = float(target_mean - donor_mean * scale)

    audit["donor_original_mean"] = float(np.mean(donor)) if donor.size else audit["donor_original_mean"]
    audit["donor_original_std"] = float(robust_std(donor)) if donor.size else audit["donor_original_std"]
    audit["target_context_std"] = float(
        audit["target_context_std"] if audit["target_context_std"] > 0 else robust_std(target)
    )
    return calibrated.astype(float), audit
