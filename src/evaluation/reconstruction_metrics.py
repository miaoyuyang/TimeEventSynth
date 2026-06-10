"""Metrics for masked event-window reconstruction fidelity."""

from __future__ import annotations

import numpy as np


def compute_mask_interval(
    event_length: int,
    mask_fraction: float,
    policy: str,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Return half-open mask interval [start, end) inside an event window."""
    if event_length < 2:
        raise ValueError("event_length must be at least 2 for masking")
    fraction = float(mask_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1)")
    mask_len = max(1, int(round(event_length * fraction)))
    mask_len = min(mask_len, event_length - 1)

    base_policy = policy.split("_")[0] if policy else "middle"
    if base_policy == "prefix":
        return 0, mask_len
    if base_policy == "suffix":
        return event_length - mask_len, event_length
    if base_policy == "middle":
        start = (event_length - mask_len) // 2
        return start, start + mask_len
    if base_policy == "random":
        generator = rng or np.random.default_rng(0)
        start = int(generator.integers(0, event_length - mask_len + 1))
        return start, start + mask_len
    raise ValueError(f"Unsupported mask policy: {policy}")


def build_masked_observation(
    original: np.ndarray,
    mask_start: int,
    mask_end: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build masked observation and partial labels (1=observed, 0=masked)."""
    values = np.asarray(original, dtype=float).reshape(-1)
    if mask_start < 0 or mask_end > len(values) or mask_start >= mask_end:
        raise ValueError("Mask interval must lie strictly inside the event window")
    partial_labels = np.ones(len(values), dtype=int)
    partial_labels[mask_start:mask_end] = 0
    masked_observed = values.copy()
    masked_observed[mask_start:mask_end] = np.nan
    return masked_observed, partial_labels


def masked_region_mae(original: np.ndarray, reconstructed: np.ndarray, mask_start: int, mask_end: int) -> float:
    original_arr = np.asarray(original, dtype=float).reshape(-1)
    recon_arr = np.asarray(reconstructed, dtype=float).reshape(-1)
    region = original_arr[mask_start:mask_end]
    pred = recon_arr[mask_start:mask_end]
    return float(np.mean(np.abs(region - pred)))


def masked_region_mse(original: np.ndarray, reconstructed: np.ndarray, mask_start: int, mask_end: int) -> float:
    original_arr = np.asarray(original, dtype=float).reshape(-1)
    recon_arr = np.asarray(reconstructed, dtype=float).reshape(-1)
    region = original_arr[mask_start:mask_end]
    pred = recon_arr[mask_start:mask_end]
    return float(np.mean((region - pred) ** 2))


def normalized_mae_by_event_std(original: np.ndarray, reconstructed: np.ndarray, mask_start: int, mask_end: int) -> float:
    mae = masked_region_mae(original, reconstructed, mask_start, mask_end)
    event_std = float(np.std(np.asarray(original, dtype=float).reshape(-1)))
    scale = max(event_std, 1e-8)
    return float(mae / scale)


def shape_correlation(original: np.ndarray, reconstructed: np.ndarray, mask_start: int, mask_end: int) -> float:
    original_arr = np.asarray(original, dtype=float).reshape(-1)[mask_start:mask_end]
    recon_arr = np.asarray(reconstructed, dtype=float).reshape(-1)[mask_start:mask_end]
    if original_arr.size < 2:
        return 1.0 if original_arr.size == 1 and np.allclose(original_arr, recon_arr) else 0.0
    if float(np.std(original_arr)) <= 1e-12 or float(np.std(recon_arr)) <= 1e-12:
        return 1.0 if np.allclose(original_arr, recon_arr) else 0.0
    return float(np.corrcoef(original_arr, recon_arr)[0, 1])


def boundary_continuity_error(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask_start: int,
    mask_end: int,
) -> float:
    """Mean absolute mismatch in first differences at mask boundaries."""
    original_arr = np.asarray(original, dtype=float).reshape(-1)
    recon_arr = np.asarray(reconstructed, dtype=float).reshape(-1)
    errors: list[float] = []
    if mask_start > 0:
        orig_step = original_arr[mask_start] - original_arr[mask_start - 1]
        recon_step = recon_arr[mask_start] - recon_arr[mask_start - 1]
        errors.append(abs(recon_step - orig_step))
    if mask_end < len(original_arr):
        orig_step = original_arr[mask_end] - original_arr[mask_end - 1]
        recon_step = recon_arr[mask_end] - recon_arr[mask_end - 1]
        errors.append(abs(recon_step - orig_step))
    return float(np.mean(errors)) if errors else 0.0


def evaluate_masked_reconstruction(
    original: np.ndarray,
    reconstructed: np.ndarray,
    mask_start: int,
    mask_end: int,
) -> dict[str, float]:
    """Compute all reconstruction metrics for a masked event window."""
    metrics = {
        "masked_region_MAE": masked_region_mae(original, reconstructed, mask_start, mask_end),
        "masked_region_MSE": masked_region_mse(original, reconstructed, mask_start, mask_end),
        "normalized_MAE_by_event_std": normalized_mae_by_event_std(original, reconstructed, mask_start, mask_end),
        "shape_correlation": shape_correlation(original, reconstructed, mask_start, mask_end),
        "boundary_continuity_error": boundary_continuity_error(original, reconstructed, mask_start, mask_end),
    }
    return metrics


def mean_squared_error(x: list[float] | np.ndarray, y: list[float] | np.ndarray) -> float:
    """Backward-compatible MSE helper used by smoke tests."""
    x_arr = np.asarray(x, dtype=float).reshape(-1)
    y_arr = np.asarray(y, dtype=float).reshape(-1)
    max_len = max(len(x_arr), len(y_arr))
    if len(x_arr) < max_len:
        x_arr = np.pad(x_arr, (0, max_len - len(x_arr)), mode="edge")
    if len(y_arr) < max_len:
        y_arr = np.pad(y_arr, (0, max_len - len(y_arr)), mode="edge")
    return float(np.mean((x_arr - y_arr) ** 2))
