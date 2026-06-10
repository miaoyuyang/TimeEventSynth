"""Synthesize new event windows from normalized-time donor windows."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..alignment.dtw_alignment import warp_source_to_target
from ..alignment.learned_event_time import LearnedEventTimeAligner
from ..alignment.normalized_time import (
    apply_partial_target_synthesis,
    calibrate_amplitude_to_context,
    denormalize_event_window,
    fill_masked_from_donor,
)
from .donor_retrieval import EventWindow


def _stack_donors(donors: list[EventWindow], target_length: int) -> np.ndarray:
    resampled = [np.asarray(denormalize_event_window(donor.values, target_length), dtype=float) for donor in donors]
    arrays = [item.reshape(target_length, -1) if item.ndim == 1 else item for item in resampled]
    return np.stack(arrays, axis=0)


def _series_values(window: EventWindow) -> np.ndarray | None:
    raw = window.metadata.get("series_values")
    return None if raw is None else np.asarray(raw, dtype=float)


def partial_labels_for_event_window(window: EventWindow) -> np.ndarray | None:
    raw = window.metadata.get("series_labels")
    if raw is None:
        return None
    labels = np.asarray(raw, dtype=int).reshape(-1)
    return labels[window.start : window.end]


def synthetic_training_labels_for_window(
    target_length: int,
    partial_labels: np.ndarray | None,
    *,
    partial_target_mode: str = "full_replacement",
) -> np.ndarray:
    """Point labels for supervised training on a synthesized event window.

    Full replacement windows represent a wholly synthetic anomaly segment, so every
    point is positive. Partial-target modes keep observed anchors as positives and
    treat donor-filled or interpolated regions as normal context inside the window.
    """
    length = int(target_length)
    if length <= 0:
        return np.zeros(0, dtype=int)
    if partial_target_mode == "full_replacement" or partial_labels is None:
        return np.ones(length, dtype=int)
    labels = np.asarray(partial_labels, dtype=int).reshape(-1)
    if labels.size != length:
        return np.ones(length, dtype=int)
    if partial_target_mode in {"residual_baseline", "anchor_interpolate"}:
        return (labels > 0).astype(int)
    return np.ones(length, dtype=int)


def synthesize_by_mean_donor(target_window: EventWindow, donors: list[EventWindow], target_length: int) -> np.ndarray:
    """Average donor windows after normalized-time resampling."""
    if not donors:
        raise ValueError("At least one donor is required.")
    stacked = _stack_donors(donors, target_length)
    synthetic = np.mean(stacked, axis=0)
    return synthetic[:, 0] if synthetic.shape[1] == 1 else synthetic


def synthesize_by_random_donor(
    target_window: EventWindow,
    donors: list[EventWindow],
    target_length: int,
    noise_std: float = 0.0,
) -> np.ndarray:
    """Sample one donor and optionally add Gaussian noise."""
    if not donors:
        raise ValueError("At least one donor is required.")
    donor = donors[0]
    synthetic = np.asarray(denormalize_event_window(donor.values, target_length), dtype=float)
    if noise_std > 0.0:
        rng = np.random.default_rng(42)
        synthetic = synthetic + rng.normal(scale=noise_std, size=synthetic.shape)
    return synthetic


def synthesize_by_weighted_donor(
    target_window: EventWindow,
    donors: list[EventWindow],
    similarities: list[float],
    target_length: int,
) -> np.ndarray:
    """Weighted average of donor windows using retrieval similarities."""
    if not donors:
        raise ValueError("At least one donor is required.")
    if len(donors) != len(similarities):
        raise ValueError("donors and similarities must have the same length.")
    stacked = _stack_donors(donors, target_length)
    weights = np.asarray(similarities, dtype=float)
    weights = np.maximum(weights, 0.0)
    if float(weights.sum()) <= 0.0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()
    synthetic = np.tensordot(weights, stacked, axes=(0, 0))
    return synthetic[:, 0] if synthetic.shape[1] == 1 else synthetic


def synthesize_by_dtw_donor(target_window: EventWindow, donors: list[EventWindow], target_length: int) -> np.ndarray:
    """Warp donor windows to the target temporal pattern using DTW and average them."""
    if not donors:
        raise ValueError("At least one donor is required.")
    warped = []
    target_values = target_window.values
    for donor in donors:
        warped_values = np.asarray(warp_source_to_target(donor.values, target_values), dtype=float)
        warped_values = np.asarray(denormalize_event_window(warped_values, target_length), dtype=float)
        warped.append(warped_values.reshape(target_length, -1) if warped_values.ndim == 1 else warped_values)
    synthetic = np.mean(np.stack(warped, axis=0), axis=0)
    return synthetic[:, 0] if synthetic.shape[1] == 1 else synthetic


def synthesize_by_learned_prototype_event_time(
    target_window: EventWindow,
    donors: list[EventWindow],
    target_length: int,
    grid_size: int = 64,
) -> np.ndarray:
    """Average donor windows in a learned prototype event-time space.

    The universal event-time representation is conditional on the compatible donor
    set for the current target timeline (``donors`` must already be filtered upstream).
    """
    if not donors:
        raise ValueError("At least one donor is required.")
    aligner = LearnedEventTimeAligner(grid_size=grid_size)
    # Prototype is fit only on the per-target compatible donor windows, not a global pool.
    aligner.fit(donors)
    synthetic = aligner.synthesize(target_window, donors, k=len(donors), target_length=target_length)
    return np.asarray(synthetic, dtype=float)


def _base_synthesis(
    target_window: EventWindow,
    donors: list[EventWindow],
    *,
    base_method: str,
    target_length: int,
    similarities: list[float] | None = None,
    grid_size: int = 64,
) -> np.ndarray:
    if base_method in {"normalized_time_mean_donor", "normalized_time_context_calibrated"}:
        return synthesize_by_mean_donor(target_window, donors, target_length)
    if base_method in {"dtw_aligned_donor", "dtw_context_calibrated"}:
        return synthesize_by_dtw_donor(target_window, donors, target_length)
    if base_method in {"learned_prototype_event_time", "learned_prototype_context_calibrated"}:
        return synthesize_by_learned_prototype_event_time(target_window, donors, target_length, grid_size=grid_size)
    if similarities is not None:
        return synthesize_by_weighted_donor(target_window, donors, similarities, target_length)
    return synthesize_by_mean_donor(target_window, donors, target_length)


def synthesize_context_aware(
    target_window: EventWindow,
    donors: list[EventWindow],
    *,
    base_method: str,
    target_length: int,
    similarities: list[float] | None = None,
    amplitude_calibration: str = "none",
    partial_target_mode: str = "full_replacement",
    context_size: int = 5,
    match_partial_amplitude: bool = True,
    grid_size: int = 64,
) -> tuple[np.ndarray, dict[str, float]]:
    """Synthesize an event window with partial-target blending and context calibration."""
    donor_shape = _base_synthesis(
        target_window,
        donors,
        base_method=base_method,
        target_length=target_length,
        similarities=similarities,
        grid_size=grid_size,
    )
    partial_labels = partial_labels_for_event_window(target_window)
    blended = apply_partial_target_synthesis(
        donor_shape,
        target_window.values,
        partial_labels,
        mode=partial_target_mode,
    )
    reference_donor = np.mean(_stack_donors(donors, target_length), axis=0)
    if reference_donor.ndim > 1:
        reference_donor = reference_donor[:, 0]
    calibrated, audit = calibrate_amplitude_to_context(
        blended,
        donor_values=reference_donor,
        target_event_values=np.asarray(target_window.values, dtype=float),
        series_values=_series_values(target_window),
        start=int(target_window.start),
        end=int(target_window.end),
        mode=amplitude_calibration,
        context_size=context_size,
        partial_labels=partial_labels,
        match_partial_amplitude=match_partial_amplitude,
    )
    audit["partial_target_mode"] = partial_target_mode
    audit["base_method"] = base_method
    point_labels = synthetic_training_labels_for_window(
        target_length,
        partial_labels,
        partial_target_mode=partial_target_mode,
    )
    audit["synthetic_point_labels"] = point_labels.tolist()
    return calibrated, audit


def reconstruct_masked_event_window(
    target_window: EventWindow,
    donors: list[EventWindow],
    *,
    method: str,
    partial_labels: np.ndarray,
    target_length: int | None = None,
    similarities: list[float] | None = None,
    grid_size: int = 64,
    seed: int = 42,
) -> np.ndarray:
    """Reconstruct a partially masked event window using a completion method."""
    length = int(target_length or len(target_window.values))
    labels = np.asarray(partial_labels, dtype=int).reshape(-1)
    if labels.size != length:
        raise ValueError("partial_labels length must match target event length")

    if method == "linear_interpolation":
        return apply_partial_target_synthesis(
            target_window.values,
            target_window.values,
            labels,
            mode="anchor_interpolate",
        )

    if not donors:
        raise ValueError(f"At least one donor is required for method={method}")

    if method == "random_donor":
        rng = np.random.default_rng(seed)
        picked = donors[int(rng.integers(0, len(donors)))]
        donor_shape = synthesize_by_random_donor(target_window, [picked], length)
    else:
        donor_shape = _base_synthesis(
            target_window,
            donors,
            base_method=method,
            target_length=length,
            similarities=similarities,
            grid_size=grid_size,
        )

    return fill_masked_from_donor(donor_shape, target_window.values, labels)


def synthesize_event_window(
    query_event: dict[str, Any],
    donor_matches: list[dict[str, Any]],
    *,
    strategy: str = "copy_top1",
) -> dict[str, Any] | None:
    """Backward-compatible wrapper for the smoke path."""
    if not donor_matches:
        return None
    target_length = len(query_event["values"])
    donors = [
        EventWindow(
            series_id=str(item["donor"]["series_id"]),
            start=0,
            end=len(item["donor"]["values"]),
            values=np.asarray(item["donor"]["values"], dtype=float),
            label=str(item["donor"].get("label", "anomaly")),
            metadata={"event_id": item["donor"].get("event_id")},
        )
        for item in donor_matches
    ]
    target = EventWindow(
        series_id=str(query_event["series_id"]),
        start=int(query_event.get("start_idx", 0)),
        end=int(query_event.get("end_idx", target_length)),
        values=np.asarray(query_event["values"], dtype=float),
        label=str(query_event.get("event_label", "anomaly")),
        metadata={"event_id": query_event.get("event_id")},
    )
    if strategy == "blend_topk":
        values = synthesize_by_weighted_donor(target, donors, [float(item["score"]) for item in donor_matches], target_length)
    elif strategy == "dtw_donor":
        values = synthesize_by_dtw_donor(target, donors, target_length)
    elif strategy == "learned_prototype_event_time":
        values = synthesize_by_learned_prototype_event_time(target, donors, target_length)
    elif strategy == "random_donor":
        values = synthesize_by_random_donor(target, donors, target_length)
    else:
        values = synthesize_by_mean_donor(target, donors[:1], target_length)
    return {
        "synthetic_event_id": f"synthetic_for_{query_event['event_id']}",
        "series_id": query_event["series_id"],
        "values": np.asarray(values, dtype=float).tolist(),
        "labels": list(query_event["labels"]),
        "local_time": list(query_event["local_time"]),
        "source_donor_event_ids": [donor.metadata.get("event_id") for donor in donors],
        "support_score": float(np.mean([item["score"] for item in donor_matches])),
    }
