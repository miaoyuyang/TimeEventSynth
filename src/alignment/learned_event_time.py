"""Prototype-based learned universal event-time alignment.

Each event window is aligned to a learned prototype shape on a shared event-time
axis. This module implements a **warped prototype baseline**: per-window forward
maps phi_i (local -> prototype) and inverse maps psi_i (prototype -> local) are
stored as DTW paths at fit time. Synthesis averages donors in prototype space and
pulls the result back through psi_target.

This is not yet a jointly trained bidirectional neural aligner. psi for windows
outside the donor fit set, or at a different ``target_length`` than fit time, is
re-estimated on the fly via DTW rather than loaded from a learned parametric map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .dtw_alignment import dtw_path
from .normalized_time import normalize_event_window, resample_window


@dataclass
class PrototypeWarp:
    """Bidirectional warp metadata between a local window and prototype event time."""

    series_id: str
    start: int
    end: int
    local_length: int
    path_to_prototype: list[tuple[int, int]]
    path_from_prototype: list[tuple[int, int]]


@dataclass
class EventTimeMappingResult:
    """Generic reusable alignment result for event-time methods.

    Ported abstraction from the crisis project:
    crisis event -> time series
    universal timeline -> universal event time
    bin positions -> event-window positions
    """

    window_to_control_points: dict[str, list[float]]
    mapped_positions: dict[str, list[float]]
    mode: str
    diagnostics: dict[str, Any] | None = None


def build_control_points(num_control_points: int) -> list[float]:
    """Return evenly spaced control points in event time."""
    if num_control_points < 2:
        raise ValueError("num_control_points must be at least 2")
    return np.linspace(0.0, 1.0, num_control_points, dtype=float).tolist()


def piecewise_linear_map(control_x: np.ndarray, control_y: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Map local normalized times to event time by piecewise-linear interpolation."""
    return np.interp(values, control_x, control_y)


class MonotoneEventTimeMap:
    """Simple numpy monotone map scaffold for future learned event-time models.

    This ports the crisis project's monotone control-point idea without bringing
    over task-specific assumptions. The current implementation is deterministic
    and is mainly here so future learners can swap in trained control points.

    TODO: replace identity-like control_y with learned monotone increments from
    a neural or optimization-based event-time learner.
    """

    def __init__(self, control_points: list[float], step_logits: np.ndarray | None = None) -> None:
        self.control_x = np.asarray(control_points, dtype=float)
        if step_logits is None:
            base_steps = np.diff(self.control_x)
            base_probs = np.clip(base_steps, 1e-8, None)
            base_probs = base_probs / base_probs.sum()
            step_logits = np.log(base_probs)
        self.step_logits = np.asarray(step_logits, dtype=float)
        if self.step_logits.ndim != 1 or len(self.step_logits) != len(self.control_x) - 1:
            raise ValueError("step_logits must have length len(control_points) - 1")

    def control_y(self) -> np.ndarray:
        shifted = self.step_logits - np.max(self.step_logits)
        probs = np.exp(shifted)
        probs = probs / np.maximum(probs.sum(), 1e-8)
        cumulative = np.cumsum(probs)
        return np.concatenate([np.asarray([0.0], dtype=float), cumulative], axis=0)

    def map_values(self, values: np.ndarray | list[float]) -> np.ndarray:
        values_array = np.asarray(values, dtype=float)
        return piecewise_linear_map(self.control_x, self.control_y(), values_array)


class LearnedEventTimeAligner:
    """Warped prototype event-time aligner (research scaffold, not a full learned bi-map).

    Implemented today:
    - prototype_: mean of normalized donor shapes on a fixed grid
    - phi_i (``path_to_prototype``): DTW map local window -> prototype, stored per fitted window
    - psi_i (``path_from_prototype``): DTW map prototype -> local window, stored per fitted window

    ``transform_window`` applies phi_i; ``inverse_transform`` applies psi_i when the target
    window was fitted (or registered) at the requested ``target_length``. Otherwise psi is
    re-estimated by DTW(prototype, resampled target) — an approximation, not a learned inverse.

    Future work (not implemented):
    - neural / optimization-based jointly learned phi and psi
    - parametric monotone control-point maps (see ``MonotoneEventTimeMap``)
    - uncertainty-aware warps and pattern-specific prototype families
    """

    def __init__(self, grid_size: int = 64) -> None:
        self.grid_size = int(grid_size)
        self.prototype_: np.ndarray | None = None
        self.window_warps_: dict[tuple[str, int, int], PrototypeWarp] = {}

    def fit(self, event_windows: list[Any]) -> "LearnedEventTimeAligner":
        """Fit a prototype event-time representation from donor windows.

        The universal event-time representation is conditional on the compatible donor
        set for the current target timeline. Callers must pass only compatible donors
        when compatibility filtering is enabled; do not pool all timelines globally.
        """
        if not event_windows:
            raise ValueError("At least one event window is required to fit the aligner.")
        normalized = [self._to_2d(normalize_event_window(window.values, grid_size=self.grid_size)) for window in event_windows]
        self.prototype_ = np.mean(np.stack(normalized, axis=0), axis=0)
        self.window_warps_.clear()
        for window in event_windows:
            self._store_window_maps(window)
        return self

    def register_window_maps(self, window: Any) -> None:
        """Learn and cache phi_i / psi_i for a window not seen during ``fit`` (e.g. target)."""
        self._require_fit()
        self._store_window_maps(window)

    def transform_window(self, window: Any, grid_size: int = 64) -> np.ndarray:
        """Map a local event window into the learned prototype event-time space."""
        self._require_fit()
        if int(grid_size) != self.grid_size:
            raise ValueError(
                f"transform_window grid_size={grid_size} does not match fit grid_size={self.grid_size}."
            )
        window_values = self._to_2d(window.values)
        path = self._path_for_window(window)
        return self._warp_onto_target_axis(window_values, path, target_dim=len(self.prototype_), from_first_index=True)

    def inverse_transform(self, aligned_values: np.ndarray, target_window: Any, target_length: int) -> np.ndarray:
        """Map prototype-space values back to target local time via psi_i."""
        self._require_fit()
        target_length = int(target_length)
        if target_length < 1:
            raise ValueError("target_length must be at least 1")
        proto_values = self._to_2d(aligned_values)
        if len(proto_values) != len(self.prototype_):
            proto_values = self._to_2d(resample_window(proto_values, len(self.prototype_)))
        path, _ = self._inverse_path_for_window(target_window, target_length)
        return self._warp_onto_target_axis(
            proto_values,
            path,
            target_dim=target_length,
            from_first_index=True,
        )

    def synthesize(
        self,
        target_window: Any,
        donor_windows: list[Any],
        k: int = 5,
        target_length: int | None = None,
    ) -> np.ndarray:
        """Synthesize a target window by averaging donor windows in prototype time."""
        self._require_fit()
        if not donor_windows:
            raise ValueError("At least one donor window is required.")
        target_length = int(target_length or len(target_window.values))
        ranked_donors = self._rank_donors(target_window, donor_windows, k=k)
        aligned = [self.transform_window(donor, grid_size=self.grid_size) for donor in ranked_donors]
        prototype_average = np.mean(np.stack(aligned, axis=0), axis=0)
        self.register_window_maps(target_window)
        restored = self.inverse_transform(prototype_average, target_window, target_length)
        return restored[:, 0] if restored.ndim == 2 and restored.shape[1] == 1 else restored

    def _rank_donors(self, target_window: Any, donor_windows: list[Any], k: int) -> list[Any]:
        target_embedding = self.transform_window(target_window, grid_size=self.grid_size).reshape(-1)
        scored: list[tuple[Any, float]] = []
        for donor in donor_windows:
            donor_embedding = self.transform_window(donor, grid_size=self.grid_size).reshape(-1)
            similarity = self._cosine(target_embedding, donor_embedding)
            scored.append((donor, similarity))
        scored.sort(key=lambda item: (-item[1], str(item[0].series_id), int(item[0].start), int(item[0].end)))
        return [donor for donor, _ in scored[: max(int(k), 1)]]

    def _require_fit(self) -> None:
        if self.prototype_ is None:
            raise ValueError("LearnedEventTimeAligner.fit(...) must be called before use.")

    def _window_key(self, window: Any) -> tuple[str, int, int]:
        return (str(window.series_id), int(window.start), int(window.end))

    def _path_for_window(self, window: Any) -> list[tuple[int, int]]:
        key = self._window_key(window)
        stored = self.window_warps_.get(key)
        if stored is not None:
            return stored.path_to_prototype
        window_values = self._to_2d(window.values)
        return dtw_path(window_values, self.prototype_)

    def _inverse_path_for_window(
        self,
        window: Any,
        target_length: int,
    ) -> tuple[list[tuple[int, int]], str]:
        """Return psi_i for ``window`` at ``target_length`` and how it was obtained."""
        key = self._window_key(window)
        stored = self.window_warps_.get(key)
        if stored is not None and stored.local_length == int(target_length):
            return stored.path_from_prototype, "stored"

        target_ref = self._to_2d(resample_window(window.values, int(target_length)))
        return dtw_path(self.prototype_, target_ref), "estimated"

    def _store_window_maps(self, window: Any) -> None:
        window_values = self._to_2d(window.values)
        local_length = len(window_values)
        path_to_prototype = dtw_path(window_values, self.prototype_)
        path_from_prototype = dtw_path(self.prototype_, window_values)
        key = self._window_key(window)
        self.window_warps_[key] = PrototypeWarp(
            series_id=str(window.series_id),
            start=int(window.start),
            end=int(window.end),
            local_length=local_length,
            path_to_prototype=path_to_prototype,
            path_from_prototype=path_from_prototype,
        )

    @staticmethod
    def _warp_onto_target_axis(
        source_values: np.ndarray,
        path: list[tuple[int, int]],
        *,
        target_dim: int,
        from_first_index: bool,
    ) -> np.ndarray:
        """Aggregate ``source_values`` onto ``target_dim`` bins using a DTW path."""
        source = LearnedEventTimeAligner._to_2d(source_values)
        buckets: list[list[np.ndarray]] = [[] for _ in range(target_dim)]
        for first_idx, second_idx in path:
            if from_first_index:
                source_idx, target_idx = first_idx, second_idx
            else:
                target_idx, source_idx = first_idx, second_idx
            if 0 <= source_idx < len(source) and 0 <= target_idx < target_dim:
                buckets[target_idx].append(source[source_idx])
        rows: list[np.ndarray] = []
        last = source[0]
        for bucket in buckets:
            if bucket:
                last = np.mean(np.stack(bucket, axis=0), axis=0)
            rows.append(last)
        return np.stack(rows, axis=0)

    @staticmethod
    def _to_2d(values: Any) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.ndim == 1:
            return array.reshape(-1, 1)
        if array.ndim == 2:
            return array
        raise ValueError(f"Expected 1D or 2D event window, got shape {array.shape}")

    @staticmethod
    def _cosine(x: np.ndarray, y: np.ndarray) -> float:
        denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-8)
        return float(np.dot(x, y) / denom)


def assign_learned_event_time(events: list[dict[str, Any]], num_buckets: int) -> list[dict[str, Any]]:
    """Backward-compatible identity-style assignment for event metadata rows.

    This helper keeps earlier scaffold scripts working. The future learned
    event-time implementation for metadata rows can replace this with a real
    event-level alignment pass.
    """
    aligned: list[dict[str, Any]] = []
    for event in events:
        row = dict(event)
        local_time = float(row.get("local_time_center", 0.0))
        local_time = min(max(local_time, 0.0), 1.0)
        row["universal_position"] = local_time
        row["universal_bucket"] = min(int(local_time * num_buckets), max(num_buckets - 1, 0))
        aligned.append(row)
    return aligned
