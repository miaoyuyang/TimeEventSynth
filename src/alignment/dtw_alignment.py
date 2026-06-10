"""DTW alignment utilities for event-window synthesis baselines."""

from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy.spatial.distance import cdist


def _as_2d(values: Sequence[float] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D sequence, got shape {array.shape}")


def dtw_path(a: Sequence[float] | np.ndarray, b: Sequence[float] | np.ndarray) -> list[tuple[int, int]]:
    """Return a monotone DTW alignment path between two sequences."""
    a_arr = _as_2d(a)
    b_arr = _as_2d(b)
    cost = cdist(a_arr, b_arr, metric="euclidean")
    n_a, n_b = len(a_arr), len(b_arr)
    dp = np.full((n_a + 1, n_b + 1), np.inf, dtype=float)
    back = np.full((n_a + 1, n_b + 1, 2), -1, dtype=int)
    dp[0, 0] = 0.0
    for i in range(1, n_a + 1):
        for j in range(1, n_b + 1):
            options = [
                (dp[i - 1, j], (i - 1, j)),
                (dp[i, j - 1], (i, j - 1)),
                (dp[i - 1, j - 1], (i - 1, j - 1)),
            ]
            best_cost, best_prev = min(options, key=lambda item: item[0])
            dp[i, j] = cost[i - 1, j - 1] + best_cost
            back[i, j] = best_prev

    path: list[tuple[int, int]] = []
    i, j = n_a, n_b
    while i > 0 and j > 0:
        path.append((i - 1, j - 1))
        i, j = tuple(back[i, j])
    path.reverse()
    return path


def dtw_distance(x: Sequence[float] | np.ndarray, y: Sequence[float] | np.ndarray) -> float:
    """Compute DTW distance using the same dynamic program as the path extractor."""
    path = dtw_path(x, y)
    if not path:
        return 0.0
    x_arr = _as_2d(x)
    y_arr = _as_2d(y)
    return float(sum(np.linalg.norm(x_arr[i] - y_arr[j]) for i, j in path))


def warp_source_to_target(source_values: Sequence[float] | np.ndarray, target_length_or_values: int | Sequence[float] | np.ndarray) -> np.ndarray:
    """Warp a source sequence to a target length or a target temporal pattern."""
    source = _as_2d(source_values)
    if isinstance(target_length_or_values, int):
        target_length = int(target_length_or_values)
        if target_length < 1:
            raise ValueError("target length must be at least 1")
        if len(source) == target_length:
            return source[:, 0] if source.shape[1] == 1 else source
        target_reference = np.linspace(0.0, 1.0, target_length, dtype=float).reshape(-1, 1)
        source_reference = np.linspace(0.0, 1.0, len(source), dtype=float).reshape(-1, 1)
        path = dtw_path(source_reference, target_reference)
        target_dim = target_length
    else:
        target = _as_2d(target_length_or_values)
        path = dtw_path(source, target)
        target_dim = len(target)

    aligned: list[list[np.ndarray]] = [[] for _ in range(target_dim)]
    for source_idx, target_idx in path:
        aligned[target_idx].append(source[source_idx])
    warped_rows = []
    last_value = source[0]
    for bucket in aligned:
        if bucket:
            last_value = np.mean(np.stack(bucket, axis=0), axis=0)
        warped_rows.append(last_value)
    warped = np.stack(warped_rows, axis=0)
    return warped[:, 0] if warped.shape[1] == 1 else warped
