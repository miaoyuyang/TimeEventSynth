"""Event-family grouping for groupwise universal event-time transfer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.cluster import KMeans

from ..synthesis.donor_retrieval import EventWindow, compute_window_embedding


def event_window_id(window: EventWindow) -> str:
    return f"{window.series_id}:{int(window.start)}:{int(window.end)}"


@dataclass
class EventGroupingResult:
    donor_assignments: dict[str, str]
    target_assignments: dict[str, str]
    num_groups: int
    group_sizes: dict[str, int]
    method: str = "kmeans_embedding"


def compute_event_group_features(
    event_windows: list[EventWindow],
    *,
    grid_size: int = 32,
    context_size: int = 5,
) -> tuple[list[str], np.ndarray]:
    ids: list[str] = []
    features: list[np.ndarray] = []
    for window in event_windows:
        ids.append(event_window_id(window))
        features.append(compute_window_embedding(window, grid_size=grid_size, context_size=context_size))
    if not features:
        return ids, np.zeros((0, 0), dtype=float)
    return ids, np.asarray(features, dtype=float)


def assign_event_groups(
    donor_windows: list[EventWindow],
    target_windows: list[EventWindow],
    *,
    config: dict[str, Any] | None = None,
    group_key: str = "event_group_id",
) -> EventGroupingResult:
    """Cluster donor windows and assign targets to nearest event family.

    In groupwise augmentation policies this is called once per augmentation run
    so donor and target windows share stable event-family IDs within the run.
    """
    cfg = config or {}
    grid_size = int(cfg.get("grid_size", 32))
    context_size = int(cfg.get("context_size", 5))
    random_state = int(cfg.get("random_state", 42))

    donor_ids, donor_features = compute_event_group_features(
        donor_windows,
        grid_size=grid_size,
        context_size=context_size,
    )
    target_ids, target_features = compute_event_group_features(
        target_windows,
        grid_size=grid_size,
        context_size=context_size,
    )

    if donor_features.shape[0] == 0:
        return EventGroupingResult({}, {}, 0, {}, method="empty")

    requested_groups = int(cfg.get("num_groups", 4))
    num_groups = max(1, min(requested_groups, donor_features.shape[0]))
    if num_groups == 1:
        donor_labels = np.zeros(len(donor_ids), dtype=int)
        target_labels = np.zeros(len(target_ids), dtype=int)
    else:
        model = KMeans(n_clusters=num_groups, random_state=random_state, n_init=10)
        donor_labels = model.fit_predict(donor_features)
        target_labels = model.predict(target_features) if len(target_ids) else np.zeros(0, dtype=int)

    donor_assignments = {event_id: f"group_{int(label)}" for event_id, label in zip(donor_ids, donor_labels)}
    target_assignments = {event_id: f"group_{int(label)}" for event_id, label in zip(target_ids, target_labels)}

    for window in donor_windows:
        window.metadata[group_key] = donor_assignments[event_window_id(window)]
    for window in target_windows:
        if event_window_id(window) in target_assignments:
            window.metadata[group_key] = target_assignments[event_window_id(window)]

    group_sizes: dict[str, int] = {}
    for group_id in donor_assignments.values():
        group_sizes[group_id] = group_sizes.get(group_id, 0) + 1

    return EventGroupingResult(
        donor_assignments=donor_assignments,
        target_assignments=target_assignments,
        num_groups=len(group_sizes),
        group_sizes=group_sizes,
    )
