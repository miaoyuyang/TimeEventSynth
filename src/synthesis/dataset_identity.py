"""Canonical dataset/domain identity for timelines and donor-pair policies."""

from __future__ import annotations

import logging
from typing import Any

from ..datasets.split_builder import series_parent_folder

logger = logging.getLogger(__name__)


def resolve_dataset_name(record: Any, *, fallback: str | None = None) -> str:
    """Resolve dataset_name for a series record without collapsing heterogeneous loads."""
    metadata = getattr(record, "metadata", None) if hasattr(record, "metadata") else record.get("metadata")
    metadata = metadata or {}
    if metadata.get("parent_folder"):
        return str(metadata["parent_folder"])
    if metadata.get("dataset_name"):
        return str(metadata["dataset_name"])
    if metadata.get("dataset"):
        return str(metadata["dataset"])
    parent = series_parent_folder(record)
    series_id = str(record.series_id if hasattr(record, "series_id") else record["series_id"])
    if parent and "/" in series_id:
        return parent
    if fallback:
        return str(fallback)
    return parent or "unknown"


def feature_dataset_name(features: dict[str, Any]) -> str | None:
    if features.get("dataset_name") is not None:
        return str(features["dataset_name"])
    if features.get("dataset") is not None:
        return str(features["dataset"])
    return None


def pair_same_dataset(
    target_series_id: str,
    source_series_id: str,
    feature_table: dict[str, dict[str, Any]],
) -> bool:
    target_dataset = feature_dataset_name(feature_table.get(target_series_id, {}))
    source_dataset = feature_dataset_name(feature_table.get(source_series_id, {}))
    if not target_dataset or not source_dataset:
        return False
    return target_dataset == source_dataset


def count_dataset_pairs(
    target_series_id: str,
    source_series_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
) -> dict[str, int]:
    """Count same- vs cross-dataset donor-series candidates for one target."""
    same = 0
    cross = 0
    for source_id in source_series_ids:
        if pair_same_dataset(target_series_id, source_id, feature_table):
            same += 1
        else:
            cross += 1
    return {
        "same_dataset_pairs_considered": same,
        "cross_dataset_pairs_considered": cross,
        "total_donor_pairs_considered": same + cross,
    }


def log_dataset_identity(
    feature_table: dict[str, dict[str, Any]],
    *,
    sample_size: int = 5,
) -> dict[str, Any]:
    """Log and return dataset_name distribution for loaded timelines."""
    dataset_counts: dict[str, int] = {}
    pairs: list[tuple[str, str]] = []
    for series_id, features in feature_table.items():
        dataset_name = feature_dataset_name(features) or "unknown"
        dataset_counts[dataset_name] = dataset_counts.get(dataset_name, 0) + 1
        if len(pairs) < sample_size:
            pairs.append((series_id, dataset_name))

    logger.info(
        "dataset identity: %d unique dataset_name(s), distribution=%s",
        len(dataset_counts),
        dataset_counts,
    )
    for series_id, dataset_name in pairs:
        logger.info("  series_id=%s dataset_name=%s", series_id, dataset_name)

    return {
        "num_unique_datasets": len(dataset_counts),
        "dataset_counts": dataset_counts,
        "sample_series": [{"series_id": sid, "dataset_name": ds} for sid, ds in pairs],
    }
