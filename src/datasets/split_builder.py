"""Series-level split building for leakage-safe time-series experiments."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any

from .event_extractor import labels_to_events


def series_parent_folder(record: Any) -> str:
    """Return a stable parent dataset/domain label for group-aware splitting."""
    if hasattr(record, "metadata") and record.metadata.get("parent_folder"):
        return str(record.metadata["parent_folder"])
    series_id = str(record.series_id if hasattr(record, "series_id") else record["series_id"])
    parts = series_id.replace("\\", "/").split("/")
    return parts[0] if len(parts) > 1 else series_id


def _record_has_event(record: Any) -> bool:
    labels = record.labels if hasattr(record, "labels") else record["labels"]
    return len(labels_to_events(labels)) > 0


def _record_dataset_name(record: Any) -> str:
    if hasattr(record, "metadata") and record.metadata.get("dataset_name"):
        return str(record.metadata["dataset_name"])
    if hasattr(record, "metadata") and record.metadata.get("dataset"):
        return str(record.metadata["dataset"])
    return series_parent_folder(record)


def _split_group_names(
    group_names: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[str], list[str], list[str]]:
    groups = list(group_names)
    rng.shuffle(groups)
    n = len(groups)
    if n == 0:
        return [], [], []
    if n == 1:
        return groups, [], []
    if n == 2:
        return [groups[0]], [], [groups[1]]

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if n >= 3 and val_ratio > 0 else int(round(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - max(n_val, 0) - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        n_test = n - n_train - n_val

    train_groups = groups[:n_train]
    val_groups = groups[n_train : n_train + n_val]
    test_groups = groups[n_train + n_val :]
    return train_groups, val_groups, test_groups


def _split_items_by_ratio(
    items: list[tuple[str, bool]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
    *,
    require_test: bool = False,
) -> dict[str, list[str]]:
    """Split series ids into train/val/test.

    When ``require_test`` is True and there are at least two items, the returned
    split always includes at least one test timeline.
    """
    rng.shuffle(items)
    n = len(items)
    if n == 0:
        return {"train": [], "val": [], "test": []}
    if n == 1:
        if require_test:
            raise ValueError(
                "Cannot assign a test timeline when the subgroup has only one series; "
                "use at least two timelines per dataset for dataset-stratified splits."
            )
        return {"train": [items[0][0]], "val": [], "test": []}
    if n == 2:
        # Two timelines cannot cover train, val, and test; prioritize a test holdout.
        return {
            "train": [items[0][0]],
            "val": [],
            "test": [items[1][0]],
        }

    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio))) if val_ratio > 0 else int(round(n * val_ratio))
    if n_train + n_val >= n:
        n_train = max(1, n - max(n_val, 0) - 1)
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        if n_train > 1:
            n_train -= 1
        elif n_val > 0:
            n_val -= 1
        n_test = n - n_train - n_val

    splits = {
        "train": [series_id for series_id, _ in items[:n_train]],
        "val": [series_id for series_id, _ in items[n_train : n_train + n_val]],
        "test": [series_id for series_id, _ in items[n_train + n_val :]],
    }
    if require_test and not splits["test"]:
        move_id = splits["train"].pop() if splits["train"] else splits["val"].pop()
        splits["test"].append(move_id)
    return splits


def _ensure_per_dataset_test_coverage(
    records: list[Any],
    splits: dict[str, list[str]],
    *,
    require_all_datasets_in_test: bool = True,
) -> dict[str, list[str]]:
    """Ensure each dataset with 2+ timelines contributes at least one timeline to test."""
    dataset_map: dict[str, list[str]] = defaultdict(list)
    for record in records:
        series_id = str(record.series_id if hasattr(record, "series_id") else record["series_id"])
        dataset_map[_record_dataset_name(record)].append(series_id)

    train = set(splits.get("train", []))
    val = set(splits.get("val", []))
    test = set(splits.get("test", []))
    missing: list[str] = []

    for dataset_name, series_ids in sorted(dataset_map.items()):
        if len(series_ids) <= 1:
            if require_all_datasets_in_test and series_ids:
                missing.append(f"{dataset_name} (timelines=1)")
            continue
        if any(series_id in test for series_id in series_ids):
            continue
        move_candidates = [series_id for series_id in series_ids if series_id in train]
        if not move_candidates:
            move_candidates = [series_id for series_id in series_ids if series_id in val]
        if not move_candidates:
            missing.append(f"{dataset_name} (timelines={len(series_ids)})")
            continue
        move_id = move_candidates[0]
        train.discard(move_id)
        val.discard(move_id)
        test.add(move_id)

    if require_all_datasets_in_test and missing:
        raise ValueError(
            "Dataset-stratified split could not place a test timeline for: "
            + ", ".join(missing)
            + ". Increase min_timelines_per_dataset or per-dataset caps."
        )

    return {
        "train": sorted(train),
        "val": sorted(val),
        "test": sorted(test),
    }


def build_series_split(
    records: list[Any],
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by_has_event: bool = True,
    group_by_parent_folder: bool = False,
    stratify_by_dataset: bool = False,
) -> dict[str, list[str]]:
    """Split whole time series into train/val/test by series id."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    rng = random.Random(seed)
    items = [
        (str(record.series_id if hasattr(record, "series_id") else record["series_id"]), _record_has_event(record))
        for record in records
    ]
    if len({series_id for series_id, _ in items}) != len(items):
        raise ValueError("Series ids must be unique for split building.")

    splits = {"train": [], "val": [], "test": []}

    if stratify_by_dataset:
        # Dataset-stratified evaluation takes precedence over parent-folder grouping.
        # Older configs can inherit ``group_by_parent_folder: true`` from low-label
        # baselines; silently honoring that flag here would disable per-dataset test
        # coverage even when ``stratify_by_dataset`` is requested explicitly.
        dataset_map: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for record in records:
            series_id = str(record.series_id if hasattr(record, "series_id") else record["series_id"])
            dataset_map[_record_dataset_name(record)].append((series_id, _record_has_event(record)))

        for dataset_items in dataset_map.values():
            # Split each dataset as one pool so small balanced subgroups still reach test.
            partial = _split_items_by_ratio(
                dataset_items,
                train_ratio,
                val_ratio,
                test_ratio,
                rng,
                require_test=len(dataset_items) >= 2,
            )
            for split_name in splits:
                splits[split_name].extend(partial[split_name])

        splits = _ensure_per_dataset_test_coverage(records, splits, require_all_datasets_in_test=True)

    elif group_by_parent_folder:
        group_map: dict[str, list[tuple[str, bool]]] = defaultdict(list)
        for record in records:
            series_id = str(record.series_id if hasattr(record, "series_id") else record["series_id"])
            group_map[series_parent_folder(record)].append((series_id, _record_has_event(record)))

        group_lists: list[list[str]]
        if stratify_by_has_event:
            event_groups = [name for name, members in group_map.items() if any(has_event for _, has_event in members)]
            normal_groups = [name for name, members in group_map.items() if not any(has_event for _, has_event in members)]
            group_lists = [event_groups, normal_groups]
        else:
            group_lists = [list(group_map.keys())]

        for group_names in group_lists:
            train_groups, val_groups, test_groups = _split_group_names(
                group_names,
                train_ratio,
                val_ratio,
                test_ratio,
                rng,
            )
            for group_name in train_groups:
                splits["train"].extend(series_id for series_id, _ in group_map[group_name])
            for group_name in val_groups:
                splits["val"].extend(series_id for series_id, _ in group_map[group_name])
            for group_name in test_groups:
                splits["test"].extend(series_id for series_id, _ in group_map[group_name])
    else:
        if stratify_by_has_event:
            positives = [item for item in items if item[1]]
            negatives = [item for item in items if not item[1]]
            item_groups = [positives, negatives]
        else:
            item_groups = [items]

        for group in item_groups:
            if not group:
                continue
            partial = _split_items_by_ratio(group, train_ratio, val_ratio, test_ratio, rng)
            for split_name in splits:
                splits[split_name].extend(partial[split_name])

    for split_name in splits:
        splits[split_name] = sorted(set(splits[split_name]))
    return splits


def build_series_splits(
    timelines: list[dict[str, Any]],
    *,
    train_ratio: float,
    dev_ratio: float,
    test_ratio: float,
    seed: int,
    group_by_parent_folder: bool = False,
    stratify_by_dataset: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    """Backward-compatible helper that returns records instead of series ids."""
    split_ids = build_series_split(
        timelines,
        train_ratio=train_ratio,
        val_ratio=dev_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_by_has_event=True,
        group_by_parent_folder=group_by_parent_folder,
        stratify_by_dataset=stratify_by_dataset,
    )
    by_id = {str(item["series_id"]): item for item in timelines}
    return {
        "train": [by_id[series_id] for series_id in split_ids["train"]],
        "dev": [by_id[series_id] for series_id in split_ids["val"]],
        "test": [by_id[series_id] for series_id in split_ids["test"]],
    }
