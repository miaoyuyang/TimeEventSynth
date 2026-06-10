"""Timeline compatibility filtering before donor retrieval and synthesis."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from ..datasets.split_builder import series_parent_folder
from .dataset_identity import count_dataset_pairs, feature_dataset_name, pair_same_dataset, resolve_dataset_name
from .compatibility import (
    DEFAULT_CONFIG as DEFAULT_COMPATIBILITY_CONFIG,
    attach_event_group_distributions,
    filter_compatible_donors,
    compute_timeline_features,
    learn_compatibility_weights,
)
from .donor_retrieval import EventWindow, compute_window_embedding, retrieve_topk_donors

logger = logging.getLogger(__name__)

DONOR_POLICY_PRESETS: dict[str, dict[str, Any]] = {
    "all_donors_no_filter": {
        "enabled": False,
        "allow_cross_dataset": True,
        "restrict_same_dataset": False,
        "require_cross_dataset": False,
        "require_same_dataset": False,
    },
    "same_dataset_only": {
        "enabled": False,
        "allow_cross_dataset": False,
        "restrict_same_dataset": True,
        "require_cross_dataset": False,
        "require_same_dataset": True,
    },
    "cross_dataset_all": {
        "enabled": False,
        "allow_cross_dataset": True,
        "restrict_same_dataset": False,
        "require_cross_dataset": True,
        "require_same_dataset": False,
    },
    "cross_dataset_compatible": {
        "enabled": True,
        "allow_cross_dataset": True,
        "restrict_same_dataset": False,
        "require_cross_dataset": True,
        "require_same_dataset": False,
    },
    "compatibility_top50": {
        "enabled": True,
        "allow_cross_dataset": True,
        "restrict_same_dataset": False,
        "require_cross_dataset": False,
        "require_same_dataset": False,
        "top_quantile": 0.5,
    },
    "compatibility_strict": {
        "enabled": True,
        "allow_cross_dataset": True,
        "restrict_same_dataset": False,
        "require_cross_dataset": False,
        "require_same_dataset": False,
        "min_score": 0.7,
    },
}


def is_cross_dataset_policy(donor_policy: str | None) -> bool:
    policy = str(donor_policy or "")
    if policy.startswith("cross_dataset"):
        return True
    preset = DONOR_POLICY_PRESETS.get(policy, {})
    return bool(preset.get("require_cross_dataset", False))


def is_same_dataset_only_policy(donor_policy: str | None) -> bool:
    policy = str(donor_policy or "")
    if policy == "same_dataset_only":
        return True
    preset = DONOR_POLICY_PRESETS.get(policy, {})
    return bool(preset.get("require_same_dataset", False)) or bool(preset.get("restrict_same_dataset", False))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key == "weights" and isinstance(value, dict):
            weights = dict(merged.get("weights", {}))
            weights.update(value)
            merged["weights"] = weights
        else:
            merged[key] = value
    return merged


def resolve_compatibility_config(
    synthesis_cfg: dict[str, Any] | None,
    policy_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge synthesis.compatibility, donor-policy preset, and per-run overrides."""
    synthesis_cfg = synthesis_cfg or {}
    policy_config = policy_config or {}
    merged = dict(DEFAULT_COMPATIBILITY_CONFIG)
    merged.update(synthesis_cfg.get("compatibility", {}) or {})

    donor_policy = policy_config.get("donor_policy") or synthesis_cfg.get("donor_policy")
    if donor_policy:
        preset = DONOR_POLICY_PRESETS.get(str(donor_policy), {})
        merged = _deep_merge(merged, preset)

    if policy_config.get("compatibility"):
        merged = _deep_merge(merged, policy_config["compatibility"])

    if policy_config.get("top_k") is not None and merged.get("top_k") is None:
        merged["top_k"] = policy_config["top_k"]
    if donor_policy:
        merged["donor_policy"] = str(donor_policy)
    return merged


def compatibility_is_enabled(compatibility_cfg: dict[str, Any]) -> bool:
    return bool(compatibility_cfg.get("enabled", False))


def _record_series_id(record: Any) -> str:
    return str(record.series_id if hasattr(record, "series_id") else record["series_id"])


def _record_values_labels_timestamps(record: Any) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    if hasattr(record, "values"):
        values = np.asarray(record.values, dtype=float)
        labels = np.asarray(record.labels, dtype=int)
        timestamps = None if record.timestamps is None else np.asarray(record.timestamps, dtype=float)
    else:
        values = np.asarray(record["values"], dtype=float)
        labels = np.asarray(record["labels"], dtype=int)
        timestamps = record.get("timestamps")
        timestamps = None if timestamps is None else np.asarray(timestamps, dtype=float)
    return values, labels, timestamps


def _record_dataset_name(record: Any, *, fallback: str | None = None) -> str:
    return resolve_dataset_name(record, fallback=fallback)


def build_timeline_feature_table(
    records: list[Any],
    *,
    synthesis_cfg: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute timeline-level features for all unique series in the record list."""
    synthesis_cfg = synthesis_cfg or {}
    last_resort_fallback = synthesis_cfg.get("dataset_name") or synthesis_cfg.get("default_dataset")
    if last_resort_fallback is None:
        dataset_cfg = synthesis_cfg.get("dataset")
        if isinstance(dataset_cfg, dict):
            last_resort_fallback = dataset_cfg.get("name")
        elif isinstance(dataset_cfg, str):
            last_resort_fallback = dataset_cfg

    feature_table: dict[str, dict[str, Any]] = {}
    for record in records:
        series_id = _record_series_id(record)
        if series_id in feature_table:
            continue
        values, labels, timestamps = _record_values_labels_timestamps(record)
        dataset_name = resolve_dataset_name(record, fallback=None) or series_parent_folder(record)
        if dataset_name in {"", "unknown"} and last_resort_fallback:
            dataset_name = str(last_resort_fallback)
        feature_config = {
            "series_id": series_id,
            "dataset": dataset_name,
            "dataset_name": dataset_name,
        }
        feature_table[series_id] = compute_timeline_features(
            values,
            labels=labels,
            timestamps=timestamps,
            config=feature_config,
        )
    return feature_table


def _unique_donor_series_ids(donor_windows: list[EventWindow], *, target_series_id: str) -> list[str]:
    series_ids = sorted({str(window.series_id) for window in donor_windows if window.series_id != target_series_id})
    return series_ids


def _restrict_same_dataset(
    target_series_id: str,
    candidate_series_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
) -> list[str]:
    target_dataset = feature_dataset_name(feature_table.get(target_series_id, {}))
    if target_dataset is None:
        return candidate_series_ids
    return [
        series_id
        for series_id in candidate_series_ids
        if feature_dataset_name(feature_table.get(series_id, {})) == target_dataset
    ]


def _restrict_cross_dataset(
    target_series_id: str,
    candidate_series_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
) -> list[str]:
    target_dataset = feature_dataset_name(feature_table.get(target_series_id, {}))
    if target_dataset is None:
        return candidate_series_ids
    return [
        series_id
        for series_id in candidate_series_ids
        if feature_dataset_name(feature_table.get(series_id, {})) != target_dataset
    ]


def _apply_donor_policy_candidate_filter(
    target_series_id: str,
    candidate_series_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
    donor_policy: str | None,
) -> list[str]:
    policy = str(donor_policy or "")
    if is_same_dataset_only_policy(policy):
        return _restrict_same_dataset(target_series_id, candidate_series_ids, feature_table)
    if is_cross_dataset_policy(policy):
        return _restrict_cross_dataset(target_series_id, candidate_series_ids, feature_table)
    return candidate_series_ids


def _filter_windows_by_series(donor_windows: list[EventWindow], kept_series_ids: set[str]) -> list[EventWindow]:
    return [window for window in donor_windows if str(window.series_id) in kept_series_ids]


def _window_group(window: EventWindow, group_key: str) -> str:
    return str(window.metadata.get(group_key, "") or "")


def _select_nearest_compatible_group(
    target_window: EventWindow,
    donor_windows: list[EventWindow],
    *,
    group_key: str,
    context_size: int,
) -> str | None:
    """Pick the nearest donor group among currently compatible donor windows."""
    grouped: dict[str, list[EventWindow]] = {}
    for donor in donor_windows:
        group = _window_group(donor, group_key)
        if not group:
            continue
        grouped.setdefault(group, []).append(donor)
    if not grouped:
        return None

    target_embedding = compute_window_embedding(target_window, context_size=context_size)
    target_vec = np.asarray(target_embedding, dtype=float).reshape(-1)
    target_norm = float(np.linalg.norm(target_vec))
    if target_norm <= 1e-12:
        target_norm = 1e-12

    best_group: str | None = None
    best_score = -np.inf
    for group, members in grouped.items():
        embeddings = [compute_window_embedding(item, context_size=context_size) for item in members]
        centroid = np.mean(np.asarray(embeddings, dtype=float), axis=0).reshape(-1)
        denom = max(target_norm * float(np.linalg.norm(centroid)), 1e-12)
        score = float(np.dot(target_vec, centroid) / denom)
        if score > best_score:
            best_score = score
            best_group = group
    return best_group


def select_donors_for_target(
    target_window: EventWindow,
    donor_windows: list[EventWindow],
    *,
    feature_table: dict[str, dict[str, Any]],
    compatibility_cfg: dict[str, Any],
    top_k: int,
    retrieval_kwargs: dict[str, Any] | None = None,
) -> tuple[list[EventWindow], list[float], dict[str, Any]]:
    """Retrieve donors for one target window with optional compatibility filtering."""
    retrieval_kwargs = retrieval_kwargs or {}
    target_series_id = str(target_window.series_id)
    donor_policy = compatibility_cfg.get("donor_policy")
    candidate_series_ids = _unique_donor_series_ids(donor_windows, target_series_id=target_series_id)
    candidate_series_ids = _apply_donor_policy_candidate_filter(
        target_series_id,
        candidate_series_ids,
        feature_table,
        donor_policy,
    )
    pair_counts = count_dataset_pairs(target_series_id, candidate_series_ids, feature_table)
    audit: dict[str, Any] = {
        "num_candidate_donor_series": len(candidate_series_ids),
        "num_compatible_donor_series": len(candidate_series_ids),
        "compatibility_enabled": compatibility_is_enabled(compatibility_cfg),
        "donor_policy": donor_policy,
        "compatibility_records": [],
        "evaluated_candidate_series_ids": list(candidate_series_ids),
        **pair_counts,
    }

    if compatibility_cfg.get("restrict_same_dataset") and not is_same_dataset_only_policy(donor_policy):
        candidate_series_ids = _restrict_same_dataset(target_series_id, candidate_series_ids, feature_table)
        audit["num_candidate_donor_series"] = len(candidate_series_ids)

    kept_series_ids = set(candidate_series_ids)
    if compatibility_is_enabled(compatibility_cfg):
        kept_series_ids_list, compatibility_records = filter_compatible_donors(
            target_series_id,
            candidate_series_ids,
            feature_table,
            compatibility_cfg,
        )
        audit["compatibility_records"] = compatibility_records
        kept_series_ids = set(kept_series_ids_list)
        audit["num_compatible_donor_series"] = len(kept_series_ids)

        if not kept_series_ids:
            fallback = str(compatibility_cfg.get("fallback_when_no_compatible_donor", "skip"))
            if fallback == "all_donors" and bool(compatibility_cfg.get("allow_incompatible_donors", False)):
                kept_series_ids = set(candidate_series_ids)
                audit["compatibility_fallback"] = "all_donors"
            else:
                audit["compatibility_fallback"] = "skip"
                logger.debug(
                    "compatibility: no compatible donors for target=%s candidates=%d -> skip",
                    target_series_id,
                    len(candidate_series_ids),
                )
                return [], [], audit

    filtered_windows = _filter_windows_by_series(donor_windows, kept_series_ids)
    group_key = retrieval_kwargs.get("group_key")
    if retrieval_kwargs.get("restrict_to_target_group") and group_key:
        selection_mode = str(retrieval_kwargs.get("group_selection_mode", "nearest_compatible_centroid"))
        target_group = target_window.metadata.get(group_key)
        if selection_mode == "nearest_compatible_centroid":
            resolved = _select_nearest_compatible_group(
                target_window,
                filtered_windows,
                group_key=str(group_key),
                context_size=int(retrieval_kwargs.get("context_size", 5)),
            )
            if resolved:
                target_group = resolved
                target_window.metadata[group_key] = resolved
        if target_group is not None:
            filtered_windows = [window for window in filtered_windows if window.metadata.get(group_key) == target_group]
        audit["target_event_group"] = str(target_group or "")
        audit["num_group_matched_donor_windows"] = len(filtered_windows)
        audit["group_selection_mode"] = selection_mode
    logger.debug(
        "compatibility: target=%s donor_series %d -> %d; donor_windows %d -> %d",
        target_series_id,
        len(candidate_series_ids),
        len(kept_series_ids),
        len(donor_windows),
        len(filtered_windows),
    )

    retrieve_params = {
        key: value
        for key, value in retrieval_kwargs.items()
        if key not in {"grouping_config", "exclude_same_series", "group_selection_mode"}
    }
    exclude_same_series = bool(retrieval_kwargs.get("exclude_same_series", True))
    donors_with_scores = retrieve_topk_donors(
        target_window,
        filtered_windows,
        k=top_k,
        exclude_same_series=exclude_same_series,
        **retrieve_params,
    )
    donors = [donor for donor, _ in donors_with_scores]
    similarities = [float(score) for _, score in donors_with_scores]

    if is_cross_dataset_policy(donor_policy):
        for donor in donors:
            if pair_same_dataset(target_series_id, str(donor.series_id), feature_table):
                raise ValueError(
                    f"cross_dataset policy {donor_policy!r} retrieved same-dataset donor "
                    f"{donor.series_id} for target {target_series_id}"
                )

    audit["num_retrieved_donor_windows"] = len(donors)
    return donors, similarities, audit


def donor_policy_experiment_specs(synthesis_cfg: dict[str, Any]) -> list[dict[str, str]]:
    """Optional ablation specs for donor-policy comparison methods."""
    experiments = synthesis_cfg.get("donor_policy_experiments") or []
    specs: list[dict[str, str]] = []
    for entry in experiments:
        if isinstance(entry, str):
            parts = entry.split(":", 2)
            if len(parts) == 3:
                name, synthesis_method, donor_policy = parts
            else:
                continue
        elif isinstance(entry, dict):
            name = str(entry["name"])
            synthesis_method = str(entry["synthesis_method"])
            donor_policy = str(entry["donor_policy"])
        else:
            continue
        specs.append(
            {
                "name": name,
                "synthesis_method": synthesis_method,
                "donor_policy": donor_policy,
            }
        )
    return specs
