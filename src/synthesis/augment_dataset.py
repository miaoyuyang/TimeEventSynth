"""Augment training data with oversampled and synthesized event windows."""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..datasets.event_extractor import extract_event_segments
from ..datasets.tsb_loader import TimeSeriesRecord
from .donor_retrieval import EventWindow, collect_event_windows, donor_similarity_stats
from .donor_selection import (
    attach_event_group_distributions,
    build_timeline_feature_table,
    compatibility_is_enabled,
    is_cross_dataset_policy,
    learn_compatibility_weights,
    resolve_compatibility_config,
    select_donors_for_target,
)
from .event_window_synthesizer import (
    partial_labels_for_event_window,
    synthesize_context_aware,
    synthetic_training_labels_for_window,
)
from .dataset_identity import feature_dataset_name, log_dataset_identity, pair_same_dataset
from .donor_policy_sanity import validate_cross_dataset_accepted_rows, validate_donor_policy_pair_counts
from .synthetic_audit import (
    build_candidate_pair_audit_rows,
    build_donor_pair_audit_rows,
    compatibility_fields_from_record,
    flatten_audit_row,
)
from .uncertainty_filter import apply_filter_policy, overlay_confidence_prior, resolve_filter_policy, score_synthetic_window

# Re-export for backward compatibility
__all__ = [
    "build_augmented_training_records",
    "build_augmented_training_records_with_audit",
    "flatten_audit_row",
    "random_oversample_events",
    "resolve_oversample_target_multiplier",
    "records_from_oversampled_events",
]

DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER = 2.0


def random_oversample_events(events: list[dict[str, Any]], *, target_multiplier: float) -> list[dict[str, Any]]:
    """Duplicate anomaly windows as a basic oversampling baseline."""
    if not events:
        return []
    counts = Counter(str(event.get("event_label", "anomaly")) for event in events)
    max_count = max(counts.values())
    augmented: list[dict[str, Any]] = list(events)
    for event in events:
        label = str(event.get("event_label", "anomaly"))
        target = int(max_count * target_multiplier)
        needed = max(target - counts[label], 0)
        for copy_idx in range(min(needed, 2)):
            augmented.append({**dict(event), "synthetic_copy": True, "synthetic_copy_index": copy_idx})
    return augmented


def resolve_oversample_target_multiplier(config: dict[str, Any] | None = None) -> float:
    """Read ``synthesis.random_oversample_target_multiplier`` from an experiment config."""
    if not config:
        return DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER
    synthesis_cfg = config.get("synthesis", {})
    if not isinstance(synthesis_cfg, dict):
        return DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER
    return float(synthesis_cfg.get("random_oversample_target_multiplier", DEFAULT_OVERSAMPLE_TARGET_MULTIPLIER))


def records_from_oversampled_events(
    train_records: list[TimeSeriesRecord],
    *,
    target_multiplier: float | None = None,
    config: dict[str, Any] | None = None,
) -> list[TimeSeriesRecord]:
    """Append-only synthetic copies of labeled train event windows.

    Original event windows already appear in ``train_records``; only rows marked
    ``synthetic_copy`` by :func:`random_oversample_events` are materialized here.
    """
    if target_multiplier is None:
        target_multiplier = resolve_oversample_target_multiplier(config)

    events: list[dict[str, Any]] = []
    for record in train_records:
        timeline = {"series_id": record.series_id, "values": record.values, "labels": record.labels}
        series_values = np.asarray(record.values, dtype=float)
        series_labels = np.asarray(record.labels, dtype=int)
        for segment in extract_event_segments(timeline, min_event_length=1, merge_gap=0, context_padding=0):
            events.append(
                {
                    "series_id": record.series_id,
                    "event_label": "anomaly",
                    "values": segment.values,
                    "labels": segment.labels,
                    "start_idx": int(segment.start_idx),
                    "end_idx": int(segment.end_idx),
                    "series_values": series_values,
                    "series_labels": series_labels,
                }
            )

    oversampled = random_oversample_events(events, target_multiplier=float(target_multiplier))
    records: list[TimeSeriesRecord] = []
    for idx, row in enumerate(oversampled):
        if not bool(row.get("synthetic_copy", False)):
            continue
        records.append(
            TimeSeriesRecord(
                series_id=f"{row['series_id']}__oversampled__{idx}",
                values=np.asarray(row["values"], dtype=float),
                labels=np.asarray(row["labels"], dtype=int),
                timestamps=None,
                source_path="oversampled",
                metadata={
                    "synthetic": True,
                    "synthesis_method": "random_event_oversampling",
                    "target_series": str(row["series_id"]),
                    "target_event_interval": [int(row["start_idx"]), int(row["end_idx"])],
                    "target_series_values": np.asarray(row["series_values"], dtype=float).tolist(),
                    "target_series_labels": np.asarray(row["series_labels"], dtype=int).tolist(),
                },
            )
        )
    return records


def _candidate_id(window: EventWindow, method: str) -> str:
    return f"{window.series_id}:{window.start}:{window.end}:{method}"


def _donor_audit_fields(donors: list[EventWindow], similarities: list[float]) -> dict[str, Any]:
    stats = donor_similarity_stats(similarities)
    return {
        "donor_series_ids": [donor.series_id for donor in donors],
        "donor_starts": [int(donor.start) for donor in donors],
        "donor_ends": [int(donor.end) for donor in donors],
        "donor_lengths": [int(donor.end - donor.start) for donor in donors],
        "donor_similarity_mean": stats["mean"],
        "donor_similarity_min": stats["min"],
        "donor_similarity_max": stats["max"],
        "donor_ids": [f"{donor.series_id}:{donor.start}:{donor.end}" for donor in donors],
    }


def _empty_confidence_components() -> dict[str, float]:
    return {
        "donor_similarity_confidence": 0.0,
        "reconstruction_consistency_confidence": 0.0,
        "donor_agreement_confidence": 0.0,
        "amplitude_compatibility_score": 0.0,
        "aggregate_confidence": 0.0,
    }


def _filter_method_name(filter_policy: dict[str, Any] | str | None) -> str:
    if isinstance(filter_policy, dict):
        return str(filter_policy.get("name", "no_filter"))
    if isinstance(filter_policy, str):
        return filter_policy
    return "no_filter"


def _audit_identity_fields(
    window: EventWindow,
    *,
    feature_table: dict[str, dict[str, Any]],
    policy_config: dict[str, Any],
    synthesis_method: str,
    source_series_id: str | None = None,
) -> dict[str, Any]:
    target_series_id = str(window.series_id)
    target_dataset = feature_dataset_name(feature_table.get(target_series_id, {}))
    source_dataset = feature_dataset_name(feature_table.get(source_series_id, {})) if source_series_id else None
    cross_policy = is_cross_dataset_policy(policy_config.get("donor_policy"))
    same_dataset = False if cross_policy else bool(target_dataset and source_dataset and target_dataset == source_dataset)
    return {
        "target_series_id": target_series_id,
        "source_series_id": source_series_id or "",
        "target_event_group": str(window.metadata.get("event_group_id", "")),
        "target_dataset": target_dataset,
        "source_dataset": source_dataset,
        "dataset_name": target_dataset,
        "same_dataset": same_dataset,
        "donor_policy": policy_config.get("donor_policy"),
        "synthesis_method": synthesis_method,
        "filter_method": _filter_method_name(policy_config.get("filter_policy")),
    }


def _mean_kept_compatibility_record(compatibility_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    kept = [row for row in compatibility_records if str(row.get("final_decision")) == "kept"]
    if not kept:
        return None
    keys = (
        "compatibility_score",
        "shape_similarity",
        "amplitude_compatibility",
        "duration_compatibility",
        "context_similarity",
        "frequency_similarity",
        "trend_similarity",
        "group_compatibility",
    )
    averaged: dict[str, Any] = {"final_decision": "kept", "rejection_reason": ""}
    for key in keys:
        averaged[key] = float(np.mean([float(row[key]) for row in kept]))
    averaged["compatibility_weighting"] = str(kept[0].get("compatibility_weighting", ""))
    return averaged


def _build_no_donor_audit_row(
    window: EventWindow,
    *,
    method: str,
    labeled_fraction: float | None,
    synthesis_method: str,
    feature_table: dict[str, dict[str, Any]],
    policy_config: dict[str, Any],
    rejection_reason: str,
    rejection_stage: str,
    compatibility_enabled: bool,
    compatibility_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    compat_record = _mean_kept_compatibility_record((compatibility_audit or {}).get("compatibility_records", []))
    return {
        "candidate_id": _candidate_id(window, method),
        "method": method,
        "labeled_fraction": labeled_fraction,
        "target_start": int(window.start),
        "target_end": int(window.end),
        "target_length": int(window.end - window.start),
        "target_id": f"{window.series_id}:{window.start}:{window.end}",
        "series_id": f"{window.series_id}__synthetic__{window.start}_{window.end}",
        "values": [],
        "labels": [],
        "record_type": "synthesis_candidate",
        "metadata": {
            "synthetic": True,
            "target_series": window.series_id,
            "target_event_interval": [window.start, window.end],
            "synthesis_method": synthesis_method,
            "confidence": 0.0,
        },
        "confidence_components": _empty_confidence_components(),
        **_donor_audit_fields([], []),
        **_audit_identity_fields(window, feature_table=feature_table, policy_config=policy_config, synthesis_method=synthesis_method),
        **compatibility_fields_from_record(compat_record, enabled=compatibility_enabled),
        "compatibility_enabled": compatibility_enabled,
        "kept": False,
        "accepted": False,
        "rejection_reason": rejection_reason,
        "rejection_stage": rejection_stage,
    }


def _resolve_synthesis_options(policy_config: dict[str, Any], synthesis_cfg: dict[str, Any] | None) -> dict[str, Any]:
    cfg = synthesis_cfg or {}
    return {
        "amplitude_calibration": str(
            policy_config.get("amplitude_calibration", cfg.get("amplitude_calibration", "none"))
        ),
        "partial_target_mode": str(
            policy_config.get("partial_target_mode", cfg.get("partial_target_synthesis", "full_replacement"))
        ),
        "context_size": int(policy_config.get("context_size", cfg.get("context_size", 5))),
        "match_partial_amplitude": bool(
            policy_config.get("match_partial_amplitude", cfg.get("match_partial_amplitude", True))
        ),
    }


def _build_synthetic_candidate(
    window: EventWindow,
    donors: list[EventWindow],
    similarities: list[float],
    policy_config: dict[str, Any],
    synthesis_cfg: dict[str, Any] | None = None,
    *,
    feature_table: dict[str, dict[str, Any]],
    compatibility_enabled: bool,
    compatibility_audit: dict[str, Any] | None = None,
) -> dict[str, Any]:
    synthesis_method = str(policy_config.get("method", "normalized_time_mean_donor"))
    method_name = str(policy_config.get("method_name", synthesis_method))
    labeled_fraction = policy_config.get("labeled_fraction")
    options = _resolve_synthesis_options(policy_config, synthesis_cfg)
    synthetic_values, calibration_audit = synthesize_context_aware(
        window,
        donors,
        base_method=synthesis_method,
        target_length=len(window.values),
        similarities=similarities,
        amplitude_calibration=options["amplitude_calibration"],
        partial_target_mode=options["partial_target_mode"],
        context_size=options["context_size"],
        match_partial_amplitude=options["match_partial_amplitude"],
        grid_size=int(policy_config.get("grid_size", 64)),
    )
    synthetic_point_labels = synthetic_training_labels_for_window(
        len(synthetic_values),
        partial_labels_for_event_window(window),
        partial_target_mode=options["partial_target_mode"],
    )

    confidence_components = score_synthetic_window(
        np.asarray(window.values, dtype=float),
        np.asarray(synthetic_values, dtype=float),
        [np.asarray(donor.values, dtype=float) for donor in donors],
        similarities,
        grid_size=int(policy_config.get("grid_size", 64)),
    )
    diversity_status = "diverse" if len({str(donor.series_id) for donor in donors}) > 1 else "single_series_only"
    confidence_components = overlay_confidence_prior(
        confidence_components,
        donor_count=len(donors),
        synthesis_method=synthesis_method,
        diversity_status=diversity_status,
    )
    compat_record = _mean_kept_compatibility_record((compatibility_audit or {}).get("compatibility_records", []))
    primary_source = str(donors[0].series_id) if donors else ""
    donor_event_groups = [str(donor.metadata.get("event_group_id", "")) for donor in donors]
    if is_cross_dataset_policy(policy_config.get("donor_policy")):
        for donor in donors:
            if pair_same_dataset(str(window.series_id), str(donor.series_id), feature_table):
                raise ValueError(
                    f"cross_dataset policy cannot synthesize from same-dataset donor {donor.series_id}"
                )
    return {
        "candidate_id": _candidate_id(window, method_name),
        "method": method_name,
        "labeled_fraction": labeled_fraction,
        "target_start": int(window.start),
        "target_end": int(window.end),
        "target_length": int(window.end - window.start),
        "target_id": f"{window.series_id}:{window.start}:{window.end}",
        "series_id": f"{window.series_id}__synthetic__{window.start}_{window.end}",
        "values": synthetic_values.tolist() if hasattr(synthetic_values, "tolist") else list(synthetic_values),
        "labels": synthetic_point_labels.tolist(),
        "record_type": "synthesis_candidate",
        "metadata": {
            "synthetic": True,
            "source_donor_series": [donor.series_id for donor in donors],
            "source_donor_event_groups": donor_event_groups,
            "target_series": window.series_id,
            "target_event_interval": [window.start, window.end],
            "target_event_group": str(window.metadata.get("event_group_id", "")),
            "target_series_values": (
                np.asarray(window.metadata.get("series_values"), dtype=float).tolist()
                if window.metadata.get("series_values") is not None
                else None
            ),
            "target_series_labels": (
                np.asarray(window.metadata.get("series_labels"), dtype=int).tolist()
                if window.metadata.get("series_labels") is not None
                else None
            ),
            "synthesis_method": synthesis_method,
            "confidence": float(confidence_components["aggregate_confidence"]),
            "prior_adjusted_confidence": float(confidence_components["prior_adjusted_confidence"]),
            "diversity_status": diversity_status,
            **calibration_audit,
        },
        "confidence_components": confidence_components,
        **_donor_audit_fields(donors, similarities),
        "donor_event_groups": donor_event_groups,
        **calibration_audit,
        **_audit_identity_fields(
            window,
            feature_table=feature_table,
            policy_config=policy_config,
            synthesis_method=synthesis_method,
            source_series_id=primary_source,
        ),
        **compatibility_fields_from_record(compat_record, enabled=compatibility_enabled),
        "compatibility_enabled": compatibility_enabled,
        "compatibility_record": compat_record,
        "kept": True,
        "accepted": True,
        "rejection_reason": None,
        "rejection_stage": "synthesis",
    }


def build_augmented_training_records_with_audit(
    records: list[Any],
    split: str,
    policy_config: dict[str, Any],
    *,
    donor_pool_records: list[Any] | None = None,
    synthesis_cfg: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create synthetic windows and return both kept windows and full audit rows."""
    synthesis_method = str(policy_config.get("method", "normalized_time_mean_donor"))
    method_name = str(policy_config.get("method_name", synthesis_method))
    top_k = int(policy_config.get("top_k", 3))
    if split != "train":
        return [], []

    synthesis_cfg = synthesis_cfg or {}
    synthesis_cfg = dict(synthesis_cfg)
    if synthesis_cfg.get("dataset_name") is None:
        dataset_cfg = synthesis_cfg.get("dataset")
        if isinstance(dataset_cfg, str):
            synthesis_cfg["dataset_name"] = dataset_cfg
    compatibility_cfg = resolve_compatibility_config(synthesis_cfg, policy_config)
    policy_config = dict(policy_config)
    policy_config["compatibility"] = compatibility_cfg
    policy_config["donor_policy"] = policy_config.get("donor_policy") or synthesis_cfg.get("donor_policy")
    compat_enabled = compatibility_is_enabled(compatibility_cfg)

    target_windows = collect_event_windows(records)
    donor_pool = donor_pool_records if donor_pool_records is not None else records
    donor_windows = collect_event_windows(donor_pool)
    feature_table = build_timeline_feature_table(list(records) + list(donor_pool), synthesis_cfg=synthesis_cfg)
    log_dataset_identity(feature_table)

    donor_policy = str(policy_config.get("donor_policy") or "")
    pair_totals = {"same_dataset_pairs_considered": 0, "cross_dataset_pairs_considered": 0}
    if compat_enabled:
        print(
            f"compatibility enabled: feature_table={len(feature_table)} series, "
            f"donor_windows={len(donor_windows)}, target_windows={len(target_windows)}"
        )

    groupwise_matching = bool(policy_config.get("groupwise_matching", False))
    retrieval_kwargs = {
        "max_donors_per_source_series": policy_config.get("max_donors_per_source_series"),
        "avoid_single_series_dominance": bool(policy_config.get("avoid_single_series_dominance", True)),
        "group_key": policy_config.get("group_key", "event_group_id") if groupwise_matching else None,
        "max_donors_per_group": policy_config.get("max_donors_per_group"),
        "restrict_to_target_group": groupwise_matching,
        "context_size": int(synthesis_cfg.get("context_size", policy_config.get("context_size", 5))),
    }
    grouping_summary = None
    if groupwise_matching:
        from ..alignment.event_grouping import assign_event_groups

        grouping_cfg = dict(synthesis_cfg.get("grouping", {}))
        grouping_cfg.setdefault("context_size", int(retrieval_kwargs.get("context_size", 5)))
        grouping_summary = assign_event_groups(
            donor_windows,
            [],
            config=grouping_cfg,
            group_key=str(retrieval_kwargs["group_key"]),
        )
        retrieval_kwargs["group_selection_mode"] = "nearest_compatible_centroid"

    learned_weight_summary: dict[str, Any] | None = None
    if compat_enabled and bool((compatibility_cfg.get("learned_weights") or {}).get("enabled", False)):
        if groupwise_matching and retrieval_kwargs.get("group_key"):
            attach_event_group_distributions(
                feature_table,
                donor_windows + target_windows,
                group_key=str(retrieval_kwargs["group_key"]),
            )
        learned_weight_summary = learn_compatibility_weights(feature_table, compatibility_cfg)
        if bool(learned_weight_summary.get("enabled", False)):
            compatibility_cfg = dict(compatibility_cfg)
            compatibility_cfg["weights"] = dict(learned_weight_summary["weights"])
            compatibility_cfg["learned_weight_summary"] = learned_weight_summary
            compatibility_cfg["weighting_method"] = "learned"
            policy_config["compatibility"] = compatibility_cfg
            weights_text = ", ".join(
                f"{key}={value:.3f}" for key, value in learned_weight_summary.get("learned_alpha", {}).items()
            )
            print(f"learned compatibility weights: {weights_text}")
        else:
            compatibility_cfg = dict(compatibility_cfg)
            compatibility_cfg["learned_weight_summary"] = learned_weight_summary
            compatibility_cfg["weighting_method"] = "fixed_fallback"
            policy_config["compatibility"] = compatibility_cfg
            print(f"learned compatibility weights disabled: {learned_weight_summary.get('reason')}")

    audit_rows: list[dict[str, Any]] = []
    for window in target_windows:
        donors, similarities, compatibility_audit = select_donors_for_target(
            window,
            donor_windows,
            feature_table=feature_table,
            compatibility_cfg=compatibility_cfg,
            top_k=top_k,
            retrieval_kwargs=retrieval_kwargs,
        )

        pair_totals["same_dataset_pairs_considered"] += int(
            compatibility_audit.get("same_dataset_pairs_considered", 0)
        )
        pair_totals["cross_dataset_pairs_considered"] += int(
            compatibility_audit.get("cross_dataset_pairs_considered", 0)
        )

        if compat_enabled:
            audit_rows.extend(
                build_donor_pair_audit_rows(
                    window,
                    method_name=method_name,
                    synthesis_method=synthesis_method,
                    labeled_fraction=policy_config.get("labeled_fraction"),
                    donor_policy=policy_config.get("donor_policy"),
                    filter_method=_filter_method_name(policy_config.get("filter_policy")),
                    feature_table=feature_table,
                    compatibility_records=compatibility_audit.get("compatibility_records", []),
                    compatibility_enabled=True,
                )
            )
        else:
            audit_rows.extend(
                build_candidate_pair_audit_rows(
                    window,
                    method_name=method_name,
                    synthesis_method=synthesis_method,
                    labeled_fraction=policy_config.get("labeled_fraction"),
                    donor_policy=policy_config.get("donor_policy"),
                    filter_method=_filter_method_name(policy_config.get("filter_policy")),
                    feature_table=feature_table,
                    candidate_series_ids=compatibility_audit.get("evaluated_candidate_series_ids", []),
                )
            )

        if not donors:
            reason = "no_compatible_donor" if compat_enabled else "synthesis_failed"
            stage = "compatibility" if compat_enabled else "synthesis"
            audit_rows.append(
                _build_no_donor_audit_row(
                    window,
                    method=method_name,
                    labeled_fraction=policy_config.get("labeled_fraction"),
                    synthesis_method=synthesis_method,
                    feature_table=feature_table,
                    policy_config=policy_config,
                    rejection_reason=reason,
                    rejection_stage=stage,
                    compatibility_enabled=compat_enabled,
                    compatibility_audit=compatibility_audit,
                )
            )
            continue

        audit_rows.append(
            _build_synthetic_candidate(
                window,
                donors,
                similarities,
                policy_config,
                synthesis_cfg,
                feature_table=feature_table,
                compatibility_enabled=compat_enabled,
                compatibility_audit=compatibility_audit,
            )
        )

    filter_policy = policy_config.get("filter_policy")
    if isinstance(filter_policy, str):
        filter_policy = resolve_filter_policy(synthesis_cfg or {}, override=filter_policy)
    elif filter_policy is None and synthesis_cfg is not None:
        filter_policy = resolve_filter_policy(synthesis_cfg)

    filtered_rows = apply_filter_policy(audit_rows, filter_policy, synthesis_cfg=synthesis_cfg)

    validate_donor_policy_pair_counts(
        donor_policy,
        same_dataset_pairs_considered=pair_totals["same_dataset_pairs_considered"],
        cross_dataset_pairs_considered=pair_totals["cross_dataset_pairs_considered"],
    )
    validate_cross_dataset_accepted_rows(filtered_rows, donor_policy=donor_policy, feature_table=feature_table)

    kept_records = [row for row in filtered_rows if bool(row.get("accepted", row.get("kept", False))) and row.get("values")]
    if groupwise_matching:
        for row in filtered_rows:
            row["event_grouping_enabled"] = True
            row["event_grouping_method"] = grouping_summary.method if grouping_summary is not None else "unknown"
            row["num_event_groups"] = int(grouping_summary.num_groups) if grouping_summary is not None else 0
    if learned_weight_summary is not None:
        for row in filtered_rows:
            row["compatibility_weight_learning_enabled"] = bool(learned_weight_summary.get("enabled", False))
            row["compatibility_weight_learning_reason"] = str(learned_weight_summary.get("reason", ""))
            for key, value in (learned_weight_summary.get("learned_alpha") or {}).items():
                row[f"compatibility_alpha_{key}"] = float(value)
    return kept_records, filtered_rows


def build_augmented_training_records(
    records: list[Any],
    split: str,
    policy_config: dict[str, Any],
    *,
    donor_pool_records: list[Any] | None = None,
    synthesis_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Create synthetic anomaly windows without mutating the original records."""
    kept_records, _ = build_augmented_training_records_with_audit(
        records,
        split,
        policy_config,
        donor_pool_records=donor_pool_records,
        synthesis_cfg=synthesis_cfg,
    )
    return kept_records
