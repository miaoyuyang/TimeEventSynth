"""Aggregate compatibility-transfer audit rows into compatibility_summary.json."""

from __future__ import annotations

from typing import Any

import numpy as np

from src.synthesis.dataset_identity import feature_dataset_name


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(number):
        return None
    return number


def _is_same_dataset_row(row: dict[str, Any]) -> bool:
    same = row.get("same_dataset")
    if same is not None and not (isinstance(same, float) and np.isnan(same)):
        return bool(same)
    target_dataset = row.get("target_dataset")
    source_dataset = row.get("source_dataset")
    if target_dataset and source_dataset:
        return str(target_dataset) == str(source_dataset)
    return False


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kept_scores: list[float] = []
    rejected_scores: list[float] = []
    same_considered = cross_considered = 0
    same_kept = same_rejected = 0
    cross_kept = cross_rejected = 0
    no_compatible_donor = 0

    for row in rows:
        is_pair = row.get("record_type") in {"donor_pair", "donor_rejection"} or (
            str(row.get("rejection_stage", "")) == "compatibility" and row.get("source_series_id")
        )
        if not is_pair:
            reason = str(row.get("rejection_reason", "") or "")
            if reason in {"no_compatible_donor", "skip"} and not bool(row.get("accepted", False)):
                no_compatible_donor += 1
            continue

        is_same = _is_same_dataset_row(row)
        if is_same:
            same_considered += 1
        else:
            cross_considered += 1

        if not bool(row.get("compatibility_enabled", False)):
            continue

        score = _safe_float(row.get("compatibility_score"))
        accepted = bool(row.get("accepted", row.get("kept", False)))

        if accepted:
            if score is not None:
                kept_scores.append(score)
            if is_same:
                same_kept += 1
            else:
                cross_kept += 1
        else:
            if score is not None:
                rejected_scores.append(score)
            if is_same:
                same_rejected += 1
            else:
                cross_rejected += 1

    return {
        "total_donor_pairs_considered": same_considered + cross_considered,
        "same_dataset_pairs_considered": same_considered,
        "cross_dataset_pairs_considered": cross_considered,
        "donor_pairs_kept": same_kept + cross_kept,
        "donor_pairs_rejected": same_rejected + cross_rejected,
        "same_dataset_kept": same_kept,
        "same_dataset_rejected": same_rejected,
        "cross_dataset_kept": cross_kept,
        "cross_dataset_rejected": cross_rejected,
        "mean_compatibility_score_kept": float(np.mean(kept_scores)) if kept_scores else None,
        "mean_compatibility_score_rejected": float(np.mean(rejected_scores)) if rejected_scores else None,
        "no_compatible_donor_count": no_compatible_donor,
    }


def _global_from_audit(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    targets: set[str] = set()
    sources: set[str] = set()
    dataset_counts: dict[str, int] = {}

    for row in audit_rows:
        target_id = row.get("target_series_id")
        if target_id:
            targets.add(str(target_id))
            td = row.get("target_dataset") or row.get("dataset_name")
            if td:
                dataset_counts[str(td)] = dataset_counts.get(str(td), 0) + 1
        source_id = row.get("source_series_id")
        if source_id and str(source_id) not in {"", "nan"}:
            sources.add(str(source_id))

    for row in audit_rows:
        for donor_id in str(row.get("donor_series_ids", "") or "").split("|"):
            if donor_id:
                sources.add(donor_id)

    return {
        "num_target_timelines": len(targets),
        "num_source_timelines": len(sources),
        "num_unique_datasets": len(dataset_counts),
        "dataset_counts": dataset_counts,
    }


def build_compatibility_summary(
    audit_rows: list[dict[str, Any]],
    *,
    rejection_summary: dict[str, Any] | None = None,
    feature_table: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Summarize donor-pair compatibility decisions with per-policy and global sections."""
    if feature_table:
        for row in audit_rows:
            if not row.get("target_dataset") and row.get("target_series_id"):
                row["target_dataset"] = feature_dataset_name(feature_table.get(str(row["target_series_id"]), {}))
            if not row.get("source_dataset") and row.get("source_series_id"):
                row["source_dataset"] = feature_dataset_name(feature_table.get(str(row["source_series_id"]), {}))

    global_stats = _global_from_audit(audit_rows)

    policies: dict[str, dict[str, Any]] = {}
    policy_names = sorted(
        {
            str(row.get("donor_policy"))
            for row in audit_rows
            if row.get("donor_policy") not in {None, "", "nan"}
        }
    )
    for policy in policy_names:
        policy_rows = [row for row in audit_rows if str(row.get("donor_policy")) == policy]
        policies[policy] = _summarize_policy_rows(policy_rows)

    legacy = {
        "num_target_timelines": global_stats["num_target_timelines"],
        "num_source_timelines": global_stats["num_source_timelines"],
        "total_donor_pairs_considered": sum(p.get("total_donor_pairs_considered", 0) for p in policies.values()),
        "donor_pairs_kept": sum(p.get("donor_pairs_kept", 0) for p in policies.values()),
        "donor_pairs_rejected": sum(p.get("donor_pairs_rejected", 0) for p in policies.values()),
        "same_dataset_kept": sum(p.get("same_dataset_kept", 0) for p in policies.values()),
        "same_dataset_rejected": sum(p.get("same_dataset_rejected", 0) for p in policies.values()),
        "cross_dataset_kept": sum(p.get("cross_dataset_kept", 0) for p in policies.values()),
        "cross_dataset_rejected": sum(p.get("cross_dataset_rejected", 0) for p in policies.values()),
        "no_compatible_donor_count": sum(p.get("no_compatible_donor_count", 0) for p in policies.values()),
    }
    kept_all = [
        s
        for p in policies.values()
        for s in [_safe_float(p.get("mean_compatibility_score_kept"))]
        if s is not None
    ]
    rejected_all = [
        s
        for p in policies.values()
        for s in [_safe_float(p.get("mean_compatibility_score_rejected"))]
        if s is not None
    ]
    legacy["mean_compatibility_score_kept"] = float(np.mean(kept_all)) if kept_all else None
    legacy["mean_compatibility_score_rejected"] = float(np.mean(rejected_all)) if rejected_all else None

    if rejection_summary:
        for entry in rejection_summary.values():
            counts = entry.get("counts", {}) if isinstance(entry, dict) else {}
            legacy["no_compatible_donor_count"] = max(
                legacy.get("no_compatible_donor_count", 0), int(counts.get("no_compatible_donor", 0))
            )

    return {
        **legacy,
        "global": global_stats,
        "policies": policies,
    }
