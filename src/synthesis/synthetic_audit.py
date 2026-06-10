"""Synthetic candidate and donor-rejection audit records for paper-ready traceability."""

from __future__ import annotations

from typing import Any

import numpy as np

from .dataset_identity import feature_dataset_name, pair_same_dataset
from .donor_selection import is_cross_dataset_policy

AUDIT_CSV_COLUMNS: tuple[str, ...] = (
    "candidate_id",
    "method",
    "labeled_fraction",
    "target_series_id",
    "source_series_id",
    "target_start",
    "target_end",
    "target_length",
    "target_dataset",
    "target_event_group",
    "source_dataset",
    "same_dataset",
    "donor_policy",
    "synthesis_method",
    "filter_method",
    "compatibility_enabled",
    "compatibility_score",
    "shape_similarity",
    "amplitude_compatibility",
    "duration_compatibility",
    "context_similarity",
    "frequency_similarity",
    "trend_similarity",
    "group_compatibility",
    "compatibility_weighting",
    "compatibility_decision",
    "compatibility_rejection_reason",
    "donor_similarity",
    "reconstruction_consistency",
    "donor_agreement",
    "filter_amplitude_compatibility",
    "final_confidence",
    "calibration_method",
    "scale",
    "shift",
    "boundary_continuity_error",
    "accepted",
    "rejection_reason",
    "rejection_stage",
    "donor_series_ids",
    "donor_starts",
    "donor_ends",
    "donor_lengths",
    "donor_event_groups",
    "donor_similarity_mean",
    "donor_similarity_min",
    "donor_similarity_max",
)

NORMALIZED_REJECTION_COUNTS = (
    "kept",
    "rejected_incompatible_donor",
    "no_compatible_donor",
    "below_compatibility_threshold",
    "below_top_quantile_0.5",
    "cross_dataset_disallowed",
    "synthesis_failed",
    "invalid_candidate",
    "evaluator_invalid",
)


def _nan_if_missing(value: Any) -> Any:
    if value is None:
        return np.nan
    return value


def normalize_rejection_reason(raw: str | None, *, rejection_stage: str | None = None) -> str:
    """Map internal reasons to stable summary keys."""
    if raw is None or raw == "" or raw == "kept":
        return ""
    text = str(raw)
    if text in {"cross_dataset_disallowed", "cross_dataset_not_allowed"}:
        return "cross_dataset_disallowed"
    if text.startswith("below_min_score"):
        return "below_compatibility_threshold"
    if text == "filtered_by_top_k_or_quantile":
        return "below_top_quantile_0.5"
    if text in {"skip", "no_compatible_donors", "no_compatible_donor"}:
        return "no_compatible_donor"
    if text == "no_donors_found":
        return "synthesis_failed"
    if text.startswith("strict_filter_failed"):
        return text
    if text.startswith("aggregate_confidence_below"):
        return "below_quality_threshold"
    if text.startswith("below_top_quantile"):
        return "below_top_quantile_0.5"
    if rejection_stage == "compatibility":
        return "rejected_incompatible_donor"
    return text


def normalized_reason_count_key(row: dict[str, Any]) -> str:
    if bool(row.get("accepted", row.get("kept", False))):
        return "kept"
    stage = str(row.get("rejection_stage", "") or "")
    reason = normalize_rejection_reason(row.get("rejection_reason"), rejection_stage=stage)
    if stage == "compatibility" and reason in {"", "below_compatibility_threshold", "cross_dataset_disallowed"}:
        if reason == "cross_dataset_disallowed":
            return "cross_dataset_disallowed"
        if reason == "below_compatibility_threshold":
            return "below_compatibility_threshold"
        return "rejected_incompatible_donor"
    if reason == "no_compatible_donor":
        return "no_compatible_donor"
    if reason.startswith("strict_filter_failed"):
        return reason
    if reason == "below_top_quantile_0.5":
        return "below_top_quantile_0.5"
    if reason == "below_quality_threshold":
        return "below_quality_threshold"
    if reason == "synthesis_failed":
        return "synthesis_failed"
    if stage == "evaluator":
        return "evaluator_invalid"
    if stage == "synthesis":
        return "synthesis_failed"
    return reason or "invalid_candidate"


def _dataset_from_feature_table(series_id: str, feature_table: dict[str, dict[str, Any]]) -> str | None:
    return feature_dataset_name(feature_table.get(series_id, {}))


def compatibility_fields_from_record(
    record: dict[str, Any] | None,
    *,
    enabled: bool,
) -> dict[str, Any]:
    if not enabled or not record:
        return {
            "compatibility_enabled": False,
            "compatibility_score": np.nan,
            "shape_similarity": np.nan,
            "amplitude_compatibility": np.nan,
            "duration_compatibility": np.nan,
            "context_similarity": np.nan,
            "frequency_similarity": np.nan,
            "trend_similarity": np.nan,
            "group_compatibility": np.nan,
            "compatibility_weighting": "not_applied",
            "compatibility_decision": "not_applied",
            "compatibility_rejection_reason": "",
        }
    return {
        "compatibility_enabled": True,
        "compatibility_score": float(record.get("compatibility_score", np.nan)),
        "shape_similarity": float(record.get("shape_similarity", np.nan)),
        "amplitude_compatibility": float(record.get("amplitude_compatibility", np.nan)),
        "duration_compatibility": float(record.get("duration_compatibility", np.nan)),
        "context_similarity": float(record.get("context_similarity", np.nan)),
        "frequency_similarity": float(record.get("frequency_similarity", np.nan)),
        "trend_similarity": float(record.get("trend_similarity", np.nan)),
        "group_compatibility": float(record.get("group_compatibility", np.nan)),
        "compatibility_weighting": str(record.get("compatibility_weighting", "")),
        "compatibility_decision": str(record.get("final_decision", "")),
        "compatibility_rejection_reason": str(record.get("rejection_reason", "") or ""),
    }


def build_candidate_pair_audit_rows(
    window: Any,
    *,
    method_name: str,
    synthesis_method: str,
    labeled_fraction: float | None,
    donor_policy: str | None,
    filter_method: str | None,
    feature_table: dict[str, dict[str, Any]],
    candidate_series_ids: list[str],
) -> list[dict[str, Any]]:
    """Audit rows for donor-policy candidate pairs when compatibility scoring is disabled."""
    target_series_id = str(window.series_id)
    target_dataset = _dataset_from_feature_table(target_series_id, feature_table)
    rows: list[dict[str, Any]] = []
    for source_series_id in candidate_series_ids:
        source_dataset = _dataset_from_feature_table(source_series_id, feature_table)
        is_same = pair_same_dataset(target_series_id, source_series_id, feature_table)
        if is_cross_dataset_policy(donor_policy) and is_same:
            raise ValueError(f"cross_dataset policy logged same-dataset candidate {source_series_id}")
        rows.append(
            {
                "candidate_id": f"{target_series_id}:{window.start}:{window.end}:candidate:{source_series_id}:{method_name}",
                "method": method_name,
                "labeled_fraction": labeled_fraction,
                "target_series_id": target_series_id,
                "source_series_id": source_series_id,
                "target_start": int(window.start),
                "target_end": int(window.end),
                "target_length": int(window.end - window.start),
                "target_dataset": target_dataset,
                "target_event_group": str(window.metadata.get("event_group_id", "")),
                "source_dataset": source_dataset,
                "same_dataset": is_same,
                "donor_policy": donor_policy,
                "synthesis_method": synthesis_method,
                "filter_method": filter_method,
                "record_type": "donor_pair",
                **compatibility_fields_from_record(None, enabled=False),
                "accepted": True,
                "rejection_reason": None,
                "rejection_stage": "compatibility",
            }
        )
    return rows


def build_donor_pair_audit_rows(
    window: Any,
    *,
    method_name: str,
    synthesis_method: str,
    labeled_fraction: float | None,
    donor_policy: str | None,
    filter_method: str | None,
    feature_table: dict[str, dict[str, Any]],
    compatibility_records: list[dict[str, Any]],
    compatibility_enabled: bool,
) -> list[dict[str, Any]]:
    """One audit row per donor-series pair scored at the compatibility stage (kept or rejected)."""
    if not compatibility_enabled or not compatibility_records:
        return []
    target_series_id = str(window.series_id)
    target_dataset = _dataset_from_feature_table(target_series_id, feature_table)
    rows: list[dict[str, Any]] = []
    for record in compatibility_records:
        source_series_id = str(record.get("source_series_id"))
        source_dataset = _dataset_from_feature_table(source_series_id, feature_table)
        is_same = pair_same_dataset(target_series_id, source_series_id, feature_table)
        if is_cross_dataset_policy(donor_policy) and is_same:
            raise ValueError(
                f"cross_dataset policy {donor_policy!r} logged same-dataset pair "
                f"{target_series_id} <- {source_series_id}"
            )
        kept = str(record.get("final_decision")) == "kept"
        raw_reason = str(record.get("rejection_reason", "") or "")
        normalized = normalize_rejection_reason(raw_reason, rejection_stage="compatibility")
        rows.append(
            {
                "candidate_id": f"{target_series_id}:{window.start}:{window.end}:donor:{source_series_id}:{method_name}",
                "method": method_name,
                "labeled_fraction": labeled_fraction,
                "target_series_id": target_series_id,
                "source_series_id": source_series_id,
                "target_start": int(window.start),
                "target_end": int(window.end),
                "target_length": int(window.end - window.start),
                "target_dataset": target_dataset,
                "target_event_group": str(window.metadata.get("event_group_id", "")),
                "source_dataset": source_dataset,
                "dataset_name": target_dataset,
                "same_dataset": is_same,
                "donor_policy": donor_policy,
                "synthesis_method": synthesis_method,
                "filter_method": filter_method,
                "record_type": "donor_pair",
                **compatibility_fields_from_record(record, enabled=True),
                "accepted": kept,
                "rejection_reason": None if kept else (normalized or raw_reason),
                "rejection_stage": "compatibility",
                "donor_series_ids": [],
                "donor_starts": [],
                "donor_ends": [],
                "donor_lengths": [],
                "confidence_components": {},
                "values": [],
                "labels": [],
            }
        )
    return rows


def build_donor_rejection_audit_rows(
    window: Any,
    *,
    method_name: str,
    synthesis_method: str,
    labeled_fraction: float | None,
    donor_policy: str | None,
    filter_method: str | None,
    feature_table: dict[str, dict[str, Any]],
    compatibility_records: list[dict[str, Any]],
    compatibility_enabled: bool,
) -> list[dict[str, Any]]:
    """Backward-compatible alias: rejected donor pairs only."""
    return [
        row
        for row in build_donor_pair_audit_rows(
            window,
            method_name=method_name,
            synthesis_method=synthesis_method,
            labeled_fraction=labeled_fraction,
            donor_policy=donor_policy,
            filter_method=filter_method,
            feature_table=feature_table,
            compatibility_records=compatibility_records,
            compatibility_enabled=compatibility_enabled,
        )
        if not row.get("accepted")
    ]


def flatten_audit_row(row: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal audit row into a CSV-friendly record."""
    components = row.get("confidence_components", {}) or {}
    enabled = bool(row.get("compatibility_enabled", False))
    compat_source = row.get("compatibility_record")
    if compat_source is None and enabled:
        compat_source = row
    compat = compatibility_fields_from_record(compat_source, enabled=enabled)

    donor_ids = row.get("donor_series_ids", [])
    rejection_stage = row.get("rejection_stage")
    if rejection_stage is None:
        rejection_stage = "kept" if bool(row.get("accepted", row.get("kept", False))) else "synthesis"

    flat = {
        "candidate_id": row.get("candidate_id"),
        "method": row.get("method"),
        "record_type": row.get("record_type"),
        "labeled_fraction": row.get("labeled_fraction"),
        "target_series_id": row.get("target_series_id"),
        "source_series_id": row.get("source_series_id", ""),
        "target_start": row.get("target_start"),
        "target_end": row.get("target_end"),
        "target_length": row.get("target_length"),
        "target_dataset": row.get("target_dataset"),
        "target_event_group": row.get("target_event_group", ""),
        "source_dataset": row.get("source_dataset"),
        "same_dataset": row.get("same_dataset"),
        "donor_policy": row.get("donor_policy"),
        "synthesis_method": row.get("synthesis_method"),
        "filter_method": row.get("filter_method"),
        **compat,
        "compatibility_enabled": bool(compat["compatibility_enabled"]),
        "donor_similarity": _nan_if_missing(row.get("donor_similarity_mean")),
        "reconstruction_consistency": components.get("reconstruction_consistency_confidence"),
        "donor_agreement": components.get("donor_agreement_confidence"),
        "filter_amplitude_compatibility": components.get("amplitude_compatibility_score"),
        "final_confidence": components.get("aggregate_confidence"),
        "calibration_method": row.get("amplitude_calibration"),
        "scale": row.get("scale"),
        "shift": row.get("shift"),
        "boundary_continuity_error": row.get("boundary_continuity_error"),
        "accepted": bool(row.get("accepted", row.get("kept", False))),
        "rejection_reason": normalize_rejection_reason(row.get("rejection_reason"), rejection_stage=str(rejection_stage)),
        "rejection_stage": rejection_stage,
        "donor_series_ids": "|".join(str(value) for value in donor_ids),
        "donor_starts": "|".join(str(value) for value in row.get("donor_starts", [])),
        "donor_ends": "|".join(str(value) for value in row.get("donor_ends", [])),
        "donor_lengths": "|".join(str(value) for value in row.get("donor_lengths", [])),
        "donor_event_groups": "|".join(str(value) for value in row.get("donor_event_groups", [])),
        "donor_similarity_mean": row.get("donor_similarity_mean"),
        "donor_similarity_min": row.get("donor_similarity_min"),
        "donor_similarity_max": row.get("donor_similarity_max"),
    }
    return flat


def summarize_rejections_enhanced(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build rejection_summary.json with normalized count keys."""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in audit_rows:
        by_method.setdefault(str(row.get("method", "unknown")), []).append(row)

    summary: dict[str, Any] = {}
    for method, rows in by_method.items():
        donor_policies = sorted({str(row.get("donor_policy")) for row in rows if row.get("donor_policy")})
        counts = {key: 0 for key in NORMALIZED_REJECTION_COUNTS}
        raw_reasons: dict[str, int] = {}
        for row in rows:
            key = normalized_reason_count_key(row)
            counts[key] = counts.get(key, 0) + 1
            raw = str(row.get("rejection_reason") or "kept")
            raw_reasons[raw] = raw_reasons.get(raw, 0) + 1
        entry = {
            "donor_policy": donor_policies[0] if len(donor_policies) == 1 else donor_policies,
            "num_candidates": len(rows),
            "num_kept": counts.get("kept", 0),
            "num_rejected": len(rows) - counts.get("kept", 0),
            "counts": counts,
            "reasons": raw_reasons,
        }
        summary[method] = entry
    return summary
