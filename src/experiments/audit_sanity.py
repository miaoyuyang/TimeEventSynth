"""Sanity checks for experiment artifacts and suite manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class AuditSanityError(RuntimeError):
    """Raised when required audit artifacts fail validation."""


def validate_synthetic_audit_csv(
    audit_rows: list[dict[str, Any]],
    *,
    config: dict[str, Any],
    synthesis_requested: bool = True,
) -> None:
    """Ensure synthetic audit exists and contains compatibility columns when enabled."""
    if synthesis_requested and not audit_rows:
        raise AuditSanityError("synthesis was requested but synthetic_audit is empty")

    if not audit_rows:
        return

    compatibility_cfg = config.get("synthesis", {}).get("compatibility", {}) or {}
    enabled = bool(compatibility_cfg.get("enabled", False))
    required_compat_cols = {
        "compatibility_enabled",
        "compatibility_score",
        "compatibility_decision",
    }
    sample = audit_rows[0]
    missing = [col for col in required_compat_cols if col not in sample]
    if missing:
        raise AuditSanityError(f"synthetic_audit missing required columns: {missing}")

    if enabled:
        has_compat_signal = any(
            bool(row.get("compatibility_enabled"))
            or str(row.get("compatibility_decision", "")) not in {"", "not_applied", "nan"}
            for row in audit_rows
        )
        if not has_compat_signal:
            raise AuditSanityError(
                "compatibility.enabled=true but no compatibility fields appear in synthetic_audit"
            )


def validate_suite_manifest(
    manifest: dict[str, Any],
    *,
    streams_requested: list[str],
    fail_on_missing_stream: bool = False,
) -> list[str]:
    """Warn or fail when requested suite streams are missing from the manifest."""
    warnings: list[str] = []
    profile_runs = manifest.get("profiles", {})
    completed_streams: set[str] = set()
    for profile_entry in profile_runs.values():
        for run in profile_entry.get("runs", []):
            if run.get("success"):
                completed_streams.add(str(run.get("stream")))

    for stream in streams_requested:
        if stream not in completed_streams:
            message = f"suite stream missing or failed: {stream}"
            if fail_on_missing_stream:
                raise AuditSanityError(message)
            warnings.append(message)
    return warnings


def validate_masked_completion_outputs(metrics: dict[str, Any]) -> None:
    """Fail loudly when masked completion produced no evaluations."""
    num_events = int(metrics.get("num_test_events", 0))
    num_evaluations = int(metrics.get("num_evaluations", 0))
    if num_events == 0:
        raise AuditSanityError("masked completion: num_test_events == 0")
    if num_evaluations == 0:
        raise AuditSanityError("masked completion: num_evaluations == 0")


def annotate_invalid_point_metrics(
    y_true: list[int] | np.ndarray,
    y_score: list[float] | np.ndarray,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """Record explicit invalidity reasons for degenerate validation/test scores."""
    labels = np.asarray(y_true, dtype=int).reshape(-1)
    scores = np.asarray(y_score, dtype=float).reshape(-1)
    updated = dict(metrics)
    if labels.size == 0:
        updated["metrics_valid"] = False
        updated["metrics_invalid_reason"] = "empty_labels"
        return updated
    if len(np.unique(labels)) < 2:
        updated["metrics_valid"] = False
        updated["metrics_invalid_reason"] = "y_true_single_class"
        return updated
    if scores.size and float(np.std(scores)) <= 1e-12:
        updated["metrics_valid"] = False
        updated["metrics_invalid_reason"] = "y_score_constant"
        return updated
    updated["metrics_valid"] = True
    updated["metrics_invalid_reason"] = ""
    return updated


def load_and_validate_audit_csv(path: Path, *, config: dict[str, Any], synthesis_requested: bool = True) -> pd.DataFrame:
    if not path.exists():
        if synthesis_requested:
            raise AuditSanityError(f"synthetic_audit.csv not found: {path}")
        return pd.DataFrame()
    frame = pd.read_csv(path)
    validate_synthetic_audit_csv(frame.to_dict(orient="records"), config=config, synthesis_requested=synthesis_requested)
    return frame


def verify_compatibility_donor_invariant(
    audit_rows: list[dict[str, Any]],
    *,
    method_names: list[str] | None = None,
    min_score: float = 0.5,
) -> list[str]:
    """Verify accepted synthesis never uses donors rejected for the same target timeline.

    Returns human-readable violation messages (empty if invariant holds).
    """
    violations: list[str] = []
    methods = method_names or sorted({str(row.get("method")) for row in audit_rows if row.get("compatibility_enabled")})

    for method in methods:
        method_rows = [row for row in audit_rows if str(row.get("method")) == method]
        if not method_rows:
            continue
        accepted = [
            row
            for row in method_rows
            if bool(row.get("accepted", row.get("kept", False)))
            and str(row.get("rejection_stage", "")) != "compatibility"
        ]
        for row in accepted:
            target = str(row.get("target_series_id", "") or "")
            if not target or target == "nan":
                continue
            donors = [
                part
                for part in str(row.get("donor_series_ids", "") or "").split("|")
                if part and part not in {"nan", "None"}
            ]
            agg = row.get("compatibility_score")
            if agg is not None and agg == agg and float(agg) < float(min_score) - 1e-9:  # noqa: PLW0127
                violations.append(
                    f"{method}: candidate {row.get('candidate_id')} aggregate compatibility_score={agg} < min_score={min_score}"
                )
            rejected_for_target = {
                str(r.get("source_series_id"))
                for r in method_rows
                if str(r.get("target_series_id", "")) == target
                and str(r.get("rejection_stage")) == "compatibility"
                and not bool(r.get("accepted", r.get("kept", False)))
            }
            overlap = set(donors) & rejected_for_target
            if overlap:
                violations.append(
                    f"{method}: target={target} candidate={row.get('candidate_id')} used compatibility-rejected donors {sorted(overlap)}"
                )
    return violations


def validate_compatibility_transfer_outputs(
    *,
    comparison_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    experiments: list[dict[str, Any]],
    allow_no_donor_rejections: bool = False,
    min_compatibility_score: float = 0.5,
) -> None:
    """Sanity checks specific to compatibility-transfer experiments."""
    if not comparison_rows:
        raise AuditSanityError("compatibility transfer produced empty comparison metrics")

    compat_methods: list[str] = []
    for spec in experiments:
        if spec.get("kind") != "synthetic":
            continue
        policy = str(spec.get("donor_policy", ""))
        if policy in {"cross_dataset_compatible", "compatibility_top50", "compatibility_strict"}:
            compat_methods.append(str(spec["name"]))

    if not compat_methods:
        return

    if not audit_rows:
        raise AuditSanityError("compatibility-enabled policies require synthetic_audit rows")

    sample = audit_rows[0]
    for col in ("compatibility_enabled", "compatibility_score", "compatibility_decision", "donor_policy"):
        if col not in sample:
            raise AuditSanityError(f"synthetic_audit missing compatibility column: {col}")

    for method_name in compat_methods:
        method_rows = [row for row in audit_rows if str(row.get("method")) == method_name]
        if not method_rows:
            raise AuditSanityError(f"no audit rows for compatibility-enabled method: {method_name}")
        has_compat = any(
            bool(row.get("compatibility_enabled"))
            or str(row.get("compatibility_decision", "")) not in {"", "not_applied", "nan", "NaN"}
            for row in method_rows
        )
        if not has_compat:
            raise AuditSanityError(
                f"compatibility-enabled policy {method_name} produced no compatibility audit fields"
            )
        if allow_no_donor_rejections:
            continue
        rejected = [
            row
            for row in method_rows
            if str(row.get("rejection_stage")) == "compatibility"
            and not bool(row.get("accepted", row.get("kept", False)))
        ]
        if not rejected:
            raise AuditSanityError(
                f"compatibility-enabled policy {method_name} kept all donors and rejected none; "
                "set compatibility_transfer.allow_no_donor_rejections=true to allow"
            )

    min_score = float(min_compatibility_score)
    invariant_violations = verify_compatibility_donor_invariant(
        audit_rows, method_names=compat_methods, min_score=min_score
    )
    if invariant_violations:
        raise AuditSanityError("compatibility donor invariant violated:\n" + "\n".join(invariant_violations[:20]))
