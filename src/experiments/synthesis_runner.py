"""Shared synthesis experiment helpers."""

from __future__ import annotations

from typing import Any

from src.synthesis.augment_dataset import flatten_audit_row
from src.synthesis.uncertainty_filter import resolve_filter_policy, summarize_rejections_by_method, warn_if_strict_filter_empty

CONTEXT_CALIBRATED_METHODS = {
    "normalized_time_context_calibrated": "normalized_time_mean_donor",
    "dtw_context_calibrated": "dtw_aligned_donor",
    "learned_prototype_context_calibrated": "learned_prototype_event_time",
}


def resolve_context_calibrated_method(method_name: str, synthesis_method: str) -> tuple[str, bool]:
    """Map context-calibrated ablation names to their base synthesis method."""
    if method_name in CONTEXT_CALIBRATED_METHODS:
        return CONTEXT_CALIBRATED_METHODS[method_name], True
    if synthesis_method in CONTEXT_CALIBRATED_METHODS:
        return CONTEXT_CALIBRATED_METHODS[synthesis_method], True
    return synthesis_method, False


def synthesis_policy_config(
    config: dict[str, Any],
    *,
    synthesis_method: str,
    method_name: str,
    labeled_fraction: float,
    filter_policy: str | dict[str, Any] | None = None,
    donor_policy: str | None = None,
) -> dict[str, Any]:
    synthesis_cfg = config.get("synthesis", {})
    top_k = int(config.get("experiment", {}).get("top_k_donors", synthesis_cfg.get("top_k", 3)))
    grid_size = int(config.get("experiment", {}).get("alignment_grid_size", synthesis_cfg.get("alignment_grid_size", 64)))
    resolved_filter = (
        resolve_filter_policy(synthesis_cfg, override=filter_policy)
        if filter_policy is not None
        else resolve_filter_policy(synthesis_cfg)
    )
    base_method, is_context_calibrated = resolve_context_calibrated_method(method_name, synthesis_method)
    amplitude_calibration = str(synthesis_cfg.get("amplitude_calibration", "none"))
    if is_context_calibrated:
        amplitude_calibration = str(
            synthesis_cfg.get("context_amplitude_calibration", "robust_scale_shift")
        )
    policy: dict[str, Any] = {
        "method": base_method,
        "method_name": method_name,
        "labeled_fraction": labeled_fraction,
        "top_k": top_k,
        "grid_size": grid_size,
        "max_donors_per_source_series": 1,
        "avoid_single_series_dominance": True,
        "filter_policy": resolved_filter,
        "amplitude_calibration": amplitude_calibration,
        "partial_target_mode": str(synthesis_cfg.get("partial_target_synthesis", "full_replacement")),
        "context_size": int(synthesis_cfg.get("context_size", 5)),
        "match_partial_amplitude": bool(synthesis_cfg.get("match_partial_amplitude", True)),
        "donor_policy": donor_policy or synthesis_cfg.get("donor_policy"),
    }
    # Global synthesis.compatibility.enabled=false must not override donor-policy presets.
    if donor_policy is None:
        policy["compatibility"] = dict(synthesis_cfg.get("compatibility", {}))
    return policy


def append_audit_rows(
    synthetic_audit: list[dict[str, Any]],
    *,
    method_name: str,
    audit_rows: list[dict[str, Any]],
    filter_policy: dict[str, Any],
) -> None:
    warn_if_strict_filter_empty(method_name, audit_rows, filter_policy)
    synthetic_audit.extend(flatten_audit_row(row) for row in audit_rows)


def build_rejection_summary(synthetic_audit: list[dict[str, Any]]) -> dict[str, Any]:
    return summarize_rejections_by_method(synthetic_audit)
