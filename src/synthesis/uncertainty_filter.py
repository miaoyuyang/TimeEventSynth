"""Confidence scoring and filtering for synthetic event windows.

Synthetic event windows can poison downstream detectors when donor retrieval or
alignment is poor. This module makes the acceptance policy explicit so we can
audit which synthetic windows are kept, rejected, and why.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from ..alignment.normalized_time import normalize_event_window


def _clip01(value: float) -> float:
    return float(min(max(value, 0.0), 1.0))


_QUALITY_FILTER_SKIP_RECORD_TYPES = frozenset({"donor_rejection", "donor_pair"})


def _quality_filterable_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Synthetic-window rows eligible for confidence-based quality filtering."""
    return [row for row in rows if row.get("record_type") not in _QUALITY_FILTER_SKIP_RECORD_TYPES]


def _aggregate_confidence(row: dict[str, Any]) -> float:
    components = row.get("confidence_components") or {}
    return float(components.get("aggregate_confidence", 0.0))


def donor_similarity_confidence(similarities: list[float]) -> float:
    """Map donor retrieval similarities into a [0, 1] confidence."""
    if not similarities:
        return 0.0
    sims = np.asarray(similarities, dtype=float)
    return _clip01(float(np.mean((sims + 1.0) / 2.0)))


def reconstruction_consistency_confidence(
    synthetic_values: np.ndarray,
    donor_values: list[np.ndarray],
    grid_size: int = 64,
) -> float:
    """Score whether the synthetic window is close to the donor consensus shape."""
    if not donor_values:
        return 0.0
    synthetic_norm = np.asarray(normalize_event_window(synthetic_values, grid_size=grid_size), dtype=float).reshape(grid_size, -1)
    donor_norms = [
        np.asarray(normalize_event_window(values, grid_size=grid_size), dtype=float).reshape(grid_size, -1)
        for values in donor_values
    ]
    donor_mean = np.mean(np.stack(donor_norms, axis=0), axis=0)
    mse = float(np.mean((synthetic_norm - donor_mean) ** 2))
    return _clip01(float(np.exp(-mse)))


def donor_agreement_confidence(donor_values: list[np.ndarray], grid_size: int = 64) -> float:
    """Score whether multiple donors agree on the event pattern."""
    if not donor_values:
        return 0.0
    if len(donor_values) == 1:
        return 1.0
    donor_norms = [
        np.asarray(normalize_event_window(values, grid_size=grid_size), dtype=float).reshape(-1)
        for values in donor_values
    ]
    pairwise: list[float] = []
    for idx in range(len(donor_norms)):
        for jdx in range(idx + 1, len(donor_norms)):
            x = donor_norms[idx]
            y = donor_norms[jdx]
            denom = max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-8)
            pairwise.append(float(np.dot(x, y) / denom))
    if not pairwise:
        return 1.0
    sims = np.asarray(pairwise, dtype=float)
    return _clip01(float(np.mean((sims + 1.0) / 2.0)))


def amplitude_compatibility_score(target_values: np.ndarray, synthetic_values: np.ndarray) -> float:
    """Score whether the synthetic event amplitude matches the target context scale."""
    target = np.asarray(target_values, dtype=float).reshape(len(target_values), -1)
    synthetic = np.asarray(synthetic_values, dtype=float).reshape(len(synthetic_values), -1)
    target_mean = float(np.mean(target))
    synthetic_mean = float(np.mean(synthetic))
    target_std = float(np.std(target))
    synthetic_std = float(np.std(synthetic))
    mean_gap = abs(target_mean - synthetic_mean)
    std_gap = abs(target_std - synthetic_std)
    scale = max(target_std, synthetic_std, 1e-6)
    score = np.exp(-(mean_gap + std_gap) / scale)
    return _clip01(float(score))


def score_synthetic_window(
    target_values: np.ndarray,
    synthetic_values: np.ndarray,
    donor_values: list[np.ndarray],
    donor_similarities: list[float],
    grid_size: int = 64,
) -> dict[str, float]:
    """Compute confidence components and an aggregate score."""
    components = {
        "donor_similarity_confidence": donor_similarity_confidence(donor_similarities),
        "reconstruction_consistency_confidence": reconstruction_consistency_confidence(
            synthetic_values,
            donor_values,
            grid_size=grid_size,
        ),
        "donor_agreement_confidence": donor_agreement_confidence(donor_values, grid_size=grid_size),
        "amplitude_compatibility_score": amplitude_compatibility_score(target_values, synthetic_values),
    }
    components["aggregate_confidence"] = float(np.mean(list(components.values())))
    return components


def overlay_confidence_prior(
    confidence_components: dict[str, float],
    *,
    donor_count: int,
    synthesis_method: str,
    diversity_status: str | None = None,
) -> dict[str, float]:
    """Apply a generic prior-style overlay to confidence components."""
    donor_count = max(int(donor_count), 0)
    method_weight = {
        "normalized_time_mean_donor": 0.85,
        "dtw_aligned_donor": 0.80,
        "learned_prototype_event_time": 0.90,
    }.get(str(synthesis_method), 0.75)
    donor_weight = min(1.0, 0.6 + 0.1 * donor_count)
    diversity_weight = 0.85 if diversity_status == "single_series_only" else 1.0
    overlaid = dict(confidence_components)
    overlaid["prior_weight"] = float(method_weight * donor_weight * diversity_weight)
    overlaid["prior_adjusted_confidence"] = float(
        confidence_components.get("aggregate_confidence", 0.0) * overlaid["prior_weight"]
    )
    return overlaid


def resolve_filter_policy(synthesis_cfg: dict[str, Any], override: str | dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve a filtering policy dict from synthesis config."""
    if isinstance(override, dict):
        return dict(override)
    policy_name = str(override or synthesis_cfg.get("filter_policy", "none")).lower()
    if policy_name in {"none", "no_filter"}:
        return {"name": "no_filter"}
    if policy_name == "strict":
        thresholds = dict(synthesis_cfg.get("strict_thresholds", {}))
        return {
            "name": "strict",
            "min_donor_similarity_confidence": float(thresholds.get("donor_similarity", 0.5)),
            "min_donor_agreement_confidence": float(thresholds.get("donor_agreement", 0.5)),
            "min_amplitude_compatibility_score": float(thresholds.get("amplitude_compatibility", 0.5)),
            "min_reconstruction_consistency_confidence": float(
                thresholds.get("reconstruction_consistency", thresholds.get("donor_similarity", 0.5))
            ),
            "min_confidence": float(synthesis_cfg.get("min_confidence", 0.5)),
        }
    if policy_name == "top_quantile":
        return {
            "name": "top_quantile",
            "quantile": float(synthesis_cfg.get("confidence_quantile", 0.5)),
        }
    if policy_name == "min_confidence":
        return {"name": "min_confidence", "min_confidence": float(synthesis_cfg.get("min_confidence", 0.5))}
    raise ValueError(f"Unsupported synthesis filter policy: {policy_name}")


def apply_filter_policy(
    audit_rows: list[dict[str, Any]],
    policy: dict[str, Any] | str | None,
    *,
    synthesis_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply an explicit filtering policy and annotate keep/reject decisions."""
    if not audit_rows:
        return []
    if isinstance(policy, str):
        policy = resolve_filter_policy(synthesis_cfg or {}, override=policy)
    policy = dict(policy or {"name": "no_filter"})
    name = str(policy.get("name", "no_filter"))
    rows = [dict(row) for row in audit_rows]

    if name in {"no_filter", "none"}:
        for row in rows:
            if row.get("record_type") in _QUALITY_FILTER_SKIP_RECORD_TYPES:
                continue
            row["kept"] = True
            row["accepted"] = True
            row["rejection_reason"] = None
            row["rejection_stage"] = "kept"
        return rows

    if name == "top_quantile":
        quantile = float(policy.get("quantile", synthesis_cfg.get("confidence_quantile", 0.5) if synthesis_cfg else 0.5))
        synth_rows = _quality_filterable_rows(rows)
        scores = np.asarray([_aggregate_confidence(row) for row in synth_rows], dtype=float)
        cutoff = float(np.quantile(scores, max(min(quantile, 1.0), 0.0))) if scores.size else 0.0
        for row in rows:
            if row.get("record_type") in _QUALITY_FILTER_SKIP_RECORD_TYPES:
                continue
            keep = _aggregate_confidence(row) >= cutoff
            row["kept"] = bool(keep)
            row["accepted"] = bool(keep)
            row["rejection_reason"] = None if keep else "below_top_quantile_0.5"
            row["rejection_stage"] = "kept" if keep else "quality_filter"
        return rows

    if name == "min_confidence":
        threshold = float(policy.get("min_confidence", synthesis_cfg.get("min_confidence", 0.5) if synthesis_cfg else 0.5))
        for row in rows:
            if row.get("record_type") in _QUALITY_FILTER_SKIP_RECORD_TYPES:
                continue
            keep = _aggregate_confidence(row) >= threshold
            row["kept"] = bool(keep)
            row["accepted"] = bool(keep)
            row["rejection_reason"] = None if keep else f"below_quality_threshold"
            row["rejection_stage"] = "kept" if keep else "quality_filter"
        return rows

    if name == "strict":
        min_thresholds = {
            "donor_similarity_confidence": float(
                policy.get("min_donor_similarity_confidence", policy.get("donor_similarity", 0.5))
            ),
            "reconstruction_consistency_confidence": float(
                policy.get(
                    "min_reconstruction_consistency_confidence",
                    policy.get("reconstruction_consistency", 0.5),
                )
            ),
            "donor_agreement_confidence": float(
                policy.get("min_donor_agreement_confidence", policy.get("donor_agreement", 0.5))
            ),
            "amplitude_compatibility_score": float(
                policy.get("min_amplitude_compatibility_score", policy.get("amplitude_compatibility", 0.5))
            ),
            "aggregate_confidence": float(policy.get("min_confidence", 0.5)),
        }
        for row in rows:
            if row.get("record_type") in _QUALITY_FILTER_SKIP_RECORD_TYPES:
                continue
            failed = [
                key
                for key, threshold in min_thresholds.items()
                if float((row.get("confidence_components") or {}).get(key, 0.0)) < threshold
            ]
            keep = len(failed) == 0
            row["kept"] = bool(keep)
            row["accepted"] = bool(keep)
            row["rejection_reason"] = None if keep else "strict_filter_failed:" + ",".join(failed)
            row["rejection_stage"] = "kept" if keep else "strict_filter"
        return rows

    raise ValueError(f"Unsupported filtering policy: {name}")


def summarize_rejections(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize candidate counts and rejection reasons for one method."""
    reasons = Counter(str(row.get("rejection_reason") or "kept") for row in audit_rows)
    return {
        "num_candidates": len(audit_rows),
        "num_kept": sum(1 for row in audit_rows if bool(row.get("accepted", row.get("kept", False)))),
        "num_rejected": sum(1 for row in audit_rows if not bool(row.get("accepted", row.get("kept", False)))),
        "reasons": dict(reasons),
    }


def summarize_rejections_by_method(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build rejection_summary.json payload grouped by synthesis method."""
    from .synthetic_audit import summarize_rejections_enhanced

    return summarize_rejections_enhanced(audit_rows)


def warn_if_strict_filter_empty(method_name: str, audit_rows: list[dict[str, Any]], policy: dict[str, Any]) -> None:
    """Print a non-silent warning when a strict filter accepts zero windows."""
    if str(policy.get("name")) != "strict":
        return
    kept = sum(1 for row in audit_rows if bool(row.get("accepted", row.get("kept", False))))
    if kept == 0 and audit_rows:
        print(
            f"WARNING: strict filter accepted 0 synthetic windows for method={method_name}; "
            "see synthetic_audit.csv and rejection_summary.json."
        )


def keep_if_confident(synthetic_event: dict[str, Any] | None, min_support: float = 0.0) -> bool:
    """Backward-compatible confidence gate for older scaffold code paths."""
    if synthetic_event is None:
        return False
    if "confidence_components" in synthetic_event:
        score = float(synthetic_event["confidence_components"].get("aggregate_confidence", 0.0))
    elif "metadata" in synthetic_event and isinstance(synthetic_event["metadata"], dict):
        score = float(synthetic_event["metadata"].get("confidence", synthetic_event.get("support_score", 0.0)))
    else:
        score = float(synthetic_event.get("support_score", 0.0))
    return bool(score >= float(min_support))
