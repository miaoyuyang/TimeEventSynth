"""Detector-agnostic augmentation policies.

These policies expose the synthesis layer as a reusable transfer module. The
downstream detector backbone can then decide whether synthetic event windows are
used for training augmentation, threshold calibration, or only audit analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..datasets.tsb_loader import TimeSeriesRecord
from ..experiments.synthesis_runner import build_rejection_summary, synthesis_policy_config
from ..synthesis.augment_dataset import build_augmented_training_records_with_audit, records_from_oversampled_events
from ..synthesis.synthetic_audit import flatten_audit_row

COMPATIBILITY_POLICIES = frozenset(
    {
        "cross_dataset_compatible",
        "compatibility_top50",
        "compatibility_strict",
        "groupwise_cross_dataset_compatible",
        "groupwise_compatibility_strict",
        "adaptive_groupwise_transfer",
    }
)

ADAPTIVE_POLICIES = frozenset({"adaptive_groupwise_transfer"})

POLICY_TO_DONOR_PRESET: dict[str, str | None] = {
    "real_only": None,
    "random_event_oversampling": None,
    "all_donors_no_filter": "all_donors_no_filter",
    "same_dataset_only": "same_dataset_only",
    "cross_dataset_all": "cross_dataset_all",
    "cross_dataset_compatible": "cross_dataset_compatible",
    "compatibility_top50": "compatibility_top50",
    "compatibility_strict": "compatibility_strict",
    "groupwise_cross_dataset_all": "cross_dataset_all",
    "groupwise_cross_dataset_compatible": "cross_dataset_compatible",
    "groupwise_compatibility_strict": "compatibility_strict",
    "adaptive_groupwise_transfer": None,
}


@dataclass
class AugmentationResult:
    """Outputs of an augmentation policy."""

    policy_name: str
    synthetic_windows: list[TimeSeriesRecord] = field(default_factory=list)
    synthetic_labels: list[np.ndarray] = field(default_factory=list)
    synthetic_metadata: list[dict[str, Any]] = field(default_factory=list)
    audit_records: list[dict[str, Any]] = field(default_factory=list)
    rejection_summary: dict[str, Any] = field(default_factory=dict)
    compatibility_summary: dict[str, Any] = field(default_factory=dict)
    synthesis_method: str = "none"
    compatibility_enabled: bool = False
    donor_policy: str | None = None
    filter_policy: str | None = None
    selected_policy_name: str | None = None
    selection_reason: str | None = None
    fallback_used: bool = False


def is_adaptive_policy(policy_name: str) -> bool:
    return str(policy_name) in ADAPTIVE_POLICIES


def _normalize_backbone_name(name: str | None) -> str:
    return str(name or "").strip().lower()


def _adaptive_policy_overrides(
    detector_backbone: str | None,
    *,
    config: dict[str, Any],
    policy_config: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    adaptive_cfg = dict(config.get("augmentation", {}).get("adaptive_groupwise", {}))
    adaptive_cfg.update(dict(policy_config.get("adaptive_groupwise", {})))

    density_backbones = {
        _normalize_backbone_name(name)
        for name in adaptive_cfg.get("density_backbones", ["iforest", "lof", "ocsvm"])
    }
    broad_backbones = {
        _normalize_backbone_name(name)
        for name in adaptive_cfg.get(
            "broad_backbones",
            ["internal_classifier", "current_internal_classifier", "cnn", "timesnet", "autoencoder"],
        )
    }
    detector_name = _normalize_backbone_name(detector_backbone)
    family = "default"
    if detector_name in density_backbones:
        family = "density"
    elif detector_name in broad_backbones:
        family = "broad"

    primary_by_family = {
        "density": "groupwise_cross_dataset_compatible",
        "broad": "groupwise_cross_dataset_all",
        "default": "groupwise_cross_dataset_compatible",
    }
    primary_by_family.update(dict(adaptive_cfg.get("primary_policy_by_family", {})))
    fallback_by_family = {
        "density": "groupwise_cross_dataset_all",
        "broad": "groupwise_cross_dataset_compatible",
        "default": "groupwise_cross_dataset_all",
    }
    fallback_by_family.update(dict(adaptive_cfg.get("fallback_policy_by_family", {})))

    selected = str(primary_by_family.get(family, primary_by_family["default"]))
    fallback = str(fallback_by_family.get(family, fallback_by_family["default"]))
    overrides = {
        "selected_policy_name": selected,
        "fallback_policy_name": fallback,
        "family": family,
        "min_synthetic_windows": int(adaptive_cfg.get("min_synthetic_windows", 1)),
    }
    return overrides, f"detector_family:{family}"


def _records_from_synthetic_rows(rows: list[dict[str, Any]]) -> list[TimeSeriesRecord]:
    outputs: list[TimeSeriesRecord] = []
    for row in rows:
        outputs.append(
            TimeSeriesRecord(
                series_id=str(row["series_id"]),
                values=np.asarray(row["values"], dtype=float),
                labels=np.asarray(row["labels"], dtype=int),
                timestamps=None,
                source_path="synthetic",
                metadata=dict(row.get("metadata", {})),
            )
        )
    return outputs


def _compatibility_summary(audit_rows: list[dict[str, Any]], policy_name: str) -> dict[str, Any]:
    donor_rows = [row for row in audit_rows if str(row.get("record_type")) == "donor_pair"]
    candidate_rows = [row for row in audit_rows if str(row.get("record_type")) == "synthesis_candidate"]
    scores = [
        float(row["compatibility_score"])
        for row in donor_rows
        if row.get("compatibility_score") is not None and np.isfinite(float(row["compatibility_score"]))
    ]
    return {
        "policy_name": policy_name,
        "compatibility_enabled": policy_name in COMPATIBILITY_POLICIES,
        "num_donor_pairs": len(donor_rows),
        "num_candidate_windows": len(candidate_rows),
        "num_kept_donor_pairs": sum(1 for row in donor_rows if bool(row.get("accepted", False))),
        "num_rejected_donor_pairs": sum(1 for row in donor_rows if not bool(row.get("accepted", False))),
        "mean_compatibility_score": float(np.mean(scores)) if scores else float("nan"),
        "min_compatibility_score": float(np.min(scores)) if scores else float("nan"),
        "max_compatibility_score": float(np.max(scores)) if scores else float("nan"),
    }


def _flatten_audit(audit_rows: list[dict[str, Any]], policy_name: str) -> list[dict[str, Any]]:
    flattened = [flatten_audit_row(row) for row in audit_rows]
    for row in flattened:
        row["augmentation_policy"] = policy_name
    return flattened


def _build_fixed_augmentation_result(
    train_records: list[TimeSeriesRecord],
    *,
    donor_records: list[TimeSeriesRecord],
    policy_name: str,
    config: dict[str, Any],
    labeled_fraction: float,
    policy_config: dict[str, Any] | None = None,
) -> AugmentationResult:
    policy_config = policy_config or {}

    if policy_name == "real_only":
        return AugmentationResult(
            policy_name=policy_name,
            synthesis_method="none",
            compatibility_enabled=False,
            donor_policy=None,
            filter_policy="no_filter",
            selected_policy_name=policy_name,
            selection_reason="fixed_policy",
        )

    if policy_name == "random_event_oversampling":
        synthetic_windows = records_from_oversampled_events(train_records, config=config)
        return AugmentationResult(
            policy_name=policy_name,
            synthetic_windows=synthetic_windows,
            synthetic_labels=[np.asarray(record.labels, dtype=int) for record in synthetic_windows],
            synthetic_metadata=[dict(record.metadata) for record in synthetic_windows],
            audit_records=[],
            rejection_summary={},
            compatibility_summary={"policy_name": policy_name, "compatibility_enabled": False},
            synthesis_method="random_event_oversampling",
            compatibility_enabled=False,
            donor_policy=None,
            filter_policy="no_filter",
            selected_policy_name=policy_name,
            selection_reason="fixed_policy",
        )

    synthesis_method = str(
        policy_config.get("synthesis_method")
        or config.get("augmentation", {}).get("default_synthesis_method")
        or config.get("synthesis", {}).get("default_synthesis_method")
        or config.get("synthesis", {}).get("methods", ["normalized_time_mean_donor"])[0]
    )
    filter_override = policy_config.get("filter_policy")
    if filter_override is None and policy_name == "compatibility_strict":
        filter_override = "strict"
    elif filter_override is None and policy_name == "groupwise_compatibility_strict":
        filter_override = "strict"
    elif filter_override is None and policy_name == "compatibility_top50":
        filter_override = "top_quantile"
    donor_policy = str(policy_config.get("donor_policy") or POLICY_TO_DONOR_PRESET[policy_name] or "")

    policy_cfg = synthesis_policy_config(
        config,
        synthesis_method=synthesis_method,
        method_name=policy_name,
        labeled_fraction=labeled_fraction,
        filter_policy=filter_override,
        donor_policy=donor_policy or None,
    )
    if policy_name.startswith("groupwise_"):
        policy_cfg["groupwise_matching"] = True
        policy_cfg["group_key"] = "event_group_id"
    kept_rows, audit_rows = build_augmented_training_records_with_audit(
        train_records,
        split="train",
        policy_config=policy_cfg,
        donor_pool_records=donor_records,
        synthesis_cfg=config.get("synthesis", {}),
    )
    synthetic_windows = _records_from_synthetic_rows(kept_rows)
    flat_audit = _flatten_audit(audit_rows, policy_name)

    return AugmentationResult(
        policy_name=policy_name,
        synthetic_windows=synthetic_windows,
        synthetic_labels=[np.asarray(record.labels, dtype=int) for record in synthetic_windows],
        synthetic_metadata=[dict(record.metadata) for record in synthetic_windows],
        audit_records=flat_audit,
        rejection_summary=build_rejection_summary(flat_audit),
        compatibility_summary=_compatibility_summary(flat_audit, policy_name),
        synthesis_method=synthesis_method,
        compatibility_enabled=policy_name in COMPATIBILITY_POLICIES,
        donor_policy=donor_policy or None,
        filter_policy=str(policy_cfg.get("filter_policy", {}).get("name", "no_filter")),
        selected_policy_name=policy_name,
        selection_reason="fixed_policy",
    )


def build_augmentation_result(
    train_records: list[TimeSeriesRecord],
    *,
    donor_records: list[TimeSeriesRecord],
    policy_name: str,
    config: dict[str, Any],
    labeled_fraction: float,
    policy_config: dict[str, Any] | None = None,
    detector_backbone: str | None = None,
) -> AugmentationResult:
    """Run one augmentation policy and return a detector-agnostic result."""
    policy_config = dict(policy_config or {})
    if policy_name not in POLICY_TO_DONOR_PRESET:
        raise ValueError(f"Unsupported augmentation policy: {policy_name}")

    if not is_adaptive_policy(policy_name):
        return _build_fixed_augmentation_result(
            train_records,
            donor_records=donor_records,
            policy_name=policy_name,
            config=config,
            labeled_fraction=labeled_fraction,
            policy_config=policy_config,
        )

    adaptive_overrides, selection_reason = _adaptive_policy_overrides(
        detector_backbone,
        config=config,
        policy_config=policy_config,
    )
    selected_policy_name = str(adaptive_overrides["selected_policy_name"])
    fallback_policy_name = str(adaptive_overrides["fallback_policy_name"])
    min_synthetic_windows = int(adaptive_overrides["min_synthetic_windows"])

    primary = _build_fixed_augmentation_result(
        train_records,
        donor_records=donor_records,
        policy_name=selected_policy_name,
        config=config,
        labeled_fraction=labeled_fraction,
        policy_config=policy_config,
    )
    primary.policy_name = policy_name
    primary.selected_policy_name = selected_policy_name
    primary.selection_reason = selection_reason
    for row in primary.audit_records:
        row["augmentation_policy"] = policy_name
        row["selected_policy_name"] = selected_policy_name
        row["selection_reason"] = selection_reason
        row["fallback_used"] = False

    if len(primary.synthetic_windows) >= min_synthetic_windows or fallback_policy_name == selected_policy_name:
        return primary

    fallback = _build_fixed_augmentation_result(
        train_records,
        donor_records=donor_records,
        policy_name=fallback_policy_name,
        config=config,
        labeled_fraction=labeled_fraction,
        policy_config=policy_config,
    )
    fallback.policy_name = policy_name
    fallback.selected_policy_name = fallback_policy_name
    fallback.selection_reason = (
        f"{selection_reason};fallback:{selected_policy_name}->"
        f"{fallback_policy_name};insufficient_synthetics:{len(primary.synthetic_windows)}<{min_synthetic_windows}"
    )
    fallback.fallback_used = True
    for row in fallback.audit_records:
        row["augmentation_policy"] = policy_name
        row["selected_policy_name"] = fallback_policy_name
        row["selection_reason"] = fallback.selection_reason
        row["fallback_used"] = True
    return fallback
