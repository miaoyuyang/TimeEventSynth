"""Experiment specs for compatibility-aware donor-policy transfer studies."""

from __future__ import annotations

from typing import Any

from src.synthesis.donor_selection import DONOR_POLICY_PRESETS

COMPATIBILITY_ENABLED_POLICIES = frozenset(
    {
        "cross_dataset_compatible",
        "compatibility_top50",
        "compatibility_strict",
    }
)


def compatibility_transfer_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config.get("compatibility_transfer", {}))


def build_compatibility_transfer_specs(
    config: dict[str, Any],
    *,
    smoke: bool = False,
) -> list[dict[str, Any]]:
    """Return method specs for the compatibility-transfer experiment."""
    transfer_cfg = compatibility_transfer_cfg(config)
    if smoke:
        experiments = list(transfer_cfg.get("smoke", {}).get("experiments", []))
        if not experiments:
            experiments = list(transfer_cfg.get("experiments", []))[:5]
    else:
        experiments = list(transfer_cfg.get("experiments", []))
    if not experiments:
        raise ValueError("compatibility_transfer.experiments is empty")
    specs: list[dict[str, Any]] = []
    for entry in experiments:
        if isinstance(entry, str):
            raise ValueError(f"compatibility_transfer experiment entry must be a mapping, got: {entry!r}")
        spec = dict(entry)
        spec["name"] = str(spec["name"])
        spec["kind"] = str(spec.get("kind", "synthetic"))
        if spec["kind"] == "synthetic":
            spec.setdefault("synthesis_method", "learned_prototype_event_time")
            spec.setdefault("filter_policy", "none")
            spec.setdefault("donor_policy", "all_donors_no_filter")
        specs.append(spec)
    return specs


def donor_policy_compatibility_enabled(donor_policy: str | None) -> bool:
    if not donor_policy:
        return False
    preset = DONOR_POLICY_PRESETS.get(str(donor_policy), {})
    return bool(preset.get("enabled", False))


def experiment_compatibility_enabled(spec: dict[str, Any]) -> bool:
    if spec.get("kind") != "synthetic":
        return False
    if spec.get("compatibility") is not None:
        return bool(spec["compatibility"].get("enabled", False))
    return donor_policy_compatibility_enabled(spec.get("donor_policy"))
