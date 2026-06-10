"""Event-pattern-specific synthesis policy helpers.

This ports the reusable selection-policy idea from the crisis project while
renaming the abstraction:

- label -> event pattern
- rare label -> rare event window / rare event pattern
- donor event -> donor window

The current benchmark mostly uses one anomaly pattern, so these functions are a
generic scaffold for future multi-pattern datasets.
"""

from __future__ import annotations

from collections import Counter
from typing import Any


POLICY_MODES = frozenset(
    {
        "train_high_confidence",
        "prior_only",
        "diagnostic_only",
        "disabled",
    }
)

DEFAULT_POLICY_THRESHOLDS = {
    "min_high_confidence_windows": 5,
    "min_unique_donor_series": 2,
    "max_top_donor_fraction": 0.6,
}


def compute_event_pattern_pool_diagnostics(
    synthetic_pool: list[dict[str, Any]],
    *,
    pattern_keys: list[str],
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for pattern_key in pattern_keys:
        rows = [row for row in synthetic_pool if str(row.get("event_pattern", "anomaly")) == pattern_key]
        donor_counter: Counter[str] = Counter()
        high_confidence = 0
        moderate_confidence = 0
        for row in rows:
            confidence = float(row.get("confidence_components", {}).get("aggregate_confidence", 0.0))
            if confidence >= 0.75:
                high_confidence += 1
            elif confidence >= 0.50:
                moderate_confidence += 1
            for donor_id in row.get("donor_ids", []):
                donor_series = str(donor_id).split(":")[0]
                if donor_series:
                    donor_counter[donor_series] += 1
        total = len(rows)
        top_donor_fraction = 0.0
        if donor_counter and total:
            top_donor_fraction = donor_counter.most_common(1)[0][1] / max(total, 1)
        diagnostics[pattern_key] = {
            "num_synthetic_total": total,
            "num_high_confidence": high_confidence,
            "num_moderate_confidence": moderate_confidence,
            "num_unique_donor_series": len(donor_counter),
            "top_donor_fraction": float(top_donor_fraction),
            "top_donor_series_id": donor_counter.most_common(1)[0][0] if donor_counter else "",
        }
    return diagnostics


def select_event_pattern_policy_mode(
    diag: dict[str, Any],
    *,
    min_high_confidence_windows: int = 5,
    min_unique_donor_series: int = 2,
    max_top_donor_fraction: float = 0.6,
) -> tuple[str, str]:
    num_high_confidence = int(diag.get("num_high_confidence", 0))
    num_moderate_confidence = int(diag.get("num_moderate_confidence", 0))
    num_donors = int(diag.get("num_unique_donor_series", 0))
    top_frac = float(diag.get("top_donor_fraction", 1.0))

    if (
        num_high_confidence >= min_high_confidence_windows
        and num_donors >= min_unique_donor_series
        and top_frac <= max_top_donor_fraction
    ):
        return "train_high_confidence", "enough_confident_diverse_windows"
    if num_high_confidence > 0 or num_moderate_confidence > 0:
        return "prior_only", "some_synthetic_signal_but_not_diverse_enough"
    if int(diag.get("num_synthetic_total", 0)) > 0:
        return "diagnostic_only", "only_low_confidence_windows_available"
    return "disabled", "no_synthetic_windows_available"


def build_event_pattern_policy(
    synthetic_pool: list[dict[str, Any]],
    *,
    pattern_keys: list[str],
    thresholds: dict[str, Any] | None = None,
) -> dict[str, Any]:
    cfg = {**DEFAULT_POLICY_THRESHOLDS, **(thresholds or {})}
    diagnostics = compute_event_pattern_pool_diagnostics(synthetic_pool, pattern_keys=pattern_keys)
    policy: dict[str, Any] = {}
    for pattern_key in pattern_keys:
        mode, reason = select_event_pattern_policy_mode(
            diagnostics[pattern_key],
            min_high_confidence_windows=int(cfg["min_high_confidence_windows"]),
            min_unique_donor_series=int(cfg["min_unique_donor_series"]),
            max_top_donor_fraction=float(cfg["max_top_donor_fraction"]),
        )
        policy[pattern_key] = {
            "mode": mode,
            "selection_reason": reason,
            "diagnostics": diagnostics[pattern_key],
        }
    return {
        "event_pattern_policy": policy,
        "selection_thresholds": cfg,
    }
