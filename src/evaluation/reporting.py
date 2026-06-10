"""Reporting utilities for paper-quality result aggregation and low-label diagnostics."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

SUPERVISED_DETECTORS = frozenset(
    {
        "random_forest_window",
        "random_forest",
        "randomforest",
        "logistic_regression_window",
        "logistic_regression",
        "logistic",
    }
)
UNSUPERVISED_DETECTORS = frozenset({"isolation_forest", "zscore"})

UNSUPPORTED_LOW_LABEL_WARNING = (
    "Low-label sweep with unsupervised detector does not test label scarcity; "
    "use this only as an unsupervised augmentation stress test."
)
REAL_ONLY_INVARIANT_WARNING = (
    "real_only invariant across fractions; experiment may not be label-sensitive."
)


def mean_std_table(
    frame: pd.DataFrame,
    group_cols: Sequence[str],
    metric_cols: Sequence[str],
) -> pd.DataFrame:
    """Aggregate metrics into mean/std columns per experimental group."""
    if frame.empty:
        columns = list(group_cols)
        for metric in metric_cols:
            columns.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_count"])
        return pd.DataFrame(columns=columns)
    grouped = frame.groupby(list(group_cols), dropna=False)
    rows: list[dict[str, float | str | int]] = []
    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        for metric in metric_cols:
            values = pd.to_numeric(group[metric], errors="coerce").dropna()
            row[f"{metric}_mean"] = float(values.mean()) if not values.empty else np.nan
            row[f"{metric}_std"] = float(values.std(ddof=0)) if not values.empty else np.nan
            row[f"{metric}_count"] = int(values.shape[0])
        rows.append(row)
    return pd.DataFrame(rows)


def rank_methods(
    frame: pd.DataFrame,
    metric: str,
    higher_is_better: bool = True,
    group_cols: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Rank methods by a chosen metric, optionally within groups."""
    if frame.empty:
        return frame.copy()
    working = frame.copy()
    group_cols = list(group_cols or [])
    ascending = not higher_is_better
    if group_cols:
        working["rank"] = working.groupby(group_cols)[metric].rank(method="dense", ascending=ascending)
    else:
        working["rank"] = working[metric].rank(method="dense", ascending=ascending)
    return working.sort_values(group_cols + ["rank", metric], ascending=[True] * len(group_cols) + [True, not ascending])


def relative_improvement(
    frame: pd.DataFrame,
    baseline_method: str = "real_only",
    metric_cols: Sequence[str] = ("auprc", "event_f1"),
    group_cols: Sequence[str] = ("dataset", "split_seed", "label_fraction", "detector"),
) -> pd.DataFrame:
    """Compute per-row relative improvement over a baseline method."""
    if frame.empty:
        return frame.copy()
    group_cols = list(group_cols)
    metric_cols = list(metric_cols)
    baseline = frame[frame["method"] == baseline_method][group_cols + metric_cols].copy()
    rename_map = {metric: f"{metric}_baseline" for metric in metric_cols}
    baseline = baseline.rename(columns=rename_map)
    merged = frame.merge(baseline, on=group_cols, how="left")
    for metric in metric_cols:
        base_col = f"{metric}_baseline"
        out_col = f"{metric}_relative_improvement"
        merged[out_col] = np.where(
            pd.to_numeric(merged[base_col], errors="coerce").abs() > 1e-12,
            (pd.to_numeric(merged[metric], errors="coerce") - pd.to_numeric(merged[base_col], errors="coerce"))
            / pd.to_numeric(merged[base_col], errors="coerce").abs(),
            np.nan,
        )
    return merged


def format_for_latex_table(
    frame: pd.DataFrame,
    metric_cols: Sequence[str],
    precision: int = 3,
) -> pd.DataFrame:
    """Format numeric metric columns as compact LaTeX-friendly strings."""
    formatted = frame.copy()
    for metric in metric_cols:
        if metric in formatted.columns:
            formatted[metric] = formatted[metric].map(
                lambda value: "-" if pd.isna(value) else f"{float(value):.{precision}f}"
            )
    return formatted


def balance_score(
    frame: pd.DataFrame,
    *,
    performance_col: str = "event_f1",
    synthetic_col: str = "num_synthetic_windows",
) -> pd.DataFrame:
    """Score methods by performance while discouraging excessive synthetic use."""
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    synthetic = pd.to_numeric(output[synthetic_col], errors="coerce").fillna(0.0)
    performance = pd.to_numeric(output[performance_col], errors="coerce").fillna(0.0)
    output["balance_score"] = performance / (1.0 + np.log1p(synthetic))
    return output


def detector_is_supervised(detector_cfg: dict[str, Any]) -> bool:
    model_type = str(detector_cfg.get("model_type", detector_cfg.get("name", "random_forest_window"))).lower()
    if model_type in UNSUPERVISED_DETECTORS:
        return False
    if model_type in SUPERVISED_DETECTORS:
        return True
    return False


def build_method_diagnostics(
    *,
    labeled_fraction: float,
    method: str,
    train_label_stats: dict[str, Any],
    num_synthetic_windows: int,
    detector_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Build per-method low-label diagnostics for metrics.json."""
    supervised = detector_is_supervised(detector_cfg)
    negative_sampled = train_label_stats.get("detector_train_negative_count")
    if negative_sampled is None:
        negative_sampled = train_label_stats.get("num_real_negative_train_points", 0)
    return {
        "labeled_fraction": float(labeled_fraction),
        "method": str(method),
        "train_positive_points": int(train_label_stats.get("num_real_positive_train_points", 0)),
        "train_positive_windows": int(train_label_stats.get("num_positive_train_event_windows", 0)),
        "train_negative_points_sampled": int(negative_sampled),
        "num_synthetic_windows": int(num_synthetic_windows),
        "detector_is_supervised": bool(supervised),
    }


def build_run_diagnostics(
    results: dict[str, Any],
    *,
    labeled_fraction: float,
) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for method, payload in results.items():
        diagnostics[method] = build_method_diagnostics(
            labeled_fraction=labeled_fraction,
            method=method,
            train_label_stats=payload.get("train_label_stats", {}),
            num_synthetic_windows=int(payload.get("num_synthetic_windows", 0)),
            detector_cfg=payload.get("detector", {}),
        )
    return diagnostics


def unsupervised_low_label_warning(detector_cfg: dict[str, Any]) -> str | None:
    if not detector_is_supervised(detector_cfg):
        return UNSUPPORTED_LOW_LABEL_WARNING
    return None


def check_real_only_train_positives_monotonic(diagnostics_rows: list[dict[str, Any]]) -> str | None:
    """Warn when real_only labeled train positives do not increase with labeled_fraction."""
    real_rows = [row for row in diagnostics_rows if row.get("method") == "real_only"]
    if len(real_rows) < 2:
        return None
    if not all(bool(row.get("detector_is_supervised")) for row in real_rows):
        return None

    sorted_rows = sorted(real_rows, key=lambda row: float(row["labeled_fraction"]))
    series = [(float(row["labeled_fraction"]), int(row["train_positive_points"])) for row in sorted_rows]
    points = [value for _, value in series]
    fractions = [value for value, _ in series]

    for (prev_fraction, prev_points), (curr_fraction, curr_points) in zip(series, series[1:]):
        if curr_fraction > prev_fraction and curr_points < prev_points:
            return (
                "real_only train_positive_points decreased as labeled_fraction increased; "
                f"diagnostics={series}"
            )

    if len(set(fractions)) > 1 and len(set(points)) == 1:
        return (
            "real_only train_positive_points unchanged across label fractions despite different labeled_fraction; "
            f"diagnostics={series}"
        )
    return None


def _metric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        if key in row:
            values.append(float(row[key]))
        elif key == "event_f1" and "best_point_f1" in row:
            values.append(float(row["best_point_f1"]))
    return values


def check_real_only_metrics_invariant(
    comparison_rows: list[dict[str, Any]],
    *,
    tolerance: float = 1e-12,
) -> str | None:
    """Warn when real_only test metrics are identical across label fractions."""
    real_rows = [row for row in comparison_rows if row.get("method") == "real_only"]
    if len(real_rows) < 2:
        return None

    metric_keys = ("auroc", "auprc", "event_f1")
    invariant_metrics: list[str] = []
    for metric_key in metric_keys:
        values = [value for value in _metric_values(real_rows, metric_key) if np.isfinite(value)]
        if len(values) < 2:
            continue
        if max(values) - min(values) <= tolerance:
            invariant_metrics.append(metric_key)

    if len(invariant_metrics) == len(metric_keys):
        return REAL_ONLY_INVARIANT_WARNING
    return None


def collect_low_label_warnings(
    *,
    detector_cfg: dict[str, Any],
    diagnostics_by_method: dict[str, dict[str, Any]] | None = None,
    diagnostics_rows: list[dict[str, Any]] | None = None,
    comparison_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Collect low-label experiment warnings for metrics.json and console output."""
    warnings: list[str] = []
    unsupervised_warning = unsupervised_low_label_warning(detector_cfg)
    if unsupervised_warning:
        warnings.append(unsupervised_warning)

    rows = list(diagnostics_rows or [])
    if diagnostics_by_method and not rows:
        rows = list(diagnostics_by_method.values())

    monotonic_warning = check_real_only_train_positives_monotonic(rows)
    if monotonic_warning:
        warnings.append(monotonic_warning)

    if comparison_rows:
        invariant_warning = check_real_only_metrics_invariant(comparison_rows)
        if invariant_warning:
            warnings.append(invariant_warning)
    return warnings


def print_low_label_warnings(warnings: list[str]) -> None:
    for warning in warnings:
        print(f"WARNING: {warning}")


def warn_if_real_only_invariant(comparison_rows: list[dict[str, Any]]) -> None:
    """Backward-compatible helper used by sweep runners."""
    warning = check_real_only_metrics_invariant(comparison_rows)
    if warning:
        print(f"WARNING: {warning}")


STRICT_FILTER_REJECT_ALL_WARNING = "strict filter rejected all synthetic windows for at least one method/fraction."
ONE_SERIES_EVALUATION_WARNING = "evaluation uses fewer than 5 test series; metrics may be unstable."
LOW_EVENT_PRECISION_WARNING = "event precision below 1% for at least one method/fraction."
POINT_UP_EVENT_DOWN_WARNING = (
    "synthesis improves point metric but worsens event metric relative to real_only."
)


def _best_method_per_fraction(
    comparison: pd.DataFrame,
    metric: str,
    *,
    higher_is_better: bool = True,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    if comparison.empty or metric not in comparison.columns:
        return results
    for fraction, group in comparison.groupby("labeled_fraction", dropna=False):
        values = pd.to_numeric(group[metric], errors="coerce")
        if values.dropna().empty:
            continue
        idx = values.idxmax() if higher_is_better else values.idxmin()
        row = group.loc[idx]
        results[str(float(fraction))] = {
            "method": str(row["method"]),
            "value": float(row[metric]),
        }
    return results


def _real_only_metrics_by_fraction(comparison: pd.DataFrame) -> dict[str, dict[str, float]]:
    real = comparison[comparison["method"] == "real_only"].copy()
    output: dict[str, dict[str, float]] = {}
    metric_cols = ("auroc", "auprc", "best_point_f1", "event_f1", "false_positive_events", "num_real_positive_train_points")
    for _, row in real.iterrows():
        fraction = str(float(row["labeled_fraction"]))
        output[fraction] = {
            col: float(row[col])
            for col in metric_cols
            if col in row and pd.notna(row[col])
        }
    return output


def _strict_filter_summary(comparison: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strict = comparison[comparison["method"].astype(str).str.contains("strict_filter", na=False)]
    for _, row in strict.iterrows():
        kept = int(row.get("num_synthetic_windows", 0) or 0)
        rejected = int(row.get("num_rejected_synthetic_windows", 0) or 0)
        rows.append(
            {
                "labeled_fraction": float(row["labeled_fraction"]),
                "method": str(row["method"]),
                "accepted": kept,
                "rejected": rejected,
                "all_rejected": kept == 0 and rejected > 0,
            }
        )
    return rows


def _collect_sweep_analysis_warnings(
    comparison: pd.DataFrame,
    per_series: pd.DataFrame,
    *,
    comparison_rows: list[dict[str, Any]] | None = None,
) -> list[str]:
    warnings: list[str] = []
    rows = comparison_rows if comparison_rows is not None else comparison.to_dict(orient="records")

    invariant = check_real_only_metrics_invariant(rows)
    if invariant:
        warnings.append(invariant)

    for item in _strict_filter_summary(comparison):
        if item["all_rejected"]:
            warnings.append(
                f"{STRICT_FILTER_REJECT_ALL_WARNING} method={item['method']} "
                f"fraction={item['labeled_fraction']} rejected={item['rejected']}"
            )

    if not per_series.empty and "series_id" in per_series.columns:
        num_series = int(per_series["series_id"].nunique())
        if num_series < 5:
            warnings.append(f"{ONE_SERIES_EVALUATION_WARNING} num_test_series={num_series}")

    if "event_precision" in comparison.columns:
        low_precision = comparison[pd.to_numeric(comparison["event_precision"], errors="coerce") < 0.01]
        for _, row in low_precision.iterrows():
            warnings.append(
                f"{LOW_EVENT_PRECISION_WARNING} method={row['method']} "
                f"fraction={row['labeled_fraction']} precision={float(row['event_precision']):.4f}"
            )

    for fraction, group in comparison.groupby("labeled_fraction", dropna=False):
        baseline_rows = group[group["method"] == "real_only"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        base_point = float(baseline.get("best_point_f1", np.nan))
        base_event = float(baseline.get("event_f1", np.nan))
        if not np.isfinite(base_point) or not np.isfinite(base_event):
            continue
        synth = group[
            ~group["method"].isin({"real_only", "random_event_oversampling"})
            & (pd.to_numeric(group["num_synthetic_windows"], errors="coerce").fillna(0) > 0)
        ]
        for _, row in synth.iterrows():
            point = float(row.get("best_point_f1", np.nan))
            event = float(row.get("event_f1", np.nan))
            if np.isfinite(point) and np.isfinite(event) and point > base_point and event < base_event:
                warnings.append(
                    f"{POINT_UP_EVENT_DOWN_WARNING} method={row['method']} "
                    f"fraction={fraction} point_f1={point:.3f}>{base_point:.3f} "
                    f"event_f1={event:.3f}<{base_event:.3f}"
                )
    return warnings


def _build_interpretation(analysis: dict[str, Any]) -> dict[str, list[str]]:
    bullets: dict[str, list[str]] = {
        "pipeline_sanity": [],
        "label_sensitivity": [],
        "synthesis_effect": [],
        "filtering_behavior": [],
        "next_recommended_experiment": [],
    }

    real_only = analysis.get("real_only_behavior", {})
    if real_only.get("metrics_invariant"):
        bullets["pipeline_sanity"].append(
            "real_only test metrics are identical across label fractions; verify label masking and supervised training path."
        )
    else:
        bullets["pipeline_sanity"].append(
            "real_only metrics vary across fractions, indicating the sweep is exercising different label budgets."
        )

    train_positives = real_only.get("train_positives_by_fraction", {})
    if len(set(train_positives.values())) > 1:
        bullets["label_sensitivity"].append(
            "real_only train positive counts increase with labeled fraction as expected for a supervised low-label setup."
        )
    elif train_positives:
        bullets["label_sensitivity"].append(
            "real_only train positive counts are flat across fractions; label masking may not be reducing usable supervision."
        )

    synth_rows = analysis.get("synthetic_usage", [])
    improved = [
        row
        for row in analysis.get("synthesis_vs_real_only", [])
        if row.get("delta_event_f1", 0.0) > 0 or row.get("delta_auprc", 0.0) > 0
    ]
    if improved:
        best = max(improved, key=lambda row: row.get("delta_event_f1", float("-inf")))
        bullets["synthesis_effect"].append(
            f"Synthesis helps in some settings; best event-F1 gain vs real_only is "
            f"{best['delta_event_f1']:+.3f} for {best['method']} at fraction={best['labeled_fraction']}."
        )
    elif synth_rows:
        bullets["synthesis_effect"].append(
            "No synthesis method consistently beats real_only on event F1; donor windows may be mismatched or filters too permissive."
        )
    else:
        bullets["synthesis_effect"].append("No synthetic windows were accepted in this sweep.")

    strict = analysis.get("strict_filter_summary", [])
    if any(item.get("all_rejected") for item in strict):
        bullets["filtering_behavior"].append(
            "Strict filters rejected all candidates for at least one method; thresholds may be too aggressive for this benchmark."
        )
    elif strict:
        bullets["filtering_behavior"].append(
            "Strict filters reduced accepted synthetic windows compared with unfiltered synthesis methods."
        )
    else:
        bullets["filtering_behavior"].append("No strict-filter methods were present in this sweep.")

    if analysis.get("per_series", {}).get("num_test_series", 0) < 5:
        bullets["next_recommended_experiment"].append(
            "Increase test-series count (>=5) or run grouped cross-validation before drawing paper conclusions."
        )
    bullets["next_recommended_experiment"].append(
        "Run masked event completion to evaluate synthesis shape fidelity independent of detector thresholding."
    )
    if any(row.get("false_positive_events", 0) > 0 for row in analysis.get("false_positive_comparison", [])):
        bullets["next_recommended_experiment"].append(
            "Compare context-calibrated synthesis and event-aware threshold tuning to reduce false-positive events."
        )
    return bullets


def analyze_low_label_sweep(
    comparison: pd.DataFrame,
    per_series: pd.DataFrame,
    summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Analyze aggregated low-label sweep outputs into a structured report payload."""
    summary = summary or {}
    comparison = comparison.copy()
    per_series = per_series.copy()
    if "labeled_fraction" not in comparison.columns and summary.get("comparison_rows"):
        comparison = pd.DataFrame(summary["comparison_rows"])

    comparison_rows = comparison.to_dict(orient="records")
    fractions = sorted({float(value) for value in comparison["labeled_fraction"].dropna().unique()}) if not comparison.empty else []
    methods = sorted(comparison["method"].dropna().unique().tolist()) if not comparison.empty else []

    real_only_by_fraction = _real_only_metrics_by_fraction(comparison)
    real_only_invariant = check_real_only_metrics_invariant(comparison_rows) is not None

    synthetic_usage: list[dict[str, Any]] = []
    for _, row in comparison.iterrows():
        synthetic_usage.append(
            {
                "labeled_fraction": float(row["labeled_fraction"]),
                "method": str(row["method"]),
                "num_synthetic_windows": int(row.get("num_synthetic_windows", 0) or 0),
                "num_synthetic_points": int(row.get("num_synthetic_points", 0) or 0),
                "num_rejected_synthetic_windows": int(row.get("num_rejected_synthetic_windows", 0) or 0),
            }
        )

    synthesis_vs_real_only: list[dict[str, Any]] = []
    for fraction, group in comparison.groupby("labeled_fraction", dropna=False):
        baseline_rows = group[group["method"] == "real_only"]
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        for _, row in group.iterrows():
            if row["method"] == "real_only":
                continue
            synthesis_vs_real_only.append(
                {
                    "labeled_fraction": float(fraction),
                    "method": str(row["method"]),
                    "delta_auprc": float(row.get("auprc", np.nan)) - float(baseline.get("auprc", np.nan)),
                    "delta_event_f1": float(row.get("event_f1", np.nan)) - float(baseline.get("event_f1", np.nan)),
                    "delta_false_positive_events": float(row.get("false_positive_events", np.nan))
                    - float(baseline.get("false_positive_events", np.nan)),
                }
            )

    num_test_series = int(per_series["series_id"].nunique()) if not per_series.empty and "series_id" in per_series.columns else 0
    warnings = list(summary.get("warnings", []))
    warnings.extend(_collect_sweep_analysis_warnings(comparison, per_series, comparison_rows=comparison_rows))
    warnings = list(dict.fromkeys(warnings))

    analysis = {
        "fractions": fractions,
        "methods": methods,
        "best_by_auprc": _best_method_per_fraction(comparison, "auprc"),
        "best_by_event_f1": _best_method_per_fraction(comparison, "event_f1"),
        "real_only_behavior": {
            "metrics_by_fraction": real_only_by_fraction,
            "metrics_invariant": real_only_invariant,
            "train_positives_by_fraction": {
                fraction: values.get("num_real_positive_train_points")
                for fraction, values in real_only_by_fraction.items()
                if "num_real_positive_train_points" in values
            },
        },
        "synthetic_usage": synthetic_usage,
        "strict_filter_summary": _strict_filter_summary(comparison),
        "false_positive_comparison": comparison[
            ["labeled_fraction", "method", "false_positive_events", "event_precision", "event_f1"]
        ].to_dict(orient="records")
        if not comparison.empty and "false_positive_events" in comparison.columns
        else [],
        "synthesis_vs_real_only": synthesis_vs_real_only,
        "per_series": {
            "num_test_series": num_test_series,
            "series_ids": sorted(per_series["series_id"].dropna().unique().tolist()) if num_test_series else [],
            "warning_small_benchmark": num_test_series < 5,
        },
        "warnings": warnings,
    }
    analysis["interpretation"] = _build_interpretation(analysis)
    return analysis


def format_low_label_sweep_markdown(analysis: dict[str, Any], *, title: str = "Low-Label Sweep Analysis") -> str:
    """Render a human-readable markdown report from analyze_low_label_sweep output."""
    lines: list[str] = [f"# {title}", ""]

    lines.append("## Summary")
    lines.append(f"- Label fractions: {', '.join(str(value) for value in analysis.get('fractions', []))}")
    lines.append(f"- Methods: {', '.join(analysis.get('methods', []))}")
    lines.append(f"- Test series count: {analysis.get('per_series', {}).get('num_test_series', 0)}")
    lines.append("")

    lines.append("## Best Method by Labeled Fraction")
    lines.append("")
    lines.append("### AUPRC")
    for fraction, payload in sorted(analysis.get("best_by_auprc", {}).items(), key=lambda item: float(item[0])):
        lines.append(f"- fraction={fraction}: **{payload['method']}** ({payload['value']:.4f})")
    lines.append("")
    lines.append("### Event F1")
    for fraction, payload in sorted(analysis.get("best_by_event_f1", {}).items(), key=lambda item: float(item[0])):
        lines.append(f"- fraction={fraction}: **{payload['method']}** ({payload['value']:.4f})")
    lines.append("")

    lines.append("## real_only Behavior")
    real_only = analysis.get("real_only_behavior", {})
    if real_only.get("metrics_invariant"):
        lines.append("- Metrics appear **invariant** across label fractions.")
    else:
        lines.append("- Metrics **change** across label fractions.")
    lines.append("")
    lines.append("| fraction | AUPRC | event F1 | FP events | train positives |")
    lines.append("|---:|---:|---:|---:|---:|")
    for fraction, metrics in sorted(real_only.get("metrics_by_fraction", {}).items(), key=lambda item: float(item[0])):
        lines.append(
            f"| {fraction} | {metrics.get('auprc', float('nan')):.4f} | "
            f"{metrics.get('event_f1', float('nan')):.4f} | "
            f"{metrics.get('false_positive_events', float('nan')):.1f} | "
            f"{metrics.get('num_real_positive_train_points', float('nan')):.0f} |"
        )
    lines.append("")

    lines.append("## Synthetic Windows / Points")
    lines.append("")
    lines.append("| fraction | method | windows | points | rejected |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in analysis.get("synthetic_usage", []):
        if row["num_synthetic_windows"] or row["num_rejected_synthetic_windows"] or "strict_filter" in row["method"]:
            lines.append(
                f"| {row['labeled_fraction']} | {row['method']} | {row['num_synthetic_windows']} | "
                f"{row['num_synthetic_points']} | {row['num_rejected_synthetic_windows']} |"
            )
    lines.append("")

    lines.append("## Strict Filter Acceptance")
    strict_rows = analysis.get("strict_filter_summary", [])
    if strict_rows:
        lines.append("| fraction | method | accepted | rejected | all rejected |")
        lines.append("|---:|---|---:|---:|---|")
        for row in strict_rows:
            lines.append(
                f"| {row['labeled_fraction']} | {row['method']} | {row['accepted']} | {row['rejected']} | {row['all_rejected']} |"
            )
    else:
        lines.append("- No strict-filter methods found.")
    lines.append("")

    lines.append("## False-Positive Event Comparison")
    lines.append("")
    lines.append("| fraction | method | FP events | event precision | event F1 |")
    lines.append("|---:|---|---:|---:|---:|")
    for row in analysis.get("false_positive_comparison", []):
        lines.append(
            f"| {row['labeled_fraction']} | {row['method']} | {row['false_positive_events']:.1f} | "
            f"{row.get('event_precision', float('nan')):.4f} | {row.get('event_f1', float('nan')):.4f} |"
        )
    lines.append("")

    per_series = analysis.get("per_series", {})
    if per_series.get("warning_small_benchmark"):
        lines.append("## Per-Series Warning")
        lines.append(
            f"- Only **{per_series.get('num_test_series', 0)}** test series evaluated "
            f"({', '.join(per_series.get('series_ids', []))}). Interpret event metrics cautiously."
        )
        lines.append("")

    warnings = analysis.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    interpretation = analysis.get("interpretation", {})
    for section, heading in (
        ("pipeline_sanity", "Pipeline Sanity"),
        ("label_sensitivity", "Label Sensitivity"),
        ("synthesis_effect", "Synthesis Effect"),
        ("filtering_behavior", "Filtering Behavior"),
        ("next_recommended_experiment", "Next Recommended Experiment"),
    ):
        bullets = interpretation.get(section, [])
        if bullets:
            lines.append(f"## {heading}")
            for bullet in bullets:
                lines.append(f"- {bullet}")
            lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _method_metric_row(comparison: pd.DataFrame, method: str) -> dict[str, Any] | None:
    rows = comparison[comparison["method"] == method]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def _delta_metric(
    comparison: pd.DataFrame,
    method_a: str,
    method_b: str,
    metric: str,
) -> float | None:
    row_a = _method_metric_row(comparison, method_a)
    row_b = _method_metric_row(comparison, method_b)
    if not row_a or not row_b:
        return None
    try:
        return float(row_a.get(metric, np.nan)) - float(row_b.get(metric, np.nan))
    except (TypeError, ValueError):
        return None


def analyze_compatibility_transfer(
    comparison: pd.DataFrame,
    rejection_summary: dict[str, Any],
    compatibility_summary: dict[str, Any],
    *,
    masked_per_event: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Analyze compatibility-transfer outputs for paper-facing comparisons."""
    comparison = comparison.copy()
    methods = comparison["method"].dropna().unique().tolist() if not comparison.empty else []

    def _best(metric: str) -> dict[str, Any] | None:
        if comparison.empty or metric not in comparison.columns:
            return None
        if "metrics_valid" in comparison.columns:
            valid = comparison[comparison["metrics_valid"].eq(True)]
        else:
            valid = comparison.iloc[0:0]
        if valid.empty:
            return None
        idx = valid[metric].astype(float).idxmax()
        row = valid.loc[idx]
        return {"method": str(row["method"]), "value": float(row[metric]), "donor_policy": row.get("donor_policy")}

    def _find_method(*substrings: str) -> str | None:
        for method in methods:
            if all(part in str(method) for part in substrings):
                return str(method)
        return None

    all_donors = _find_method("all_donors")
    cross_compat = _find_method("cross_dataset_compatible")
    same_dataset = _find_method("same_dataset_only")
    top50_filter = _find_method("top50_filter") or _find_method("compatibility_top50")
    strict_filter = _find_method("strict_filter") or _find_method("compatibility_strict")

    rejection_counts: dict[str, int] = {}
    for entry in rejection_summary.values():
        if not isinstance(entry, dict):
            continue
        for key, value in (entry.get("counts") or {}).items():
            rejection_counts[key] = rejection_counts.get(key, 0) + int(value)

    masked_ranking: list[dict[str, Any]] = []
    reconstruction_downstream_agreement: list[dict[str, Any]] = []
    if masked_per_event:
        frame = pd.DataFrame(masked_per_event)
        metric_cols = [col for col in ("normalized_MAE_by_event_std", "shape_correlation") if col in frame.columns]
        if metric_cols and "method" in frame.columns:
            grouped = frame.groupby("method")[metric_cols].mean(numeric_only=True).reset_index()
            if "shape_correlation" in grouped.columns:
                grouped = grouped.sort_values("shape_correlation", ascending=False)
            masked_ranking = grouped.to_dict(orient="records")

            downstream = comparison[["method", "event_f1"]].copy() if "event_f1" in comparison.columns else pd.DataFrame()
            if not downstream.empty:
                merged = grouped.merge(downstream, on="method", how="inner")
                for _, row in merged.iterrows():
                    reconstruction_downstream_agreement.append(
                        {
                            "method": str(row["method"]),
                            "shape_correlation": float(row.get("shape_correlation", np.nan)),
                            "event_f1": float(row.get("event_f1", np.nan)),
                        }
                    )

    compat_beats_all = None
    if all_donors and cross_compat:
        compat_beats_all = {
            "delta_event_f1": _delta_metric(comparison, cross_compat, all_donors, "event_f1"),
            "delta_best_point_f1": _delta_metric(comparison, cross_compat, all_donors, "best_point_f1"),
        }

    cross_beats_same = None
    if same_dataset and cross_compat:
        cross_beats_same = {
            "delta_event_f1": _delta_metric(comparison, cross_compat, same_dataset, "event_f1"),
            "delta_best_point_f1": _delta_metric(comparison, cross_compat, same_dataset, "best_point_f1"),
        }

    analysis = {
        "methods": methods,
        "best_by_event_f1": _best("event_f1"),
        "best_by_best_point_f1": _best("best_point_f1"),
        "compatibility_filtered_beats_all_donors": compat_beats_all,
        "cross_dataset_compatible_beats_same_dataset_only": cross_beats_same,
        "rejection_counts": rejection_counts,
        "compatibility_summary": compatibility_summary,
        "masked_completion_ranking": masked_ranking,
        "downstream_ranking": comparison.sort_values("event_f1", ascending=False)[
            ["method", "event_f1", "best_point_f1", "donor_policy"]
        ].to_dict(orient="records")
        if not comparison.empty and "event_f1" in comparison.columns
        else [],
        "reconstruction_vs_downstream": reconstruction_downstream_agreement,
        "strict_filter_note": {
            "top50_filter_method": top50_filter,
            "strict_filter_method": strict_filter,
            "num_rejected_top50": rejection_counts.get("below_top_quantile_0.5", 0),
            "strict_filter_failures": sum(
                value for key, value in rejection_counts.items() if str(key).startswith("strict_filter_failed")
            ),
        },
    }

    warnings: list[str] = []
    if compat_beats_all and compat_beats_all.get("delta_event_f1") is not None and compat_beats_all["delta_event_f1"] < 0:
        warnings.append("Compatibility-filtered transfer did not beat all-donor transfer on event F1.")
    if reconstruction_downstream_agreement:
        shape_scores = [row["shape_correlation"] for row in reconstruction_downstream_agreement]
        f1_scores = [row["event_f1"] for row in reconstruction_downstream_agreement]
        if len(shape_scores) >= 2 and np.std(shape_scores) > 1e-6 and np.std(f1_scores) > 1e-6:
            corr = float(np.corrcoef(shape_scores, f1_scores)[0, 1])
            analysis["reconstruction_downstream_correlation"] = corr
            if corr < 0:
                warnings.append("Masked-completion shape correlation disagrees with downstream event F1 ranking.")
    analysis["warnings"] = warnings
    return analysis


def format_compatibility_transfer_markdown(
    analysis: dict[str, Any],
    *,
    title: str = "Compatibility Transfer Analysis",
) -> str:
    """Render analysis_report.md for compatibility-transfer experiments."""
    lines = [f"# {title}", ""]

    best_f1 = analysis.get("best_by_event_f1")
    best_point = analysis.get("best_by_best_point_f1")
    lines.append("## Best Methods")
    if best_f1:
        lines.append(f"- Best **event_f1**: `{best_f1['method']}` ({best_f1['value']:.4f}, policy={best_f1.get('donor_policy')})")
    if best_point:
        lines.append(
            f"- Best **point F1**: `{best_point['method']}` ({best_point['value']:.4f}, policy={best_point.get('donor_policy')})"
        )
    lines.append("")

    compat_vs_all = analysis.get("compatibility_filtered_beats_all_donors")
    if compat_vs_all:
        lines.append("## Compatibility-Filtered vs All-Donor Transfer")
        delta = compat_vs_all.get("delta_event_f1")
        if delta is not None:
            verdict = "beats" if delta > 0 else "does not beat"
            lines.append(f"- Cross-dataset compatible transfer **{verdict}** naive all-donor on event F1 (Δ={delta:+.4f}).")
        lines.append("")

    cross_vs_same = analysis.get("cross_dataset_compatible_beats_same_dataset_only")
    if cross_vs_same:
        lines.append("## Cross-Dataset Compatible vs Same-Dataset Only")
        delta = cross_vs_same.get("delta_event_f1")
        if delta is not None:
            verdict = "helps" if delta > 0 else "does not help"
            lines.append(f"- Cross-dataset compatible transfer **{verdict}** vs same-dataset-only (Δ={delta:+.4f}).")
        lines.append("")

    summary = analysis.get("compatibility_summary", {})
    if summary:
        lines.append("## Donor Compatibility Summary")
        lines.append(f"- Target timelines: {summary.get('num_target_timelines')}")
        lines.append(f"- Source timelines: {summary.get('num_source_timelines')}")
        lines.append(f"- Donor pairs kept / rejected: {summary.get('donor_pairs_kept')} / {summary.get('donor_pairs_rejected')}")
        lines.append(f"- No compatible donor (targets): {summary.get('no_compatible_donor_count')}")
        if summary.get("mean_compatibility_score_kept") is not None:
            lines.append(f"- Mean compatibility (kept): {summary['mean_compatibility_score_kept']:.4f}")
        if summary.get("mean_compatibility_score_rejected") is not None:
            lines.append(f"- Mean compatibility (rejected): {summary['mean_compatibility_score_rejected']:.4f}")
        lines.append("")

    rejection_counts = analysis.get("rejection_counts", {})
    if rejection_counts:
        lines.append("## Major Rejection Reasons")
        for key, value in sorted(rejection_counts.items(), key=lambda item: (-item[1], item[0])):
            if key == "kept" or value == 0:
                continue
            lines.append(f"- `{key}`: {value}")
        lines.append("")

    downstream = analysis.get("downstream_ranking", [])
    if downstream:
        lines.append("## Downstream Detection Ranking (event F1)")
        for row in downstream[:10]:
            lines.append(
                f"- `{row['method']}`: event_f1={float(row.get('event_f1', 0)):.4f} "
                f"(policy={row.get('donor_policy')})"
            )
        lines.append("")

    masked = analysis.get("masked_completion_ranking", [])
    if masked:
        lines.append("## Masked-Completion Ranking")
        for row in masked[:10]:
            lines.append(
                f"- `{row['method']}`: shape_correlation={float(row.get('shape_correlation', 0)):.4f}, "
                f"normalized_MAE={float(row.get('normalized_MAE_by_event_std', 0)):.4f}"
            )
        lines.append("")

    corr = analysis.get("reconstruction_downstream_correlation")
    if corr is not None:
        lines.append("## Reconstruction vs Downstream Utility")
        lines.append(f"- Pearson correlation (shape_correlation vs event_f1): **{corr:.3f}**")
        lines.append("")

    strict_note = analysis.get("strict_filter_note", {})
    if strict_note:
        lines.append("## Strict / Top-Quantile Filtering")
        lines.append(f"- Top-quantile rejections: {strict_note.get('num_rejected_top50', 0)}")
        lines.append(f"- Strict-filter failures: {strict_note.get('strict_filter_failures', 0)}")
        lines.append("")

    warnings = analysis.get("warnings", [])
    if warnings:
        lines.append("## Warnings")
        for warning in warnings:
            lines.append(f"- {warning}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
