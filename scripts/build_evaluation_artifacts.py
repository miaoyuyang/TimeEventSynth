#!/usr/bin/env python3
"""Build paper-facing Evaluation artifacts from completed backbone runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POLICY_LABELS = {
    "real_only": "No Aug.",
    "random_event_oversampling": "Random Aug.",
    "all_donors_no_filter": "Sampling Aug.",
    "adaptive_groupwise_transfer": "TimeEventSynth",
    "groupwise_cross_dataset_all": "TES-All",
    "groupwise_cross_dataset_compatible": "TES-Compatible",
    "groupwise_compatibility_strict": "TES-Strict",
    "cross_dataset_all": "TES-All",
    "cross_dataset_compatible": "TES-Compatible",
    "compatibility_strict": "TES-Strict",
}

DETECTOR_LABELS = {
    "internal_classifier": "Supervised",
    "iforest": "IForest",
    "lof": "LOF",
    "ocsvm": "OCSVM",
    "autoencoder": "AutoEncoder",
    "cnn": "CNN",
    "timesnet": "TimesNet",
}

DETECTOR_ORDER = ["iforest", "lof", "ocsvm", "autoencoder", "cnn", "timesnet", "internal_classifier"]
DETECTOR_LABEL_ORDER = {DETECTOR_LABELS[name]: idx for idx, name in enumerate(DETECTOR_ORDER)}
POLICY_ORDER = [
    "real_only",
    "random_event_oversampling",
    "all_donors_no_filter",
    "groupwise_cross_dataset_all",
    "groupwise_cross_dataset_compatible",
    "groupwise_compatibility_strict",
    "adaptive_groupwise_transfer",
    "cross_dataset_all",
    "cross_dataset_compatible",
    "compatibility_strict",
]
TES_FAMILY = {
    "adaptive_groupwise_transfer",
    "groupwise_cross_dataset_all",
    "groupwise_cross_dataset_compatible",
    "groupwise_compatibility_strict",
    "cross_dataset_all",
    "cross_dataset_compatible",
    "compatibility_strict",
}


def _detector_label(name: object) -> str:
    return DETECTOR_LABELS.get(str(name), str(name))


def _policy_label(name: object) -> str:
    return POLICY_LABELS.get(str(name), str(name))


def _detector_sort_key(name: object) -> tuple[int, str]:
    raw = str(name)
    if raw in DETECTOR_ORDER:
        return (DETECTOR_ORDER.index(raw), raw)
    if raw in DETECTOR_LABEL_ORDER:
        return (DETECTOR_LABEL_ORDER[raw], raw)
    return (len(DETECTOR_ORDER), raw)


def _policy_sort_key(name: object) -> tuple[int, str]:
    raw = str(name)
    return (POLICY_ORDER.index(raw) if raw in POLICY_ORDER else len(POLICY_ORDER), raw)


def _fmt(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_delta(value: object, digits: int = 4) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):+.{digits}f}"


def _fmt_mean_std(row: pd.Series, metric: str) -> str:
    mean = row.get(f"{metric}_mean", row.get(metric))
    std = row.get(f"{metric}_std")
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return _fmt(mean)
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def _to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    lines = [
        "| " + " | ".join(str(col) for col in frame.columns) + " |",
        "| " + " | ".join(["---"] * len(frame.columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines) + "\n"


def _write_table(frame: pd.DataFrame, out_dir: Path, stem: str) -> None:
    frame.to_csv(out_dir / f"{stem}.csv", index=False)
    (out_dir / f"{stem}.md").write_text(_to_markdown(frame), encoding="utf-8")


def _read_csv(run_dir: Path, candidates: list[str]) -> pd.DataFrame:
    for name in candidates:
        path = run_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(f"Missing required CSV in {run_dir}: {', '.join(candidates)}")


def _metric_col(row: pd.Series, metric: str) -> float:
    for col in (f"{metric}_mean", metric):
        if col in row.index and pd.notna(row[col]):
            return float(row[col])
    return float("nan")


def _valid_seed_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "metrics_valid" not in frame.columns:
        return frame.copy()
    return frame[frame["metrics_valid"].eq(True)].copy()


def build_timesynth_vs_random_win_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare adaptive TimeEventSynth against Random Aug. per detector."""
    working = _valid_seed_rows(frame)
    if working.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    metrics = ["event_f1", "event_precision", "event_recall", "false_positive_events"]
    for detector, group in working.groupby("detector_backbone", dropna=False):
        adaptive = group[group["augmentation_policy"] == "adaptive_groupwise_transfer"].set_index("split_seed")
        random = group[group["augmentation_policy"] == "random_event_oversampling"].set_index("split_seed")
        common = adaptive.index.intersection(random.index)
        if common.empty:
            continue
        row: dict[str, Any] = {
            "detector_backbone": str(detector),
            "timeeventsynth_policy": "adaptive_groupwise_transfer",
            "random_policy": "random_event_oversampling",
            "num_seeds": int(len(common)),
        }
        for metric in metrics:
            delta = pd.to_numeric(adaptive.loc[common, metric], errors="coerce") - pd.to_numeric(random.loc[common, metric], errors="coerce")
            row[f"{metric}_delta_mean"] = float(delta.mean())
            row[f"{metric}_delta_std"] = float(delta.std(ddof=0))
            if metric == "false_positive_events":
                row["false_positive_events_reduction_mean"] = float((-delta).mean())
                row["false_positive_events_lower_fp_seeds"] = int((delta < 0).sum())
                row["false_positive_events_lower_fp_rate"] = float((delta < 0).mean())
            else:
                row[f"{metric}_wins_timeeventsynth"] = int((delta > 0).sum())
                row[f"{metric}_win_rate_timeeventsynth"] = float((delta > 0).mean())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_threshold_tradeoff(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize saved threshold tradeoff curves, if a run contains them."""
    if frame.empty:
        return pd.DataFrame()
    required = {
        "split_seed",
        "detector_backbone",
        "augmentation_policy",
        "selected_policy_name",
        "threshold",
        "event_f1",
        "event_precision",
        "event_recall",
        "false_positive_events",
        "event_count_true",
    }
    if not required.issubset(set(frame.columns)):
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_cols = ["split_seed", "detector_backbone", "augmentation_policy", "selected_policy_name"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        numeric = group.copy()
        for col in ("event_f1", "event_precision", "event_recall", "false_positive_events", "threshold", "event_count_true"):
            numeric[col] = pd.to_numeric(numeric[col], errors="coerce")
        numeric = numeric.dropna(subset=["event_f1", "false_positive_events", "threshold"])
        if numeric.empty:
            continue
        best_f1 = numeric.sort_values(["event_f1", "false_positive_events", "threshold"], ascending=[False, True, False]).iloc[0]
        fp_limit = max(1.0, float(numeric["event_count_true"].max()))
        fp_limited = numeric[numeric["false_positive_events"] <= fp_limit]
        if fp_limited.empty:
            fp_limited = numeric.sort_values(["false_positive_events", "event_f1"], ascending=[True, False]).head(1)
        else:
            fp_limited = fp_limited.sort_values(["event_f1", "false_positive_events", "threshold"], ascending=[False, True, False]).head(1)
        fp_row = fp_limited.iloc[0]
        row = {col: value for col, value in zip(group_cols, keys)}
        row.update(
            {
                "best_event_f1_threshold": float(best_f1["threshold"]),
                "best_event_f1": float(best_f1["event_f1"]),
                "best_event_precision": float(best_f1["event_precision"]),
                "best_event_recall": float(best_f1["event_recall"]),
                "best_false_positive_events": float(best_f1["false_positive_events"]),
                "fp_limited_threshold": float(fp_row["threshold"]),
                "fp_limited_event_f1": float(fp_row["event_f1"]),
                "fp_limited_event_precision": float(fp_row["event_precision"]),
                "fp_limited_event_recall": float(fp_row["event_recall"]),
                "fp_limited_false_positive_events": float(fp_row["false_positive_events"]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_dataset_setup(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "dataset_balance_summary.json"
    if not path.exists():
        return pd.DataFrame()
    payload = json.loads(path.read_text(encoding="utf-8"))
    per_seed = payload.get("per_seed") or {}
    rows: list[dict[str, Any]] = []
    for seed, info in sorted(per_seed.items(), key=lambda item: int(item[0])):
        after = ((info or {}).get("after") or {}).get("per_dataset") or {}
        coverage = (info or {}).get("split_dataset_coverage") or {}
        split_per_dataset = coverage.get("per_dataset") or {}
        for dataset, stats in sorted(after.items()):
            split_stats = split_per_dataset.get(dataset) or {}
            rows.append(
                {
                    "split_seed": int(seed),
                    "domain": dataset,
                    "series": int(stats.get("num_timelines", 0)),
                    "points": int(stats.get("num_points", 0)),
                    "anomalous_series": int(stats.get("num_anomalous_timelines", 0)),
                    "anomaly_points": int(stats.get("num_anomaly_points", 0)),
                    "train_series": int(split_stats.get("num_train_timelines", split_stats.get("train", 0))),
                    "val_series": int(split_stats.get("num_val_timelines", split_stats.get("val", 0))),
                    "test_series": int(split_stats.get("num_test_timelines", split_stats.get("test", 0))),
                }
            )
    return pd.DataFrame(rows)


def build_dataset_setup_summary(dataset_setup: pd.DataFrame) -> pd.DataFrame:
    if dataset_setup.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for domain, group in dataset_setup.groupby("domain", dropna=False):
        rows.append(
            {
                "domain": str(domain),
                "series_mean": float(group["series"].mean()),
                "series_min": int(group["series"].min()),
                "series_max": int(group["series"].max()),
                "points_mean": float(group["points"].mean()),
                "anomaly_points_mean": float(group["anomaly_points"].mean()),
                "train_series_mean": float(group["train_series"].mean()),
                "val_series_mean": float(group["val_series"].mean()),
                "test_series_mean": float(group["test_series"].mean()),
            }
        )
    total = dataset_setup.groupby("split_seed", dropna=False)[["series", "points", "anomaly_points", "train_series", "val_series", "test_series"]].sum()
    rows.append(
        {
            "domain": "TOTAL",
            "series_mean": float(total["series"].mean()),
            "series_min": int(total["series"].min()),
            "series_max": int(total["series"].max()),
            "points_mean": float(total["points"].mean()),
            "anomaly_points_mean": float(total["anomaly_points"].mean()),
            "train_series_mean": float(total["train_series"].mean()),
            "val_series_mean": float(total["val_series"].mean()),
            "test_series_mean": float(total["test_series"].mean()),
        }
    )
    return pd.DataFrame(rows)


def build_full_metrics_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    working = aggregate.copy()
    if "metrics_valid_all" in working.columns:
        working = working[working["metrics_valid_all"].eq(True)]
    working = working.sort_values(
        by=["detector_backbone", "augmentation_policy"],
        key=lambda col: col.map(_detector_sort_key) if col.name == "detector_backbone" else col.map(_policy_sort_key),
    )
    for _, row in working.iterrows():
        rows.append(
            {
                "Detector": _detector_label(row["detector_backbone"]),
                "Method": _policy_label(row["augmentation_policy"]),
                "Event-F1": _fmt_mean_std(row, "event_f1"),
                "Precision": _fmt_mean_std(row, "event_precision"),
                "Recall": _fmt_mean_std(row, "event_recall"),
                "False Positive Events": _fmt_mean_std(row, "false_positive_events"),
                "Seeds": str(int(row.get("num_seeds_valid", row.get("num_seeds", 0)) or 0)),
                "Selected Policy": _policy_label(row.get("selected_policy_name", row["augmentation_policy"])),
            }
        )
    return pd.DataFrame(rows)


def build_policy_ablation_table(aggregate: pd.DataFrame) -> pd.DataFrame:
    working = aggregate.copy()
    if "metrics_valid_all" in working.columns:
        working = working[working["metrics_valid_all"].eq(True)]
    rows: list[dict[str, str]] = []
    for detector, group in working.groupby("detector_backbone", dropna=False):
        row: dict[str, str] = {"Detector": _detector_label(detector)}
        for policy in sorted(group["augmentation_policy"].dropna().unique(), key=_policy_sort_key):
            policy_row = group[group["augmentation_policy"] == policy].iloc[0]
            row[_policy_label(policy)] = _fmt(_metric_col(policy_row, "event_f1"))
        rows.append(row)
    return pd.DataFrame(sorted(rows, key=lambda item: _detector_sort_key(item["Detector"])))


def _one_policy(group: pd.DataFrame, policy: str) -> pd.Series | None:
    subset = group[group["augmentation_policy"] == policy]
    if subset.empty:
        return None
    return subset.iloc[0]


def build_adaptive_vs_baselines(aggregate: pd.DataFrame) -> pd.DataFrame:
    return _build_fixed_policy_vs_baselines(aggregate, "adaptive_groupwise_transfer", "Adaptive TimeEventSynth")


def _build_fixed_policy_vs_baselines(aggregate: pd.DataFrame, policy: str, label: str) -> pd.DataFrame:
    working = aggregate.copy()
    if "metrics_valid_all" in working.columns:
        working = working[working["metrics_valid_all"].eq(True)]
    rows: list[dict[str, str]] = []
    for detector, group in working.groupby("detector_backbone", dropna=False):
        base = _one_policy(group, "real_only")
        random = _one_policy(group, "random_event_oversampling")
        target = _one_policy(group, policy)
        if target is None:
            continue
        target_f1 = _metric_col(target, "event_f1")
        base_f1 = _metric_col(base, "event_f1") if base is not None else float("nan")
        random_f1 = _metric_col(random, "event_f1") if random is not None else float("nan")
        target_fp = _metric_col(target, "false_positive_events")
        base_fp = _metric_col(base, "false_positive_events") if base is not None else float("nan")
        random_fp = _metric_col(random, "false_positive_events") if random is not None else float("nan")
        rows.append(
            {
                "Detector": _detector_label(detector),
                "No Aug. Event-F1": _fmt(base_f1),
                "Random Aug. Event-F1": _fmt(random_f1),
                f"{label} Event-F1": _fmt(target_f1),
                "Delta vs No Aug.": _fmt_delta(target_f1 - base_f1),
                "Delta vs Random": _fmt_delta(target_f1 - random_f1),
                "FP Delta vs No Aug.": _fmt_delta(target_fp - base_fp, digits=1),
                "FP Delta vs Random": _fmt_delta(target_fp - random_fp, digits=1),
            }
        )
    return pd.DataFrame(sorted(rows, key=lambda item: _detector_sort_key(item["Detector"])))


def build_best_tes_vs_baselines(aggregate: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    working = aggregate.copy()
    if "metrics_valid_all" in working.columns:
        working = working[working["metrics_valid_all"].eq(True)]
    rows: list[dict[str, str]] = []
    selected: dict[str, str] = {}
    for detector, group in working.groupby("detector_backbone", dropna=False):
        base = _one_policy(group, "real_only")
        random = _one_policy(group, "random_event_oversampling")
        tes = group[group["augmentation_policy"].isin(TES_FAMILY)].copy()
        if tes.empty:
            continue
        tes["event_f1_value"] = tes.apply(lambda row: _metric_col(row, "event_f1"), axis=1)
        best = tes.sort_values(["event_f1_value", "augmentation_policy"], ascending=[False, True]).iloc[0]
        selected[str(detector)] = str(best["augmentation_policy"])
        best_f1 = _metric_col(best, "event_f1")
        base_f1 = _metric_col(base, "event_f1") if base is not None else float("nan")
        random_f1 = _metric_col(random, "event_f1") if random is not None else float("nan")
        best_fp = _metric_col(best, "false_positive_events")
        base_fp = _metric_col(base, "false_positive_events") if base is not None else float("nan")
        random_fp = _metric_col(random, "false_positive_events") if random is not None else float("nan")
        rows.append(
            {
                "Detector": _detector_label(detector),
                "No Aug. Event-F1": _fmt(base_f1),
                "Random Aug. Event-F1": _fmt(random_f1),
                "Best TimeEventSynth Event-F1": _fmt(best_f1),
                "Best TES Policy": _policy_label(best["augmentation_policy"]),
                "Delta vs No Aug.": _fmt_delta(best_f1 - base_f1),
                "Delta vs Random": _fmt_delta(best_f1 - random_f1),
                "FP Delta vs No Aug.": _fmt_delta(best_fp - base_fp, digits=1),
                "FP Delta vs Random": _fmt_delta(best_fp - random_fp, digits=1),
            }
        )
    return pd.DataFrame(sorted(rows, key=lambda item: _detector_sort_key(item["Detector"]))), selected


def build_selected_tes_seed_win_table(seed_metrics: pd.DataFrame, selected: dict[str, str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    valid = seed_metrics.copy()
    if "metrics_valid" in valid.columns:
        valid = valid[valid["metrics_valid"].eq(True)]
    for detector, policy in selected.items():
        group = valid[valid["detector_backbone"].astype(str) == detector]
        tes = group[group["augmentation_policy"] == policy].set_index("split_seed")
        random = group[group["augmentation_policy"] == "random_event_oversampling"].set_index("split_seed")
        common = tes.index.intersection(random.index)
        if common.empty:
            continue
        f1_delta = pd.to_numeric(tes.loc[common, "event_f1"], errors="coerce") - pd.to_numeric(random.loc[common, "event_f1"], errors="coerce")
        precision_delta = pd.to_numeric(tes.loc[common, "event_precision"], errors="coerce") - pd.to_numeric(random.loc[common, "event_precision"], errors="coerce")
        recall_delta = pd.to_numeric(tes.loc[common, "event_recall"], errors="coerce") - pd.to_numeric(random.loc[common, "event_recall"], errors="coerce")
        fp_delta = pd.to_numeric(tes.loc[common, "false_positive_events"], errors="coerce") - pd.to_numeric(random.loc[common, "false_positive_events"], errors="coerce")
        rows.append(
            {
                "Detector": _detector_label(detector),
                "TES Policy": _policy_label(policy),
                "Seeds": int(len(common)),
                "Event-F1 Delta Mean": float(f1_delta.mean()),
                "Event-F1 Wins": int((f1_delta > 0).sum()),
                "Event-F1 Win Rate": float((f1_delta > 0).mean()),
                "Precision Delta Mean": float(precision_delta.mean()),
                "Recall Delta Mean": float(recall_delta.mean()),
                "FP Reduction Mean": float((-fp_delta).mean()),
                "Lower-FP Wins": int((fp_delta < 0).sum()),
                "Lower-FP Win Rate": float((fp_delta < 0).mean()),
            }
        )
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table = table.sort_values("Detector", key=lambda col: col.map(_detector_sort_key)).reset_index(drop=True)
    for col in ["Event-F1 Delta Mean", "Precision Delta Mean", "Recall Delta Mean", "Event-F1 Win Rate", "Lower-FP Win Rate"]:
        table[col] = table[col].map(lambda value: f"{float(value):.4f}")
    table["FP Reduction Mean"] = table["FP Reduction Mean"].map(lambda value: f"{float(value):.1f}")
    return table


def build_adaptive_win_table(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    raw = build_timesynth_vs_random_win_table(seed_metrics)
    if raw.empty:
        return raw
    rows: list[dict[str, str]] = []
    for _, row in raw.iterrows():
        rows.append(
            {
                "Detector": _detector_label(row["detector_backbone"]),
                "Seeds": str(int(row["num_seeds"])),
                "Event-F1 Delta Mean": _fmt_delta(row.get("event_f1_delta_mean")),
                "Event-F1 Wins": f"{int(row.get('event_f1_wins_timeeventsynth', 0))}/{int(row['num_seeds'])}",
                "Precision Delta Mean": _fmt_delta(row.get("event_precision_delta_mean")),
                "Recall Delta Mean": _fmt_delta(row.get("event_recall_delta_mean")),
                "FP Reduction Mean": _fmt_delta(row.get("false_positive_events_reduction_mean"), digits=1),
                "Lower-FP Wins": f"{int(row.get('false_positive_events_lower_fp_seeds', 0))}/{int(row['num_seeds'])}",
            }
        )
    return pd.DataFrame(sorted(rows, key=lambda item: _detector_sort_key(item["Detector"])))


def build_threshold_tables(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    diag_path = run_dir / "threshold_diagnostics.csv"
    diagnostics = pd.read_csv(diag_path) if diag_path.exists() else pd.DataFrame()
    curve_path = run_dir / "threshold_tradeoff_curve.csv"
    if curve_path.exists():
        curve = pd.read_csv(curve_path)
        summary = summarize_threshold_tradeoff(curve)
    else:
        summary = pd.DataFrame()
    return diagnostics, summary


def build_compatibility_summary(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "synthetic_audit.csv"
    if not path.exists():
        return pd.DataFrame()
    usecols = [
        "record_type",
        "accepted",
        "compatibility_enabled",
        "compatibility_score",
        "shape_similarity",
        "amplitude_compatibility",
        "duration_compatibility",
        "context_similarity",
        "frequency_similarity",
        "trend_similarity",
        "compatibility_rejection_reason",
        "augmentation_policy",
        "selected_policy_name",
        "detector_backbone",
        "split_seed",
    ]
    header = pd.read_csv(path, nrows=0).columns
    present = [col for col in usecols if col in header]
    audit = pd.read_csv(path, usecols=present, low_memory=False)
    if "record_type" not in audit.columns:
        return pd.DataFrame()
    donor = audit[audit["record_type"].astype(str) == "donor_pair"].copy()
    if donor.empty:
        return pd.DataFrame()
    donor["accepted_bool"] = donor["accepted"].astype(str).str.lower().isin(["true", "1", "yes"])
    for col in [
        "compatibility_score",
        "shape_similarity",
        "amplitude_compatibility",
        "duration_compatibility",
        "context_similarity",
        "frequency_similarity",
        "trend_similarity",
    ]:
        if col in donor.columns:
            donor[col] = pd.to_numeric(donor[col], errors="coerce")
    group_cols = [col for col in ["detector_backbone", "augmentation_policy", "selected_policy_name", "compatibility_enabled"] if col in donor.columns]
    rows: list[dict[str, Any]] = []
    for keys, group in donor.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {col: value for col, value in zip(group_cols, keys)}
        kept = group[group["accepted_bool"]]
        rejected = group[~group["accepted_bool"]]
        row.update(
            {
                "donor_pairs": int(len(group)),
                "kept_pairs": int(len(kept)),
                "rejected_pairs": int(len(rejected)),
                "kept_rate": float(len(kept) / len(group)) if len(group) else float("nan"),
            }
        )
        for col in [
            "compatibility_score",
            "shape_similarity",
            "amplitude_compatibility",
            "duration_compatibility",
            "context_similarity",
            "frequency_similarity",
            "trend_similarity",
        ]:
            if col in group.columns:
                row[f"kept_{col}_mean"] = float(kept[col].mean()) if not kept.empty else float("nan")
                row[f"rejected_{col}_mean"] = float(rejected[col].mean()) if not rejected.empty else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def build_manifest(
    *,
    run_dir: Path,
    out_dir: Path,
    seed_metrics: pd.DataFrame,
    aggregate: pd.DataFrame,
    dataset_setup_summary: pd.DataFrame,
    threshold_tradeoff_summary: pd.DataFrame,
) -> dict[str, Any]:
    policies = sorted(seed_metrics["augmentation_policy"].dropna().astype(str).unique(), key=_policy_sort_key)
    detectors = sorted(seed_metrics["detector_backbone"].dropna().astype(str).unique(), key=_detector_sort_key)
    seeds = sorted(int(seed) for seed in seed_metrics["split_seed"].dropna().unique()) if "split_seed" in seed_metrics.columns else []
    missing: list[str] = []
    for name in [
        "seed_level_backbone_metrics.csv",
        "aggregate_backbone_comparison_metrics_reportable.csv",
        "dataset_balance_summary.json",
        "synthetic_audit.csv",
        "threshold_diagnostics.csv",
    ]:
        if not (run_dir / name).exists():
            missing.append(name)
    if threshold_tradeoff_summary.empty:
        missing.append("threshold_tradeoff_curve.csv or threshold_tradeoff_summary.csv")
    total_row = dataset_setup_summary[dataset_setup_summary["domain"] == "TOTAL"]
    total_series = float(total_row["series_mean"].iloc[0]) if not total_row.empty else float("nan")
    return {
        "run_dir": str(run_dir),
        "artifact_dir": str(out_dir),
        "num_seeds": len(seeds),
        "seeds": seeds,
        "num_detectors": len(detectors),
        "detectors": detectors,
        "num_policies": len(policies),
        "policies": policies,
        "mean_series_per_seed": total_series,
        "aggregate_rows": int(len(aggregate)),
        "seed_metric_rows": int(len(seed_metrics)),
        "missing_or_not_reconstructable": missing,
        "evaluation_tables": [
            "dataset_setup_summary",
            "paper_main_full_metrics",
            "paper_policy_ablation_event_f1",
            "paper_adaptive_timeeventsynth_vs_baselines",
            "paper_best_timeeventsynth_vs_baselines",
            "paper_adaptive_timeeventsynth_vs_random_win_table",
            "paper_best_timeeventsynth_vs_random_win_table",
            "paper_compatibility_filter_summary",
            "paper_threshold_diagnostics",
            "paper_threshold_tradeoff_summary",
        ],
    }


def write_manifest_md(manifest: dict[str, Any], out_dir: Path) -> None:
    lines = [
        "# Evaluation Artifact Manifest",
        "",
        f"- Run dir: `{manifest['run_dir']}`",
        f"- Artifact dir: `{manifest['artifact_dir']}`",
        f"- Seeds: {manifest['num_seeds']} ({manifest['seeds']})",
        f"- Detectors: {manifest['num_detectors']} ({', '.join(manifest['detectors'])})",
        f"- Policies: {manifest['num_policies']} ({', '.join(manifest['policies'])})",
        f"- Mean series per seed: {manifest['mean_series_per_seed']:.1f}",
        "",
        "## Paper Tables",
    ]
    for table in manifest["evaluation_tables"]:
        lines.append(f"- `{table}.csv` / `{table}.md`")
    if manifest["missing_or_not_reconstructable"]:
        lines.extend(["", "## Missing Or Not Reconstructable From Saved Outputs"])
        for item in manifest["missing_or_not_reconstructable"]:
            lines.append(f"- `{item}`")
    (out_dir / "evaluation_artifact_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_run(run_dir: Path, out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_metrics = _read_csv(run_dir, ["seed_level_backbone_metrics.csv", "backbone_comparison_metrics.csv"])
    aggregate = _read_csv(
        run_dir,
        [
            "aggregate_backbone_comparison_metrics_reportable.csv",
            "aggregate_metrics_reportable.csv",
            "aggregate_backbone_comparison_metrics.csv",
            "aggregate_metrics.csv",
        ],
    )

    dataset_setup = build_dataset_setup(run_dir)
    dataset_setup_summary = build_dataset_setup_summary(dataset_setup)
    _write_table(dataset_setup, out_dir, "dataset_setup_by_seed_domain")
    _write_table(dataset_setup_summary, out_dir, "dataset_setup_summary")

    _write_table(build_full_metrics_table(aggregate), out_dir, "paper_main_full_metrics")
    _write_table(build_policy_ablation_table(aggregate), out_dir, "paper_policy_ablation_event_f1")
    _write_table(build_adaptive_vs_baselines(aggregate), out_dir, "paper_adaptive_timeeventsynth_vs_baselines")

    best_table, selected = build_best_tes_vs_baselines(aggregate)
    _write_table(best_table, out_dir, "paper_best_timeeventsynth_vs_baselines")
    _write_table(build_adaptive_win_table(seed_metrics), out_dir, "paper_adaptive_timeeventsynth_vs_random_win_table")
    _write_table(build_selected_tes_seed_win_table(seed_metrics, selected), out_dir, "paper_best_timeeventsynth_vs_random_win_table")

    threshold_diagnostics, threshold_tradeoff_summary = build_threshold_tables(run_dir)
    _write_table(threshold_diagnostics, out_dir, "paper_threshold_diagnostics")
    if threshold_tradeoff_summary.empty and (run_dir / "threshold_tradeoff_summary.csv").exists():
        threshold_tradeoff_summary = pd.read_csv(run_dir / "threshold_tradeoff_summary.csv")
    _write_table(threshold_tradeoff_summary, out_dir, "paper_threshold_tradeoff_summary")

    compatibility_summary = build_compatibility_summary(run_dir)
    _write_table(compatibility_summary, out_dir, "paper_compatibility_filter_summary")

    manifest = build_manifest(
        run_dir=run_dir,
        out_dir=out_dir,
        seed_metrics=seed_metrics,
        aggregate=aggregate,
        dataset_setup_summary=dataset_setup_summary,
        threshold_tradeoff_summary=threshold_tradeoff_summary,
    )
    (out_dir / "evaluation_artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_manifest_md(manifest, out_dir)
    return manifest


def build_cross_run_index(manifests: list[dict[str, Any]], out_dir: Path) -> None:
    rows = [
        {
            "run_dir": item["run_dir"],
            "artifact_dir": item["artifact_dir"],
            "num_seeds": item["num_seeds"],
            "seeds": ",".join(str(seed) for seed in item["seeds"]),
            "num_detectors": item["num_detectors"],
            "num_policies": item["num_policies"],
            "mean_series_per_seed": item["mean_series_per_seed"],
            "missing_or_not_reconstructable": "; ".join(item["missing_or_not_reconstructable"]),
        }
        for item in manifests
    ]
    _write_table(pd.DataFrame(rows), out_dir, "cross_run_evaluation_index")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper Evaluation tables from completed experiment output directories.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="Completed experiment output directory. Can be passed multiple times.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Directory for combined artifacts. Defaults to <first-run>/evaluation_artifacts.")
    args = parser.parse_args()

    run_dirs = [path.resolve() for path in args.run_dir]
    if len(run_dirs) == 1:
        out_root = args.out_dir.resolve() if args.out_dir else run_dirs[0] / "evaluation_artifacts"
        manifest = process_run(run_dirs[0], out_root)
        print(manifest["artifact_dir"])
        return

    out_root = args.out_dir.resolve() if args.out_dir else Path("outputs") / "evaluation_artifacts_combined"
    out_root.mkdir(parents=True, exist_ok=True)
    manifests: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        run_out = out_root / run_dir.name
        manifests.append(process_run(run_dir, run_out))
    build_cross_run_index(manifests, out_root)
    print(out_root)


if __name__ == "__main__":
    main()
