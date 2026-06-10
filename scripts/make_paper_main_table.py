#!/usr/bin/env python3
"""Export a compact paper-facing main-result table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


POLICY_LABELS = {
    "real_only": "No Aug.",
    "random_event_oversampling": "Random Aug.",
    "all_donors_no_filter": "Sampling Aug.",
    "adaptive_groupwise_transfer": "TimeEventSynth",
    "groupwise_cross_dataset_all": "TES-All",
    "groupwise_cross_dataset_compatible": "TES-Compatible",
    "groupwise_compatibility_strict": "TES-Strict",
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
POLICY_ORDER = ["real_only", "random_event_oversampling", "all_donors_no_filter", "adaptive_groupwise_transfer"]


def _fmt_mean_std(row: pd.Series, metric: str) -> str:
    mean = row.get(f"{metric}_mean")
    std = row.get(f"{metric}_std")
    if pd.isna(mean):
        return ""
    if pd.isna(std):
        return f"{float(mean):.4f}"
    return f"{float(mean):.4f} +/- {float(std):.4f}"


def build_table(run_dir: Path) -> pd.DataFrame:
    src = run_dir / "aggregate_backbone_comparison_metrics_reportable.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing aggregate CSV: {src}")
    frame = pd.read_csv(src)
    rows: list[dict[str, str]] = []
    for detector in DETECTOR_ORDER:
        for policy in POLICY_ORDER:
            subset = frame[
                (frame["detector_backbone"] == detector)
                & (frame["augmentation_policy"] == policy)
            ]
            if subset.empty:
                continue
            row = subset.iloc[0]
            rows.append(
                {
                    "Detector": DETECTOR_LABELS.get(detector, detector),
                    "Method": POLICY_LABELS.get(policy, policy),
                    "Event-F1": _fmt_mean_std(row, "event_f1"),
                    "Precision": _fmt_mean_std(row, "event_precision"),
                    "Recall": _fmt_mean_std(row, "event_recall"),
                    "False Positive Events": _fmt_mean_std(row, "false_positive_events"),
                    "Selected Policy": str(row.get("selected_policy_name", "")),
                }
            )
    return pd.DataFrame(rows)


def _fmt_delta(value: object, *, invert: bool = False) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    if invert:
        number = -number
    return f"{number:+.4f}"


def _fmt_win_rate(wins: object, num_seeds: object, rate: object) -> str:
    if pd.isna(wins) or pd.isna(num_seeds):
        return ""
    if pd.isna(rate):
        return f"{int(wins)}/{int(num_seeds)}"
    return f"{int(wins)}/{int(num_seeds)} ({100.0 * float(rate):.1f}%)"


def build_timesynth_vs_random_table(run_dir: Path) -> pd.DataFrame:
    src = run_dir / "timeeventsynth_vs_random_win_table.csv"
    if not src.exists():
        return pd.DataFrame()
    frame = pd.read_csv(src)
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        detector = str(row.get("detector_backbone", ""))
        num_seeds = row.get("num_seeds")
        rows.append(
            {
                "Detector": DETECTOR_LABELS.get(detector, detector),
                "Seeds": "" if pd.isna(num_seeds) else str(int(num_seeds)),
                "Event-F1 Delta": _fmt_delta(row.get("event_f1_delta_mean")),
                "Event-F1 Wins": _fmt_win_rate(
                    row.get("event_f1_wins_timeeventsynth"),
                    num_seeds,
                    row.get("event_f1_win_rate_timeeventsynth"),
                ),
                "Precision Delta": _fmt_delta(row.get("event_precision_delta_mean")),
                "Recall Delta": _fmt_delta(row.get("event_recall_delta_mean")),
                "FP Reduction": _fmt_delta(row.get("false_positive_events_reduction_mean")),
                "Lower-FP Wins": _fmt_win_rate(
                    row.get("false_positive_events_lower_fp_seeds"),
                    num_seeds,
                    row.get("false_positive_events_lower_fp_rate"),
                ),
            }
        )
    order = {DETECTOR_LABELS.get(detector, detector): idx for idx, detector in enumerate(DETECTOR_ORDER)}
    table = pd.DataFrame(rows)
    if not table.empty:
        table["_order"] = table["Detector"].map(order).fillna(len(order))
        table = table.sort_values(["_order", "Detector"]).drop(columns=["_order"])
    return table


def build_threshold_tradeoff_table(run_dir: Path) -> pd.DataFrame:
    src = run_dir / "threshold_tradeoff_summary.csv"
    if not src.exists():
        return pd.DataFrame()
    frame = pd.read_csv(src)
    rows: list[dict[str, str]] = []
    for _, row in frame.iterrows():
        detector = str(row.get("detector_backbone", ""))
        policy = str(row.get("augmentation_policy", ""))
        if policy not in {"random_event_oversampling", "adaptive_groupwise_transfer"}:
            continue
        rows.append(
            {
                "Detector": DETECTOR_LABELS.get(detector, detector),
                "Method": POLICY_LABELS.get(policy, policy),
                "Seed": str(int(row["split_seed"])) if "split_seed" in row and pd.notna(row["split_seed"]) else "",
                "Best-F1 Threshold": "" if pd.isna(row.get("best_event_f1_threshold")) else f"{float(row.get('best_event_f1_threshold')):.6g}",
                "Best-F1": "" if pd.isna(row.get("best_event_f1")) else f"{float(row.get('best_event_f1')):.4f}",
                "Best-F1 FP": "" if pd.isna(row.get("best_false_positive_events")) else f"{float(row.get('best_false_positive_events')):.0f}",
                "FP-Limited Threshold": "" if pd.isna(row.get("fp_limited_threshold")) else f"{float(row.get('fp_limited_threshold')):.6g}",
                "FP-Limited F1": "" if pd.isna(row.get("fp_limited_event_f1")) else f"{float(row.get('fp_limited_event_f1')):.4f}",
                "FP-Limited Precision": "" if pd.isna(row.get("fp_limited_event_precision")) else f"{float(row.get('fp_limited_event_precision')):.4f}",
                "FP-Limited Recall": "" if pd.isna(row.get("fp_limited_event_recall")) else f"{float(row.get('fp_limited_event_recall')):.4f}",
                "FP-Limited FP": "" if pd.isna(row.get("fp_limited_false_positive_events")) else f"{float(row.get('fp_limited_false_positive_events')):.0f}",
            }
        )
    table = pd.DataFrame(rows)
    if not table.empty:
        detector_order = {DETECTOR_LABELS.get(detector, detector): idx for idx, detector in enumerate(DETECTOR_ORDER)}
        method_order = {"Random Aug.": 0, "TimeEventSynth": 1}
        table["_detector_order"] = table["Detector"].map(detector_order).fillna(len(detector_order))
        table["_method_order"] = table["Method"].map(method_order).fillna(len(method_order))
        table = table.sort_values(["_detector_order", "Seed", "_method_order"]).drop(columns=["_detector_order", "_method_order"])
    return table


def to_markdown(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        values = [str(value).replace("|", "\\|") for value in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper main-result table from a backbone run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--wins-csv", type=Path, default=None)
    parser.add_argument("--wins-md", type=Path, default=None)
    parser.add_argument("--tradeoff-csv", type=Path, default=None)
    parser.add_argument("--tradeoff-md", type=Path, default=None)
    args = parser.parse_args()

    table = build_table(args.run_dir)
    out_csv = args.out_csv or (args.run_dir / "paper_main_table.csv")
    out_md = args.out_md or (args.run_dir / "paper_main_table.md")
    table.to_csv(out_csv, index=False)
    out_md.write_text(to_markdown(table), encoding="utf-8")
    print(out_csv)
    print(out_md)

    wins_table = build_timesynth_vs_random_table(args.run_dir)
    if not wins_table.empty:
        wins_csv = args.wins_csv or (args.run_dir / "paper_timeeventsynth_vs_random_win_table.csv")
        wins_md = args.wins_md or (args.run_dir / "paper_timeeventsynth_vs_random_win_table.md")
        wins_table.to_csv(wins_csv, index=False)
        wins_md.write_text(to_markdown(wins_table), encoding="utf-8")
        print(wins_csv)
        print(wins_md)

    tradeoff_table = build_threshold_tradeoff_table(args.run_dir)
    if not tradeoff_table.empty:
        tradeoff_csv = args.tradeoff_csv or (args.run_dir / "paper_threshold_tradeoff_summary.csv")
        tradeoff_md = args.tradeoff_md or (args.run_dir / "paper_threshold_tradeoff_summary.md")
        tradeoff_table.to_csv(tradeoff_csv, index=False)
        tradeoff_md.write_text(to_markdown(tradeoff_table), encoding="utf-8")
        print(tradeoff_csv)
        print(tradeoff_md)


if __name__ == "__main__":
    main()
