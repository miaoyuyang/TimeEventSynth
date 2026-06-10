from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _save(fig: plt.Figure, stem: Path) -> None:
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    plt.close(fig)


def _line_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty:
        return
    metrics = [("auprc", "AUPRC"), ("event_f1", "Event F1"), ("auroc", "AUROC")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (metric, title) in zip(axes, metrics):
        for method, group in frame.groupby("method"):
            summary = group.groupby("label_fraction", dropna=False)[metric].mean().reset_index()
            ax.plot(summary["label_fraction"], summary[metric], marker="o", label=method)
        ax.set_title(f"Label Fraction vs {title}")
        ax.set_xlabel("Label Fraction")
        ax.set_ylabel(title)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.05))
    _save(fig, out_dir / "label_fraction_vs_metrics")


def _bar_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty:
        return
    summary = frame.groupby("method", dropna=False)[["auprc", "event_f1"]].mean().sort_values("event_f1", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(summary))
    ax.bar([idx - 0.18 for idx in x], summary["auprc"], width=0.36, label="AUPRC")
    ax.bar([idx + 0.18 for idx in x], summary["event_f1"], width=0.36, label="Event F1")
    ax.set_xticks(list(x))
    ax.set_xticklabels(summary.index, rotation=30, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Method Comparison")
    ax.legend()
    _save(fig, out_dir / "method_comparison")


def _scatter_plot(frame: pd.DataFrame, out_dir: Path) -> None:
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in frame.groupby("method"):
        ax.scatter(group["num_synthetic_windows"], group["event_f1"], label=method, alpha=0.8)
    ax.set_xlabel("Synthetic Windows")
    ax.set_ylabel("Event F1")
    ax.set_title("Synthetic Windows vs Performance")
    ax.legend(loc="best", fontsize=8)
    _save(fig, out_dir / "synthetic_windows_vs_performance")


def main() -> None:
    summary_csv = PROJECT_ROOT / "outputs" / "summary" / "all_results.csv"
    out_dir = PROJECT_ROOT / "outputs" / "summary" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")
    frame = pd.read_csv(summary_csv)
    _line_plot(frame, out_dir)
    _bar_plot(frame, out_dir)
    _scatter_plot(frame, out_dir)
    print(str(out_dir))


if __name__ == "__main__":
    main()
