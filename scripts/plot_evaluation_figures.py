#!/usr/bin/env python3
"""Plot paper-facing Evaluation figures from generated artifact tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace("+", "", regex=False), errors="coerce")


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_best_tes_main(artifact_dir: Path, out_dir: Path) -> None:
    table = pd.read_csv(artifact_dir / "paper_best_timeeventsynth_vs_baselines.csv")
    detectors = table["Detector"].tolist()
    x = np.arange(len(detectors))
    width = 0.24

    no_aug = _numeric(table["No Aug. Event-F1"])
    random_aug = _numeric(table["Random Aug. Event-F1"])
    best_tes = _numeric(table["Best TimeEventSynth Event-F1"])

    fig, ax = plt.subplots(figsize=(8.2, 3.2))
    ax.bar(x - width, no_aug, width, label="No Aug.", color="#8c8c8c")
    ax.bar(x, random_aug, width, label="Random Aug.", color="#5b8cc0")
    ax.bar(x + width, best_tes, width, label="Best TimeEventSynth", color="#d5853b")
    ax.set_ylabel("Event-F1")
    ax.set_xticks(x)
    ax.set_xticklabels(detectors, rotation=25, ha="right")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "best_timeeventsynth_event_f1_by_detector")


def plot_fp_tradeoff(artifact_dir: Path, out_dir: Path) -> None:
    table = pd.read_csv(artifact_dir / "paper_best_timeeventsynth_vs_baselines.csv")
    detectors = table["Detector"].tolist()
    f1_delta = _numeric(table["Delta vs Random"])
    fp_delta = _numeric(table["FP Delta vs Random"])

    colors = ["#2f8f5b" if fp <= 0 else "#b24a3a" for fp in fp_delta]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.scatter(f1_delta, fp_delta, s=58, c=colors, edgecolor="white", linewidth=0.8)
    for detector, x, y in zip(detectors, f1_delta, fp_delta):
        ax.annotate(detector, (x, y), textcoords="offset points", xytext=(5, 4), fontsize=8)
    ax.set_xlabel("Event-F1 gain over Random Aug.")
    ax.set_ylabel("FP-event change vs Random Aug.")
    ax.grid(alpha=0.25, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "timeeventsynth_fp_tradeoff_vs_random")


def plot_policy_ablation(artifact_dir: Path, out_dir: Path) -> None:
    table = pd.read_csv(artifact_dir / "paper_policy_ablation_event_f1.csv")
    detector_col = table["Detector"]
    method_cols = [col for col in table.columns if col != "Detector"]
    values = table[method_cols].apply(pd.to_numeric, errors="coerce")

    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    im = ax.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(method_cols)))
    ax.set_xticklabels(method_cols, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(detector_col)))
    ax.set_yticklabels(detector_col)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Event-F1")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _save(fig, out_dir, "policy_ablation_event_f1_heatmap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Evaluation figures from generated TimeEventSynth artifact tables.")
    parser.add_argument("--artifact-dir", type=Path, required=True, help="Directory containing paper_*.csv artifacts.")
    parser.add_argument("--out-dir", type=Path, default=None, help="Figure output directory. Defaults to <artifact-dir>/figures.")
    args = parser.parse_args()

    artifact_dir = args.artifact_dir.resolve()
    out_dir = args.out_dir.resolve() if args.out_dir else artifact_dir / "figures"
    plot_best_tes_main(artifact_dir, out_dir)
    plot_fp_tradeoff(artifact_dir, out_dir)
    plot_policy_ablation(artifact_dir, out_dir)
    print(out_dir)


if __name__ == "__main__":
    main()
