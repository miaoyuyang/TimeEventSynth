"""Publication-style real-data motivation figure for TimeEventSynth.

This script uses the same audit-driven example as ``plot_intro_real_motivation.py``
but draws it as an introduction figure: donor candidates, compatibility check,
and recipient event. The unfiltered donor is not labeled as rejected; it is a
real example of what an unfiltered policy may use.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from plot_intro_real_motivation import (
    _first_donor,
    _load_series_values,
    _resolve_pointer_path,
    _select_example,
    _window,
)


BLUE = "#4C78A8"
ORANGE = "#D9853B"
GRAY = "#333333"
LIGHT_GRAY = "#777777"


def _plot_series(
    ax: plt.Axes,
    values: np.ndarray,
    start: int,
    end: int,
    *,
    shade_color: str,
    title: str,
    subtitle: str,
) -> None:
    x = np.linspace(0.0, 1.0, len(values))
    ax.plot(x, values, color=GRAY, linewidth=1.25)
    left = start / max(len(values) - 1, 1)
    right = max(end - 1, start) / max(len(values) - 1, 1)
    ax.axvspan(left, right, color=shade_color, alpha=0.20, linewidth=0)
    ax.set_title(title, fontsize=8.7, pad=2.0)
    ax.text(0.02, 0.96, subtitle, transform=ax.transAxes, va="top", fontsize=6.8, color=LIGHT_GRAY)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#8a8a8a")
    ax.spines["bottom"].set_color("#8a8a8a")
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)


def _box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str) -> None:
    x, y = xy
    rect = FancyBboxPatch(
        (x, y),
        0.82,
        0.20,
        boxstyle="round,pad=0.02,rounding_size=0.025",
        linewidth=0.9,
        edgecolor=color,
        facecolor=color,
        alpha=0.10,
        transform=ax.transAxes,
    )
    ax.add_patch(rect)
    ax.text(x + 0.41, y + 0.10, text, ha="center", va="center", fontsize=7.0, color=GRAY, transform=ax.transAxes)


def _figure_arrow(fig: plt.Figure, start: tuple[float, float], end: tuple[float, float], color: str, style: str = "-") -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=fig.transFigure,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=0.8,
        linestyle=style,
        color=color,
        alpha=0.90,
    )
    fig.add_artist(arrow)


def _save_snippets_csv(path: Path, panels: dict[str, tuple[np.ndarray, np.ndarray, int, int]]) -> None:
    rows: list[dict[str, object]] = []
    for panel, (values, labels, start, end) in panels.items():
        denom = max(len(values) - 1, 1)
        for idx, value in enumerate(values):
            rows.append(
                {
                    "panel": panel,
                    "x": idx / denom,
                    "value": float(value),
                    "is_event": int(start <= idx < end),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/TSB-UAD-Public-v2"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/backbone_augmentation/TSB-UAD-Public-v2/paper_main_100_seed0/synthetic_audit.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/intro_real_motivation_v2.pdf"))
    parser.add_argument("--png", action="store_true")
    args = parser.parse_args()

    data_root = _resolve_pointer_path(args.data_root)
    audit = pd.read_csv(args.audit_csv, low_memory=False)
    naive_row, comp_row = _select_example(audit)

    target_id = str(comp_row["target_series_id"])
    target_start = int(float(comp_row["target_start"]))
    target_end = int(float(comp_row["target_end"]))
    naive_donor = _first_donor(naive_row)
    comp_donor = _first_donor(comp_row)
    assert naive_donor is not None and comp_donor is not None

    target_values, target_labels = _load_series_values(data_root, target_id)
    naive_values, naive_labels = _load_series_values(data_root, naive_donor[0])
    comp_values, comp_labels = _load_series_values(data_root, comp_donor[0])

    target_win = _window(target_values, target_labels, target_start, target_end)
    naive_win = _window(naive_values, naive_labels, naive_donor[1], naive_donor[2])
    comp_win = _window(comp_values, comp_labels, comp_donor[1], comp_donor[2])

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 2.28))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[1.45, 0.82, 1.45],
        height_ratios=[1, 1],
        left=0.045,
        right=0.985,
        top=0.90,
        bottom=0.14,
        wspace=0.28,
        hspace=0.58,
    )
    ax_unfiltered = fig.add_subplot(gs[0, 0])
    ax_compatible = fig.add_subplot(gs[1, 0])
    ax_filter = fig.add_subplot(gs[:, 1])
    ax_recipient = fig.add_subplot(gs[:, 2])

    target_len = target_end - target_start
    naive_len = naive_donor[2] - naive_donor[1]
    comp_len = comp_donor[2] - comp_donor[1]
    score = float(comp_row.get("compatibility_score", np.nan))

    _plot_series(
        ax_unfiltered,
        naive_win[0],
        naive_win[2],
        naive_win[3],
        shade_color=ORANGE,
        title="Unfiltered donor candidate",
        subtitle=f"{naive_len} points; no compatibility check",
    )
    _plot_series(
        ax_compatible,
        comp_win[0],
        comp_win[2],
        comp_win[3],
        shade_color=BLUE,
        title="Compatibility-selected donor",
        subtitle=f"{comp_len} points; score={score:.3f}",
    )
    _plot_series(
        ax_recipient,
        target_win[0],
        target_win[2],
        target_win[3],
        shade_color=BLUE,
        title="Recipient event context",
        subtitle=f"{target_len} point target event",
    )

    ax_filter.axis("off")
    ax_filter.text(0.50, 0.91, "Compatibility\nfilter", ha="center", va="top", fontsize=8.5, color=GRAY, transform=ax_filter.transAxes)
    _box(ax_filter, (0.09, 0.58), "blind transfer\nmismatch risk", ORANGE)
    _box(ax_filter, (0.09, 0.27), "recipient-aware\nkeep", BLUE)
    ax_filter.text(
        0.50,
        0.08,
        "shape | scale | duration | context",
        ha="center",
        va="center",
        fontsize=6.8,
        color=LIGHT_GRAY,
        transform=ax_filter.transAxes,
    )

    _figure_arrow(fig, (0.318, 0.67), (0.395, 0.67), ORANGE, style="--")
    _figure_arrow(fig, (0.318, 0.31), (0.395, 0.31), BLUE)
    _figure_arrow(fig, (0.595, 0.31), (0.665, 0.47), BLUE)

    fig.text(0.045, 0.965, "Real TSB-UAD donor examples", ha="left", va="top", fontsize=8.8, color=GRAY)
    fig.text(0.675, 0.965, "Target series", ha="left", va="top", fontsize=8.8, color=GRAY)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")

    _save_snippets_csv(
        args.out.with_name(args.out.stem + "_snippets.csv"),
        {"unfiltered": naive_win, "compatible": comp_win, "recipient": target_win},
    )
    args.out.with_suffix(".txt").write_text(
        "\n".join(
            [
                f"target_series_id={target_id}",
                f"target_window=[{target_start}, {target_end})",
                f"unfiltered_donor={naive_donor[0]}:[{naive_donor[1]}, {naive_donor[2]})",
                f"compatible_donor={comp_donor[0]}:[{comp_donor[1]}, {comp_donor[2]})",
                f"compatibility_score={score}",
                "note=Unfiltered donor is an audit example selected by an unfiltered policy; it is not claimed to be rejected by the compatibility filter.",
                f"audit_csv={args.audit_csv}",
                f"data_root={data_root}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved {args.out}")
    if args.png:
        print(f"Saved {args.out.with_suffix('.png')}")
    print(f"Saved {args.out.with_name(args.out.stem + '_snippets.csv')}")
    print(f"Saved {args.out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
