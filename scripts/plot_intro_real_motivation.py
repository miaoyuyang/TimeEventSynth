"""Plot a real-data motivation example for TimeEventSynth.

The figure uses real TSB-UAD snippets selected from an experiment audit:
one recipient event, one donor selected by unfiltered transfer, and one donor
selected by compatibility-aware transfer for the same recipient event.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.datasets.tsb_loader import load_tsb_records


SUPPORTED_SUFFIXES = (".csv", ".tsv", ".txt", ".out", "")


def _resolve_pointer_path(path: Path) -> Path:
    """Resolve text-file path pointers used by local dataset placeholders."""
    path = path.expanduser()
    if path.is_dir():
        return path
    if path.is_file():
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text and "\n" not in text and len(text) < 512:
            candidate = (path.parent / text).resolve()
            if candidate.exists():
                return candidate
    return path


def _split_pipe(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    text = str(value)
    if not text:
        return []
    return [part for part in text.split("|") if part != ""]


def _first_donor(row: pd.Series) -> tuple[str, int, int] | None:
    series_ids = _split_pipe(row.get("donor_series_ids"))
    starts = _split_pipe(row.get("donor_starts"))
    ends = _split_pipe(row.get("donor_ends"))
    if not series_ids or not starts or not ends:
        return None
    return series_ids[0], int(float(starts[0])), int(float(ends[0]))


def _find_series_file(data_root: Path, series_id: str) -> Path:
    base = data_root / series_id
    for suffix in SUPPORTED_SUFFIXES:
        candidate = base.with_suffix(suffix) if suffix else base
        if candidate.exists() and candidate.is_file():
            return candidate
    matches = list(data_root.rglob(Path(series_id).name + ".*"))
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find raw file for series_id={series_id!r} under {data_root}")


def _load_series_values(data_root: Path, series_id: str) -> tuple[np.ndarray, np.ndarray]:
    file_path = _find_series_file(data_root, series_id)
    records = load_tsb_records(file_path)
    for record in records:
        if record.series_id == Path(series_id).name or record.series_id == series_id or record.source_path == str(file_path):
            return np.asarray(record.values, dtype=float).reshape(-1), np.asarray(record.labels, dtype=int)
    record = records[0]
    return np.asarray(record.values, dtype=float).reshape(-1), np.asarray(record.labels, dtype=int)


def _window(values: np.ndarray, labels: np.ndarray, start: int, end: int, pad_scale: float = 2.0) -> tuple[np.ndarray, np.ndarray, int, int]:
    length = max(end - start, 1)
    pad = max(20, int(length * pad_scale))
    left = max(0, start - pad)
    right = min(len(values), end + pad)
    snippet = values[left:right].astype(float)
    snippet_labels = labels[left:right].astype(int)
    mean = float(np.nanmean(snippet))
    std = float(np.nanstd(snippet))
    if std <= 1e-12:
        std = 1.0
    return (snippet - mean) / std, snippet_labels, start - left, end - left


def _select_example(audit: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    usable = audit[audit["accepted"].astype(str).str.lower().eq("true")].copy()
    usable = usable[usable["donor_starts"].fillna("").astype(str).ne("")]
    no_filter = usable[usable["method"].isin(["all_donors_no_filter", "groupwise_cross_dataset_all"])]
    compatible = usable[
        usable["method"].isin(
            [
                "groupwise_cross_dataset_compatible",
                "groupwise_compatibility_strict",
                "adaptive_groupwise_transfer",
            ]
        )
        | usable["compatibility_enabled"].astype(str).str.lower().eq("true")
    ]
    key_cols = ["target_series_id", "target_start", "target_end"]
    for _, naive_row in no_filter.iterrows():
        mask = np.ones(len(compatible), dtype=bool)
        for col in key_cols:
            mask &= compatible[col].astype(str).eq(str(naive_row[col])).to_numpy()
        matches = compatible[mask]
        if matches.empty:
            continue
        comp_row = matches.sort_values("compatibility_score", ascending=False).iloc[0]
        if _first_donor(naive_row) and _first_donor(comp_row):
            return naive_row, comp_row

    naive_candidates = no_filter[no_filter.apply(lambda row: _first_donor(row) is not None, axis=1)]
    comp_candidates = compatible[compatible.apply(lambda row: _first_donor(row) is not None, axis=1)]
    if not naive_candidates.empty and not comp_candidates.empty:
        naive_row = naive_candidates.sort_values("donor_similarity", ascending=False).iloc[0]
        comp_row = comp_candidates.sort_values("compatibility_score", ascending=False).iloc[0]
        return naive_row, comp_row

    raise ValueError("Could not find usable unfiltered and compatibility-aware donor rows in the audit.")


def _plot_panel(ax: plt.Axes, title: str, values: np.ndarray, labels: np.ndarray, start: int, end: int, color: str) -> None:
    x = np.linspace(0.0, 1.0, len(values))
    ax.plot(x, values, color="#333333", linewidth=1.2)
    left = start / max(len(values) - 1, 1)
    right = max(end - 1, start) / max(len(values) - 1, 1)
    ax.axvspan(left, right, color=color, alpha=0.22, linewidth=0)
    ax.set_title(title, fontsize=9, pad=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#777777")
    ax.spines["bottom"].set_color("#777777")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/raw/TSB-UAD-Public-v2"))
    parser.add_argument(
        "--audit-csv",
        type=Path,
        default=Path("outputs/backbone_augmentation/TSB-UAD-Public-v2/paper_main_100_seed0/synthetic_audit.csv"),
    )
    parser.add_argument("--out", type=Path, default=Path("outputs/figures/intro_real_motivation.pdf"))
    parser.add_argument("--png", action="store_true", help="Also save a PNG next to the PDF.")
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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.9), constrained_layout=True)
    _plot_panel(axes[0], "Recipient event", *target_win, color="#4C78A8")
    _plot_panel(axes[1], "Unfiltered donor", *naive_win, color="#D9853B")
    _plot_panel(axes[2], "Compatible donor", *comp_win, color="#4C78A8")
    fig.suptitle("Real TSB-UAD example motivating compatibility-aware event transfer", fontsize=10, y=1.06)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight")
    if args.png:
        fig.savefig(args.out.with_suffix(".png"), dpi=300, bbox_inches="tight")

    meta = args.out.with_suffix(".txt")
    meta.write_text(
        "\n".join(
            [
                f"target_series_id={target_id}",
                f"target_window=[{target_start}, {target_end})",
                f"unfiltered_donor={naive_donor[0]}:[{naive_donor[1]}, {naive_donor[2]})",
                f"compatible_donor={comp_donor[0]}:[{comp_donor[1]}, {comp_donor[2]})",
                f"compatibility_score={comp_row.get('compatibility_score', '')}",
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
    print(f"Saved {meta}")


if __name__ == "__main__":
    main()
