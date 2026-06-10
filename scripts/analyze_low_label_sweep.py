#!/usr/bin/env python3
"""Generate a human-readable analysis report from a completed low-label sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporting import analyze_low_label_sweep, format_low_label_sweep_markdown


def _load_summary(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze a completed low-label sweep and write reports.")
    parser.add_argument("--comparison", type=Path, required=True, help="Path to low_label_sweep_comparison.csv")
    parser.add_argument("--per-series", type=Path, required=True, help="Path to low_label_sweep_per_series.csv")
    parser.add_argument("--summary", type=Path, required=True, help="Path to low_label_sweep_summary.json")
    parser.add_argument("--out", type=Path, required=True, help="Output markdown path (analysis_report.md)")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional machine-readable JSON output path (defaults to analysis_report.json beside --out)",
    )
    args = parser.parse_args()

    comparison = pd.read_csv(args.comparison)
    per_series = pd.read_csv(args.per_series)
    summary = _load_summary(args.summary)

    analysis = analyze_low_label_sweep(comparison, per_series, summary)
    analysis["input_files"] = {
        "comparison": str(args.comparison.resolve()),
        "per_series": str(args.per_series.resolve()),
        "summary": str(args.summary.resolve()),
    }

    markdown = format_low_label_sweep_markdown(analysis)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(markdown, encoding="utf-8")

    json_path = args.json_out or args.out.with_name("analysis_report.json")
    json_path.write_text(json.dumps(analysis, indent=2), encoding="utf-8")

    print(f"Wrote markdown report to {args.out}")
    print(f"Wrote JSON report to {json_path}")
    if analysis.get("warnings"):
        print(f"Warnings: {len(analysis['warnings'])}")


if __name__ == "__main__":
    main()
