#!/usr/bin/env python3
"""Regenerate analysis_report.md/json from a completed compatibility-transfer run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporting import analyze_compatibility_transfer, format_compatibility_transfer_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze compatibility-transfer experiment outputs.")
    parser.add_argument("--run-dir", type=Path, required=True, help="Directory containing comparison_metrics.csv")
    parser.add_argument("--out", type=Path, default=None, help="Markdown output (default: run-dir/analysis_report.md)")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    comparison = pd.read_csv(run_dir / "comparison_metrics.csv")
    rejection_summary = json.loads((run_dir / "rejection_summary.json").read_text(encoding="utf-8"))
    compatibility_summary = json.loads((run_dir / "compatibility_summary.json").read_text(encoding="utf-8"))
    per_event_path = run_dir / "per_event_metrics.csv"
    masked_rows = pd.read_csv(per_event_path).to_dict(orient="records") if per_event_path.exists() else None

    analysis = analyze_compatibility_transfer(
        comparison,
        rejection_summary,
        compatibility_summary,
        masked_per_event=masked_rows,
    )
    markdown = format_compatibility_transfer_markdown(analysis)
    out_md = args.out or (run_dir / "analysis_report.md")
    out_md.write_text(markdown, encoding="utf-8")
    out_json = out_md.with_name("analysis_report.json")
    out_json.write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
