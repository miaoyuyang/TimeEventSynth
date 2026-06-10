from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.dataset_stats import build_dataset_stats
from src.datasets.load_options import add_dataset_load_arguments, infer_dataset_name, load_options_from_args
from src.datasets.tsb_loader import load_tsb_records_with_summary
from src.experiments.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a raw time-series anomaly dataset.")
    add_dataset_load_arguments(parser)
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Optional experiment YAML (applies dataset include/exclude filters from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for machine-readable stats JSON (default: outputs/inspect/<dataset-name>).",
    )
    args = parser.parse_args()

    if args.data is None:
        parser.error("--data is required")

    data_path = Path(args.data)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path

    load_options = load_options_from_args(
        args,
        config=load_config(args.config) if args.config is not None else None,
    )
    if load_options.dataset_name is None:
        load_options.dataset_name = infer_dataset_name(data_path)

    records, load_summary = load_tsb_records_with_summary(data_path, options=load_options)
    stats = build_dataset_stats(
        records,
        data_path=str(data_path),
        load_options=load_options,
        load_summary=load_summary,
    )

    output_dir = args.output_dir
    if output_dir is None:
        output_dir = PROJECT_ROOT / "outputs" / "inspect" / str(load_options.dataset_name)
    else:
        output_dir = output_dir if output_dir.is_absolute() else PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = output_dir / "dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"number of series: {stats['num_series']}")
    print("length statistics:", stats["length_statistics"])
    print(f"number of event windows: {stats['num_event_windows']}")
    print("event length statistics:", stats["event_length_statistics"])
    print(f"anomaly/event point ratio: {stats['anomaly_point_ratio']:.6f}")
    print("top 10 longest series:")
    for row in stats["top_longest_series"]:
        print(f"  {row['series_id']}: {row['length']}")
    print("top 10 series by event count:")
    for row in stats["top_series_by_event_count"]:
        print(f"  {row['series_id']}: {row['num_events']}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
