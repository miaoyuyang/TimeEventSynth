"""Run a real-only anomaly/event detection baseline."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.experiments.artifacts import save_standard_experiment_bundle
from src.experiments.config import load_config
from src.experiments.pipeline import evaluate_detector, load_options_from_config, load_records_for_experiment, split_records


def _write_pointwise_scores(path: Path, records, scores) -> None:
    rows = []
    for record in records:
        for index, (score, label) in enumerate(zip(scores[record.series_id], record.labels)):
            rows.append(
                {
                    "series_id": record.series_id,
                    "time_index": index,
                    "label": int(label),
                    "score": float(score),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def run_real_only_experiment(config: dict[str, Any], *, use_synthetic: bool = False) -> dict[str, Any]:
    records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
    )
    train_records, val_records, test_records, split_ids = split_records(records, config)
    result = evaluate_detector(
        config,
        train_records,
        val_records,
        test_records,
        labeled_fraction=1.0,
        real_train_count=len(train_records),
    )
    return {
        "records": records,
        "data_path": data_path,
        "load_summary": load_summary,
        "split_ids": split_ids,
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real-only anomaly/event detection baseline.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    payload = run_real_only_experiment(config, use_synthetic=args.use_synthetic)
    result = payload["result"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "synthetic" if args.use_synthetic else config.get("dataset", {}).get("name", "real_only")
    output_dir = PROJECT_ROOT / "outputs" / "real_only" / str(run_label) / timestamp

    metrics = {
        "split_sizes": result["split_sizes"],
        "threshold_selection": result["threshold_selection"],
        "test_point_metrics": result["test_point_metrics"],
        "test_event_metrics": result["test_event_metrics"],
        "detector": result["detector"],
        "train_label_stats": result["train_label_stats"],
    }
    test_records = [record for record in payload["records"] if record.series_id in set(payload["split_ids"]["test"])]
    save_standard_experiment_bundle(
        output_dir,
        config=config,
        project_root=PROJECT_ROOT,
        experiment_name="real_only",
        records=payload["records"],
        data_path=payload["data_path"],
        load_options=load_options_from_config(config),
        load_summary=payload["load_summary"],
        split_ids=payload["split_ids"],
        metrics=metrics,
        per_series_rows=result["per_series_metrics"],
        synthesis_requested=False,
        extra_files={"train_label_stats.json": result["train_label_stats"]},
    )
    _write_pointwise_scores(output_dir / "per_series_scores.csv", test_records, result["scores"]["test"])

    print(json.dumps(metrics, indent=2))
    print(str(output_dir))


def _load_config(path: Path) -> dict[str, Any]:
    return load_config(path)


def _make_synthetic_records(seed: int = 42):
    from src.datasets.synthetic_data import make_synthetic_records

    return make_synthetic_records(seed)


def _flatten_labels_and_scores(records, scores):
    from src.experiments.pipeline import _flatten_labels_and_scores as flatten

    return flatten(records, scores)


if __name__ == "__main__":
    main()
