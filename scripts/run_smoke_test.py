from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alignment.learned_event_time import assign_learned_event_time
from src.alignment.normalized_time import assign_normalized_event_time
from src.datasets.event_extractor import extract_event_segments
from src.datasets.split_builder import build_series_splits
from src.datasets.tsb_loader import load_tsb_like_csv, timelines_from_frame
from src.evaluation.reconstruction_metrics import mean_squared_error
from src.experiments.run_ablation import compare_retrieval_methods
from src.experiments.run_masked_completion import run_masked_completion
from src.experiments.run_real_only import run_real_only_experiment
from src.synthesis.donor_retrieval import retrieve_donors
from src.synthesis.event_window_synthesizer import synthesize_event_window
from src.synthesis.uncertainty_filter import keep_if_confident


def _load_config() -> dict:
    with (PROJECT_ROOT / "configs" / "tsb_uad.yaml").open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _make_synthetic_dataset(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    num_series = 8
    series_length = 120
    for series_idx in range(num_series):
        base = np.sin(np.linspace(0, 4 * np.pi, series_length)) + 0.1 * rng.normal(size=series_length)
        labels = np.zeros(series_length, dtype=int)
        for event_idx in range(2):
            start = 15 + event_idx * 40 + int(rng.integers(-3, 4))
            end = min(start + int(rng.integers(6, 12)), series_length - 1)
            base[start:end] += 2.5 + 0.5 * rng.normal(size=end - start)
            labels[start:end] = 1
        for time_idx, (value, label) in enumerate(zip(base, labels)):
            rows.append(
                {
                    "series_id": f"series_{series_idx}",
                    "time_index": time_idx,
                    "value": float(value),
                    "label": int(label),
                }
            )
    return pd.DataFrame(rows)


def _event_dicts(timelines: list[dict], config: dict) -> list[dict]:
    events = []
    for timeline in timelines:
        segments = extract_event_segments(
            timeline,
            min_event_length=int(config["event_extraction"]["min_event_length"]),
            merge_gap=int(config["event_extraction"]["merge_gap"]),
            context_padding=int(config["event_extraction"]["context_padding"]),
        )
        for segment in segments:
            center = 0.0 if not segment.local_time else float(np.mean(segment.local_time))
            events.append(
                {
                    "event_id": segment.event_id,
                    "series_id": segment.series_id,
                    "start_idx": segment.start_idx,
                    "end_idx": segment.end_idx,
                    "values": list(segment.values),
                    "labels": list(segment.labels),
                    "local_time": list(segment.local_time),
                    "local_time_center": center,
                    "context_values": list(segment.context_values),
                    "event_label": "anomaly",
                }
            )
    return events


def main() -> None:
    config = _load_config()
    processed_dir = PROJECT_ROOT / "data" / "processed"
    outputs_dir = PROJECT_ROOT / "outputs"
    processed_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    frame = _make_synthetic_dataset(int(config["seed"]))
    csv_path = processed_dir / "smoke_dataset.csv"
    frame.to_csv(csv_path, index=False)

    loaded = load_tsb_like_csv(csv_path)
    timelines = timelines_from_frame(loaded)
    splits = build_series_splits(
        timelines,
        train_ratio=float(config["data"]["train_ratio"]),
        dev_ratio=float(config["data"]["dev_ratio"]),
        test_ratio=float(config["data"]["test_ratio"]),
        seed=int(config["seed"]),
    )

    train_events = _event_dicts(splits["train"], config)
    dev_events = _event_dicts(splits["dev"], config)
    _ = assign_normalized_event_time(train_events, int(config["alignment"]["normalized_num_bins"]))
    learned_events = assign_learned_event_time(train_events, int(config["alignment"]["normalized_num_bins"]))

    real_only_result = run_real_only_experiment(config, use_synthetic=True)
    real_only_metrics = {
        **real_only_result["test_point_metrics"],
        **real_only_result["test_event_metrics"],
    }

    donor_matches = retrieve_donors(
        train_events[0],
        train_events[1:],
        top_k=int(config["synthesis"]["retrieval_top_k"]),
        method="normalized_time",
    ) if len(train_events) > 1 else []
    synthetic_event = synthesize_event_window(train_events[0], donor_matches, strategy="blend_topk") if donor_matches else None
    kept = keep_if_confident(synthetic_event, min_support=-10.0)
    masked_outputs = run_masked_completion(train_events, dev_events[:2]) if dev_events else []
    ablation = compare_retrieval_methods(train_events[0], train_events[1:]) if len(train_events) > 1 else {}

    reconstruction_mse = None
    if kept and synthetic_event is not None:
        reconstruction_mse = mean_squared_error(train_events[0]["values"], synthetic_event["values"])

    summary = {
        "num_train_series": len(splits["train"]),
        "num_dev_series": len(splits["dev"]),
        "num_test_series": len(splits["test"]),
        "num_train_events": len(train_events),
        "num_dev_events": len(dev_events),
        "real_only_metrics": real_only_metrics,
        "num_learned_events": len(learned_events),
        "num_masked_outputs": len(masked_outputs),
        "synthetic_event_kept": kept,
        "reconstruction_mse": reconstruction_mse,
        "ablation_methods": sorted(ablation),
    }

    with (outputs_dir / "smoke_test_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
