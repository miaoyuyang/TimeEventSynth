from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.reporting import balance_score, mean_std_table, relative_improvement


def _safe_load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _extract_run_metadata(run_dir: Path) -> dict[str, Any]:
    config = _safe_load_yaml(run_dir / "config_resolved.yaml")
    data_cfg = config.get("data", {})
    experiment_cfg = config.get("experiment", {})
    detector_cfg = config.get("detector", {})
    raw_path = str(data_cfg.get("raw_path", "synthetic"))
    dataset = Path(raw_path).name if raw_path else "synthetic"
    if dataset in {"", ".", "benchmark"}:
        dataset = Path(raw_path).parent.name if raw_path else "synthetic"
    return {
        "dataset": dataset or "synthetic",
        "split_seed": int(config.get("seed", 42)),
        "label_fraction": float(experiment_cfg.get("labeled_fraction", 1.0)),
        "detector": str(detector_cfg.get("model_type", "unknown")),
        "run_type": run_dir.parent.name,
        "run_id": run_dir.name,
    }


def _rows_from_comparison_csv(path: Path) -> list[dict[str, Any]]:
    run_dir = path.parent
    metadata = _extract_run_metadata(run_dir)
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for row in frame.to_dict(orient="records"):
        rows.append({**metadata, **row})
    return rows


def _rows_from_metrics_json(path: Path) -> list[dict[str, Any]]:
    run_dir = path.parent
    if (run_dir / "comparison_metrics.csv").exists():
        return []
    metadata = _extract_run_metadata(run_dir)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    if run_dir.parent.name == "real_only":
        rows.append(
            {
                **metadata,
                "method": "real_only",
                "auroc": payload.get("test_point_metrics", {}).get("point_auroc"),
                "auprc": payload.get("test_point_metrics", {}).get("point_auprc"),
                "best_point_f1": payload.get("test_point_metrics", {}).get("point_f1"),
                "event_precision": payload.get("test_event_metrics", {}).get("event_precision"),
                "event_recall": payload.get("test_event_metrics", {}).get("event_recall"),
                "event_f1": payload.get("test_event_metrics", {}).get("event_f1"),
                "false_positive_events": payload.get("test_event_metrics", {}).get("false_positive_event_count"),
                "num_synthetic_windows": 0,
                "num_synthetic_points": 0,
            }
        )
        return rows
    if isinstance(payload, dict):
        for method, method_payload in payload.items():
            if not isinstance(method_payload, dict):
                continue
            rows.append(
                {
                    **metadata,
                    "method": method,
                    "auroc": method_payload.get("test_point_metrics", {}).get("point_auroc"),
                    "auprc": method_payload.get("test_point_metrics", {}).get("point_auprc"),
                    "best_point_f1": method_payload.get("test_point_metrics", {}).get("point_f1"),
                    "event_precision": method_payload.get("test_event_metrics", {}).get("event_precision"),
                    "event_recall": method_payload.get("test_event_metrics", {}).get("event_recall"),
                    "event_f1": method_payload.get("test_event_metrics", {}).get("event_f1"),
                    "false_positive_events": method_payload.get("test_event_metrics", {}).get("false_positive_event_count"),
                    "num_synthetic_windows": method_payload.get("num_synthetic_windows", 0),
                    "num_synthetic_points": method_payload.get("num_synthetic_points", 0),
                    "num_rejected_synthetic_windows": method_payload.get("num_rejected_synthetic_windows", 0),
                }
            )
    return rows


def collect_results(outputs_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seen_comparison_dirs: set[Path] = set()
    for csv_path in sorted(outputs_dir.glob("*/*/comparison_metrics.csv")):
        seen_comparison_dirs.add(csv_path.parent)
        rows.extend(_rows_from_comparison_csv(csv_path))
    for metrics_path in sorted(outputs_dir.glob("*/*/metrics.json")):
        rows.extend(_rows_from_metrics_json(metrics_path))
    frame = pd.DataFrame(rows)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "dataset",
                "split_seed",
                "label_fraction",
                "detector",
                "method",
                "auroc",
                "auprc",
                "best_point_f1",
                "event_precision",
                "event_recall",
                "event_f1",
                "false_positive_events",
                "num_synthetic_windows",
                "num_synthetic_points",
            ]
        )
    return frame.sort_values(["dataset", "split_seed", "label_fraction", "detector", "method"]).reset_index(drop=True)


def main() -> None:
    outputs_dir = PROJECT_ROOT / "outputs"
    summary_dir = outputs_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    all_results = collect_results(outputs_dir)
    all_results = balance_score(all_results, performance_col="event_f1", synthetic_col="num_synthetic_windows")
    all_results_path = summary_dir / "all_results.csv"
    all_results.to_csv(all_results_path, index=False)

    main_table = mean_std_table(
        all_results,
        group_cols=["dataset", "split_seed", "label_fraction", "detector", "method"],
        metric_cols=["auprc", "event_f1", "auroc", "best_point_f1", "num_synthetic_windows"],
    )
    main_table = relative_improvement(
        main_table,
        baseline_method="real_only",
        metric_cols=["auprc_mean", "event_f1_mean"],
        group_cols=["dataset", "split_seed", "label_fraction", "detector"],
    )
    main_table_path = summary_dir / "main_table.csv"
    main_table.to_csv(main_table_path, index=False)

    print(str(all_results_path))
    print(str(main_table_path))


if __name__ == "__main__":
    main()
