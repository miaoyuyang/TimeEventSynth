"""Low-label experiment with normalized-time event-window synthesis."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.tsb_loader import TimeSeriesRecord
from src.experiments.low_label_masking import mask_train_labels_for_low_label
from src.datasets.load_options import add_dataset_load_arguments, apply_dataset_cli_to_config
from src.experiments.artifacts import save_standard_experiment_bundle
from src.experiments.config import load_config
from src.experiments.pipeline import (
    build_experiment_dataset_stats_payload,
    donor_pool_records,
    prepare_low_label_train_and_donor_pool,
    evaluate_detector,
    load_options_from_config,
    load_records_for_experiment,
    split_records,
    warn_if_small_test_benchmark,
)
from src.evaluation.reporting import (
    build_run_diagnostics,
    collect_low_label_warnings,
    print_low_label_warnings,
)
from src.experiments.synthesis_runner import append_audit_rows, build_rejection_summary, synthesis_policy_config
from src.synthesis.donor_selection import donor_policy_experiment_specs
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit, records_from_oversampled_events


def _mask_train_labels(records: list[TimeSeriesRecord], labeled_fraction: float, seed: int) -> list[TimeSeriesRecord]:
    return mask_train_labels_for_low_label(records, labeled_fraction, seed)


def _records_from_synthetic_windows(windows: list[dict[str, Any]]) -> list[TimeSeriesRecord]:
    outputs: list[TimeSeriesRecord] = []
    for row in windows:
        outputs.append(
            TimeSeriesRecord(
                series_id=str(row["series_id"]),
                values=row["values"],
                labels=row["labels"],
                timestamps=None,
                source_path="synthetic",
                metadata=row.get("metadata", {}),
            )
        )
    return outputs


def synthetic_window_metrics(
    masked_train: list[TimeSeriesRecord],
    method_train: list[TimeSeriesRecord],
    audit_rows: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Count synthetic windows/points for comparison tables.

    Donor-synthesis methods report kept/rejected counts from ``audit_rows``.
    Non-audit augmentations (e.g. random event oversampling) derive kept windows
    from appended training records so ``num_synthetic_windows`` matches
    ``num_synthetic_points``.
    """
    appended = method_train[len(masked_train) :]
    num_synthetic_points = int(sum(len(record.labels) for record in appended))
    if audit_rows is not None:
        num_synthetic_windows = sum(1 for row in audit_rows if bool(row.get("kept", False)))
        num_rejected = sum(1 for row in audit_rows if not bool(row.get("kept", False)))
    else:
        num_synthetic_windows = len(appended)
        num_rejected = 0
    return {
        "num_synthetic_windows": num_synthetic_windows,
        "num_synthetic_points": num_synthetic_points,
        "num_rejected_synthetic_windows": num_rejected,
    }


def _build_method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    synthesis_cfg = config.get("synthesis", {})
    specs: list[dict[str, Any]] = [
        {"name": "real_only", "kind": "real"},
        {"name": "random_event_oversampling", "kind": "oversample"},
    ]
    for method in synthesis_cfg.get("methods", ["normalized_time_mean_donor", "dtw_aligned_donor", "learned_prototype_event_time"]):
        specs.append({"name": str(method), "kind": "synthetic", "synthesis_method": str(method), "filter_policy": None})
    filter_variants = [
        ("normalized_time_top50_filter", "normalized_time_mean_donor", "top_quantile"),
        ("learned_prototype_event_time_top50_filter", "learned_prototype_event_time", "top_quantile"),
        ("normalized_time_minconf_filter", "normalized_time_mean_donor", "min_confidence"),
        ("learned_prototype_event_time_minconf_filter", "learned_prototype_event_time", "min_confidence"),
    ]
    for method_name, synthesis_method, filter_name in filter_variants:
        specs.append(
            {"name": method_name, "kind": "synthetic", "synthesis_method": synthesis_method, "filter_policy": filter_name}
        )
    for method_name, synthesis_method in [
        ("normalized_time_context_calibrated", "normalized_time_mean_donor"),
        ("dtw_context_calibrated", "dtw_aligned_donor"),
        ("learned_prototype_context_calibrated", "learned_prototype_event_time"),
    ]:
        specs.append(
            {"name": method_name, "kind": "synthetic", "synthesis_method": synthesis_method, "filter_policy": None}
        )
    for entry in donor_policy_experiment_specs(synthesis_cfg):
        specs.append(
            {
                "name": entry["name"],
                "kind": "synthetic",
                "synthesis_method": entry["synthesis_method"],
                "filter_policy": None,
                "donor_policy": entry["donor_policy"],
            }
        )
    return specs


def _build_method_training_sets(
    masked_train: list[TimeSeriesRecord],
    donor_records: list[TimeSeriesRecord],
    config: dict[str, Any],
    labeled_fraction: float,
) -> tuple[dict[str, list[TimeSeriesRecord]], list[dict[str, Any]]]:
    synthetic_audit: list[dict[str, Any]] = []
    methods: dict[str, list[TimeSeriesRecord]] = {}
    for spec in _build_method_specs(config):
        method_name = str(spec["name"])
        if spec["kind"] == "real":
            methods[method_name] = masked_train
            continue
        if spec["kind"] == "oversample":
            methods[method_name] = masked_train + records_from_oversampled_events(masked_train, config=config)
            continue
        policy = synthesis_policy_config(
            config,
            synthesis_method=str(spec["synthesis_method"]),
            method_name=method_name,
            labeled_fraction=labeled_fraction,
            filter_policy=spec.get("filter_policy"),
            donor_policy=spec.get("donor_policy"),
        )
        kept, audit_rows = build_augmented_training_records_with_audit(
            masked_train,
            split="train",
            policy_config=policy,
            donor_pool_records=donor_records,
            synthesis_cfg=config.get("synthesis", {}),
        )
        methods[method_name] = masked_train + _records_from_synthetic_windows(kept)
        append_audit_rows(
            synthetic_audit,
            method_name=method_name,
            audit_rows=audit_rows,
            filter_policy=policy["filter_policy"],
        )
    return methods, synthetic_audit


def run_low_label_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
) -> dict[str, Any]:
    records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    train_records, val_records, test_records, split_ids = split_records(records, config)
    warn_if_small_test_benchmark(test_records, all_records=records)
    dataset_stats = build_experiment_dataset_stats_payload(records, split_ids, load_summary)
    labeled_fraction = float(
        config.get("low_label", {}).get(
            "default_fraction",
            config.get("experiment", {}).get("labeled_fraction", 0.2),
        )
    )
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=labeled_fraction,
        seed=int(config.get("seed", 42)),
    )

    results: dict[str, Any] = {}
    per_series_by_method: dict[str, list[dict[str, Any]]] = {}
    train_label_stats_by_method: dict[str, Any] = {}
    methods, synthetic_audit = _build_method_training_sets(masked_train, donor_records, config, labeled_fraction)
    rejection_summary = build_rejection_summary(synthetic_audit)
    for method_name, method_train_records in methods.items():
        result = evaluate_detector(
            config,
            method_train_records,
            val_records,
            test_records,
            labeled_fraction=labeled_fraction,
            real_train_count=len(masked_train),
        )
        per_series_by_method[method_name] = result["per_series_metrics"]
        train_label_stats_by_method[method_name] = result["train_label_stats"]
        results[method_name] = {
            "threshold_selection": result["threshold_selection"],
            "test_point_metrics": result["test_point_metrics"],
            "test_event_metrics": result["test_event_metrics"],
            "detector": result["detector"],
            "train_label_stats": result["train_label_stats"],
            "num_training_records": len(method_train_records),
            **synthetic_window_metrics(masked_train, method_train_records),
        }
    diagnostics = build_run_diagnostics(results, labeled_fraction=labeled_fraction)
    detector_cfg = results.get("real_only", {}).get("detector", config.get("detector", {}))
    warnings = collect_low_label_warnings(
        detector_cfg=detector_cfg,
        diagnostics_by_method=diagnostics,
    )
    print_low_label_warnings(warnings)
    return {
        "records": records,
        "data_path": data_path,
        "load_summary": load_summary,
        "split_ids": split_ids,
        "dataset_stats": dataset_stats,
        "labeled_fraction": labeled_fraction,
        "results": results,
        "per_series_by_method": per_series_by_method,
        "train_label_stats_by_method": train_label_stats_by_method,
        "synthetic_audit": synthetic_audit,
        "rejection_summary": rejection_summary,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the low-label normalized-time synthesis experiment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    load_options = apply_dataset_cli_to_config(config, args)
    payload = run_low_label_experiment(
        config,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_label = "synthetic" if args.use_synthetic else config.get("dataset", {}).get("name", "low_label")
    output_dir = PROJECT_ROOT / "outputs" / "low_label" / str(run_label) / timestamp

    comparison_rows = []
    per_series_rows = []
    for method, method_payload in payload["results"].items():
        comparison_rows.append(
            {
                "method": method,
                "threshold_mode": method_payload["threshold_selection"]["mode"],
                "threshold": method_payload["threshold_selection"]["threshold"],
                "auroc": method_payload["test_point_metrics"]["point_auroc"],
                "auprc": method_payload["test_point_metrics"]["point_auprc"],
                "best_point_f1": method_payload["test_point_metrics"]["point_f1"],
                "event_precision": method_payload["test_event_metrics"]["event_precision"],
                "event_recall": method_payload["test_event_metrics"]["event_recall"],
                "event_f1": method_payload["test_event_metrics"]["event_f1"],
                "false_positive_events": method_payload["test_event_metrics"]["false_positive_event_count"],
                "num_synthetic_windows": method_payload["num_synthetic_windows"],
                "num_synthetic_points": method_payload["num_synthetic_points"],
                "num_real_positive_train_points": method_payload.get("train_label_stats", {}).get(
                    "num_real_positive_train_points", 0
                ),
            }
        )
        for row in payload["per_series_by_method"][method]:
            per_series_rows.append(
                {
                    "series_id": row["series_id"],
                    "method": method,
                    "labeled_fraction": payload["labeled_fraction"],
                    "num_points": row["num_points"],
                    "num_true_event_windows": row["num_true_event_windows"],
                    "num_pred_event_windows": row["num_pred_event_windows"],
                    "auprc": row["auprc"],
                    "event_precision": row["event_precision"],
                    "event_recall": row["event_recall"],
                    "event_f1": row["event_f1"],
                }
            )

    save_standard_experiment_bundle(
        output_dir,
        config=config,
        project_root=PROJECT_ROOT,
        experiment_name="low_label",
        records=payload["records"],
        data_path=payload["data_path"],
        load_options=load_options_from_config(config),
        load_summary=payload["load_summary"],
        split_ids=payload["split_ids"],
        metrics={
            "labeled_fraction": payload["labeled_fraction"],
            "warnings": payload["warnings"],
            "diagnostics": payload["diagnostics"],
            "methods": payload["results"],
        },
        per_series_rows=per_series_rows,
        extra_files={
            "comparison_metrics.csv": pd.DataFrame(comparison_rows),
            "train_label_stats.json": payload["train_label_stats_by_method"],
            "low_label_diagnostics.json": payload["diagnostics"],
            "rejection_summary.json": payload["rejection_summary"],
        },
        synthetic_audit=payload["synthetic_audit"],
        synthesis_requested=True,
    )
    print(json.dumps(payload["results"], indent=2))
    print(str(output_dir))


# Backward-compatible helper used by run_ablation imports.
def _evaluate_method(config, train_records, val_records, test_records):
    return evaluate_detector(config, train_records, val_records, test_records)


if __name__ == "__main__":
    main()
