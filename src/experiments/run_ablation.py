"""Run the main alignment/synthesis ablation table for TimeEventSynth."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.load_options import add_dataset_load_arguments, apply_dataset_cli_to_config
from src.experiments.artifacts import save_standard_experiment_bundle
from src.experiments.config import load_config
from src.experiments.pipeline import (
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
from src.synthesis.donor_selection import donor_policy_experiment_specs
from src.experiments.synthesis_runner import append_audit_rows, build_rejection_summary, synthesis_policy_config
from src.experiments.run_low_label import (
    _mask_train_labels,
    _records_from_synthetic_windows,
    synthetic_window_metrics,
)
from src.synthesis.augment_dataset import records_from_oversampled_events
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit
from src.synthesis.donor_retrieval import retrieve_donors
from src.synthesis.event_pattern_policy import build_event_pattern_policy
from src.utils.seeds import set_global_seed


def compare_retrieval_methods(query_event: dict[str, Any], donor_pool: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Backward-compatible smoke helper for retrieval method comparison."""
    return {
        "normalized_time": retrieve_donors(query_event, donor_pool, top_k=3, method="normalized_time"),
        "dtw": retrieve_donors(query_event, donor_pool, top_k=3, method="dtw"),
        "learned_prototype_event_time": retrieve_donors(
            query_event,
            donor_pool,
            top_k=3,
            method="learned_prototype_event_time",
        ),
    }


def _build_comparison_rows(results: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method, payload in results.items():
        rows.append(
            {
                "method": method,
                "auroc": payload["test_point_metrics"]["point_auroc"],
                "auprc": payload["test_point_metrics"]["point_auprc"],
                "best_point_f1": payload["test_point_metrics"]["point_f1"],
                "event_precision": payload["test_event_metrics"]["event_precision"],
                "event_recall": payload["test_event_metrics"]["event_recall"],
                "event_f1": payload["test_event_metrics"]["event_f1"],
                "false_positive_events": payload["test_event_metrics"]["false_positive_event_count"],
                "num_synthetic_windows": payload["num_synthetic_windows"],
                "num_synthetic_points": payload["num_synthetic_points"],
                "num_rejected_synthetic_windows": payload.get("num_rejected_synthetic_windows", 0),
                "num_real_positive_train_points": payload.get("train_label_stats", {}).get(
                    "num_real_positive_train_points", 0
                ),
            }
        )
    return rows


def _build_method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    synthesis_cfg = config.get("synthesis", {})
    base_methods = synthesis_cfg.get(
        "methods",
        ["normalized_time_mean_donor", "dtw_aligned_donor", "learned_prototype_event_time"],
    )
    specs: list[dict[str, Any]] = [
        {"name": "real_only", "kind": "real"},
        {"name": "random_event_oversampling", "kind": "oversample"},
    ]
    for method in base_methods:
        specs.append(
            {
                "name": str(method),
                "kind": "synthetic",
                "synthesis_method": str(method),
                "filter_policy": None,
            }
        )
    filter_variants = [
        ("normalized_time_top50_filter", "normalized_time_mean_donor", "top_quantile"),
        ("learned_prototype_event_time_top50_filter", "learned_prototype_event_time", "top_quantile"),
        ("normalized_time_minconf_filter", "normalized_time_mean_donor", "min_confidence"),
        ("learned_prototype_event_time_minconf_filter", "learned_prototype_event_time", "min_confidence"),
        ("normalized_time_strict_filter", "normalized_time_mean_donor", "strict"),
        ("learned_prototype_event_time_strict_filter", "learned_prototype_event_time", "strict"),
    ]
    for method_name, synthesis_method, filter_name in filter_variants:
        specs.append(
            {
                "name": method_name,
                "kind": "synthetic",
                "synthesis_method": synthesis_method,
                "filter_policy": filter_name,
            }
        )
    for method_name, synthesis_method in [
        ("normalized_time_context_calibrated", "normalized_time_mean_donor"),
        ("dtw_context_calibrated", "dtw_aligned_donor"),
        ("learned_prototype_context_calibrated", "learned_prototype_event_time"),
    ]:
        specs.append(
            {
                "name": method_name,
                "kind": "synthetic",
                "synthesis_method": synthesis_method,
                "filter_policy": None,
            }
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


def run_ablation_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
    labeled_fraction: float | None = None,
    prepared: dict[str, Any] | None = None,
) -> dict[str, Any]:
    set_global_seed(int(config.get("seed", 42)))

    if prepared is not None:
        records = prepared["records"]
        load_summary = prepared.get("load_summary", {})
        data_path = prepared.get("data_path", "synthetic")
        train_records = prepared["train_records"]
        val_records = prepared["val_records"]
        test_records = prepared["test_records"]
        split_ids = prepared["split_ids"]
    else:
        records, load_summary, data_path = load_records_for_experiment(
            config,
            project_root=PROJECT_ROOT,
            use_synthetic=use_synthetic,
            cli_data=cli_data,
            load_options=load_options,
        )
        train_records, val_records, test_records, split_ids = split_records(records, config)
        warn_if_small_test_benchmark(test_records, all_records=records)

    if labeled_fraction is None:
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
    synthetic_audit: list[dict[str, Any]] = []
    per_series_by_method: dict[str, list[dict[str, Any]]] = {}

    for spec in _build_method_specs(config):
        method_name = str(spec["name"])
        if spec["kind"] == "real":
            method_train_records = masked_train
            audit_rows = None
            filter_policy = {"name": "no_filter"}
        elif spec["kind"] == "oversample":
            method_train_records = masked_train + records_from_oversampled_events(masked_train, config=config)
            audit_rows = None
            filter_policy = {"name": "no_filter"}
        else:
            filter_policy = synthesis_policy_config(
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
                policy_config=filter_policy,
                donor_pool_records=donor_records,
                synthesis_cfg=config.get("synthesis", {}),
            )
            method_train_records = masked_train + _records_from_synthetic_windows(kept)
            append_audit_rows(
                synthetic_audit,
                method_name=method_name,
                audit_rows=audit_rows,
                filter_policy=filter_policy["filter_policy"],
            )

        result = evaluate_detector(
            config,
            method_train_records,
            val_records,
            test_records,
            labeled_fraction=labeled_fraction,
            real_train_count=len(masked_train),
        )
        synth_metrics = synthetic_window_metrics(masked_train, method_train_records, audit_rows)
        results[method_name] = {
            "threshold_selection": result["threshold_selection"],
            "test_point_metrics": result["test_point_metrics"],
            "test_event_metrics": result["test_event_metrics"],
            "detector": result["detector"],
            "train_label_stats": result["train_label_stats"],
            "num_training_records": len(method_train_records),
            **synth_metrics,
        }
        per_series_by_method[method_name] = result["per_series_metrics"]

    rejection_summary = build_rejection_summary(synthetic_audit)

    event_pattern_policy = build_event_pattern_policy(
        [{**row, "event_pattern": "anomaly"} for row in synthetic_audit],
        pattern_keys=["anomaly"],
    )

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
        "labeled_fraction": labeled_fraction,
        "results": results,
        "comparison_rows": _build_comparison_rows(results),
        "synthetic_audit": synthetic_audit,
        "rejection_summary": rejection_summary,
        "event_pattern_policy": event_pattern_policy,
        "per_series_by_method": per_series_by_method,
        "diagnostics": diagnostics,
        "warnings": warnings,
    }


def prepare_ablation_data(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
) -> dict[str, Any]:
    """Load and split data once for reuse across low-label fraction sweeps."""
    records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    train_records, val_records, test_records, split_ids = split_records(records, config)
    return {
        "records": records,
        "load_summary": load_summary,
        "data_path": data_path,
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "split_ids": split_ids,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TimeEventSynth alignment/synthesis ablation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    load_options = apply_dataset_cli_to_config(config, args)
    payload = run_ablation_experiment(
        config,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
    )

    dataset_name = load_options.dataset_name or config.get("dataset", {}).get("name") or config.get("data", {}).get("dataset_name", "dataset")
    run_label = "synthetic" if args.use_synthetic else str(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "ablation" / run_label / timestamp

    per_series_rows: list[dict[str, Any]] = []
    for method_name, rows in payload["per_series_by_method"].items():
        for row in rows:
            per_series_rows.append(
                {
                    "series_id": row["series_id"],
                    "method": method_name,
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
        experiment_name=str(config.get("experiment", {}).get("name", "ablation")),
        records=payload["records"],
        data_path=payload["data_path"],
        load_options=load_options_from_config(config),
        load_summary=payload["load_summary"],
        split_ids=payload["split_ids"],
        metrics={"warnings": payload["warnings"], "diagnostics": payload["diagnostics"], "methods": payload["results"]},
        per_series_rows=per_series_rows,
        synthetic_audit=payload["synthetic_audit"],
        synthesis_requested=True,
        extra_files={
            "comparison_metrics.csv": pd.DataFrame(payload["comparison_rows"]),
            "comparison_metrics.json": payload["comparison_rows"],
            "rejection_summary.json": payload["rejection_summary"],
            "event_pattern_policy.json": payload["event_pattern_policy"],
            "train_label_stats.json": {
                method: payload["results"][method]["train_label_stats"] for method in payload["results"]
            },
            "low_label_diagnostics.json": payload["diagnostics"],
        },
    )

    print(json.dumps(payload["results"], indent=2))
    print(str(output_dir))


if __name__ == "__main__":
    main()
