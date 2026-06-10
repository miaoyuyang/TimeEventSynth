"""Compatibility-aware cross-timeline transfer experiment for paper comparisons."""

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

from src.datasets.load_options import add_dataset_load_arguments, apply_dataset_cli_to_config
from src.evaluation.reconstruction_metrics import (
    build_masked_observation,
    compute_mask_interval,
    evaluate_masked_reconstruction,
)
from src.evaluation.reporting import (
    analyze_compatibility_transfer,
    format_compatibility_transfer_markdown,
)
from src.experiments.artifacts import save_standard_experiment_bundle
from src.experiments.audit_sanity import (
    validate_compatibility_transfer_outputs,
    validate_masked_completion_outputs,
    validate_synthetic_audit_csv,
)
from src.experiments.compatibility_summary import build_compatibility_summary
from src.experiments.compatibility_transfer_specs import (
    build_compatibility_transfer_specs,
    compatibility_transfer_cfg,
    experiment_compatibility_enabled,
)
from src.experiments.config import load_config
from src.experiments.pipeline import (
    prepare_low_label_train_and_donor_pool,
    evaluate_detector,
    load_options_from_config,
    load_records_for_experiment,
    split_records,
    warn_if_small_test_benchmark,
)
from src.experiments.run_low_label import (
    _mask_train_labels,
    _records_from_synthetic_windows,
    synthetic_window_metrics,
)
from src.experiments.run_masked_completion import build_donor_windows
from src.experiments.synthesis_runner import append_audit_rows, build_rejection_summary, synthesis_policy_config
from src.synthesis.augment_dataset import build_augmented_training_records_with_audit, records_from_oversampled_events
from src.synthesis.donor_retrieval import EventWindow, collect_event_windows
from src.synthesis.donor_selection import (
    build_timeline_feature_table,
    resolve_compatibility_config,
    select_donors_for_target,
)
from src.synthesis.event_window_synthesizer import reconstruct_masked_event_window
from src.synthesis.event_pattern_policy import build_event_pattern_policy
from src.utils.seeds import set_global_seed


def _json_safe_metrics(results: dict[str, Any]) -> dict[str, Any]:
    """Strip non-serializable detector outputs for metrics.json."""
    slim: dict[str, Any] = {}
    for method_name, payload in results.items():
        slim[method_name] = {
            key: value
            for key, value in payload.items()
            if key not in {"scores", "spec", "per_series_metrics"}
        }
    return slim


def _comparison_row(
    method_name: str,
    spec: dict[str, Any],
    payload: dict[str, Any],
    *,
    labeled_fraction: float,
) -> dict[str, Any]:
    point = payload["test_point_metrics"]
    event = payload["test_event_metrics"]
    metrics_valid = point.get("metrics_valid")
    return {
        "method": method_name,
        "donor_policy": spec.get("donor_policy"),
        "synthesis_method": spec.get("synthesis_method"),
        "filter_policy": spec.get("filter_policy"),
        "compatibility_enabled": experiment_compatibility_enabled(spec),
        "labeled_fraction": labeled_fraction,
        "auroc": point.get("point_auroc"),
        "auprc": point.get("point_auprc"),
        "best_point_f1": point.get("point_f1"),
        "event_precision": event.get("event_precision"),
        "event_recall": event.get("event_recall"),
        "event_f1": event.get("event_f1"),
        "false_positive_events": event.get("false_positive_event_count"),
        "metrics_valid": bool(metrics_valid is True),
        "metrics_invalid_reason": point.get("metrics_invalid_reason", "" if metrics_valid is True else "metrics_valid_missing"),
        "num_synthetic_windows": payload.get("num_synthetic_windows", 0),
        "num_synthetic_points": payload.get("num_synthetic_points", 0),
        "num_rejected_synthetic_windows": payload.get("num_rejected_synthetic_windows", 0),
        "num_real_positive_train_points": payload.get("train_label_stats", {}).get(
            "num_real_positive_train_points", 0
        ),
    }


def run_compatibility_transfer_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
    smoke: bool = False,
) -> dict[str, Any]:
    """Run donor-policy comparison with low-label downstream detection."""
    set_global_seed(int(config.get("seed", 42)))
    transfer_cfg = compatibility_transfer_cfg(config)
    labeled_fraction = float(
        transfer_cfg.get(
            "labeled_fraction",
            config.get("low_label", {}).get("default_fraction", config.get("experiment", {}).get("labeled_fraction", 0.2)),
        )
    )
    specs = build_compatibility_transfer_specs(config, smoke=smoke)

    if smoke:
        smoke_limits = transfer_cfg.get("smoke", {})
        if smoke_limits.get("max_series") is not None:
            config = dict(config)
            dataset = dict(config.get("dataset", {}))
            dataset["max_series"] = int(smoke_limits["max_series"])
            config["dataset"] = dataset

    records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    train_records, val_records, test_records, split_ids = split_records(records, config)
    warn_if_small_test_benchmark(test_records, all_records=records)
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

    for spec in specs:
        method_name = str(spec["name"])
        if spec["kind"] == "real":
            method_train = masked_train
            audit_rows = None
        elif spec["kind"] == "oversample":
            method_train = masked_train + records_from_oversampled_events(masked_train, config=config)
            audit_rows = None
        else:
            filter_name = spec.get("filter_policy")
            policy = synthesis_policy_config(
                config,
                synthesis_method=str(spec["synthesis_method"]),
                method_name=method_name,
                labeled_fraction=labeled_fraction,
                filter_policy=filter_name if filter_name not in {None, "none"} else None,
                donor_policy=str(spec.get("donor_policy")),
            )
            if spec.get("compatibility") is not None:
                policy["compatibility"] = dict(spec["compatibility"])
            kept, audit_rows = build_augmented_training_records_with_audit(
                masked_train,
                split="train",
                policy_config=policy,
                donor_pool_records=donor_records,
                synthesis_cfg=config.get("synthesis", {}),
            )
            method_train = masked_train + _records_from_synthetic_windows(kept)
            append_audit_rows(
                synthetic_audit,
                method_name=method_name,
                audit_rows=audit_rows,
                filter_policy=policy["filter_policy"],
            )

        result = evaluate_detector(
            config,
            method_train,
            val_records,
            test_records,
            labeled_fraction=labeled_fraction,
            real_train_count=len(masked_train),
        )
        synth_metrics = synthetic_window_metrics(masked_train, method_train, audit_rows)
        results[method_name] = {
            **result,
            "spec": spec,
            **synth_metrics,
        }
        per_series_by_method[method_name] = result["per_series_metrics"]

    comparison_rows = [_comparison_row(name, results[name]["spec"], results[name], labeled_fraction=labeled_fraction) for name in results]
    rejection_summary = build_rejection_summary(synthetic_audit)
    compatibility_summary = build_compatibility_summary(synthetic_audit, rejection_summary=rejection_summary)

    masked_payload: dict[str, Any] | None = None
    masked_cfg = transfer_cfg.get("masked_completion", {})
    if bool(masked_cfg.get("enabled", False)):
        masked_payload = _run_policy_masked_completion(
            config,
            train_records=masked_train,
            val_records=val_records,
            test_records=test_records,
            split_ids=split_ids,
            specs=[spec for spec in specs if spec.get("kind") == "synthetic"],
            masked_cfg=masked_cfg,
        )

    event_pattern_policy = build_event_pattern_policy(
        [{**row, "event_pattern": "anomaly"} for row in synthetic_audit],
        pattern_keys=["anomaly"],
    )

    return {
        "records": records,
        "data_path": data_path,
        "load_summary": load_summary,
        "split_ids": split_ids,
        "labeled_fraction": labeled_fraction,
        "experiments": specs,
        "results": results,
        "comparison_rows": comparison_rows,
        "synthetic_audit": synthetic_audit,
        "rejection_summary": rejection_summary,
        "compatibility_summary": compatibility_summary,
        "event_pattern_policy": event_pattern_policy,
        "per_series_by_method": per_series_by_method,
        "masked_completion": masked_payload,
        "allow_no_donor_rejections": bool(transfer_cfg.get("allow_no_donor_rejections", False)),
    }


def _run_policy_masked_completion(
    config: dict[str, Any],
    *,
    train_records: list[Any],
    val_records: list[Any],
    test_records: list[Any],
    split_ids: dict[str, list[str]],
    specs: list[dict[str, Any]],
    masked_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Masked completion with per-spec donor policies for reconstruction vs downstream correlation."""
    synthesis_cfg = config.get("synthesis", {})
    test_series_ids = set(split_ids["test"])
    allow_test_series_donors = bool(masked_cfg.get("allow_test_series_donors", False))
    donor_source = str(masked_cfg.get("donor_source", synthesis_cfg.get("donor_source", "train_val")))
    donor_windows = build_donor_windows(
        train_records,
        val_records,
        test_series_ids,
        donor_source=donor_source,
        allow_test_series_donors=allow_test_series_donors,
    )
    feature_table = build_timeline_feature_table(
        list(train_records) + list(val_records) + list(test_records),
        synthesis_cfg=synthesis_cfg,
    )
    test_windows = collect_event_windows(test_records)
    mask_fraction = float(masked_cfg.get("mask_fraction", 0.3))
    mask_policies = list(masked_cfg.get("mask_policies", ["middle_30"]))
    top_k = int(masked_cfg.get("top_k_donors", synthesis_cfg.get("top_k", 3)))
    grid_size = int(masked_cfg.get("alignment_grid_size", synthesis_cfg.get("alignment_grid_size", 64)))
    context_size = int(masked_cfg.get("context_size", synthesis_cfg.get("context_size", 5)))
    seed = int(config.get("seed", 42))
    use_policies = bool(masked_cfg.get("use_donor_policies", True))

    if use_policies and not specs:
        specs = [
            {
                "name": f"{method}__cross_dataset_compatible",
                "synthesis_method": method,
                "donor_policy": "cross_dataset_compatible",
            }
            for method in masked_cfg.get("methods", ["learned_prototype_event_time"])
        ]

    per_event_rows: list[dict[str, Any]] = []
    for window in test_windows:
        event_length = len(window.values)
        if event_length < 3:
            continue
        for policy_name in mask_policies:
            policy_rng = np.random.default_rng(seed + hash((window.series_id, window.start, window.end, policy_name)) % 10_000)
            mask_start, mask_end = compute_mask_interval(event_length, mask_fraction, policy_name, rng=policy_rng)
            partial_labels = build_masked_observation(window.values, mask_start, mask_end)[1]
            query_window = EventWindow(
                series_id=window.series_id,
                start=window.start,
                end=window.end,
                values=np.asarray(window.values, dtype=float),
                label=window.label,
                metadata=dict(window.metadata),
            )
            for spec in specs:
                synthesis_method = str(spec.get("synthesis_method", "learned_prototype_event_time"))
                method_label = str(spec["name"])
                donor_policy = str(spec.get("donor_policy", "all_donors_no_filter"))
                compatibility_cfg = resolve_compatibility_config(
                    synthesis_cfg,
                    {"donor_policy": donor_policy, "compatibility": spec.get("compatibility")},
                )
                donors, similarities, _ = select_donors_for_target(
                    query_window,
                    donor_windows,
                    feature_table=feature_table,
                    compatibility_cfg=compatibility_cfg,
                    top_k=top_k,
                    retrieval_kwargs={"context_size": context_size},
                )
                if not donors:
                    continue
                reconstructed = reconstruct_masked_event_window(
                    query_window,
                    donors=donors,
                    method=synthesis_method,
                    partial_labels=partial_labels,
                    similarities=similarities,
                    grid_size=grid_size,
                    seed=seed,
                )
                metrics = evaluate_masked_reconstruction(window.values, reconstructed, mask_start, mask_end)
                per_event_rows.append(
                    {
                        "event_id": window.metadata.get("event_id", f"{window.series_id}:{window.start}:{window.end}"),
                        "series_id": window.series_id,
                        "mask_policy": policy_name,
                        "method": method_label,
                        "synthesis_method": synthesis_method,
                        "donor_policy": donor_policy,
                        "mask_start": mask_start,
                        "mask_end": mask_end,
                        "event_length": event_length,
                        "num_donors": len(donors),
                        **metrics,
                    }
                )

    metrics_payload = {
        "mask_fraction": mask_fraction,
        "mask_policies": mask_policies,
        "num_test_events": len({row["event_id"] for row in per_event_rows}) if per_event_rows else 0,
        "num_evaluations": len(per_event_rows),
        "donor_source": donor_source,
    }
    return {"metrics": metrics_payload, "per_event_rows": per_event_rows}


def save_compatibility_transfer_outputs(
    payload: dict[str, Any],
    output_dir: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Validate and write standard compatibility-transfer artifacts."""
    validate_synthetic_audit_csv(
        payload["synthetic_audit"],
        config=config,
        synthesis_requested=any(spec.get("kind") == "synthetic" for spec in payload["experiments"]),
    )
    validate_compatibility_transfer_outputs(
        comparison_rows=payload["comparison_rows"],
        audit_rows=payload["synthetic_audit"],
        experiments=payload["experiments"],
        allow_no_donor_rejections=bool(payload.get("allow_no_donor_rejections", False)),
        min_compatibility_score=float(
            config.get("synthesis", {}).get("compatibility", {}).get("min_score", 0.5)
        ),
    )

    per_series_rows: list[dict[str, Any]] = []
    for method_name, rows in payload["per_series_by_method"].items():
        spec = payload["results"][method_name]["spec"]
        for row in rows:
            per_series_rows.append(
                {
                    **row,
                    "method": method_name,
                    "donor_policy": spec.get("donor_policy"),
                    "synthesis_method": spec.get("synthesis_method"),
                    "labeled_fraction": payload["labeled_fraction"],
                }
            )

    masked_completion = payload.get("masked_completion") or {}
    analysis = analyze_compatibility_transfer(
        pd.DataFrame(payload["comparison_rows"]),
        payload["rejection_summary"],
        payload["compatibility_summary"],
        masked_per_event=masked_completion.get("per_event_rows"),
    )
    report_md = format_compatibility_transfer_markdown(analysis)

    extra_files: dict[str, Any] = {
        "comparison_metrics.csv": pd.DataFrame(payload["comparison_rows"]),
        "comparison_metrics.json": payload["comparison_rows"],
        "rejection_summary.json": payload["rejection_summary"],
        "compatibility_summary.json": payload["compatibility_summary"],
        "event_pattern_policy.json": payload["event_pattern_policy"],
        "analysis_report.json": analysis,
        "analysis_report.md": report_md,
    }
    if payload.get("masked_completion"):
        validate_masked_completion_outputs(payload["masked_completion"]["metrics"])
        extra_files["masked_completion_metrics.json"] = payload["masked_completion"]["metrics"]
        extra_files["per_event_metrics.csv"] = pd.DataFrame(payload["masked_completion"]["per_event_rows"])

    save_standard_experiment_bundle(
        output_dir,
        config=config,
        project_root=PROJECT_ROOT,
        experiment_name=str(config.get("experiment", {}).get("name", "compatibility_transfer")),
        records=payload["records"],
        data_path=payload["data_path"],
        load_options=load_options_from_config(config),
        load_summary=payload["load_summary"],
        split_ids=payload["split_ids"],
        metrics={"labeled_fraction": payload["labeled_fraction"], "methods": _json_safe_metrics(payload["results"])},
        per_series_rows=per_series_rows,
        synthetic_audit=payload["synthetic_audit"],
        synthesis_requested=any(spec.get("kind") == "synthetic" for spec in payload["experiments"]),
        extra_files=extra_files,
    )
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run compatibility-aware donor-policy transfer experiment.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_compatibility_transfer.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Use smoke experiment subset and optional max_series cap.")
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    load_options = apply_dataset_cli_to_config(config, args)
    payload = run_compatibility_transfer_experiment(
        config,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
        smoke=args.smoke,
    )

    dataset_name = load_options.dataset_name or config.get("dataset", {}).get("name") or "dataset"
    run_label = "synthetic" if args.use_synthetic else str(dataset_name)
    suffix = "smoke" if args.smoke else "full"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "compatibility_transfer" / run_label / f"{timestamp}_{suffix}"

    analysis = save_compatibility_transfer_outputs(payload, output_dir, config)
    print(json.dumps({"output_dir": str(output_dir), "best_event_f1": analysis.get("best_by_event_f1")}, indent=2))
    print(str(output_dir))


if __name__ == "__main__":
    main()
