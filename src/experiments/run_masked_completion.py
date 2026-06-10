"""Masked event completion experiment for direct synthesis-fidelity evaluation."""

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
from src.datasets.tsb_loader import TimeSeriesRecord
from src.evaluation.reconstruction_metrics import (
    build_masked_observation,
    compute_mask_interval,
    evaluate_masked_reconstruction,
)
from src.experiments.audit_sanity import validate_masked_completion_outputs
from src.experiments.pipeline import (
    load_options_from_config,
    load_records_for_experiment,
    split_records,
    warn_if_small_test_benchmark,
)
from src.synthesis.donor_retrieval import EventWindow, collect_event_windows
from src.synthesis.donor_selection import (
    build_timeline_feature_table,
    resolve_compatibility_config,
    select_donors_for_target,
)
from src.synthesis.event_window_synthesizer import reconstruct_masked_event_window
from src.utils.seeds import set_global_seed

DEFAULT_METHODS = [
    "linear_interpolation",
    "random_donor",
    "normalized_time_mean_donor",
    "dtw_aligned_donor",
    "learned_prototype_event_time",
]

DEFAULT_MASK_POLICIES = ["middle_30", "suffix_30", "prefix_30", "random_contiguous"]


def _experiment_cfg(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("masked_completion", config.get("experiment", {}))


def build_donor_pool_records(
    train_records: list[TimeSeriesRecord],
    val_records: list[TimeSeriesRecord],
    test_series_ids: set[str],
    *,
    donor_source: str,
    allow_test_series_donors: bool,
) -> list[TimeSeriesRecord]:
    """Return train/val records allowed as masked-completion donors."""
    source = str(donor_source).lower()
    if source == "train_val":
        pool_records = list(train_records) + list(val_records)
    elif source == "train_only":
        pool_records = list(train_records)
    else:
        raise ValueError(f"Unsupported donor_source for masked completion: {donor_source}")

    if not allow_test_series_donors:
        pool_records = [record for record in pool_records if record.series_id not in test_series_ids]
    return pool_records


def build_donor_windows(
    train_records: list[TimeSeriesRecord],
    val_records: list[TimeSeriesRecord],
    test_series_ids: set[str],
    *,
    donor_source: str,
    allow_test_series_donors: bool,
) -> list[EventWindow]:
    """Collect donor windows from train/val only, excluding test series by default."""
    return collect_event_windows(
        build_donor_pool_records(
            train_records,
            val_records,
            test_series_ids,
            donor_source=donor_source,
            allow_test_series_donors=allow_test_series_donors,
        )
    )


def _masked_completion_selection_settings(
    exp_cfg: dict[str, Any],
    synthesis_cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build compatibility and retrieval settings for masked completion."""
    donor_policy = str(exp_cfg.get("donor_policy", "all_donors_no_filter"))
    groupwise_matching = bool(exp_cfg.get("groupwise_matching", False))
    policy_overrides: dict[str, Any] = {"donor_policy": donor_policy}
    if exp_cfg.get("compatibility") is not None:
        policy_overrides["compatibility"] = exp_cfg["compatibility"]
    compatibility_cfg = resolve_compatibility_config(synthesis_cfg, policy_overrides)
    context_size = int(exp_cfg.get("context_size", synthesis_cfg.get("context_size", 5)))
    retrieval_kwargs: dict[str, Any] = {
        "max_donors_per_source_series": exp_cfg.get("max_donors_per_source_series", 1),
        "avoid_single_series_dominance": bool(exp_cfg.get("avoid_single_series_dominance", True)),
        "context_size": context_size,
        "exclude_same_series": not bool(exp_cfg.get("allow_test_series_donors", False)),
        "restrict_to_target_group": groupwise_matching,
        "group_key": exp_cfg.get("group_key", "event_group_id") if groupwise_matching else None,
    }
    if groupwise_matching:
        retrieval_kwargs["group_selection_mode"] = "nearest_compatible_centroid"
    return compatibility_cfg, retrieval_kwargs


def _records_from_event_dicts(events: list[dict[str, Any]]) -> list[TimeSeriesRecord]:
    records: list[TimeSeriesRecord] = []
    for item in events:
        labels = item.get("labels")
        if labels is None:
            label_array = np.ones(len(item["values"]), dtype=int)
        else:
            label_array = np.asarray(labels, dtype=int)
        records.append(
            TimeSeriesRecord(
                series_id=str(item["series_id"]),
                values=np.asarray(item["values"], dtype=float),
                labels=label_array,
                timestamps=None,
                source_path="masked_completion",
                metadata={"event_id": item.get("event_id")},
            )
        )
    return records


def _assert_donor_pool_excludes_test(
    donor_windows: list[EventWindow],
    test_series_ids: set[str],
    *,
    allow_test_series_donors: bool,
) -> None:
    if allow_test_series_donors:
        return
    leaked = sorted({window.series_id for window in donor_windows if window.series_id in test_series_ids})
    if leaked:
        raise RuntimeError(f"Donor pool leaked test series: {leaked}")


def run_masked_completion_experiment(
    config: dict[str, Any],
    *,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
) -> dict[str, Any]:
    """Run masked completion for all test events, mask policies, and methods."""
    set_global_seed(int(config.get("seed", 42)))
    exp_cfg = _experiment_cfg(config)
    synthesis_cfg = config.get("synthesis", {})

    records, load_summary, data_path = load_records_for_experiment(
        config,
        project_root=PROJECT_ROOT,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    train_records, val_records, test_records, split_ids = split_records(records, config)
    warn_if_small_test_benchmark(test_records, all_records=records)

    test_series_ids = set(split_ids["test"])
    allow_test_series_donors = bool(exp_cfg.get("allow_test_series_donors", False))
    donor_source = str(exp_cfg.get("donor_source", synthesis_cfg.get("donor_source", "train_val")))
    donor_windows = build_donor_windows(
        train_records,
        val_records,
        test_series_ids,
        donor_source=donor_source,
        allow_test_series_donors=allow_test_series_donors,
    )
    donor_pool_records = build_donor_pool_records(
        train_records,
        val_records,
        test_series_ids,
        donor_source=donor_source,
        allow_test_series_donors=allow_test_series_donors,
    )
    feature_table = build_timeline_feature_table(
        list(donor_pool_records) + list(test_records),
        synthesis_cfg=synthesis_cfg,
    )
    compatibility_cfg, retrieval_kwargs = _masked_completion_selection_settings(exp_cfg, synthesis_cfg)
    donor_policy = str(exp_cfg.get("donor_policy", "all_donors_no_filter"))
    _assert_donor_pool_excludes_test(
        donor_windows,
        test_series_ids,
        allow_test_series_donors=allow_test_series_donors,
    )

    test_windows = collect_event_windows(test_records)
    if bool(retrieval_kwargs.get("restrict_to_target_group")) and retrieval_kwargs.get("group_key"):
        from src.alignment.event_grouping import assign_event_groups

        grouping_cfg = dict(exp_cfg.get("grouping") or synthesis_cfg.get("grouping", {}))
        grouping_cfg.setdefault("context_size", int(retrieval_kwargs.get("context_size", 5)))
        assign_event_groups(
            donor_windows,
            [],
            config=grouping_cfg,
            group_key=str(retrieval_kwargs["group_key"]),
        )
    mask_fraction = float(exp_cfg.get("mask_fraction", 0.3))
    mask_policies = list(exp_cfg.get("mask_policies", DEFAULT_MASK_POLICIES))
    methods = list(exp_cfg.get("methods", DEFAULT_METHODS))
    top_k = int(exp_cfg.get("top_k_donors", synthesis_cfg.get("top_k", 3)))
    grid_size = int(exp_cfg.get("alignment_grid_size", synthesis_cfg.get("alignment_grid_size", 64)))
    context_size = int(exp_cfg.get("context_size", synthesis_cfg.get("context_size", 5)))
    seed = int(config.get("seed", 42))

    per_event_rows: list[dict[str, Any]] = []
    qualitative_candidates: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for window in test_windows:
        event_length = len(window.values)
        if event_length < 3:
            continue

        for policy in mask_policies:
            policy_rng = np.random.default_rng(seed + hash((window.series_id, window.start, window.end, policy)) % 10_000)
            mask_start, mask_end = compute_mask_interval(event_length, mask_fraction, policy, rng=policy_rng)
            masked_observed, partial_labels = build_masked_observation(window.values, mask_start, mask_end)

            query_window = EventWindow(
                series_id=window.series_id,
                start=window.start,
                end=window.end,
                values=np.asarray(window.values, dtype=float),
                label=window.label,
                metadata=dict(window.metadata),
            )

            donors, similarities, _ = select_donors_for_target(
                query_window,
                donor_windows,
                feature_table=feature_table,
                compatibility_cfg=compatibility_cfg,
                top_k=top_k,
                retrieval_kwargs=retrieval_kwargs,
            )

            reconstructions: dict[str, np.ndarray] = {}
            for method in methods:
                if method == "linear_interpolation":
                    reconstructed = reconstruct_masked_event_window(
                        query_window,
                        donors=donors or [query_window],
                        method=method,
                        partial_labels=partial_labels,
                        seed=seed,
                    )
                elif not donors:
                    continue
                else:
                    reconstructed = reconstruct_masked_event_window(
                        query_window,
                        donors=donors,
                        method=method,
                        partial_labels=partial_labels,
                        similarities=similarities,
                        grid_size=grid_size,
                        seed=seed,
                    )
                reconstructions[method] = np.asarray(reconstructed, dtype=float)
                metrics = evaluate_masked_reconstruction(window.values, reconstructed, mask_start, mask_end)
                per_event_rows.append(
                    {
                        "event_id": window.metadata.get("event_id", f"{window.series_id}:{window.start}:{window.end}"),
                        "series_id": window.series_id,
                        "mask_policy": policy,
                        "method": method,
                        "mask_start": mask_start,
                        "mask_end": mask_end,
                        "event_length": event_length,
                        "num_donors": len(donors),
                        **metrics,
                    }
                )

            if reconstructions:
                qualitative_candidates.append(
                    {
                        "event_id": window.metadata.get("event_id", f"{window.series_id}:{window.start}:{window.end}"),
                        "series_id": window.series_id,
                        "mask_policy": policy,
                        "mask_start": mask_start,
                        "mask_end": mask_end,
                        "original": np.asarray(window.values, dtype=float),
                        "masked_observed": masked_observed,
                        "reconstructions": reconstructions,
                    }
                )

    aggregate_rows = _aggregate_metrics(per_event_rows)
    metrics_payload = {
        "mask_fraction": mask_fraction,
        "mask_policies": mask_policies,
        "methods": methods,
        "num_test_events": len({row["event_id"] for row in per_event_rows}) if per_event_rows else 0,
        "num_evaluations": len(per_event_rows),
        "donor_source": donor_source,
        "donor_policy": donor_policy,
        "allow_test_series_donors": allow_test_series_donors,
        "aggregate_by_method": aggregate_rows.get("by_method", {}),
        "aggregate_by_method_and_policy": aggregate_rows.get("by_method_and_policy", {}),
    }

    return {
        "records": records,
        "data_path": data_path,
        "load_summary": load_summary,
        "split_ids": split_ids,
        "metrics": metrics_payload,
        "per_event_rows": per_event_rows,
        "qualitative_candidates": qualitative_candidates,
    }


def _aggregate_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"by_method": {}, "by_method_and_policy": {}}
    frame = pd.DataFrame(rows)
    metric_cols = [
        "masked_region_MAE",
        "masked_region_MSE",
        "normalized_MAE_by_event_std",
        "shape_correlation",
        "boundary_continuity_error",
    ]
    by_method = frame.groupby("method")[metric_cols].mean(numeric_only=True).reset_index()
    by_method_policy = frame.groupby(["method", "mask_policy"])[metric_cols].mean(numeric_only=True).reset_index()
    return {
        "by_method": by_method.set_index("method").to_dict(orient="index"),
        "by_method_and_policy": {
            f"{row['method']}::{row['mask_policy']}": {col: float(row[col]) for col in metric_cols}
            for _, row in by_method_policy.iterrows()
        },
    }


def save_masked_completion_outputs(payload: dict[str, Any], output_dir: Path, config: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    validate_masked_completion_outputs(payload["metrics"])
    exp_cfg = _experiment_cfg(config)
    max_examples = int(exp_cfg.get("max_qualitative_examples", 5))

    (output_dir / "metrics.json").write_text(json.dumps(payload["metrics"], indent=2), encoding="utf-8")
    pd.DataFrame(payload["per_event_rows"]).to_csv(output_dir / "per_event_metrics.csv", index=False)

    qual_dir = output_dir / "qualitative_examples"
    qual_dir.mkdir(parents=True, exist_ok=True)
    candidates = payload["qualitative_candidates"][:max_examples]
    for idx, example in enumerate(candidates):
        length = len(example["original"])
        rows: list[dict[str, Any]] = []
        for t in range(length):
            row: dict[str, Any] = {
                "t": t,
                "original": float(example["original"][t]),
                "masked_observed": float(example["masked_observed"][t]) if np.isfinite(example["masked_observed"][t]) else "",
            }
            for method, values in example["reconstructions"].items():
                row[f"reconstructed_{method}"] = float(values[t])
            rows.append(row)
        example_id = str(example["event_id"]).replace("/", "_").replace(":", "_")
        pd.DataFrame(rows).to_csv(qual_dir / f"example_{idx:02d}_{example_id}_{example['mask_policy']}.csv", index=False)


def run_masked_completion(
    train_events: list[dict[str, Any]],
    query_events: list[dict[str, Any]],
    *,
    mask_fraction: float = 0.3,
    mask_policy: str = "middle_30",
    method: str = "normalized_time_mean_donor",
    top_k: int = 3,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Backward-compatible smoke helper for masked completion on dict events."""
    donor_windows = [
        EventWindow(
            series_id=str(item["series_id"]),
            start=int(item.get("start_idx", 0)),
            end=int(item.get("end_idx", len(item["values"]))),
            values=np.asarray(item["values"], dtype=float),
            label=str(item.get("event_label", "anomaly")),
            metadata={"event_id": item.get("event_id")},
        )
        for item in train_events
    ]
    feature_table = build_timeline_feature_table(
        _records_from_event_dicts(train_events + query_events),
        synthesis_cfg={"compatibility": {"enabled": False}},
    )
    compatibility_cfg, retrieval_kwargs = _masked_completion_selection_settings(
        {"donor_policy": "all_donors_no_filter", "allow_test_series_donors": False},
        {"compatibility": {"enabled": False}},
    )
    outputs: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)
    for query in query_events:
        values = np.asarray(query["values"], dtype=float)
        if len(values) < 3:
            continue
        mask_start, mask_end = compute_mask_interval(len(values), mask_fraction, mask_policy, rng=rng)
        masked_observed, partial_labels = build_masked_observation(values, mask_start, mask_end)
        target = EventWindow(
            series_id=str(query["series_id"]),
            start=int(query.get("start_idx", 0)),
            end=int(query.get("end_idx", len(values))),
            values=values,
            label=str(query.get("event_label", "anomaly")),
            metadata={"event_id": query.get("event_id")},
        )
        donors, similarities, _ = select_donors_for_target(
            target,
            donor_windows,
            feature_table=feature_table,
            compatibility_cfg=compatibility_cfg,
            top_k=top_k,
            retrieval_kwargs=retrieval_kwargs,
        )
        if method != "linear_interpolation" and not donors:
            continue
        reconstructed = reconstruct_masked_event_window(
            target,
            donors=donors or [target],
            method=method,
            partial_labels=partial_labels,
            similarities=similarities,
            seed=seed,
        )
        metrics = evaluate_masked_reconstruction(values, reconstructed, mask_start, mask_end)
        outputs.append(
            {
                "event_id": query.get("event_id"),
                "series_id": query.get("series_id"),
                "mask_policy": mask_policy,
                "method": method,
                "mask_start": mask_start,
                "mask_end": mask_end,
                "values": np.asarray(reconstructed, dtype=float).tolist(),
                "masked_observed": masked_observed.tolist(),
                "metrics": metrics,
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run masked event completion synthesis fidelity evaluation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_masked_completion.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    config = load_config(args.config)
    load_options = apply_dataset_cli_to_config(config, args)
    payload = run_masked_completion_experiment(
        config,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
    )

    dataset_name = load_options.dataset_name or config.get("dataset", {}).get("name") or "dataset"
    run_label = "synthetic" if args.use_synthetic else str(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = PROJECT_ROOT / "outputs" / "masked_completion" / run_label / timestamp
    save_masked_completion_outputs(payload, output_dir, config)
    print(json.dumps(payload["metrics"], indent=2))
    print(str(output_dir))


if __name__ == "__main__":
    main()
