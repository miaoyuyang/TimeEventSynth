"""Sweep low-label fractions and aggregate ablation results."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.load_options import add_dataset_load_arguments, infer_dataset_name, load_options_from_args
from src.experiments.artifacts import build_run_metadata, save_dataset_stats, save_resolved_config, save_standard_experiment_bundle
from src.experiments.config import load_config
from src.evaluation.reporting import (
    collect_low_label_warnings,
    print_low_label_warnings,
    warn_if_real_only_invariant,
)
from src.experiments.pipeline import load_options_from_config
from src.experiments.run_ablation import prepare_ablation_data, run_ablation_experiment


def _fraction_tag(fraction: float) -> str:
    text = f"{fraction:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _apply_config_overrides(config: dict[str, Any], args: argparse.Namespace, load_options) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    if args.data is not None:
        cfg.setdefault("dataset", {})["path"] = str(args.data)
        cfg.setdefault("data", {})["raw_path"] = str(args.data)
    if load_options.dataset_name is not None:
        cfg.setdefault("dataset", {})["name"] = load_options.dataset_name
        cfg.setdefault("data", {})["dataset_name"] = load_options.dataset_name
    if load_options.max_series is not None:
        cfg.setdefault("dataset", {})["max_series"] = load_options.max_series
    cfg.setdefault("data", {})["load_options"] = load_options.to_dict()
    return cfg


def _sweep_fractions(config: dict[str, Any], cli_fractions: list[float] | None) -> list[float]:
    if cli_fractions:
        return cli_fractions
    fractions = config.get("low_label", {}).get("fractions")
    if fractions:
        return [float(value) for value in fractions]
    default_fraction = float(config.get("low_label", {}).get("default_fraction", 0.2))
    return [default_fraction]


def _aggregate_rows(fraction_payloads: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction, payload in fraction_payloads:
        for row in payload["comparison_rows"]:
            rows.append({"labeled_fraction": fraction, **row})
    return rows


def _aggregate_per_series(fraction_payloads: list[tuple[float, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for fraction, payload in fraction_payloads:
        for method_name, method_rows in payload["per_series_by_method"].items():
            for row in method_rows:
                rows.append({"labeled_fraction": fraction, "method": method_name, **row})
    return rows


def run_low_label_sweep(
    config: dict[str, Any],
    *,
    fractions: list[float],
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options=None,
) -> dict[str, Any]:
    prepared = prepare_ablation_data(
        config,
        use_synthetic=use_synthetic,
        cli_data=cli_data,
        load_options=load_options,
    )
    fraction_payloads: list[tuple[float, dict[str, Any]]] = []
    per_fraction_dirs: dict[str, str] = {}
    all_diagnostics_rows: list[dict[str, Any]] = []

    for fraction in fractions:
        fraction_config = copy.deepcopy(config)
        fraction_config.setdefault("low_label", {})["default_fraction"] = fraction
        fraction_config.setdefault("experiment", {})["labeled_fraction"] = fraction
        payload = run_ablation_experiment(
            fraction_config,
            use_synthetic=use_synthetic,
            cli_data=cli_data,
            load_options=load_options,
            labeled_fraction=fraction,
            prepared=prepared,
        )
        fraction_payloads.append((fraction, payload))
        for method, diagnostic in payload.get("diagnostics", {}).items():
            all_diagnostics_rows.append(diagnostic)

    comparison_rows = _aggregate_rows(fraction_payloads)
    detector_cfg = config.get("detector", {})
    sweep_warnings = collect_low_label_warnings(
        detector_cfg=detector_cfg,
        diagnostics_rows=all_diagnostics_rows,
        comparison_rows=comparison_rows,
    )
    print_low_label_warnings(sweep_warnings)

    return {
        "prepared": prepared,
        "fractions": fractions,
        "fraction_payloads": fraction_payloads,
        "comparison_rows": comparison_rows,
        "per_series_rows": _aggregate_per_series(fraction_payloads),
        "per_fraction_dirs": per_fraction_dirs,
        "diagnostics_rows": all_diagnostics_rows,
        "warnings": sweep_warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep low-label fractions for TimeEventSynth ablation.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "experiment_low_label.yaml")
    parser.add_argument("--use-synthetic", action="store_true")
    parser.add_argument(
        "--fractions",
        type=float,
        nargs="*",
        default=None,
        help="Override config low_label.fractions (space-separated).",
    )
    add_dataset_load_arguments(parser)
    args = parser.parse_args()

    load_options = load_options_from_args(args)
    config = _apply_config_overrides(load_config(args.config), args, load_options)
    fractions = _sweep_fractions(config, args.fractions)

    sweep = run_low_label_sweep(
        config,
        fractions=fractions,
        use_synthetic=args.use_synthetic,
        cli_data=args.data,
        load_options=load_options,
    )

    dataset_name = (
        load_options.dataset_name
        or config.get("dataset", {}).get("name")
        or config.get("data", {}).get("dataset_name", "dataset")
    )
    run_label = "synthetic" if args.use_synthetic else str(dataset_name)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = PROJECT_ROOT / "outputs" / "low_label_sweep" / run_label / timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)

    prepared = sweep["prepared"]
    save_resolved_config(sweep_dir, config)
    (sweep_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                **build_run_metadata(config, project_root=PROJECT_ROOT, experiment_name="low_label_sweep"),
                "fractions": fractions,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (sweep_dir / "split_ids.json").write_text(json.dumps(prepared["split_ids"], indent=2), encoding="utf-8")
    save_dataset_stats(
        sweep_dir,
        prepared["records"],
        data_path=str(prepared["data_path"]),
        load_options=load_options_from_config(config),
        load_summary=prepared["load_summary"],
    )
    pd.DataFrame(sweep["comparison_rows"]).to_csv(sweep_dir / "low_label_sweep_comparison.csv", index=False)
    pd.DataFrame(sweep["per_series_rows"]).to_csv(sweep_dir / "low_label_sweep_per_series.csv", index=False)
    (sweep_dir / "low_label_sweep_summary.json").write_text(
        json.dumps(
            {
                "fractions": fractions,
                "methods": sorted({row["method"] for row in sweep["comparison_rows"]}),
                "warnings": sweep["warnings"],
                "comparison_rows": sweep["comparison_rows"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for fraction, payload in sweep["fraction_payloads"]:
        fraction_dir = sweep_dir / f"fraction_{_fraction_tag(fraction)}"
        per_series_rows: list[dict[str, Any]] = []
        for method_name, rows in payload["per_series_by_method"].items():
            for row in rows:
                per_series_rows.append({"method": method_name, **row})

        fraction_config = copy.deepcopy(config)
        fraction_config.setdefault("low_label", {})["default_fraction"] = fraction
        fraction_config.setdefault("experiment", {})["labeled_fraction"] = fraction

        save_standard_experiment_bundle(
            fraction_dir,
            config=fraction_config,
            project_root=PROJECT_ROOT,
            experiment_name=f"low_label_sweep_{_fraction_tag(fraction)}",
            records=prepared["records"],
            data_path=prepared["data_path"],
            load_options=load_options_from_config(config),
            load_summary=prepared["load_summary"],
            split_ids=prepared["split_ids"],
            metrics={
                "labeled_fraction": fraction,
                "warnings": payload.get("warnings", []),
                "diagnostics": payload.get("diagnostics", {}),
                "methods": payload["results"],
            },
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
                "low_label_diagnostics.json": payload.get("diagnostics", {}),
            },
        )

    print(f"Wrote sweep to {sweep_dir}")
    warn_if_real_only_invariant(sweep["comparison_rows"])


if __name__ == "__main__":
    main()
