"""Standard experiment artifact writers for paper-ready runs."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
import numpy as np

from ..synthesis.synthetic_audit import AUDIT_CSV_COLUMNS
from ..datasets.dataset_stats import build_dataset_stats, build_experiment_dataset_stats
from ..datasets.load_options import DatasetLoadOptions
from ..datasets.tsb_loader import TimeSeriesRecord
from .audit_sanity import validate_synthetic_audit_csv


def get_git_commit_hash(project_root: Path) -> str | None:
    """Return current git commit hash if the repo is available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        commit = result.stdout.strip()
        return commit or None
    except (subprocess.SubprocessError, OSError):
        return None


def build_run_metadata(config: dict[str, Any], *, project_root: Path, experiment_name: str) -> dict[str, Any]:
    return {
        "experiment_name": experiment_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": get_git_commit_hash(project_root),
        "seed": int(config.get("seed", 42)),
    }


def save_split_ids(output_dir: Path, split_ids: dict[str, list[str]]) -> Path:
    path = output_dir / "split_ids.json"
    path.write_text(json.dumps(split_ids, indent=2), encoding="utf-8")
    return path


def save_dataset_stats(
    output_dir: Path,
    records: list[TimeSeriesRecord],
    *,
    data_path: str,
    load_options: DatasetLoadOptions,
    load_summary: dict[str, Any] | None = None,
    split_ids: dict[str, list[str]] | None = None,
) -> Path:
    if split_ids is not None:
        stats = build_experiment_dataset_stats(records, split_ids, load_summary=load_summary)
        stats["dataset_name"] = load_options.dataset_name
        stats["data_path"] = data_path
        stats["load_options"] = load_options.to_dict()
    else:
        stats = build_dataset_stats(
            records,
            data_path=data_path,
            load_options=load_options,
            load_summary=load_summary,
        )
    path = output_dir / "dataset_stats.json"
    path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return path


def save_resolved_config(output_dir: Path, config: dict[str, Any]) -> Path:
    path = output_dir / "config_resolved.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def save_run_metadata(output_dir: Path, metadata: dict[str, Any]) -> Path:
    path = output_dir / "run_metadata.json"
    path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return path


def save_metrics(output_dir: Path, metrics: dict[str, Any], *, filename: str = "metrics.json") -> Path:
    path = output_dir / filename
    path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return path


def save_per_series_metrics(output_dir: Path, rows: list[dict[str, Any]], *, filename: str = "per_series_metrics.csv") -> Path:
    path = output_dir / filename
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def save_synthetic_audit(output_dir: Path, audit_rows: list[dict[str, Any]]) -> Path | None:
    if not audit_rows:
        return None
    path = output_dir / "synthetic_audit.csv"
    frame = pd.DataFrame(audit_rows)
    for column in AUDIT_CSV_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan
    ordered = [column for column in AUDIT_CSV_COLUMNS if column in frame.columns]
    extras = [column for column in frame.columns if column not in ordered]
    frame[ordered + extras].to_csv(path, index=False)
    return path


def save_standard_experiment_bundle(
    output_dir: Path,
    *,
    config: dict[str, Any],
    project_root: Path,
    experiment_name: str,
    records: list[TimeSeriesRecord],
    data_path: str,
    load_options: DatasetLoadOptions,
    load_summary: dict[str, Any],
    split_ids: dict[str, list[str]],
    metrics: dict[str, Any],
    per_series_rows: list[dict[str, Any]] | None = None,
    synthetic_audit: list[dict[str, Any]] | None = None,
    extra_files: dict[str, Any] | None = None,
    synthesis_requested: bool | None = None,
) -> Path:
    """Write the standard artifact set required for internal paper experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(output_dir, config)
    save_run_metadata(output_dir, build_run_metadata(config, project_root=project_root, experiment_name=experiment_name))
    save_dataset_stats(output_dir, records, data_path=data_path, load_options=load_options, load_summary=load_summary, split_ids=split_ids)
    save_split_ids(output_dir, split_ids)
    save_metrics(output_dir, metrics)
    if per_series_rows:
        save_per_series_metrics(output_dir, per_series_rows)
    if synthesis_requested is None:
        synthesis_requested = any(
            spec.get("kind") == "synthesis"
            for spec in (config.get("experiment", {}).get("methods") or config.get("methods") or [])
        ) or bool(config.get("synthesis", {}))
    validate_synthetic_audit_csv(
        synthetic_audit or [],
        config=config,
        synthesis_requested=bool(synthesis_requested),
    )
    if synthetic_audit is not None:
        save_synthetic_audit(output_dir, synthetic_audit)
    if extra_files:
        for name, payload in extra_files.items():
            target = output_dir / name
            if isinstance(payload, pd.DataFrame):
                payload.to_csv(target, index=False)
            elif isinstance(payload, (list, dict)):
                target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                target.write_text(str(payload), encoding="utf-8")
    return output_dir
