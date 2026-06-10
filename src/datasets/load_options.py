"""Shared dataset loading options for CLI entry points."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class DatasetLoadOptions:
    """Filters applied after parsing raw files into series records."""

    max_series: int | None = None
    min_event_windows: int = 0
    min_length: int = 1
    max_length: int | None = None
    min_event_length: int = 1
    merge_gap: int = 0
    max_event_ratio: float | None = None
    include_patterns: list[str] = field(default_factory=list)
    exclude_patterns: list[str] = field(default_factory=list)
    group_by_parent_folder: bool = True
    dataset_name: str | None = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def infer_dataset_name(data_path: Path) -> str:
    """Derive a stable dataset name from a file or directory path."""
    if data_path.is_file():
        return data_path.stem
    resolved = data_path.resolve()
    return resolved.name or "dataset"


def add_dataset_load_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=None, help="Single file or directory of series CSV/out files.")
    parser.add_argument("--max-series", type=int, default=None, help="Keep at most N series after filtering.")
    parser.add_argument(
        "--min-event-windows",
        type=int,
        default=None,
        help="Drop series with fewer than K contiguous anomaly windows.",
    )
    parser.add_argument("--min-length", type=int, default=None, help="Drop series shorter than T points.")
    parser.add_argument("--max-length", type=int, default=None, help="Drop series longer than T points.")
    parser.add_argument(
        "--dataset-name",
        type=str,
        default=None,
        help="Name used in output paths and stats JSON (defaults to data folder name).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for subsampling and splits.")


def load_options_from_args(args: argparse.Namespace, *, config: dict[str, Any] | None = None) -> DatasetLoadOptions:
    data_path = getattr(args, "data", None)
    dataset_name = getattr(args, "dataset_name", None)
    if dataset_name is None and data_path is not None:
        dataset_name = infer_dataset_name(Path(data_path))

    dataset_cfg = (config or {}).get("dataset", {})
    data_cfg = (config or {}).get("data", {})
    load_cfg = data_cfg.get("load_options", {})

    def pick(name: str, default: Any) -> Any:
        cli_value = getattr(args, name.replace("-", "_"), None) if hasattr(args, name.replace("-", "_")) else None
        if cli_value is not None:
            return cli_value
        if dataset_cfg.get(name) is not None:
            return dataset_cfg.get(name)
        if load_cfg.get(name) is not None:
            return load_cfg.get(name)
        return default

    seed = getattr(args, "seed", None)
    if seed is None:
        seed = int((config or {}).get("seed", 42))

    return DatasetLoadOptions(
        max_series=pick("max_series", None),
        min_event_windows=int(pick("min_event_windows", 0)),
        min_length=int(pick("min_length", 1)),
        max_length=pick("max_length", None),
        min_event_length=int(pick("min_event_length", 1)),
        merge_gap=int(pick("merge_gap", 0)),
        max_event_ratio=pick("max_event_ratio", None),
        include_patterns=list(pick("include_patterns", []) or []),
        exclude_patterns=list(pick("exclude_patterns", []) or []),
        group_by_parent_folder=bool(pick("group_by_parent_folder", True)),
        dataset_name=dataset_name or dataset_cfg.get("name") or data_cfg.get("dataset_name"),
        seed=int(seed),
    )


def apply_dataset_cli_to_config(config: dict[str, Any], args: argparse.Namespace) -> DatasetLoadOptions:
    """Merge CLI dataset overrides into the experiment config."""
    load_options = load_options_from_args(args, config=config)
    if getattr(args, "seed", None) is not None:
        config["seed"] = int(args.seed)
        config.setdefault("split", {})["seed"] = int(args.seed)
    if args.data is not None:
        config.setdefault("dataset", {})["path"] = str(args.data)
        config.setdefault("data", {})["raw_path"] = str(args.data)
    if load_options.dataset_name is not None:
        config.setdefault("dataset", {})["name"] = load_options.dataset_name
        config.setdefault("data", {})["dataset_name"] = load_options.dataset_name
    dataset = config.setdefault("dataset", {})
    if load_options.max_series is not None:
        dataset["max_series"] = load_options.max_series
    if load_options.min_event_windows:
        dataset["min_event_windows"] = load_options.min_event_windows
    if load_options.min_length > 1:
        dataset["min_length"] = load_options.min_length
    if load_options.max_length is not None:
        dataset["max_length"] = load_options.max_length
    dataset["max_event_ratio"] = load_options.max_event_ratio
    dataset["include_patterns"] = load_options.include_patterns
    dataset["exclude_patterns"] = load_options.exclude_patterns
    dataset["group_by_parent_folder"] = load_options.group_by_parent_folder
    config.setdefault("data", {})["load_options"] = load_options.to_dict()
    return load_options
