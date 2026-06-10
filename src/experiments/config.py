"""Config loading and normalization for TimeEventSynth experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_config(path: Path, *, project_root: Path | None = None) -> dict[str, Any]:
    """Load YAML config with optional inheritance (single path or ordered list)."""
    root = project_root or PROJECT_ROOT
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    inherits = config.get("inherits")
    if inherits:
        inherit_paths = inherits if isinstance(inherits, list) else [inherits]
        merged: dict[str, Any] = {}
        for inherit in inherit_paths:
            parent_path = (root / str(inherit)).resolve()
            parent = load_config(parent_path, project_root=root)
            merged = _deep_merge_config(merged, parent)
        for key, value in config.items():
            if key == "inherits":
                continue
            if isinstance(value, dict) and isinstance(merged.get(key), dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        config = merged
    return normalize_config(config)


def _deep_merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    """Merge paper-ready schema fields with legacy keys used by runners."""
    cfg = dict(config)

    dataset = dict(cfg.get("dataset", {}))
    data = dict(cfg.get("data", {}))
    if dataset.get("path"):
        data["raw_path"] = dataset["path"]
    if dataset.get("name"):
        data["dataset_name"] = dataset["name"]
    load_options = dict(data.get("load_options", {}))
    for key in (
        "max_series",
        "min_event_windows",
        "min_length",
        "max_length",
        "max_event_ratio",
        "include_patterns",
        "exclude_patterns",
        "group_by_parent_folder",
    ):
        if dataset.get(key) is not None:
            load_options[key] = dataset[key]
    if load_options:
        data["load_options"] = load_options
    cfg["data"] = data

    split = dict(cfg.get("split", {}))
    split_seed = split.get("seed", cfg.get("seed", 42))
    split.setdefault("seed", split_seed)
    split.setdefault("stratify_by_has_event", True)
    if dataset.get("group_by_parent_folder") is not None:
        split.setdefault("group_by_parent_folder", dataset["group_by_parent_folder"])
    cfg["split"] = split
    cfg["seed"] = int(split_seed)

    if split.get("train_ratio") is not None:
        data["train_ratio"] = split["train_ratio"]
    if split.get("val_ratio") is not None:
        data["dev_ratio"] = split["val_ratio"]
        data["val_ratio"] = split["val_ratio"]
    if split.get("test_ratio") is not None:
        data["test_ratio"] = split["test_ratio"]

    low_label = dict(cfg.get("low_label", {}))
    fractions = low_label.get("fractions", [0.2])
    default_fraction = low_label.get("default_fraction", fractions[0] if fractions else 0.2)
    experiment = dict(cfg.get("experiment", {}))
    experiment.setdefault("labeled_fraction", default_fraction)
    experiment.setdefault("low_label_fractions", fractions)
    cfg["low_label"] = {**low_label, "default_fraction": default_fraction, "fractions": fractions}
    cfg["experiment"] = experiment

    evaluation = dict(cfg.get("evaluation", {}))
    evaluation["labeled_fraction"] = float(experiment["labeled_fraction"])
    cfg["evaluation"] = evaluation

    synthesis = dict(cfg.get("synthesis", {}))
    experiment.setdefault("top_k_donors", synthesis.get("top_k", synthesis.get("retrieval_top_k", 3)))
    experiment.setdefault("alignment_grid_size", synthesis.get("alignment_grid_size", 64))
    experiment.setdefault("synthesis_methods", synthesis.get("methods", []))
    synthesis.setdefault("donor_source", "train_only")
    cfg["synthesis"] = synthesis

    detector = dict(cfg.get("detector", {}))
    params = dict(detector.get("params", {}))
    if detector.get("name") and not detector.get("model_type"):
        detector["model_type"] = detector["name"]
    for key, value in params.items():
        detector.setdefault(key, value)
    cfg["detector"] = detector

    evaluation = dict(cfg.get("evaluation", {}))
    if evaluation.get("threshold_selection") and not evaluation.get("threshold_metric"):
        evaluation["threshold_metric"] = evaluation["threshold_selection"]
    cfg["evaluation"] = evaluation

    return cfg
