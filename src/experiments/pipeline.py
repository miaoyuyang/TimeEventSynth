"""Shared experiment pipeline: loading, splitting, evaluation."""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any

import numpy as np

from ..datasets.dataset_stats import build_experiment_dataset_stats
from ..datasets.load_options import DatasetLoadOptions, infer_dataset_name
from ..datasets.split_builder import build_series_split, series_parent_folder
from ..synthesis.dataset_identity import resolve_dataset_name
from ..datasets.synthetic_data import make_synthetic_records
from ..datasets.tsb_loader import TimeSeriesRecord, load_tsb_records_with_summary
from ..detectors.classical import build_detector
from ..detectors.train_label_stats import compute_train_label_stats
from ..evaluation.event_metrics import compute_event_metrics
from ..evaluation.per_series_metrics import compute_per_series_metrics
from ..evaluation.point_metrics import compute_point_metrics, parse_prediction_smoothing, select_validation_threshold
from ..evaluation.reporting import detector_is_supervised, warn_if_real_only_invariant
from .audit_sanity import annotate_invalid_point_metrics
from ..utils.seeds import detector_random_state, set_global_seed


def _flatten_labels_and_scores(records: list[TimeSeriesRecord], scores: dict[str, np.ndarray]) -> tuple[list[int], list[float]]:
    y_true: list[int] = []
    y_score: list[float] = []
    for record in records:
        series_scores = scores[record.series_id]
        y_true.extend(int(x) for x in record.labels.tolist())
        y_score.extend(float(x) for x in series_scores.tolist())
    return y_true, y_score


def build_detector_config(config: dict[str, Any]) -> dict[str, Any]:
    detector = config.get("detector", {})
    params = dict(detector.get("params", {}))
    random_state = int(params.get("random_state", detector_random_state(config)))
    model_type = str(detector.get("name") or detector.get("model_type", "random_forest_window"))
    return {
        "model_type": model_type,
        "name": model_type,
        "window_size": int(detector.get("window_size", params.get("window_size", 15))),
        "negative_sample_ratio": float(detector.get("negative_sample_ratio", 5.0)),
        "class_weight": detector.get("class_weight", "balanced"),
        "contamination": float(detector.get("contamination", params.get("contamination", 0.1))),
        "random_state": random_state,
        "num_lags": int(detector.get("num_lags", 2)),
        "params": params,
    }


def resolve_data_path(config: dict[str, Any], cli_data: Path | None, project_root: Path) -> Path | None:
    if cli_data is not None:
        return cli_data if cli_data.is_absolute() else project_root / cli_data
    raw_path = config.get("dataset", {}).get("path") or config.get("data", {}).get("raw_path")
    if not raw_path:
        return None
    path = Path(str(raw_path))
    return path if path.is_absolute() else project_root / path


def load_options_from_config(config: dict[str, Any]) -> DatasetLoadOptions:
    data_cfg = config.get("data", {})
    dataset_cfg = config.get("dataset", {})
    load_cfg = data_cfg.get("load_options", {})
    dataset_name = dataset_cfg.get("name") or data_cfg.get("dataset_name")
    raw_path = dataset_cfg.get("path") or data_cfg.get("raw_path")
    if dataset_name is None and raw_path:
        dataset_name = infer_dataset_name(Path(str(raw_path)))
    return DatasetLoadOptions(
        max_series=load_cfg.get("max_series", dataset_cfg.get("max_series")),
        min_event_windows=int(load_cfg.get("min_event_windows", dataset_cfg.get("min_event_windows", 0))),
        min_length=int(load_cfg.get("min_length", dataset_cfg.get("min_length", 1))),
        max_length=load_cfg.get("max_length", dataset_cfg.get("max_length")),
        min_event_length=int(load_cfg.get("min_event_length", 1)),
        merge_gap=int(load_cfg.get("merge_gap", 0)),
        max_event_ratio=load_cfg.get("max_event_ratio", dataset_cfg.get("max_event_ratio")),
        include_patterns=list(load_cfg.get("include_patterns", dataset_cfg.get("include_patterns", [])) or []),
        exclude_patterns=list(load_cfg.get("exclude_patterns", dataset_cfg.get("exclude_patterns", [])) or []),
        group_by_parent_folder=bool(
            load_cfg.get("group_by_parent_folder", dataset_cfg.get("group_by_parent_folder", True))
        ),
        dataset_name=dataset_name,
        seed=int(config.get("seed", 42)),
    )


def load_records_for_experiment(
    config: dict[str, Any],
    *,
    project_root: Path,
    use_synthetic: bool = False,
    cli_data: Path | None = None,
    load_options: DatasetLoadOptions | None = None,
) -> tuple[list[TimeSeriesRecord], dict[str, Any], str]:
    set_global_seed(int(config.get("seed", 42)))
    data_path = resolve_data_path(config, cli_data, project_root)
    if use_synthetic or data_path is None:
        return make_synthetic_records(int(config.get("seed", 42))), {}, "synthetic"
    options = load_options or load_options_from_config(config)
    if options.dataset_name is None and data_path is not None:
        options.dataset_name = infer_dataset_name(data_path)
    records, summary = load_tsb_records_with_summary(data_path, options=options)
    return records, summary, str(data_path)


def warn_if_small_test_benchmark(
    test_records: list[TimeSeriesRecord],
    *,
    all_records: list[TimeSeriesRecord] | None = None,
) -> None:
    """Print warnings when the test benchmark is too small to interpret reliably."""
    from ..datasets.event_extractor import labels_to_events

    num_test_series = len(test_records)
    total_test_event_windows = sum(len(labels_to_events(record.labels)) for record in test_records)
    test_parent_folders = sorted({series_parent_folder(record) for record in test_records})
    all_parent_folders = sorted({series_parent_folder(record) for record in (all_records or test_records)})

    if num_test_series < 5:
        print(
            f"WARNING: test split has only {num_test_series} series; "
            "results may not generalize (recommended >= 5 test series)."
        )
    if total_test_event_windows < 20:
        print(
            f"WARNING: test split has only {total_test_event_windows} event windows; "
            "event-level metrics may be unstable (recommended >= 20)."
        )
    if len(test_parent_folders) <= 1 and len(all_parent_folders) > 1:
        print(
            "WARNING: only one parent dataset/domain appears in the test split: "
            f"{test_parent_folders[0] if test_parent_folders else 'unknown'}."
        )
    elif len(all_parent_folders) <= 1:
        print(
            "WARNING: only one parent dataset/domain appears in the loaded benchmark: "
            f"{all_parent_folders[0] if all_parent_folders else 'unknown'}."
        )


def summarize_split_dataset_test_coverage(
    records: list[TimeSeriesRecord],
    split_ids: dict[str, list[str]],
) -> dict[str, Any]:
    """Summarize how many datasets contribute timelines to each split."""
    test_ids = set(split_ids.get("test", []))
    train_ids = set(split_ids.get("train", []))
    val_ids = set(split_ids.get("val", []))
    per_dataset: dict[str, dict[str, int]] = {}
    for record in records:
        dataset_name = resolve_dataset_name(record, fallback="unknown")
        stats = per_dataset.setdefault(
            dataset_name,
            {"num_timelines": 0, "num_test_timelines": 0, "num_train_timelines": 0, "num_val_timelines": 0},
        )
        stats["num_timelines"] += 1
        if record.series_id in test_ids:
            stats["num_test_timelines"] += 1
        if record.series_id in train_ids:
            stats["num_train_timelines"] += 1
        if record.series_id in val_ids:
            stats["num_val_timelines"] += 1
    missing_test = sorted(name for name, stats in per_dataset.items() if stats["num_test_timelines"] <= 0)
    return {
        "num_datasets": len(per_dataset),
        "datasets_missing_from_test": missing_test,
        "per_dataset": per_dataset,
    }


def validate_dataset_balanced_test_coverage(
    records: list[TimeSeriesRecord],
    split_ids: dict[str, list[str]],
    *,
    require_all_datasets_in_test: bool = True,
) -> dict[str, Any]:
    """Fail fast when balanced cross-dataset evaluation lacks per-dataset test timelines."""
    summary = summarize_split_dataset_test_coverage(records, split_ids)
    missing = list(summary["datasets_missing_from_test"])
    if require_all_datasets_in_test and missing:
        details = ", ".join(
            f"{name} (timelines={summary['per_dataset'][name]['num_timelines']})" for name in missing
        )
        raise ValueError(
            "Dataset-balanced evaluation requires at least one test timeline per dataset, "
            f"but these datasets are missing from test: {details}. "
            "Increase max_timelines_per_dataset or relax balancing limits."
        )
    if missing:
        print(
            "WARNING: datasets missing from test split after balancing: "
            + ", ".join(missing)
        )
    return summary


def build_experiment_dataset_stats_payload(
    records: list[TimeSeriesRecord],
    split_ids: dict[str, list[str]],
    load_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return build_experiment_dataset_stats(records, split_ids, load_summary=load_summary)


def split_records(
    records: list[TimeSeriesRecord],
    config: dict[str, Any],
) -> tuple[list[TimeSeriesRecord], list[TimeSeriesRecord], list[TimeSeriesRecord], dict[str, list[str]]]:
    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})
    split_ids = build_series_split(
        records,
        train_ratio=float(data_cfg.get("train_ratio", split_cfg.get("train_ratio", 0.7))),
        val_ratio=float(data_cfg.get("dev_ratio", data_cfg.get("val_ratio", split_cfg.get("val_ratio", 0.1)))),
        test_ratio=float(data_cfg.get("test_ratio", split_cfg.get("test_ratio", 0.2))),
        seed=int(split_cfg.get("seed", config.get("seed", 42))),
        stratify_by_has_event=bool(split_cfg.get("stratify_by_has_event", True)),
        group_by_parent_folder=bool(split_cfg.get("group_by_parent_folder", False)),
        stratify_by_dataset=bool(split_cfg.get("stratify_by_dataset", False)),
    )
    train_ids = set(split_ids["train"])
    val_ids = set(split_ids["val"])
    test_ids = set(split_ids["test"])
    train_records = [record for record in records if record.series_id in train_ids]
    val_records = [record for record in records if record.series_id in val_ids]
    test_records = [record for record in records if record.series_id in test_ids]
    evaluation_cfg = config.get("evaluation", {})
    if bool(evaluation_cfg.get("dataset_balanced", False)) and bool(
        split_cfg.get("stratify_by_dataset", evaluation_cfg.get("stratify_by_dataset", False))
    ):
        validate_dataset_balanced_test_coverage(records, split_ids)
    return train_records, val_records, test_records, split_ids


def donor_pool_records(
    train_records: list[TimeSeriesRecord],
    val_records: list[TimeSeriesRecord],
    config: dict[str, Any],
) -> list[TimeSeriesRecord]:
    """Return records allowed as synthesis donors (train-only by default)."""
    source = str(config.get("synthesis", {}).get("donor_source", "train_only")).lower()
    if source == "train_val":
        return list(train_records) + list(val_records)
    if source != "train_only":
        raise ValueError(f"Unsupported synthesis.donor_source: {source}")
    return list(train_records)


def prepare_low_label_train_and_donor_pool(
    train_records: list[TimeSeriesRecord],
    val_records: list[TimeSeriesRecord],
    config: dict[str, Any],
    *,
    labeled_fraction: float,
    seed: int,
) -> tuple[list[TimeSeriesRecord], list[TimeSeriesRecord]]:
    """Apply the low-label mask to train, then build donors from the visible train labels only.

    Train donors must use the same label budget as backbone training. Building the donor
    pool from unmasked train records would let synthesis mine withheld train anomalies.
    """
    from .low_label_masking import mask_train_labels_for_low_label

    masked_train = mask_train_labels_for_low_label(train_records, labeled_fraction, seed)
    donor_source = str(config.get("synthesis", {}).get("donor_source", "train_only")).lower()
    allow_val_donors = bool(config.get("synthesis", {}).get("allow_validation_donors_for_low_label", False))
    if donor_source == "train_val" and not allow_val_donors:
        raise ValueError(
            "synthesis.donor_source=train_val is not allowed in low-label augmentation by default because "
            "it exposes fully labeled validation anomalies. Set "
            "synthesis.allow_validation_donors_for_low_label=true to override explicitly."
        )
    donor_records = donor_pool_records(masked_train, val_records, config)
    return masked_train, donor_records


def summarize_dataset_distribution(records: list[TimeSeriesRecord]) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, int]] = {}
    for record in records:
        dataset_name = resolve_dataset_name(record, fallback="unknown")
        stats = per_dataset.setdefault(
            dataset_name,
            {"num_timelines": 0, "num_points": 0, "num_anomalous_timelines": 0, "num_anomaly_points": 0},
        )
        labels = np.asarray(record.labels, dtype=int).reshape(-1)
        stats["num_timelines"] += 1
        stats["num_points"] += int(labels.size)
        stats["num_anomaly_points"] += int(labels.sum())
        if int(labels.sum()) > 0:
            stats["num_anomalous_timelines"] += 1
    return {
        "num_datasets": len(per_dataset),
        "per_dataset": per_dataset,
    }


def balance_records_by_dataset(
    records: list[TimeSeriesRecord],
    *,
    max_timelines_per_dataset: int | None = None,
    max_points_per_dataset: int | None = None,
    min_timelines_per_dataset: int | None = None,
    seed: int = 42,
    preserve_anomaly_timelines: bool = True,
) -> tuple[list[TimeSeriesRecord], dict[str, Any]]:
    """Subsample timelines per dataset so no single dataset dominates evaluation."""
    before = summarize_dataset_distribution(records)
    if max_timelines_per_dataset is None and max_points_per_dataset is None:
        return list(records), {
            "enabled": False,
            "seed": int(seed),
            "before": before,
            "after": before,
        }

    rng = random.Random(seed)
    grouped: dict[str, list[TimeSeriesRecord]] = {}
    for record in records:
        dataset_name = resolve_dataset_name(record, fallback="unknown")
        grouped.setdefault(dataset_name, []).append(record)

    balanced: list[TimeSeriesRecord] = []
    for dataset_name, dataset_records in sorted(grouped.items()):
        event_records = [record for record in dataset_records if int(np.asarray(record.labels, dtype=int).sum()) > 0]
        normal_records = [record for record in dataset_records if int(np.asarray(record.labels, dtype=int).sum()) == 0]
        rng.shuffle(event_records)
        rng.shuffle(normal_records)
        ordered = event_records + normal_records if preserve_anomaly_timelines else list(dataset_records)
        if not preserve_anomaly_timelines:
            rng.shuffle(ordered)

        selected: list[TimeSeriesRecord] = []
        point_budget = 0
        for record in ordered:
            if max_timelines_per_dataset is not None and len(selected) >= int(max_timelines_per_dataset):
                break
            record_points = int(np.asarray(record.labels, dtype=int).size)
            if max_points_per_dataset is not None and selected and point_budget + record_points > int(max_points_per_dataset):
                continue
            selected.append(record)
            point_budget += record_points

        if not selected and ordered:
            selected.append(ordered[0])
        balanced.extend(selected)

    after = summarize_dataset_distribution(balanced)
    if min_timelines_per_dataset is not None:
        undersampled = [
            dataset_name
            for dataset_name, stats in after["per_dataset"].items()
            if int(stats.get("num_timelines", 0)) < int(min_timelines_per_dataset)
        ]
        if undersampled:
            raise ValueError(
                "Dataset balancing requires at least "
                f"{int(min_timelines_per_dataset)} timelines per dataset for split coverage, "
                f"but these datasets are under quota: {', '.join(sorted(undersampled))}."
            )
    return balanced, {
        "enabled": True,
        "seed": int(seed),
        "before": before,
        "after": after,
        "max_timelines_per_dataset": None if max_timelines_per_dataset is None else int(max_timelines_per_dataset),
        "max_points_per_dataset": None if max_points_per_dataset is None else int(max_points_per_dataset),
        "min_timelines_per_dataset": None if min_timelines_per_dataset is None else int(min_timelines_per_dataset),
        "preserve_anomaly_timelines": bool(preserve_anomaly_timelines),
    }


def evaluate_detector(
    config: dict[str, Any],
    train_records: list[TimeSeriesRecord],
    val_records: list[TimeSeriesRecord],
    test_records: list[TimeSeriesRecord],
    *,
    labeled_fraction: float | None = None,
    real_train_count: int | None = None,
) -> dict[str, Any]:
    detector_cfg = build_detector_config(config)
    detector = build_detector(detector_cfg)
    train_labels = [np.asarray(record.labels, dtype=int).reshape(-1) for record in train_records]
    detector.fit(train_records, train_labels)
    val_scores = detector.score(val_records, config=config)
    test_scores = detector.score(test_records, config=config)

    val_y_true, val_y_score = _flatten_labels_and_scores(val_records, val_scores)
    eval_cfg = config.get("evaluation", {})
    smoothing = parse_prediction_smoothing(eval_cfg)
    min_event_length = smoothing["min_event_length"] if smoothing["enabled"] else 1
    merge_gap = smoothing["merge_gap"] if smoothing["enabled"] else 0
    iou_threshold = float(eval_cfg.get("event_iou_threshold", 0.1))

    threshold_info = select_validation_threshold(val_y_true, val_y_score, eval_cfg)
    threshold = (
        float(threshold_info["threshold"])
        if np.isfinite(threshold_info["threshold"])
        else float(eval_cfg.get("point_threshold", 0.5))
    )

    test_y_true, test_y_score = _flatten_labels_and_scores(test_records, test_scores)
    per_series_rows = compute_per_series_metrics(
        test_records,
        test_scores,
        threshold=threshold,
        event_iou_threshold=iou_threshold,
        min_event_length=min_event_length,
        merge_gap=merge_gap,
    )
    if labeled_fraction is None:
        labeled_fraction = float(
            config.get("low_label", {}).get(
                "default_fraction",
                config.get("experiment", {}).get("labeled_fraction", 1.0),
            )
        )
    train_label_stats = compute_train_label_stats(
        train_records,
        labeled_fraction=float(labeled_fraction),
        real_train_count=real_train_count,
    )
    if hasattr(detector, "train_positive_count"):
        train_label_stats["detector_train_positive_count"] = int(detector.train_positive_count)
        train_label_stats["detector_train_negative_count"] = int(detector.train_negative_count)
    train_label_stats["detector_is_supervised"] = detector_is_supervised(detector_cfg)

    test_point_metrics = compute_point_metrics(test_y_true, test_y_score, threshold=threshold)
    test_point_metrics = annotate_invalid_point_metrics(test_y_true, test_y_score, test_point_metrics)

    return {
        "split_sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
        "threshold_selection": threshold_info,
        "test_point_metrics": test_point_metrics,
        "test_event_metrics": compute_event_metrics(
            test_y_true,
            y_score=test_y_score,
            threshold=threshold,
            iou_threshold=iou_threshold,
            min_event_length=min_event_length,
            merge_gap=merge_gap,
        ),
        "per_series_metrics": per_series_rows,
        "detector": detector_cfg,
        "train_label_stats": train_label_stats,
        "scores": {"val": val_scores, "test": test_scores},
    }
