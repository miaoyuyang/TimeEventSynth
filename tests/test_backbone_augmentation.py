from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.augmentation.policies import AugmentationResult, build_augmentation_result
from src.datasets.synthetic_data import make_synthetic_records
from src.detectors.iforest import IForestBackbone
from src.detectors.internal_classifier import InternalClassifierBackbone
from src.detectors.cnn import CNNBackbone
from src.detectors.timesnet import TimesNetBackbone
from src.detectors.base import merge_backbone_threshold_config
from src.experiments.config import load_config
from src.experiments.pipeline import (
    balance_records_by_dataset,
    build_detector_config,
    prepare_low_label_train_and_donor_pool,
    split_records,
)
from src.experiments.run_backbone_augmentation import (
    build_backbone_from_spec,
    evaluate_backbone_with_augmentation,
    run_backbone_augmentation_experiment,
    run_backbone_augmentation_multiseed,
)
from src.experiments.run_low_label import _mask_train_labels


def _smoke_config() -> dict:
    config = load_config(PROJECT_ROOT / "configs" / "experiment_backbone_augmentation_smoke.yaml")
    config["split"]["group_by_parent_folder"] = False
    return config


def test_detector_interface_fit_score_predict() -> None:
    records = make_synthetic_records(seed=7, num_series=9, events_per_series=4)
    train = records[:6]
    test = records[6:]

    supervised = InternalClassifierBackbone(detector_config=build_detector_config(_smoke_config()))
    supervised.fit(train, train_labels=[np.asarray(record.labels, dtype=int) for record in train])
    supervised_scores = supervised.score(test)
    supervised_preds = supervised.predict(supervised_scores, threshold=0.5)
    assert set(supervised_scores.keys()) == {record.series_id for record in test}
    assert set(supervised_preds.keys()) == {record.series_id for record in test}

    unsupervised = IForestBackbone(window_size=15, contamination=0.1, random_state=7)
    unsupervised.fit(train)
    unsupervised_scores = unsupervised.score(test)
    assert set(unsupervised_scores.keys()) == {record.series_id for record in test}


def test_augmentation_policy_result_shape() -> None:
    config = _smoke_config()
    records = make_synthetic_records(seed=11, num_series=9, events_per_series=4)
    train_records, val_records, _, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=11,
    )

    result = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="cross_dataset_compatible",
        config=config,
        labeled_fraction=0.2,
    )
    assert isinstance(result, AugmentationResult)
    assert result.compatibility_enabled is True
    assert isinstance(result.synthetic_windows, list)
    assert isinstance(result.audit_records, list)
    assert isinstance(result.rejection_summary, dict)
    assert isinstance(result.compatibility_summary, dict)


def test_groupwise_cross_dataset_compatible_keeps_group_fields() -> None:
    config = _smoke_config()
    config.setdefault("synthesis", {})
    config["synthesis"]["grouping"] = {"num_groups": 2, "grid_size": 8}
    records = make_synthetic_records(seed=12, num_series=9, events_per_series=4)
    train_records, val_records, _, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=12,
    )

    result = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="groupwise_cross_dataset_compatible",
        config=config,
        labeled_fraction=0.2,
    )
    assert isinstance(result, AugmentationResult)
    assert result.compatibility_enabled is True
    candidate_rows = [row for row in result.audit_records if str(row.get("record_type")) == "synthesis_candidate"]
    assert candidate_rows
    assert all("target_event_group" in row for row in candidate_rows)


def test_adaptive_groupwise_transfer_routes_by_detector_family() -> None:
    config = _smoke_config()
    config.setdefault("synthesis", {})
    config["synthesis"]["grouping"] = {"num_groups": 2, "grid_size": 8}
    records = make_synthetic_records(seed=13, num_series=9, events_per_series=4)
    train_records, val_records, _, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=13,
    )

    iforest_result = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="adaptive_groupwise_transfer",
        config=config,
        labeled_fraction=0.2,
        detector_backbone="iforest",
    )
    clf_result = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="adaptive_groupwise_transfer",
        config=config,
        labeled_fraction=0.2,
        detector_backbone="internal_classifier",
    )
    assert iforest_result.policy_name == "adaptive_groupwise_transfer"
    assert iforest_result.selected_policy_name == "groupwise_cross_dataset_compatible"
    assert iforest_result.donor_policy == "cross_dataset_compatible"
    assert clf_result.selected_policy_name == "groupwise_cross_dataset_all"
    assert clf_result.donor_policy == "cross_dataset_all"


def test_unsupervised_threshold_calibration_with_synthetic_scores() -> None:
    detector = IForestBackbone(window_size=15, contamination=0.1, random_state=3)
    val_scores = np.asarray([0.05, 0.10, 0.15, 0.20], dtype=float)
    labels = np.asarray([0, 0, 0, 0], dtype=int)
    synthetic_scores = np.asarray([0.75, 0.80, 0.90], dtype=float)
    calibration = detector.calibrate_threshold(val_scores, labels=labels, synthetic_scores=synthetic_scores)
    assert calibration.method == "synthetic_positive_separation"
    assert 0.20 <= calibration.threshold < 0.90


def test_backbone_threshold_mode_and_quantile_are_honored() -> None:
    detector = IForestBackbone(
        window_size=15,
        contamination=0.1,
        random_state=3,
        threshold_mode="quantile",
        quantile=0.99,
    )
    val_scores = np.asarray([0.05, 0.10, 0.15, 0.20, 0.90], dtype=float)
    synthetic_scores = np.asarray([0.75, 0.80, 0.90], dtype=float)
    calibration = detector.calibrate_threshold(val_scores, synthetic_scores=synthetic_scores, config={})
    assert calibration.method == "quantile"
    assert calibration.details.get("threshold_mode") == "quantile"
    assert float(calibration.details.get("quantile", calibration.details.get("threshold_quantile", -1))) == 0.99


def test_backbone_threshold_mode_wins_over_evaluation_global() -> None:
    detector = IForestBackbone(
        window_size=15,
        contamination=0.1,
        random_state=3,
        threshold_mode="quantile",
        quantile=0.99,
    )
    val_scores = np.asarray([0.05, 0.10, 0.15, 0.20, 0.90], dtype=float)
    synthetic_scores = np.asarray([0.75, 0.80, 0.90], dtype=float)
    calibration = detector.calibrate_threshold(
        val_scores,
        synthetic_scores=synthetic_scores,
        config=merge_backbone_threshold_config(
            detector,
            {"threshold_mode": "synthetic_separation", "event_iou_threshold": 0.1},
        ),
    )
    assert calibration.method == "quantile"
    assert calibration.details.get("threshold_mode") == "quantile"


def test_merge_backbone_threshold_config_prefers_backbone_over_evaluation() -> None:
    detector = IForestBackbone(threshold_mode="quantile", quantile=0.88, random_state=1)
    merged = merge_backbone_threshold_config(
        detector,
        {"threshold_mode": "synthetic_separation", "quantile": 0.5, "event_iou_threshold": 0.1},
        backbone_config={"threshold_mode": "oracle_val", "quantile": 0.5},
    )
    assert merged["threshold_mode"] == "quantile"
    assert merged["quantile"] == 0.88
    assert merged["event_iou_threshold"] == 0.1


def test_build_backbone_from_spec_attaches_fp_aware_threshold_settings() -> None:
    config = _smoke_config()
    fp_cfg = {
        "threshold_mode": "fp_aware_synthetic_separation",
        "quantile": 0.99,
        "max_false_positive_rate": 0.02,
        "min_calibration_precision": 0.10,
        "false_positive_penalty": 1.5,
        "fallback_when_inverted_gap": True,
    }
    for name in ("ocsvm", "timesnet"):
        backbone = build_backbone_from_spec(
            config,
            {
                "name": name,
                "config": {
                    "window_size": 12,
                    "epochs": 1,
                    "batch_size": 16,
                    "max_train_windows": 128,
                    **fp_cfg,
                },
            },
        )
        merged = merge_backbone_threshold_config(backbone, {"threshold_mode": "synthetic_separation"})
        assert getattr(backbone, "threshold_mode") == "fp_aware_synthetic_separation"
        assert merged["threshold_mode"] == "fp_aware_synthetic_separation"
        assert merged["quantile"] == 0.99
        assert merged["max_false_positive_rate"] == 0.02
        assert merged["min_calibration_precision"] == 0.10
        assert merged["false_positive_penalty"] == 1.5
        assert merged["fallback_when_inverted_gap"] is True


def test_runner_honors_detector_backbones_threshold_mode_over_evaluation() -> None:
    config = _smoke_config()
    config.setdefault("evaluation", {})["threshold_mode"] = "oracle_val"
    config.setdefault("evaluation", {})["quantile"] = 0.5
    records = make_synthetic_records(seed=21, num_series=9, events_per_series=4)
    train_records, val_records, test_records, _ = split_records(records, config)
    masked_train = _mask_train_labels(train_records, 0.2, 21)
    augmentation = build_augmentation_result(
        masked_train,
        donor_records=prepare_low_label_train_and_donor_pool(
            train_records,
            val_records,
            config,
            labeled_fraction=0.2,
            seed=21,
        )[1],
        policy_name="cross_dataset_all",
        config=config,
        labeled_fraction=0.2,
    )
    backbone = build_backbone_from_spec(
        config,
        {
            "name": "iforest",
            "config": {
                "window_size": 15,
                "contamination": 0.1,
                "threshold_mode": "quantile",
                "quantile": 0.99,
            },
        },
    )
    row = evaluate_backbone_with_augmentation(
        backbone,
        augmentation,
        train_records=masked_train,
        val_records=val_records,
        test_records=test_records,
        config=config,
        labeled_fraction=0.2,
        split_seed=21,
        backbone_config={"threshold_mode": "quantile", "quantile": 0.99},
    )
    assert row["threshold_mode"] == "quantile"
    assert row["threshold_calibration_method"] == "quantile"


def test_evaluate_backbone_with_augmentation_uses_backbone_calibration_settings() -> None:
    config = _smoke_config()
    records = make_synthetic_records(seed=19, num_series=9, events_per_series=4)
    train_records, val_records, test_records, _ = split_records(records, config)
    masked_train = _mask_train_labels(train_records, 0.2, 19)
    augmentation = build_augmentation_result(
        masked_train,
        donor_records=prepare_low_label_train_and_donor_pool(
            train_records,
            val_records,
            config,
            labeled_fraction=0.2,
            seed=19,
        )[1],
        policy_name="cross_dataset_all",
        config=config,
        labeled_fraction=0.2,
    )
    backbone = IForestBackbone(
        window_size=15,
        contamination=0.1,
        random_state=19,
        threshold_mode="quantile",
        quantile=0.99,
    )
    row = evaluate_backbone_with_augmentation(
        backbone,
        augmentation,
        train_records=masked_train,
        val_records=val_records,
        test_records=test_records,
        config=config,
        labeled_fraction=0.2,
        split_seed=19,
        backbone_config={"threshold_mode": "quantile", "quantile": 0.99},
    )
    assert row["threshold_mode"] == "quantile"
    assert row["threshold_calibration_method"] == "quantile"


def test_run_backbone_augmentation_supports_adaptive_groupwise_transfer() -> None:
    config = _smoke_config()
    config["augmentation_policies"] = [{"name": "adaptive_groupwise_transfer"}]
    config["detector_backbones"] = [
        {"name": "iforest", "config": {"window_size": 15}},
        {"name": "internal_classifier", "config": {}},
    ]
    config.setdefault("synthesis", {})
    config["synthesis"]["grouping"] = {"num_groups": 2, "grid_size": 8}
    payload = run_backbone_augmentation_experiment(config, use_synthetic=True)
    frame = payload["metrics_frame"]
    assert set(frame["augmentation_policy"]) == {"adaptive_groupwise_transfer"}
    selected = dict(zip(frame["detector_backbone"], frame["selected_policy_name"]))
    assert selected["iforest"] == "groupwise_cross_dataset_compatible"
    assert selected["internal_classifier"] == "groupwise_cross_dataset_all"
    audit = pd.DataFrame(payload["synthetic_audit"])
    adaptive_audit = audit[audit["augmentation_policy"] == "adaptive_groupwise_transfer"]
    assert not adaptive_audit.empty
    assert set(adaptive_audit["detector_backbone"]) == {"iforest", "internal_classifier"}


def test_evaluate_backbone_with_augmentation_runs_cnn() -> None:
    config = _smoke_config()
    records = make_synthetic_records(seed=31, num_series=9, events_per_series=4)
    train_records, val_records, test_records, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=31,
    )
    augmentation = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="cross_dataset_all",
        config=config,
        labeled_fraction=0.2,
    )
    backbone = CNNBackbone(window_size=12, stride=2, epochs=1, batch_size=16, max_train_windows=128)
    row = evaluate_backbone_with_augmentation(
        backbone,
        augmentation,
        train_records=masked_train,
        val_records=val_records,
        test_records=test_records,
        config=config,
        labeled_fraction=0.2,
        split_seed=31,
        backbone_config={"threshold_mode": "synthetic_separation"},
    )
    assert row["detector_backbone"] == "cnn"
    assert row["threshold_calibration_method"] == "synthetic_positive_separation"
    assert row["num_synthetic_windows"] >= 0


def test_evaluate_backbone_with_augmentation_runs_timesnet() -> None:
    config = _smoke_config()
    records = make_synthetic_records(seed=37, num_series=9, events_per_series=4)
    train_records, val_records, test_records, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=37,
    )
    augmentation = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="cross_dataset_all",
        config=config,
        labeled_fraction=0.2,
    )
    backbone = TimesNetBackbone(
        window_size=12,
        stride=2,
        epochs=1,
        batch_size=16,
        d_model=16,
        num_blocks=1,
        max_train_windows=128,
    )
    row = evaluate_backbone_with_augmentation(
        backbone,
        augmentation,
        train_records=masked_train,
        val_records=val_records,
        test_records=test_records,
        config=config,
        labeled_fraction=0.2,
        split_seed=37,
        backbone_config={"threshold_mode": "synthetic_separation"},
    )
    assert row["detector_backbone"] == "timesnet"
    assert row["threshold_calibration_method"] == "synthetic_positive_separation"
    assert row["num_synthetic_windows"] >= 0


def test_merge_backbone_threshold_config_fills_missing_evaluation_keys() -> None:
    detector = IForestBackbone(threshold_mode="quantile", quantile=0.88, random_state=1)
    merged = merge_backbone_threshold_config(detector, {"event_iou_threshold": 0.1})
    assert merged["threshold_mode"] == "quantile"
    assert merged["quantile"] == 0.88
    assert merged["threshold_quantile"] == 0.88
    assert merged["event_iou_threshold"] == 0.1


def test_compatibility_strict_never_keeps_below_threshold_donors() -> None:
    config = _smoke_config()
    records = make_synthetic_records(seed=13, num_series=9, events_per_series=4)
    train_records, val_records, _, _ = split_records(records, config)
    masked_train, donor_records = prepare_low_label_train_and_donor_pool(
        train_records,
        val_records,
        config,
        labeled_fraction=0.2,
        seed=13,
    )
    result = build_augmentation_result(
        masked_train,
        donor_records=donor_records,
        policy_name="compatibility_strict",
        config=config,
        labeled_fraction=0.2,
    )
    donor_rows = [row for row in result.audit_records if str(row.get("record_type")) == "donor_pair" and bool(row.get("accepted", False))]
    for row in donor_rows:
        assert float(row["compatibility_score"]) >= 0.7


def test_runner_writes_required_metric_columns(tmp_path: Path) -> None:
    config = _smoke_config()
    payload = run_backbone_augmentation_experiment(config, use_synthetic=True)
    frame = payload["metrics_frame"]
    required = {
        "detector_backbone",
        "augmentation_policy",
        "synthesis_method",
        "compatibility_enabled",
        "labeled_fraction",
        "auroc",
        "auprc",
        "best_point_f1",
        "event_precision",
        "event_recall",
        "event_f1",
        "false_positive_events",
        "threshold",
        "threshold_calibration_method",
        "num_synthetic_windows",
        "num_synthetic_points",
        "num_rejected_synthetic_windows",
    }
    assert required.issubset(set(frame.columns))

    output_dir = tmp_path / "runner_output"
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_backbone_augmentation",
        "--config",
        str(PROJECT_ROOT / "configs" / "experiment_backbone_augmentation_smoke.yaml"),
        "--use-synthetic",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    written = output_dir / "backbone_comparison_metrics.csv"
    assert written.exists()
    header = written.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert required.issubset(set(header))
    report = json.loads((output_dir / "analysis_report.json").read_text(encoding="utf-8"))
    assert "best_augmentation_policy_per_detector" in report


def test_dataset_balancing_caps_timelines_per_dataset() -> None:
    records = make_synthetic_records(seed=17, num_series=15, events_per_series=4)
    balanced, summary = balance_records_by_dataset(records, max_timelines_per_dataset=2, seed=17)
    assert summary["enabled"] is True
    for dataset_stats in summary["after"]["per_dataset"].values():
        assert int(dataset_stats["num_timelines"]) <= 2
    assert len(balanced) <= 6


def test_multiseed_runner_writes_aggregate_artifacts(tmp_path: Path) -> None:
    config = load_config(PROJECT_ROOT / "configs" / "experiment_backbone_augmentation_balanced_smoke.yaml")
    payload = run_backbone_augmentation_multiseed(config, use_synthetic=True)
    assert not payload["seed_metrics"].empty
    assert not payload["aggregate_metrics"].empty
    assert "aggregate_improvement" in payload["multi_seed_analysis_report"]

    output_dir = tmp_path / "multiseed_output"
    command = [
        sys.executable,
        "-m",
        "src.experiments.run_backbone_augmentation",
        "--config",
        str(PROJECT_ROOT / "configs" / "experiment_backbone_augmentation_balanced_smoke.yaml"),
        "--use-synthetic",
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    for name in (
        "seed_level_backbone_metrics.csv",
        "aggregate_backbone_comparison_metrics.csv",
        "dataset_balance_summary.json",
        "multi_seed_analysis_report.json",
    ):
        assert (output_dir / name).exists()
