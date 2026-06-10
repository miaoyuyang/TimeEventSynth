from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.thresholding import (
    calibrate_threshold,
    calibrate_threshold_oracle,
    calibrate_threshold_quantile,
    calibrate_threshold_synthetic_separation,
)
from src.evaluation.point_metrics import precision_recall_f1_at_threshold


def test_quantile_threshold() -> None:
    scores = np.asarray([0.1, 0.2, 0.3, 0.4, 0.9], dtype=float)
    threshold = calibrate_threshold_quantile(scores, quantile=0.8)
    assert 0.3 <= threshold <= 0.9


def test_oracle_threshold_improves_f1_on_toy_data() -> None:
    scores = np.asarray([0.1, 0.2, 0.25, 0.7, 0.8, 0.9], dtype=float)
    labels = np.asarray([0, 0, 0, 1, 1, 1], dtype=int)
    threshold, best_f1 = calibrate_threshold_oracle(scores, labels, metric="f1")
    default_metrics = precision_recall_f1_at_threshold(labels.tolist(), scores.tolist(), 0.5)
    assert best_f1 >= default_metrics["f1"]
    assert 0.2 <= threshold <= 0.9


def test_synthetic_separation_reasonable_threshold() -> None:
    normal_scores = np.asarray([0.05, 0.10, 0.12, 0.18], dtype=float)
    synthetic_scores = np.asarray([0.7, 0.75, 0.9], dtype=float)
    threshold, diagnostics = calibrate_threshold_synthetic_separation(normal_scores, synthetic_scores, metric="f1")
    assert np.max(normal_scores) <= threshold < np.max(synthetic_scores)
    assert diagnostics["score_gap"] > 0
    assert diagnostics["best_calibration_f1"] >= 0.5


def test_synthetic_separation_fallback_when_empty() -> None:
    normal_scores = np.asarray([0.05, 0.10, 0.12, 0.18], dtype=float)
    threshold, diagnostics = calibrate_threshold_synthetic_separation(normal_scores, [], metric="f1")
    assert np.isfinite(threshold)
    assert diagnostics["fallback_used"] is True
    assert diagnostics["num_synthetic_scores"] == 0


def test_synthetic_separation_fallback_when_no_normal_scores() -> None:
    result = calibrate_threshold(
        mode="synthetic_separation",
        scores=np.asarray([0.6, 0.7, 0.8], dtype=float),
        labels=np.asarray([1, 1, 1], dtype=int),
        synthetic_scores=np.asarray([0.9, 0.95], dtype=float),
        config={"quantile": 0.95},
    )
    assert np.isfinite(result["threshold"])
    assert result["diagnostics"]["fallback_used"] is True
    assert result["diagnostics"]["fallback_reason"] == "empty_normal_scores"


def test_dispatcher_returns_diagnostics() -> None:
    result = calibrate_threshold(
        mode="synthetic_separation",
        scores=np.asarray([0.1, 0.2, 0.25], dtype=float),
        labels=np.asarray([0, 0, 0], dtype=int),
        synthetic_scores=np.asarray([0.8, 0.9], dtype=float),
        config={"metric": "f1"},
    )
    assert "threshold" in result
    assert result["threshold_mode"] == "synthetic_separation"
    assert "diagnostics" in result
    assert "synthetic_score_mean" in result["diagnostics"]


def test_fp_aware_synthetic_separation_limits_normal_false_positives() -> None:
    result = calibrate_threshold(
        mode="fp_aware_synthetic_separation",
        scores=np.asarray([0.05, 0.10, 0.20, 0.30, 0.80], dtype=float),
        labels=np.asarray([0, 0, 0, 0, 1], dtype=int),
        synthetic_scores=np.asarray([0.35, 0.90], dtype=float),
        config={"max_false_positive_rate": 0.25, "false_positive_penalty": 1.0, "quantile": 0.95},
    )
    assert result["threshold_mode"] == "fp_aware_synthetic_separation"
    assert result["diagnostics"]["best_calibration_false_positive_rate"] <= 0.25


def test_fp_aware_synthetic_separation_falls_back_on_inverted_gap() -> None:
    result = calibrate_threshold(
        mode="fp_aware_synthetic_separation",
        scores=np.asarray([0.8, 0.9, 1.0], dtype=float),
        labels=np.asarray([0, 0, 0], dtype=int),
        synthetic_scores=np.asarray([0.1, 0.2], dtype=float),
        config={"fallback_when_inverted_gap": True, "quantile": 0.9},
    )
    assert result["diagnostics"]["fallback_used"] is True
    assert result["diagnostics"]["fallback_reason"] == "inverted_synthetic_normal_gap"
