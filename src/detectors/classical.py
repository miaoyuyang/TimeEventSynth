"""Classical and supervised window-based detectors with a consistent series-level API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .base import apply_score_normalization, score_normalization_mode


def _as_2d(values: list[float] | list[list[float]] | np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return array.reshape(-1, 1)
    if array.ndim == 2:
        return array
    raise ValueError(f"Expected 1D or 2D time-series values, got shape {array.shape}")


def build_point_features(
    values: list[float] | list[list[float]] | np.ndarray,
    window: int = 5,
    *,
    num_lags: int = 2,
) -> np.ndarray:
    """Build sliding-window point features for univariate or multivariate series."""
    series = _as_2d(values)
    n_dims = series.shape[1]
    features: list[np.ndarray] = []
    for idx in range(len(series)):
        left = max(0, idx - window + 1)
        segment = series[left : idx + 1]
        raw = series[idx]
        previous = series[idx - 1] if idx > 0 else series[idx]
        rolling_mean = np.mean(segment, axis=0)
        rolling_std = np.std(segment, axis=0) if len(segment) > 1 else np.zeros(n_dims, dtype=float)
        first_diff = raw - previous
        seg_min = np.min(segment, axis=0)
        seg_max = np.max(segment, axis=0)
        lag_values = [series[idx - lag] if idx - lag >= 0 else series[0] for lag in range(1, num_lags + 1)]
        lags = np.concatenate(lag_values, axis=0) if lag_values else np.zeros(0, dtype=float)
        feature_row = np.concatenate([raw, rolling_mean, rolling_std, first_diff, seg_min, seg_max, lags], axis=0)
        features.append(feature_row.astype(float))
    return np.asarray(features, dtype=float)


def _labels_from_records(records: list[Any]) -> list[np.ndarray]:
    return [np.asarray(record.labels, dtype=int).reshape(-1) for record in records]


def _is_synthetic_record(record: Any) -> bool:
    metadata = getattr(record, "metadata", {}) or {}
    if bool(metadata.get("synthetic", False)):
        return True
    source_path = str(getattr(record, "source_path", "") or "")
    return source_path in {"synthetic", "oversampled"} or "__synthetic__" in str(getattr(record, "series_id", ""))


class SeriesDetector(Protocol):
    """Consistent detector API for real-only and synthesized experiments."""

    def fit(self, records: list[Any], labels: list[np.ndarray] | None = None) -> "SeriesDetector":
        ...

    def score(self, records: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        ...


def _normalize_scores(raw_scores: np.ndarray) -> np.ndarray:
    min_value = float(np.min(raw_scores))
    max_value = float(np.max(raw_scores))
    if max_value - min_value < 1e-8:
        return np.zeros_like(raw_scores, dtype=float)
    return ((raw_scores - min_value) / (max_value - min_value)).astype(float)


def _subsample_training_rows(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    negative_sample_ratio: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    pos_mask = labels > 0
    neg_mask = ~pos_mask
    pos_features = features[pos_mask]
    neg_features = features[neg_mask]
    if len(pos_features) == 0:
        if len(neg_features) == 0:
            raise ValueError("No training points available for supervised detector.")
        rng = np.random.default_rng(random_state)
        keep = min(len(neg_features), max(int(negative_sample_ratio), 1))
        chosen = rng.choice(len(neg_features), size=keep, replace=False)
        sampled_neg = neg_features[chosen]
        return sampled_neg, np.zeros(len(sampled_neg), dtype=int)

    max_negatives = int(max(len(pos_features) * negative_sample_ratio, 1))
    if len(neg_features) > max_negatives:
        rng = np.random.default_rng(random_state)
        chosen = rng.choice(len(neg_features), size=max_negatives, replace=False)
        neg_features = neg_features[chosen]

    x_train = np.vstack([pos_features, neg_features]) if len(neg_features) else pos_features
    y_train = np.concatenate(
        [np.ones(len(pos_features), dtype=int), np.zeros(len(neg_features), dtype=int)]
    )
    return x_train, y_train


@dataclass
class LogisticRegressionWindowDetector:
    """Supervised logistic regression on sliding-window point features."""

    window_size: int = 15
    negative_sample_ratio: float = 5.0
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    num_lags: int = 2

    def __post_init__(self) -> None:
        model_params = {
            "class_weight": self.class_weight,
            "max_iter": 1000,
            "random_state": self.random_state,
            **self.params,
        }
        self.model = LogisticRegression(**model_params)
        self._fitted = False
        self.train_positive_count: int = 0
        self.train_negative_count: int = 0

    def fit(self, records: list[Any], labels: list[np.ndarray] | None = None) -> "LogisticRegressionWindowDetector":
        label_arrays = labels if labels is not None else _labels_from_records(records)
        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for record, record_labels in zip(records, label_arrays):
            feats = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            if len(feats) != len(record_labels):
                raise ValueError(f"Feature/label length mismatch for series {record.series_id}")
            all_features.append(feats)
            all_labels.append(record_labels.astype(int))

        features = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0).astype(int)
        self.train_positive_count = int(np.sum(y > 0))
        self.train_negative_count = int(np.sum(y == 0))
        x_train, y_train = _subsample_training_rows(
            features,
            y,
            negative_sample_ratio=self.negative_sample_ratio,
            random_state=self.random_state,
        )
        self.model.fit(x_train, y_train)
        self._fitted = True
        return self

    def score(self, records: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in records:
            features = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            if hasattr(self.model, "predict_proba"):
                proba = self.model.predict_proba(features)
                if proba.shape[1] == 1:
                    scores = proba[:, 0]
                else:
                    scores = proba[:, 1]
            else:
                scores = self.model.decision_function(features)
            outputs[str(record.series_id)] = apply_score_normalization(np.asarray(scores, dtype=float), normalization)
        return outputs


@dataclass
class RandomForestWindowDetector:
    """Supervised random forest on sliding-window point features."""

    window_size: int = 15
    negative_sample_ratio: float = 5.0
    class_weight: str | dict[str, float] | None = "balanced"
    random_state: int = 42
    params: dict[str, Any] = field(default_factory=dict)
    num_lags: int = 2

    def __post_init__(self) -> None:
        model_params = {
            "n_estimators": 200,
            "max_depth": 12,
            "class_weight": self.class_weight,
            "random_state": self.random_state,
            **self.params,
        }
        self.model = RandomForestClassifier(**model_params)
        self._fitted = False
        self.train_positive_count: int = 0
        self.train_negative_count: int = 0

    def fit(self, records: list[Any], labels: list[np.ndarray] | None = None) -> "RandomForestWindowDetector":
        label_arrays = labels if labels is not None else _labels_from_records(records)
        all_features: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for record, record_labels in zip(records, label_arrays):
            feats = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            all_features.append(feats)
            all_labels.append(record_labels.astype(int))

        features = np.concatenate(all_features, axis=0)
        y = np.concatenate(all_labels, axis=0).astype(int)
        self.train_positive_count = int(np.sum(y > 0))
        self.train_negative_count = int(np.sum(y == 0))
        x_train, y_train = _subsample_training_rows(
            features,
            y,
            negative_sample_ratio=self.negative_sample_ratio,
            random_state=self.random_state,
        )
        self.model.fit(x_train, y_train)
        self._fitted = True
        return self

    def score(self, records: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in records:
            features = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            proba = self.model.predict_proba(features)
            scores = proba[:, 1] if proba.shape[1] > 1 else proba[:, 0]
            outputs[str(record.series_id)] = apply_score_normalization(np.asarray(scores, dtype=float), normalization)
        return outputs


@dataclass
class IsolationForestDetector:
    """Unsupervised isolation forest baseline on sliding-window point features."""

    window_size: int = 5
    contamination: float = 0.1
    random_state: int = 42
    num_lags: int = 2

    def __post_init__(self) -> None:
        self.model = IsolationForest(
            n_estimators=200,
            contamination=self.contamination,
            random_state=self.random_state,
        )
        self._fitted = False

    def fit(self, records: list[Any], labels: list[np.ndarray] | None = None) -> "IsolationForestDetector":
        del labels
        features = np.concatenate(
            [build_point_features(record.values, self.window_size, num_lags=self.num_lags) for record in records],
            axis=0,
        )
        self.model.fit(features)
        self._fitted = True
        return self

    def score(self, records: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if not self._fitted:
            raise RuntimeError("Detector must be fitted before scoring.")
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in records:
            features = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            raw_scores = -self.model.score_samples(features)
            outputs[str(record.series_id)] = apply_score_normalization(raw_scores, normalization)
        return outputs


@dataclass
class ZScoreDetector:
    """Simple z-score anomaly detector over sliding-window features."""

    window_size: int = 5
    num_lags: int = 2

    def __post_init__(self) -> None:
        self.feature_mean: np.ndarray | None = None
        self.feature_std: np.ndarray | None = None

    def fit(self, records: list[Any], labels: list[np.ndarray] | None = None) -> "ZScoreDetector":
        del labels
        features = np.concatenate(
            [build_point_features(record.values, self.window_size, num_lags=self.num_lags) for record in records],
            axis=0,
        )
        self.feature_mean = np.mean(features, axis=0)
        self.feature_std = np.std(features, axis=0)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)
        return self

    def score(self, records: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Detector must be fitted before scoring.")
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in records:
            features = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            z = np.abs((features - self.feature_mean) / self.feature_std)
            outputs[str(record.series_id)] = apply_score_normalization(np.max(z, axis=1).astype(float), normalization)
        return outputs


def build_detector(config: dict[str, Any]) -> SeriesDetector:
    """Instantiate a detector from config."""
    model_type = str(config.get("model_type", config.get("name", "random_forest_window"))).lower()
    window_size = int(config.get("window_size", 15))
    random_state = int(config.get("random_state", 42))
    params = dict(config.get("params", {}))
    common = {
        "window_size": window_size,
        "random_state": random_state,
        "negative_sample_ratio": float(config.get("negative_sample_ratio", 5.0)),
        "class_weight": config.get("class_weight", "balanced"),
        "params": params,
        "num_lags": int(config.get("num_lags", 2)),
    }

    if model_type in {"random_forest_window", "random_forest", "randomforest"}:
        return RandomForestWindowDetector(**common)
    if model_type in {"logistic_regression_window", "logistic_regression", "logistic"}:
        return LogisticRegressionWindowDetector(**common)
    if model_type == "isolation_forest":
        return IsolationForestDetector(
            window_size=window_size,
            contamination=float(config.get("contamination", 0.1)),
            random_state=random_state,
            num_lags=int(config.get("num_lags", 2)),
        )
    if model_type == "zscore":
        return ZScoreDetector(window_size=window_size, num_lags=int(config.get("num_lags", 2)))
    raise ValueError(f"Unsupported detector model_type: {model_type}")
