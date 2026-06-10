"""Isolation Forest detector backbone wrapper."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import (
    DetectorBackbone,
    apply_score_normalization,
    make_windows,
    map_window_scores_to_points,
    score_normalization_mode,
    window_features,
)


@dataclass
class IForestDetector(DetectorBackbone):
    window_size: int = 15
    stride: int = 1
    contamination: float = 0.1
    random_state: int = 42
    n_estimators: int = 200
    threshold_mode: str = "synthetic_separation"
    quantile: float = 0.95
    max_train_windows: int | None = None
    score_reduction: str = "max"

    def __post_init__(self) -> None:
        self.name = "iforest"
        self.supports_training_augmentation = False
        self.is_supervised = False
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
        )

    def fit(
        self,
        train_series: list[Any],
        train_labels: list[np.ndarray] | None = None,
        synthetic_windows: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "IForestDetector":
        del train_labels, synthetic_windows, config
        feature_blocks: list[np.ndarray] = []
        for record in train_series:
            windows, _ = make_windows(record.values, self.window_size, self.stride)
            feature_blocks.append(window_features(windows))
        features = np.concatenate(feature_blocks, axis=0) if feature_blocks else np.zeros((0, self.window_size), dtype=float)
        if self.max_train_windows is not None and len(features) > self.max_train_windows:
            rng = np.random.default_rng(self.random_state)
            idx = rng.choice(len(features), size=self.max_train_windows, replace=False)
            features = features[idx]
        self.model.fit(features)
        return self

    def score(self, test_series: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in test_series:
            windows, spans = make_windows(record.values, self.window_size, self.stride)
            features = window_features(windows)
            window_scores = -self.model.score_samples(features)
            point_scores = map_window_scores_to_points(
                len(np.asarray(record.values)),
                spans,
                apply_score_normalization(window_scores, normalization),
                reduction=self.score_reduction,
            )
            outputs[str(record.series_id)] = point_scores
        return outputs


IForestBackbone = IForestDetector
