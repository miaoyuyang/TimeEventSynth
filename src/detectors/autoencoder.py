"""Simple reconstruction-based autoencoder backbone using sklearn MLPRegressor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.neural_network import MLPRegressor

from .base import DetectorBackbone, apply_score_normalization, score_normalization_mode
from .classical import build_point_features


@dataclass
class AutoEncoderBackbone(DetectorBackbone):
    window_size: int = 15
    num_lags: int = 2
    hidden_layer_sizes: tuple[int, ...] = (64, 32, 64)
    random_state: int = 42
    max_iter: int = 200
    threshold_mode: str = "synthetic_separation"
    quantile: float = 0.95
    train_with_synthetic: bool = False
    max_train_points: int | None = None

    def __post_init__(self) -> None:
        self.name = "autoencoder"
        self.supports_training_augmentation = bool(self.train_with_synthetic)
        self.is_supervised = False
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            random_state=self.random_state,
            max_iter=self.max_iter,
            early_stopping=True,
        )

    def fit(
        self,
        train_series: list[Any],
        train_labels: list[np.ndarray] | None = None,
        synthetic_windows: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "AutoEncoderBackbone":
        del train_labels, config
        combined = list(train_series)
        if self.train_with_synthetic and synthetic_windows:
            combined.extend(list(synthetic_windows))
        features = np.concatenate(
            [build_point_features(record.values, self.window_size, num_lags=self.num_lags) for record in combined],
            axis=0,
        )
        if self.max_train_points is not None and len(features) > self.max_train_points:
            rng = np.random.default_rng(self.random_state)
            keep = rng.choice(len(features), size=self.max_train_points, replace=False)
            features = features[keep]
        self.model.fit(features, features)
        return self

    def score(self, test_series: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        for record in test_series:
            features = build_point_features(record.values, self.window_size, num_lags=self.num_lags)
            recon = self.model.predict(features)
            error = np.mean((recon - features) ** 2, axis=1)
            outputs[str(record.series_id)] = apply_score_normalization(error, normalization)
        return outputs
