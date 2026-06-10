"""CNN forecasting backbone for detector-agnostic augmentation experiments.

This is a lightweight PyTorch reimplementation of the forecasting-style CNN
baseline used in time-series anomaly detection benchmarks. The model predicts
the next point from a sliding context window and uses forecast error as the
anomaly score. Synthetic windows can influence threshold calibration by default
and can optionally be mixed into forecasting training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import warnings

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .base import (
    DetectorBackbone,
    _as_2d,
    apply_score_normalization,
    make_forecast_windows,
    map_forecast_scores_to_points,
    score_normalization_mode,
)


class _CNNForecastModel(nn.Module):
    def __init__(self, input_dim: int, hidden_channels: int, kernel_size: int, horizon: int) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_dim, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(hidden_channels, input_dim * horizon)
        self.input_dim = input_dim
        self.horizon = horizon

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, W, D]
        features = self.net(x.transpose(1, 2)).squeeze(-1)
        output = self.head(features)
        return output.reshape(x.shape[0], self.horizon, self.input_dim)


@dataclass
class CNNBackbone(DetectorBackbone):
    window_size: int = 32
    stride: int = 1
    horizon: int = 1
    hidden_channels: int = 32
    kernel_size: int = 3
    random_state: int = 42
    epochs: int = 10
    batch_size: int = 128
    learning_rate: float = 1e-3
    threshold_mode: str = "synthetic_separation"
    quantile: float = 0.95
    train_with_synthetic: bool = False
    max_train_windows: int | None = 20000
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.name = "cnn"
        self.supports_training_augmentation = bool(self.train_with_synthetic)
        self.is_supervised = False
        self._device = torch.device(self.device)
        self.model: _CNNForecastModel | None = None

    def _collect_training_arrays(
        self,
        train_series: list[Any],
        synthetic_windows: list[Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        windows_list: list[np.ndarray] = []
        targets_list: list[np.ndarray] = []
        source_flags_list: list[np.ndarray] = []
        synthetic_records = list(synthetic_windows or [])
        stats = {
            "train_records_seen": len(train_series),
            "synthetic_records_seen": len(synthetic_records) if self.train_with_synthetic else 0,
            "train_records_skipped_no_windows": 0,
            "synthetic_records_skipped_no_windows": 0,
            "train_windows_built": 0,
            "synthetic_windows_built": 0,
            "train_windows_used": 0,
            "synthetic_windows_used": 0,
            "total_windows_used": 0,
        }

        def _append_records(records: list[Any], *, is_synthetic: bool) -> None:
            for record in records:
                windows, targets, _ = make_forecast_windows(
                    record.values,
                    self.window_size,
                    self.stride,
                    horizon=self.horizon,
                )
                if len(windows) == 0:
                    if is_synthetic:
                        stats["synthetic_records_skipped_no_windows"] += 1
                    else:
                        stats["train_records_skipped_no_windows"] += 1
                    continue
                windows_list.append(windows)
                targets_list.append(targets)
                source_flags_list.append(np.full(len(windows), 1 if is_synthetic else 0, dtype=int))
                if is_synthetic:
                    stats["synthetic_windows_built"] += int(len(windows))
                else:
                    stats["train_windows_built"] += int(len(windows))

        _append_records(list(train_series), is_synthetic=False)
        if self.train_with_synthetic and synthetic_records:
            _append_records(synthetic_records, is_synthetic=True)

        if stats["train_records_skipped_no_windows"] > 0:
            warnings.warn(
                (
                    "CNN backbone skipped "
                    f"{stats['train_records_skipped_no_windows']} train records "
                    "because they were too short to form forecasting targets."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        if stats["synthetic_records_skipped_no_windows"] > 0:
            warnings.warn(
                (
                    "CNN backbone skipped "
                    f"{stats['synthetic_records_skipped_no_windows']} synthetic records "
                    "because they were too short to form forecasting targets."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
        if not windows_list:
            raise ValueError("CNN backbone could not build any forecasting windows from the training data.")
        windows = np.concatenate(windows_list, axis=0)
        targets = np.concatenate(targets_list, axis=0)
        source_flags = np.concatenate(source_flags_list, axis=0)
        if self.max_train_windows is not None and len(windows) > self.max_train_windows:
            rng = np.random.default_rng(self.random_state)
            train_idx = np.flatnonzero(source_flags == 0)
            synth_idx = np.flatnonzero(source_flags == 1)
            if len(train_idx) > 0 and len(synth_idx) > 0:
                target_total = int(self.max_train_windows)
                synth_quota = int(round(target_total * len(synth_idx) / len(source_flags)))
                synth_quota = max(1, min(synth_quota, len(synth_idx)))
                train_quota = max(target_total - synth_quota, 0)
                if train_quota == 0 and len(train_idx) > 0 and target_total > 1:
                    train_quota = 1
                    synth_quota = min(target_total - train_quota, len(synth_idx))
                train_quota = min(train_quota, len(train_idx))
                synth_quota = min(target_total - train_quota, len(synth_idx))
                remainder = target_total - (train_quota + synth_quota)
                if remainder > 0:
                    extra_train = min(remainder, len(train_idx) - train_quota)
                    train_quota += extra_train
                    remainder -= extra_train
                if remainder > 0:
                    extra_synth = min(remainder, len(synth_idx) - synth_quota)
                    synth_quota += extra_synth
                keep = np.concatenate(
                    [
                        rng.choice(train_idx, size=train_quota, replace=False),
                        rng.choice(synth_idx, size=synth_quota, replace=False),
                    ],
                    axis=0,
                )
            else:
                keep = rng.choice(len(windows), size=self.max_train_windows, replace=False)
            windows = windows[keep]
            targets = targets[keep]
            source_flags = source_flags[keep]
        stats["train_windows_used"] = int(np.sum(source_flags == 0))
        stats["synthetic_windows_used"] = int(np.sum(source_flags == 1))
        stats["total_windows_used"] = int(len(windows))
        self.training_window_summary_ = stats
        return windows, targets

    def fit(
        self,
        train_series: list[Any],
        train_labels: list[np.ndarray] | None = None,
        synthetic_windows: list[Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> "CNNBackbone":
        del train_labels, config
        torch.manual_seed(self.random_state)
        windows, targets = self._collect_training_arrays(train_series, synthetic_windows)
        input_dim = int(windows.shape[2])
        self.model = _CNNForecastModel(
            input_dim=input_dim,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            horizon=self.horizon,
        ).to(self._device)

        dataset = TensorDataset(
            torch.tensor(windows, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn = nn.MSELoss()

        self.model.train()
        for _ in range(max(int(self.epochs), 1)):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)
                optimizer.zero_grad()
                prediction = self.model(batch_x)
                loss = loss_fn(prediction, batch_y)
                loss.backward()
                optimizer.step()
        return self

    def score(self, test_series: list[Any], config: dict[str, Any] | None = None) -> dict[str, np.ndarray]:
        if self.model is None:
            raise RuntimeError("CNN backbone must be fitted before scoring.")
        normalization = score_normalization_mode(config)
        outputs: dict[str, np.ndarray] = {}
        self.model.eval()
        with torch.no_grad():
            for record in test_series:
                values = _as_2d(record.values)
                windows, targets, indices = make_forecast_windows(
                    values,
                    self.window_size,
                    self.stride,
                    horizon=self.horizon,
                )
                if len(windows) == 0:
                    outputs[str(record.series_id)] = np.zeros(len(values), dtype=float)
                    continue
                batch_x = torch.tensor(windows, dtype=torch.float32, device=self._device)
                prediction = self.model(batch_x).cpu().numpy()
                error = np.mean((prediction - targets) ** 2, axis=(1, 2))
                point_scores = map_forecast_scores_to_points(
                    len(values),
                    indices,
                    apply_score_normalization(error, normalization),
                    horizon=self.horizon,
                )
                outputs[str(record.series_id)] = point_scores
        return outputs


CNNDetector = CNNBackbone
