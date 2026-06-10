from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.synthetic_data import make_synthetic_records
from src.detectors.base import apply_score_normalization, make_windows, map_window_scores_to_points, window_features
from src.detectors.autoencoder import AutoEncoderBackbone
from src.detectors.iforest import IForestDetector
from src.detectors.lof import LOFDetector
from src.detectors.ocsvm import OCSVMDetector
from src.detectors.cnn import CNNBackbone
from src.detectors.timesnet import TimesNetBackbone
from src.detectors.internal_classifier import InternalClassifierBackbone
from src.datasets.tsb_loader import TimeSeriesRecord
from src.utils.seeds import set_global_seed


def _toy_records():
    records = make_synthetic_records(seed=17, num_series=6, events_per_series=3)
    return records[:4], records[4:]


def _synthetic_windows():
    train, _ = _toy_records()
    return train[:1]


def test_iforest_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = IForestDetector(window_size=12, stride=4, contamination=0.1, random_state=17)
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_ocsvm_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = OCSVMDetector(window_size=12, stride=4, nu=0.1)
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_lof_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = LOFDetector(window_size=12, stride=4, contamination=0.1, n_neighbors=10)
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_synthetic_separation_uses_synthetic_scores() -> None:
    detector = IForestDetector(window_size=8, stride=2, contamination=0.1)
    scores = np.asarray([0.05, 0.1, 0.15, 0.2], dtype=float)
    synthetic_scores = np.asarray([0.7, 0.8, 0.9], dtype=float)
    calibration = detector.calibrate_threshold(
        scores,
        labels=np.zeros_like(scores, dtype=int),
        synthetic_scores=synthetic_scores,
        config={"threshold_mode": "synthetic_separation", "quantile": 0.95},
    )
    assert calibration.method == "synthetic_positive_separation"
    assert calibration.threshold >= float(np.max(scores))
    assert calibration.threshold < float(np.max(synthetic_scores))


def test_predict_marks_higher_scores_positive() -> None:
    detector = LOFDetector(window_size=8, stride=2)
    predictions = detector.predict(np.asarray([0.1, 0.4, 0.8]), threshold=0.5)
    assert np.array_equal(predictions, np.asarray([0, 0, 1]))


def test_iforest_default_scores_are_raw_not_per_record_minmax() -> None:
    train, test = _toy_records()
    record = test[0]
    detector = IForestDetector(window_size=12, stride=4, contamination=0.1, random_state=17)
    detector.fit(train)
    windows, spans = make_windows(record.values, detector.window_size, detector.stride)
    features = window_features(windows)
    raw_window_scores = -detector.model.score_samples(features)
    expected = map_window_scores_to_points(
        len(np.asarray(record.values)),
        spans,
        apply_score_normalization(raw_window_scores, "none"),
        reduction=detector.score_reduction,
    )
    actual = detector.score([record])[record.series_id]
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-8)
    per_record = apply_score_normalization(raw_window_scores, "per_record")
    if raw_window_scores.size > 1 and float(np.max(raw_window_scores) - np.min(raw_window_scores)) > 1e-8:
        assert float(np.max(per_record)) > float(np.max(raw_window_scores)) or float(np.min(per_record)) < float(
            np.min(raw_window_scores)
        )


def test_no_crash_when_synthetic_windows_empty() -> None:
    train, test = _toy_records()
    detector = OCSVMDetector(window_size=10, stride=5)
    detector.fit(train, synthetic_windows=[])
    scores = detector.score(test)
    assert scores


def test_internal_classifier_embeds_synthetic_windows_into_full_context() -> None:
    train, _ = _toy_records()
    synthetic_window = TimeSeriesRecord(
        series_id="train0__synthetic__event",
        values=np.asarray([10.0, 11.0, 12.0], dtype=float),
        labels=np.asarray([1, 1, 1], dtype=int),
        timestamps=None,
        source_path="synthetic",
        metadata={
            "synthetic": True,
            "target_series": train[0].series_id,
            "target_event_interval": [2, 5],
            "target_series_values": np.asarray(train[0].values, dtype=float).tolist(),
            "target_series_labels": np.asarray(train[0].labels, dtype=int).tolist(),
        },
    )

    class RecordingSeriesModel:
        def __init__(self):
            self.records = None
            self.labels = None

        def fit(self, records, labels):
            self.records = records
            self.labels = labels
            return self

        def score(self, records):
            return {record.series_id: np.zeros(len(record.values), dtype=float) for record in records}

    backbone = InternalClassifierBackbone(detector_config={"embed_synthetic_context": True})
    recorder = RecordingSeriesModel()
    backbone.model = recorder
    train_labels = [np.asarray(record.labels, dtype=int) for record in train]
    backbone.fit(train, train_labels=train_labels, synthetic_windows=[synthetic_window])

    embedded = recorder.records[-1]
    assert len(embedded.values) == len(train[0].values)
    np.testing.assert_allclose(np.asarray(embedded.values)[2:5], np.asarray([10.0, 11.0, 12.0]))
    np.testing.assert_array_equal(np.asarray(embedded.labels, dtype=int)[2:5], np.asarray([1, 1, 1]))
    assert bool(embedded.metadata.get("synthetic_context_embedded", False))


def test_autoencoder_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = AutoEncoderBackbone(window_size=12, num_lags=2, max_iter=50, train_with_synthetic=True)
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_autoencoder_uses_synthetic_windows_when_enabled() -> None:
    train, _ = _toy_records()
    synthetic = _synthetic_windows()
    detector = AutoEncoderBackbone(window_size=8, num_lags=1, max_iter=10, train_with_synthetic=True)

    class RecordingModel:
        def __init__(self):
            self.fit_shapes = []

        def fit(self, x, y):
            self.fit_shapes.append((x.shape, y.shape))
            return self

        def predict(self, x):
            return np.asarray(x, dtype=float)

    recorder = RecordingModel()
    detector.model = recorder
    detector.fit(train, synthetic_windows=synthetic)
    used_points = recorder.fit_shapes[0][0][0]
    baseline_points = sum(len(record.values) for record in train)
    augmented_points = baseline_points + sum(len(record.values) for record in synthetic)
    assert used_points == augmented_points


def test_autoencoder_can_disable_training_augmentation() -> None:
    train, _ = _toy_records()
    synthetic = _synthetic_windows()
    detector = AutoEncoderBackbone(window_size=8, num_lags=1, max_iter=10, train_with_synthetic=False)

    class RecordingModel:
        def __init__(self):
            self.fit_shapes = []

        def fit(self, x, y):
            self.fit_shapes.append((x.shape, y.shape))
            return self

        def predict(self, x):
            return np.asarray(x, dtype=float)

    recorder = RecordingModel()
    detector.model = recorder
    detector.fit(train, synthetic_windows=synthetic)
    used_points = recorder.fit_shapes[0][0][0]
    baseline_points = sum(len(record.values) for record in train)
    assert used_points == baseline_points


def test_autoencoder_default_skips_training_augmentation() -> None:
    train, _ = _toy_records()
    synthetic = _synthetic_windows()
    detector = AutoEncoderBackbone(window_size=8, num_lags=1, max_iter=10)

    class RecordingModel:
        def __init__(self):
            self.fit_shapes = []

        def fit(self, x, y):
            self.fit_shapes.append((x.shape, y.shape))
            return self

        def predict(self, x):
            return np.asarray(x, dtype=float)

    recorder = RecordingModel()
    detector.model = recorder
    detector.fit(train, synthetic_windows=synthetic)
    used_points = recorder.fit_shapes[0][0][0]
    baseline_points = sum(len(record.values) for record in train)
    assert used_points == baseline_points
    assert detector.supports_training_augmentation is False


def test_cnn_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = CNNBackbone(window_size=12, stride=2, epochs=1, batch_size=16, max_train_windows=128)
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_timesnet_fit_and_score_length_matches_points() -> None:
    train, test = _toy_records()
    detector = TimesNetBackbone(
        window_size=12,
        stride=2,
        epochs=1,
        batch_size=16,
        d_model=16,
        num_blocks=1,
        max_train_windows=128,
    )
    detector.fit(train)
    scores = detector.score(test)
    for record in test:
        assert len(scores[record.series_id]) == len(record.values)


def test_cnn_training_window_summary_tracks_post_subsample_usage() -> None:
    train, _ = _toy_records()
    synthetic = train[:2]
    detector = CNNBackbone(
        window_size=12,
        stride=2,
        horizon=1,
        epochs=1,
        batch_size=16,
        train_with_synthetic=True,
        max_train_windows=16,
    )
    detector.fit(train, synthetic_windows=synthetic)
    summary = detector.training_window_summary_
    assert summary["train_windows_built"] >= summary["train_windows_used"]
    assert summary["synthetic_windows_built"] >= summary["synthetic_windows_used"]
    assert summary["train_windows_used"] + summary["synthetic_windows_used"] == summary["total_windows_used"]
    assert summary["synthetic_windows_used"] > 0


def test_timesnet_training_window_summary_tracks_post_subsample_usage() -> None:
    train, _ = _toy_records()
    synthetic = train[:2]
    detector = TimesNetBackbone(
        window_size=12,
        stride=2,
        horizon=1,
        epochs=1,
        batch_size=16,
        d_model=16,
        num_blocks=1,
        train_with_synthetic=True,
        max_train_windows=16,
    )
    detector.fit(train, synthetic_windows=synthetic)
    summary = detector.training_window_summary_
    assert summary["train_windows_built"] >= summary["train_windows_used"]
    assert summary["synthetic_windows_built"] >= summary["synthetic_windows_used"]
    assert summary["train_windows_used"] + summary["synthetic_windows_used"] == summary["total_windows_used"]
    assert summary["synthetic_windows_used"] > 0


def test_cnn_warns_and_counts_dropped_short_synthetic_records() -> None:
    train, _ = _toy_records()
    short_synthetic = [
        TimeSeriesRecord(
            series_id="short_synth",
            values=np.asarray([1.0], dtype=float),
            labels=np.asarray([1], dtype=int),
            timestamps=None,
            source_path="synthetic",
            metadata={},
        )
    ]
    detector = CNNBackbone(window_size=12, stride=2, horizon=1, epochs=1, batch_size=16, train_with_synthetic=True)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.fit(train, synthetic_windows=short_synthetic)
    assert any("skipped 1 synthetic records" in str(item.message).lower() for item in caught)
    assert detector.training_window_summary_["synthetic_records_skipped_no_windows"] == 1
    assert detector.training_window_summary_["synthetic_windows_built"] == 0


def test_timesnet_warns_and_counts_dropped_short_synthetic_records() -> None:
    train, _ = _toy_records()
    short_synthetic = [
        TimeSeriesRecord(
            series_id="short_synth",
            values=np.asarray([1.0], dtype=float),
            labels=np.asarray([1], dtype=int),
            timestamps=None,
            source_path="synthetic",
            metadata={},
        )
    ]
    detector = TimesNetBackbone(
        window_size=12,
        stride=2,
        horizon=1,
        epochs=1,
        batch_size=16,
        d_model=16,
        num_blocks=1,
        train_with_synthetic=True,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        detector.fit(train, synthetic_windows=short_synthetic)
    assert any("skipped 1 synthetic records" in str(item.message).lower() for item in caught)
    assert detector.training_window_summary_["synthetic_records_skipped_no_windows"] == 1
    assert detector.training_window_summary_["synthetic_windows_built"] == 0


def test_set_global_seed_sets_torch_seed_when_available() -> None:
    try:
        import torch
    except ImportError:
        return
    set_global_seed(123)
    first = torch.rand(4)
    set_global_seed(123)
    second = torch.rand(4)
    assert torch.allclose(first, second)


def test_cnn_warns_when_train_records_are_too_short() -> None:
    short_train = [
        TimeSeriesRecord(
            series_id="short_train",
            values=np.asarray([1.0], dtype=float),
            labels=np.asarray([0], dtype=int),
            timestamps=None,
            source_path="synthetic",
            metadata={},
        )
    ]
    detector = CNNBackbone(window_size=12, stride=2, horizon=1, epochs=1, batch_size=16)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            detector.fit(short_train)
        except ValueError:
            pass
    assert any("skipped 1 train records" in str(item.message).lower() for item in caught)


def test_timesnet_warns_when_train_records_are_too_short() -> None:
    short_train = [
        TimeSeriesRecord(
            series_id="short_train",
            values=np.asarray([1.0], dtype=float),
            labels=np.asarray([0], dtype=int),
            timestamps=None,
            source_path="synthetic",
            metadata={},
        )
    ]
    detector = TimesNetBackbone(
        window_size=12,
        stride=2,
        horizon=1,
        epochs=1,
        batch_size=16,
        d_model=16,
        num_blocks=1,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            detector.fit(short_train)
        except ValueError:
            pass
    assert any("skipped 1 train records" in str(item.message).lower() for item in caught)


def test_synthesis_metadata_enables_classifier_context_embedding() -> None:
    from src.augmentation.policies import _records_from_synthetic_rows
    from src.detectors.internal_classifier import _expand_synthetic_with_context
    from src.synthesis.augment_dataset import build_augmented_training_records_with_audit

    def make(sid: str, seed: int) -> TimeSeriesRecord:
        rng = np.random.default_rng(seed)
        values = rng.normal(size=100)
        labels = np.zeros(100, dtype=int)
        labels[30:40] = 1
        values[30:40] += 5
        return TimeSeriesRecord(series_id=sid, values=values, labels=labels, timestamps=None, source_path="t")

    train = [make("dsA/a", 0), make("dsB/b", 1)]
    policy = {
        "method": "normalized_time_mean_donor",
        "method_name": "test",
        "donor_policy": "cross_dataset_all",
        "labeled_fraction": 0.2,
        "top_k": 2,
    }
    kept, _ = build_augmented_training_records_with_audit(
        train,
        "train",
        policy,
        donor_pool_records=train,
        synthesis_cfg={"dataset_name": "dsA"},
    )
    assert kept
    record = _records_from_synthetic_rows(kept)[0]
    expanded = _expand_synthetic_with_context(record)
    assert expanded.metadata.get("synthetic_context_embedded") is True
    assert len(expanded.values) == len(record.metadata["target_series_values"])
