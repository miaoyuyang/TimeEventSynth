"""Timeline-level compatibility scoring for cross-timeline donor selection."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..datasets.event_extractor import labels_to_events

DEFAULT_WEIGHTS: dict[str, float] = {
    "shape_similarity": 0.25,
    "amplitude_compatibility": 0.25,
    "duration_compatibility": 0.15,
    "context_similarity": 0.15,
    "frequency_similarity": 0.10,
    "trend_similarity": 0.10,
    "group_compatibility": 0.0,
}

DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "min_score": 0.5,
    "top_k": None,
    "top_quantile": None,
    "weights": DEFAULT_WEIGHTS,
    "same_dataset_bonus": 0.05,
    "allow_cross_dataset": True,
    "learned_weights": {
        "enabled": False,
        "component_keys": [
            "shape_similarity",
            "amplitude_compatibility",
            "context_similarity",
            "group_compatibility",
        ],
        "epochs": 250,
        "learning_rate": 0.2,
        "temperature": 0.15,
        "l2_to_prior": 0.02,
        "max_pairs": 5000,
        "min_positive_pairs": 4,
        "min_negative_pairs": 4,
        "positive_group_overlap": 0.2,
        "seed": 0,
    },
}

_NEUTRAL_ANOMALY: dict[str, float] = {
    "anomaly_ratio": 0.0,
    "anomaly_segment_count": 0.0,
    "median_anomaly_duration": 0.0,
    "mean_anomaly_duration": 0.0,
}


def _resolve_config(config: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(DEFAULT_CONFIG)
    if config:
        merged.update(config)
        if "weights" in config:
            weights = dict(DEFAULT_WEIGHTS)
            weights.update(config["weights"])
            merged["weights"] = weights
        if "learned_weights" in config:
            learned = dict(DEFAULT_CONFIG["learned_weights"])
            learned.update(config["learned_weights"] or {})
            merged["learned_weights"] = learned
    return merged


def _clean_series(series: np.ndarray) -> np.ndarray:
    array = np.asarray(series, dtype=float).reshape(-1)
    if array.size == 0:
        return array
    finite = np.isfinite(array)
    if not np.any(finite):
        return np.zeros(0, dtype=float)
    if not np.all(finite):
        median = float(np.nanmedian(array))
        array = np.where(finite, array, median)
    return array


def _robust_amplitude(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    q75, q25 = np.percentile(values, [75, 25])
    return float(max(q75 - q25, np.std(values), 1e-6))


def _trend_strength(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)
    x = (x - x.mean()) / max(x.std(), 1e-6)
    y = (values - values.mean()) / max(values.std(), 1e-6)
    slope = float(np.dot(x, y) / max(values.size - 1, 1))
    return float(np.clip(abs(slope), 0.0, 1.0))


def _autocorr_lag1(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    a = values[:-1] - values[:-1].mean()
    b = values[1:] - values[1:].mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 0.0
    return float(np.clip(np.dot(a, b) / denom, -1.0, 1.0))


def _spectral_energy_summary(values: np.ndarray) -> float:
    """Low-frequency energy share as a simple spectral proxy."""
    if values.size < 4:
        return 0.0
    centered = values - np.mean(values)
    spectrum = np.abs(np.fft.rfft(centered))
    if spectrum.size <= 1:
        return 0.0
    total = float(np.sum(spectrum**2))
    if total <= 1e-12:
        return 0.0
    low_band = spectrum[: max(1, spectrum.size // 4)]
    return float(np.clip(np.sum(low_band**2) / total, 0.0, 1.0))


def _local_context_summary(values: np.ndarray, events: list[tuple[int, int]], *, context_size: int = 5) -> dict[str, float]:
    if not events:
        return {
            "pre_mean": float(np.mean(values)) if values.size else 0.0,
            "pre_std": float(np.std(values)) if values.size else 0.0,
            "post_mean": float(np.mean(values)) if values.size else 0.0,
            "post_std": float(np.std(values)) if values.size else 0.0,
        }
    pre_means: list[float] = []
    pre_stds: list[float] = []
    post_means: list[float] = []
    post_stds: list[float] = []
    for start, end in events:
        pre = values[max(0, start - context_size) : start]
        post = values[end : min(len(values), end + context_size)]
        if pre.size:
            pre_means.append(float(np.mean(pre)))
            pre_stds.append(float(np.std(pre)))
        if post.size:
            post_means.append(float(np.mean(post)))
            post_stds.append(float(np.std(post)))
    return {
        "pre_mean": float(np.mean(pre_means)) if pre_means else float(np.mean(values)),
        "pre_std": float(np.mean(pre_stds)) if pre_stds else float(np.std(values)),
        "post_mean": float(np.mean(post_means)) if post_means else float(np.mean(values)),
        "post_std": float(np.mean(post_stds)) if post_stds else float(np.std(values)),
    }


def compute_timeline_features(
    series: np.ndarray | list[float],
    labels: np.ndarray | list[int] | None = None,
    timestamps: np.ndarray | list[float] | None = None,
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute robust timeline-level features for compatibility scoring."""
    cfg = _resolve_config(config)
    context_size = int(cfg.get("context_size", 5))
    values = _clean_series(np.asarray(series, dtype=float))
    length = int(values.size)

    features: dict[str, Any] = {
        "length": float(length),
        "mean": float(np.mean(values)) if length else 0.0,
        "std": float(np.std(values)) if length else 0.0,
        "median": float(np.median(values)) if length else 0.0,
        "iqr": float(np.percentile(values, 75) - np.percentile(values, 25)) if length else 0.0,
        "min": float(np.min(values)) if length else 0.0,
        "max": float(np.max(values)) if length else 0.0,
        "robust_amplitude": _robust_amplitude(values),
        "trend_strength": _trend_strength(values),
        "autocorr_lag1": _autocorr_lag1(values),
        "dominant_frequency_proxy": _spectral_energy_summary(values),
        "spectral_energy_summary": _spectral_energy_summary(values),
    }

    if cfg.get("dataset") is not None:
        features["dataset"] = str(cfg["dataset"])
        features["dataset_name"] = str(cfg.get("dataset_name", cfg["dataset"]))
    elif cfg.get("dataset_name") is not None:
        features["dataset_name"] = str(cfg["dataset_name"])
        features["dataset"] = str(cfg["dataset_name"])
    if cfg.get("series_id") is not None:
        features["series_id"] = str(cfg["series_id"])

    if timestamps is not None:
        ts = _clean_series(np.asarray(timestamps, dtype=float))
        if ts.size == values.size and ts.size >= 2:
            features["timestamp_span"] = float(ts[-1] - ts[0])
            features["median_sampling_interval"] = float(np.median(np.diff(ts)))
        else:
            features["timestamp_span"] = float(length)
            features["median_sampling_interval"] = 1.0
    else:
        features["timestamp_span"] = float(length)
        features["median_sampling_interval"] = 1.0

    if labels is None:
        features.update(dict(_NEUTRAL_ANOMALY))
        features["local_context_summary"] = _local_context_summary(values, [], context_size=context_size)
        return features

    label_array = np.asarray(labels, dtype=int).reshape(-1)
    if label_array.size != length:
        raise ValueError("labels must align with series length")
    events = labels_to_events(label_array, min_length=1, merge_gap=0)
    durations = [float(end - start) for start, end in events]
    anomaly_points = int(np.sum(label_array > 0))

    features["anomaly_ratio"] = float(anomaly_points / length) if length else 0.0
    features["anomaly_segment_count"] = float(len(events))
    features["median_anomaly_duration"] = float(np.median(durations)) if durations else 0.0
    features["mean_anomaly_duration"] = float(np.mean(durations)) if durations else 0.0
    features["local_context_summary"] = _local_context_summary(values, events, context_size=context_size)
    return features


def _scale_pair(a: float, b: float) -> float:
    return max(abs(a), abs(b), 1e-6)


def _similarity_from_distance(distance: float) -> float:
    return float(np.clip(1.0 / (1.0 + max(distance, 0.0)), 0.0, 1.0))


def _scalar_similarity(a: float, b: float) -> float:
    return _similarity_from_distance(abs(a - b) / _scale_pair(a, b))


def _dict_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    keys = sorted(set(a.keys()) & set(b.keys()))
    if not keys:
        return 1.0
    scores = [_scalar_similarity(float(a[key]), float(b[key])) for key in keys]
    return float(np.mean(scores))


def _shape_similarity(source: dict[str, Any], target: dict[str, Any]) -> float:
    keys = ("mean", "std", "median", "iqr")
    scores = [_scalar_similarity(float(source[key]), float(target[key])) for key in keys]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _amplitude_compatibility(source: dict[str, Any], target: dict[str, Any]) -> float:
    keys = ("robust_amplitude", "min", "max", "std")
    scores = [_scalar_similarity(float(source[key]), float(target[key])) for key in keys]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _duration_compatibility(source: dict[str, Any], target: dict[str, Any]) -> float:
    scores = [_scalar_similarity(float(source["length"]), float(target["length"]))]
    if float(source.get("anomaly_segment_count", 0)) > 0 or float(target.get("anomaly_segment_count", 0)) > 0:
        for key in ("median_anomaly_duration", "mean_anomaly_duration", "anomaly_ratio"):
            scores.append(_scalar_similarity(float(source[key]), float(target[key])))
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _context_similarity(source: dict[str, Any], target: dict[str, Any]) -> float:
    source_ctx = source.get("local_context_summary", {})
    target_ctx = target.get("local_context_summary", {})
    if not isinstance(source_ctx, dict) or not isinstance(target_ctx, dict):
        return 1.0
    return float(np.clip(_dict_similarity(source_ctx, target_ctx), 0.0, 1.0))


def _frequency_similarity(source: dict[str, Any], target: dict[str, Any]) -> float:
    keys = ("dominant_frequency_proxy", "spectral_energy_summary", "autocorr_lag1")
    scores = [_scalar_similarity(float(source[key]), float(target[key])) for key in keys]
    return float(np.clip(np.mean(scores), 0.0, 1.0))


def _trend_similarity(source: dict[str, Any], target: dict[str, Any]) -> float:
    return _scalar_similarity(float(source["trend_strength"]), float(target["trend_strength"]))


def _group_compatibility(source: dict[str, Any], target: dict[str, Any]) -> float:
    """Overlap between train-visible event-group distributions.

    Missing group information is neutral so the component only affects runs that
    explicitly attach event groups before donor filtering.
    """
    source_dist = source.get("event_group_distribution")
    target_dist = target.get("event_group_distribution")
    if not isinstance(source_dist, dict) or not isinstance(target_dist, dict):
        return 0.5
    keys = set(source_dist) | set(target_dist)
    if not keys:
        return 0.5
    overlap = sum(min(float(source_dist.get(key, 0.0)), float(target_dist.get(key, 0.0))) for key in keys)
    return float(np.clip(overlap, 0.0, 1.0))


def _compatibility_components(source_features: dict[str, Any], target_features: dict[str, Any]) -> dict[str, float]:
    return {
        "shape_similarity": _shape_similarity(source_features, target_features),
        "amplitude_compatibility": _amplitude_compatibility(source_features, target_features),
        "duration_compatibility": _duration_compatibility(source_features, target_features),
        "context_similarity": _context_similarity(source_features, target_features),
        "frequency_similarity": _frequency_similarity(source_features, target_features),
        "trend_similarity": _trend_similarity(source_features, target_features),
        "group_compatibility": _group_compatibility(source_features, target_features),
    }


def attach_event_group_distributions(
    feature_table: dict[str, dict[str, Any]],
    windows: list[Any],
    *,
    group_key: str = "event_group_id",
) -> None:
    """Attach per-series event-group distributions computed from train-visible windows."""
    counts_by_series: dict[str, dict[str, int]] = {}
    for window in windows:
        series_id = str(getattr(window, "series_id", ""))
        metadata = getattr(window, "metadata", {}) or {}
        group = str(metadata.get(group_key, "") or "")
        if not series_id or not group:
            continue
        counts_by_series.setdefault(series_id, {})
        counts_by_series[series_id][group] = counts_by_series[series_id].get(group, 0) + 1

    for series_id, counts in counts_by_series.items():
        total = float(sum(counts.values()))
        if total <= 0 or series_id not in feature_table:
            continue
        feature_table[series_id]["event_group_distribution"] = {
            group: float(count / total) for group, count in sorted(counts.items())
        }


def _softmax(theta: np.ndarray) -> np.ndarray:
    shifted = theta - float(np.max(theta))
    values = np.exp(shifted)
    return values / max(float(np.sum(values)), 1e-12)


def learn_compatibility_weights(
    feature_table: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Learn convex compatibility weights from train-visible event-group pseudo-labels.

    Positives are series pairs with overlapping event-group distributions; negatives
    are non-overlapping pairs. This keeps learning within the training split and
    avoids using held-out labels.
    """
    cfg = _resolve_config(config)
    learn_cfg = dict(cfg.get("learned_weights", {}) or {})
    if not bool(learn_cfg.get("enabled", False)):
        return {"enabled": False, "weights": dict(cfg["weights"]), "reason": "disabled"}

    component_keys = [str(key) for key in learn_cfg.get("component_keys", []) if str(key) in DEFAULT_WEIGHTS]
    if not component_keys:
        return {"enabled": False, "weights": dict(cfg["weights"]), "reason": "no_component_keys"}

    series_ids = sorted(
        series_id
        for series_id, features in feature_table.items()
        if isinstance(features.get("event_group_distribution"), dict)
    )
    if len(series_ids) < 2:
        return {"enabled": False, "weights": dict(cfg["weights"]), "reason": "insufficient_grouped_series"}

    positive_threshold = float(learn_cfg.get("positive_group_overlap", 0.2))
    rows: list[list[float]] = []
    labels: list[float] = []
    for target_id in series_ids:
        for source_id in series_ids:
            if source_id == target_id:
                continue
            components = _compatibility_components(feature_table[source_id], feature_table[target_id])
            group_overlap = float(components.get("group_compatibility", 0.0))
            rows.append([float(components[key]) for key in component_keys])
            labels.append(1.0 if group_overlap >= positive_threshold else 0.0)

    if not rows:
        return {"enabled": False, "weights": dict(cfg["weights"]), "reason": "no_training_pairs"}

    x = np.asarray(rows, dtype=float)
    y = np.asarray(labels, dtype=float)
    pos_count = int(np.sum(y > 0.5))
    neg_count = int(np.sum(y <= 0.5))
    if pos_count < int(learn_cfg.get("min_positive_pairs", 4)) or neg_count < int(learn_cfg.get("min_negative_pairs", 4)):
        return {
            "enabled": False,
            "weights": dict(cfg["weights"]),
            "reason": "insufficient_positive_or_negative_pairs",
            "positive_pairs": pos_count,
            "negative_pairs": neg_count,
        }

    max_pairs = int(learn_cfg.get("max_pairs", 5000))
    if x.shape[0] > max_pairs:
        rng = np.random.default_rng(int(learn_cfg.get("seed", 0)))
        indices = rng.choice(x.shape[0], size=max_pairs, replace=False)
        x = x[indices]
        y = y[indices]
        pos_count = int(np.sum(y > 0.5))
        neg_count = int(np.sum(y <= 0.5))

    base_weights = dict(cfg["weights"])
    positive_prior_values = [
        float(base_weights.get(key, 0.0)) for key in component_keys if float(base_weights.get(key, 0.0)) > 0
    ]
    default_prior_value = float(np.mean(positive_prior_values)) if positive_prior_values else 1.0
    prior_raw = np.asarray(
        [
            float(base_weights.get(key, 0.0)) if float(base_weights.get(key, 0.0)) > 0 else default_prior_value
            for key in component_keys
        ],
        dtype=float,
    )
    prior = prior_raw / max(float(np.sum(prior_raw)), 1e-12)
    theta = np.log(prior)

    lr = float(learn_cfg.get("learning_rate", 0.2))
    epochs = int(learn_cfg.get("epochs", 250))
    temperature = max(float(learn_cfg.get("temperature", 0.15)), 1e-3)
    l2_to_prior = float(learn_cfg.get("l2_to_prior", 0.02))
    class_weight = np.where(y > 0.5, 0.5 / max(float(np.mean(y)), 1e-6), 0.5 / max(float(1.0 - np.mean(y)), 1e-6))

    for _ in range(max(epochs, 0)):
        alpha = _softmax(theta)
        score = x @ alpha
        logits = (score - 0.5) / temperature
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        dloss_dscore = class_weight * (prob - y) / temperature / max(float(x.shape[0]), 1.0)
        grad_alpha = x.T @ dloss_dscore + l2_to_prior * (alpha - prior)
        grad_theta = alpha * (grad_alpha - float(np.dot(grad_alpha, alpha)))
        theta -= lr * grad_theta

    learned_alpha = _softmax(theta)
    learned_component_weights = {key: float(value) for key, value in zip(component_keys, learned_alpha)}
    new_weights = dict(base_weights)
    learned_mass = float(sum(max(float(base_weights.get(key, 0.0)), 0.0) for key in component_keys))
    if learned_mass <= 0:
        learned_mass = 1.0
    for key in component_keys:
        new_weights[key] = float(learned_component_weights[key] * learned_mass)

    return {
        "enabled": True,
        "reason": "learned_from_train_visible_event_groups",
        "component_keys": component_keys,
        "weights": new_weights,
        "learned_alpha": learned_component_weights,
        "positive_pairs": pos_count,
        "negative_pairs": neg_count,
        "num_training_pairs": int(x.shape[0]),
    }


def compute_pairwise_compatibility(
    source_features: dict[str, Any],
    target_features: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score whether a source timeline is compatible with a target timeline."""
    cfg = _resolve_config(config)
    weights = cfg["weights"]
    min_score = float(cfg.get("min_score", 0.5))

    source_dataset = source_features.get("dataset")
    target_dataset = target_features.get("dataset")
    if not cfg.get("allow_cross_dataset", True) and source_dataset is not None and target_dataset is not None:
        if str(source_dataset) != str(target_dataset):
            return {
                "compatibility_score": 0.0,
                "shape_similarity": 0.0,
                "amplitude_compatibility": 0.0,
                "duration_compatibility": 0.0,
                "context_similarity": 0.0,
                "frequency_similarity": 0.0,
                "trend_similarity": 0.0,
                "group_compatibility": 0.0,
                "compatibility_weighting": str(cfg.get("weighting_method", "fixed")),
                "final_decision": "rejected",
                "rejection_reason": "cross_dataset_disallowed",
            }

    components = _compatibility_components(source_features, target_features)

    total_weight = float(sum(weights.get(key, 0.0) for key in components))
    if total_weight <= 0:
        compatibility_score = float(np.mean(list(components.values())))
    else:
        compatibility_score = float(
            sum(components[key] * weights.get(key, 0.0) for key in components) / total_weight
        )

    bonus = float(cfg.get("same_dataset_bonus", 0.0))
    if bonus > 0 and source_dataset is not None and target_dataset is not None and str(source_dataset) == str(target_dataset):
        compatibility_score = min(1.0, compatibility_score + bonus)

    compatibility_score = float(np.clip(compatibility_score, 0.0, 1.0))
    for key in components:
        components[key] = float(np.clip(components[key], 0.0, 1.0))

    if compatibility_score >= min_score:
        decision = "kept"
        reason = ""
    else:
        decision = "rejected"
        weakest = min(components, key=components.get)
        reason = f"below_min_score:{weakest}"

    return {
        "compatibility_score": compatibility_score,
        **components,
        "compatibility_weighting": str(cfg.get("weighting_method", "fixed")),
        "final_decision": decision,
        "rejection_reason": reason,
    }


def _donor_record(
    target_series_id: str,
    source_series_id: str,
    compatibility: dict[str, Any],
) -> dict[str, Any]:
    return {
        "target_series_id": target_series_id,
        "source_series_id": source_series_id,
        "compatibility_score": float(compatibility["compatibility_score"]),
        "shape_similarity": float(compatibility["shape_similarity"]),
        "amplitude_compatibility": float(compatibility["amplitude_compatibility"]),
        "duration_compatibility": float(compatibility["duration_compatibility"]),
        "context_similarity": float(compatibility["context_similarity"]),
        "frequency_similarity": float(compatibility["frequency_similarity"]),
        "trend_similarity": float(compatibility["trend_similarity"]),
        "group_compatibility": float(compatibility.get("group_compatibility", np.nan)),
        "compatibility_weighting": compatibility.get("compatibility_weighting", ""),
        "final_decision": str(compatibility["final_decision"]),
        "rejection_reason": str(compatibility.get("rejection_reason", "")),
    }


def rank_compatible_donors(
    target_series_id: str,
    candidate_donor_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Rank donor timelines by compatibility with a target timeline."""
    cfg = _resolve_config(config)
    target_features = feature_table[target_series_id]
    records: list[dict[str, Any]] = []
    for donor_id in candidate_donor_ids:
        if donor_id == target_series_id:
            continue
        if donor_id not in feature_table:
            continue
        compatibility = compute_pairwise_compatibility(feature_table[donor_id], target_features, cfg)
        records.append(_donor_record(target_series_id, donor_id, compatibility))
    records.sort(
        key=lambda row: (
            -row["compatibility_score"],
            row["source_series_id"],
        )
    )
    return records


def filter_compatible_donors(
    target_series_id: str,
    candidate_donor_ids: list[str],
    feature_table: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Filter donors by compatibility score and optional top-k / top-quantile policies."""
    cfg = _resolve_config(config)
    ranked = rank_compatible_donors(target_series_id, candidate_donor_ids, feature_table, cfg)
    min_score = float(cfg.get("min_score", 0.5))

    passing = [row for row in ranked if float(row["compatibility_score"]) >= min_score and row["final_decision"] == "kept"]
    if not passing:
        return [], ranked

    top_k = cfg.get("top_k")
    if top_k is not None:
        passing = passing[: int(top_k)]

    top_quantile = cfg.get("top_quantile")
    if top_quantile is not None:
        quantile = float(top_quantile)
        if not 0.0 < quantile <= 1.0:
            raise ValueError("top_quantile must be in (0, 1]")
        n_keep = max(1, int(np.ceil(len(passing) * quantile)))
        passing = passing[:n_keep]

    kept_ids = [str(row["source_series_id"]) for row in passing]
    kept_set = set(kept_ids)
    records = []
    for row in ranked:
        updated = dict(row)
        if row["source_series_id"] in kept_set:
            updated["final_decision"] = "kept"
            updated["rejection_reason"] = ""
        elif float(row["compatibility_score"]) >= min_score:
            updated["final_decision"] = "rejected"
            updated["rejection_reason"] = "below_top_quantile_0.5"
        records.append(updated)
    return kept_ids, records
