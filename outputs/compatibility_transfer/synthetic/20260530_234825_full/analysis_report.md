# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__compatibility_strict` (0.9808, policy=compatibility_strict)
- Best **point F1**: `learned_prototype_event_time__strict_filter` (0.9819, policy=cross_dataset_compatible)

## Compatibility-Filtered vs All-Donor Transfer
- Cross-dataset compatible transfer **beats** naive all-donor on event F1 (Δ=+0.0173).

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **helps** vs same-dataset-only (Δ=+0.0092).

## Donor Compatibility Summary
- Target timelines: 21
- Source timelines: 24
- Donor pairs kept / rejected: 451 / 4059
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.9396
- Mean compatibility (rejected): 0.8801

## Major Rejection Reasons
- `below_top_quantile_0.5`: 4059
- `strict_filter_failed:amplitude_compatibility_score`: 35

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__compatibility_strict`: event_f1=0.9808 (policy=compatibility_strict)
- `real_only`: event_f1=0.9714 (policy=None)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.9714 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__compatibility_top50`: event_f1=0.9714 (policy=compatibility_top50)
- `random_event_oversampling`: event_f1=0.9623 (policy=None)
- `learned_prototype_event_time__same_dataset_only`: event_f1=0.9623 (policy=same_dataset_only)
- `learned_prototype_event_time__cross_dataset_all`: event_f1=0.9623 (policy=cross_dataset_all)
- `learned_prototype_event_time__strict_filter`: event_f1=0.9623 (policy=cross_dataset_compatible)
- `dtw_aligned_donor__cross_dataset_compatible`: event_f1=0.9623 (policy=cross_dataset_compatible)
- `normalized_time_mean_donor__cross_dataset_compatible`: event_f1=0.9623 (policy=cross_dataset_compatible)

## Masked-Completion Ranking
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.5863, normalized_MAE=0.7341
- `learned_prototype_event_time__cross_dataset_all`: shape_correlation=0.4267, normalized_MAE=1.0423
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.4267, normalized_MAE=1.0423
- `learned_prototype_event_time__strict_filter`: shape_correlation=0.4267, normalized_MAE=1.0423
- `normalized_time_mean_donor__cross_dataset_compatible`: shape_correlation=0.4251, normalized_MAE=1.0464
- `learned_prototype_event_time__all_donors_no_filter`: shape_correlation=0.4063, normalized_MAE=1.0198
- `learned_prototype_event_time__compatibility_strict`: shape_correlation=0.4063, normalized_MAE=1.0198
- `learned_prototype_event_time__compatibility_top50`: shape_correlation=0.3735, normalized_MAE=1.0497
- `learned_prototype_event_time__same_dataset_only`: shape_correlation=0.3488, normalized_MAE=1.0015

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **-0.159**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 4059
- Strict-filter failures: 35

## Warnings
- Masked-completion shape correlation disagrees with downstream event F1 ranking.
