# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__cross_dataset_compatible` (0.8659, policy=cross_dataset_compatible)
- Best **point F1**: `learned_prototype_event_time__strict_filter` (0.9680, policy=cross_dataset_compatible)

## Compatibility-Filtered vs All-Donor Transfer
- Cross-dataset compatible transfer **beats** naive all-donor on event F1 (Δ=+0.0431).

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **helps** vs same-dataset-only (Δ=+0.0431).

## Donor Compatibility Summary
- Target timelines: 10
- Source timelines: 10
- Donor pairs kept / rejected: 530 / 817
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.9397
- Mean compatibility (rejected): 0.9069

## Major Rejection Reasons
- `below_top_quantile_0.5`: 826
- `strict_filter_failed:amplitude_compatibility_score`: 17

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.8659 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__compatibility_strict`: event_f1=0.8659 (policy=compatibility_strict)
- `learned_prototype_event_time__strict_filter`: event_f1=0.8659 (policy=cross_dataset_compatible)
- `random_event_oversampling`: event_f1=0.8589 (policy=None)
- `real_only`: event_f1=0.8447 (policy=None)
- `learned_prototype_event_time__compatibility_top50`: event_f1=0.8447 (policy=compatibility_top50)
- `learned_prototype_event_time__top50_filter`: event_f1=0.8375 (policy=cross_dataset_compatible)
- `dtw_aligned_donor__cross_dataset_compatible`: event_f1=0.8375 (policy=cross_dataset_compatible)
- `normalized_time_mean_donor__cross_dataset_compatible`: event_f1=0.8375 (policy=cross_dataset_compatible)
- `normalized_time_mean_donor__all_donors_no_filter`: event_f1=0.8302 (policy=all_donors_no_filter)

## Masked-Completion Ranking
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.8015, normalized_MAE=0.6382
- `learned_prototype_event_time__compatibility_top50`: shape_correlation=0.5962, normalized_MAE=0.9744
- `learned_prototype_event_time__all_donors_no_filter`: shape_correlation=0.4160, normalized_MAE=0.9495
- `learned_prototype_event_time__compatibility_strict`: shape_correlation=0.4160, normalized_MAE=0.9495
- `learned_prototype_event_time__cross_dataset_all`: shape_correlation=0.4160, normalized_MAE=0.9495
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.4160, normalized_MAE=0.9495
- `learned_prototype_event_time__strict_filter`: shape_correlation=0.4160, normalized_MAE=0.9495
- `learned_prototype_event_time__top50_filter`: shape_correlation=0.4160, normalized_MAE=0.9495
- `normalized_time_mean_donor__all_donors_no_filter`: shape_correlation=0.4150, normalized_MAE=0.9516
- `normalized_time_mean_donor__cross_dataset_compatible`: shape_correlation=0.4150, normalized_MAE=0.9516

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **-0.092**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 826
- Strict-filter failures: 17

## Warnings
- Masked-completion shape correlation disagrees with downstream event F1 ranking.
