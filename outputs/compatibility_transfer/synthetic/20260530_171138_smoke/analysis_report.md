# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__cross_dataset_compatible` (0.8659, policy=cross_dataset_compatible)
- Best **point F1**: `real_only` (0.9587, policy=None)

## Compatibility-Filtered vs All-Donor Transfer
- Cross-dataset compatible transfer **beats** naive all-donor on event F1 (Δ=+0.0431).

## Donor Compatibility Summary
- Target timelines: 10
- Source timelines: 10
- Donor pairs kept / rejected: 171 / 228
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.9387
- Mean compatibility (rejected): 0.9063

## Major Rejection Reasons
- `below_top_quantile_0.5`: 228

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.8659 (policy=cross_dataset_compatible)
- `random_event_oversampling`: event_f1=0.8589 (policy=None)
- `real_only`: event_f1=0.8447 (policy=None)
- `dtw_aligned_donor__cross_dataset_compatible`: event_f1=0.8375 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__all_donors_no_filter`: event_f1=0.8228 (policy=all_donors_no_filter)

## Masked-Completion Ranking
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.7376, normalized_MAE=0.6937
- `learned_prototype_event_time__all_donors_no_filter`: shape_correlation=0.3737, normalized_MAE=1.1064
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.3737, normalized_MAE=1.1064

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **-0.180**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 228
- Strict-filter failures: 0

## Warnings
- Masked-completion shape correlation disagrees with downstream event F1 ranking.
