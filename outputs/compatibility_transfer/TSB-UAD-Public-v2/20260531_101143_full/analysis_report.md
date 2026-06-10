# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `random_event_oversampling` (0.0779, policy=None)
- Best **point F1**: `random_event_oversampling` (0.1220, policy=None)

## Compatibility-Filtered vs All-Donor Transfer
- Cross-dataset compatible transfer **beats** naive all-donor on event F1 (Δ=+0.0213).

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **helps** vs same-dataset-only (Δ=+0.0015).

## Donor Compatibility Summary
- Target timelines: 32
- Source timelines: 42
- Donor pairs kept / rejected: 4295 / 48061
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.8418
- Mean compatibility (rejected): 0.6192

## Major Rejection Reasons
- `below_top_quantile_0.5`: 40246
- `rejected_incompatible_donor`: 7815
- `strict_filter_failed:reconstruction_consistency_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 144
- `strict_filter_failed:reconstruction_consistency_confidence,amplitude_compatibility_score,aggregate_confidence`: 66
- `strict_filter_failed:amplitude_compatibility_score`: 14
- `strict_filter_failed:donor_agreement_confidence,amplitude_compatibility_score`: 13
- `strict_filter_failed:donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 7
- `strict_filter_failed:donor_agreement_confidence`: 3
- `strict_filter_failed:donor_similarity_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:donor_similarity_confidence,reconstruction_consistency_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:donor_similarity_confidence,reconstruction_consistency_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 1

## Downstream Detection Ranking (event F1)
- `random_event_oversampling`: event_f1=0.0779 (policy=None)
- `real_only`: event_f1=0.0771 (policy=None)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.0603 (policy=cross_dataset_compatible)
- `normalized_time_mean_donor__cross_dataset_compatible`: event_f1=0.0592 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__same_dataset_only`: event_f1=0.0588 (policy=same_dataset_only)
- `learned_prototype_event_time__compatibility_top50`: event_f1=0.0507 (policy=compatibility_top50)
- `learned_prototype_event_time__cross_dataset_all`: event_f1=0.0469 (policy=cross_dataset_all)
- `learned_prototype_event_time__compatibility_strict`: event_f1=0.0455 (policy=compatibility_strict)
- `learned_prototype_event_time__strict_filter`: event_f1=0.0448 (policy=cross_dataset_compatible)
- `dtw_aligned_donor__cross_dataset_compatible`: event_f1=0.0414 (policy=cross_dataset_compatible)

## Masked-Completion Ranking
- `learned_prototype_event_time__all_donors_no_filter`: shape_correlation=0.2797, normalized_MAE=19.5341
- `learned_prototype_event_time__compatibility_top50`: shape_correlation=0.2622, normalized_MAE=18.0704
- `learned_prototype_event_time__same_dataset_only`: shape_correlation=0.2534, normalized_MAE=17.0676
- `learned_prototype_event_time__compatibility_strict`: shape_correlation=0.2350, normalized_MAE=17.1350
- `learned_prototype_event_time__cross_dataset_all`: shape_correlation=0.1043, normalized_MAE=14.7480
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.1043, normalized_MAE=14.7480
- `learned_prototype_event_time__strict_filter`: shape_correlation=0.1043, normalized_MAE=14.7480
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.0975, normalized_MAE=14.3918
- `normalized_time_mean_donor__cross_dataset_compatible`: shape_correlation=0.0909, normalized_MAE=14.8454

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **-0.171**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 40246
- Strict-filter failures: 250

## Warnings
- Masked-completion shape correlation disagrees with downstream event F1 ranking.
