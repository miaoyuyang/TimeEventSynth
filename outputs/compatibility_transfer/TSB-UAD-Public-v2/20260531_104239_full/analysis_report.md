# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__compatibility_strict` (0.1063, policy=compatibility_strict)
- Best **point F1**: `learned_prototype_event_time__strict_filter` (0.1253, policy=cross_dataset_compatible)

## Compatibility-Filtered vs All-Donor Transfer
- Cross-dataset compatible transfer **beats** naive all-donor on event F1 (Δ=+0.0302).

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **helps** vs same-dataset-only (Δ=+0.0306).

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
- `strict_filter_failed:reconstruction_consistency_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 162
- `strict_filter_failed:reconstruction_consistency_confidence,amplitude_compatibility_score,aggregate_confidence`: 69
- `strict_filter_failed:amplitude_compatibility_score`: 9
- `strict_filter_failed:reconstruction_consistency_confidence`: 4
- `strict_filter_failed:donor_agreement_confidence,amplitude_compatibility_score`: 2
- `strict_filter_failed:donor_agreement_confidence`: 1
- `strict_filter_failed:donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:donor_similarity_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:donor_similarity_confidence,reconstruction_consistency_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:donor_similarity_confidence,reconstruction_consistency_confidence,donor_agreement_confidence,amplitude_compatibility_score,aggregate_confidence`: 1
- `strict_filter_failed:reconstruction_consistency_confidence,amplitude_compatibility_score`: 1
- `strict_filter_failed:reconstruction_consistency_confidence,donor_agreement_confidence`: 1

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__compatibility_strict`: event_f1=0.1063 (policy=compatibility_strict)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.0864 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__strict_filter`: event_f1=0.0789 (policy=cross_dataset_compatible)
- `random_event_oversampling`: event_f1=0.0779 (policy=None)
- `real_only`: event_f1=0.0771 (policy=None)
- `normalized_time_mean_donor__cross_dataset_compatible`: event_f1=0.0592 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__compatibility_top50`: event_f1=0.0577 (policy=compatibility_top50)
- `learned_prototype_event_time__all_donors_no_filter`: event_f1=0.0562 (policy=all_donors_no_filter)
- `learned_prototype_event_time__same_dataset_only`: event_f1=0.0558 (policy=same_dataset_only)
- `learned_prototype_event_time__cross_dataset_all`: event_f1=0.0539 (policy=cross_dataset_all)

## Masked-Completion Ranking
- `learned_prototype_event_time__same_dataset_only`: shape_correlation=0.3009, normalized_MAE=17.1471
- `learned_prototype_event_time__compatibility_strict`: shape_correlation=0.2886, normalized_MAE=17.0063
- `learned_prototype_event_time__compatibility_top50`: shape_correlation=0.2450, normalized_MAE=18.0314
- `learned_prototype_event_time__all_donors_no_filter`: shape_correlation=0.2278, normalized_MAE=18.3569
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.0975, normalized_MAE=14.3918
- `normalized_time_mean_donor__cross_dataset_compatible`: shape_correlation=0.0909, normalized_MAE=14.8454
- `learned_prototype_event_time__cross_dataset_all`: shape_correlation=0.0515, normalized_MAE=13.7948
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.0515, normalized_MAE=13.7948
- `learned_prototype_event_time__strict_filter`: shape_correlation=0.0515, normalized_MAE=13.7948

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **0.120**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 40246
- Strict-filter failures: 253
