# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__compatibility_top50` (0.8936, policy=compatibility_top50)
- Best **point F1**: `learned_prototype_event_time__compatibility_top50` (0.9611, policy=compatibility_top50)

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **does not help** vs same-dataset-only (Δ=-0.0503).

## Donor Compatibility Summary
- Target timelines: 21
- Source timelines: 24
- Donor pairs kept / rejected: 287 / 2624
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.9341
- Mean compatibility (rejected): 0.8782

## Major Rejection Reasons
- `below_top_quantile_0.5`: 2624
- `strict_filter_failed:amplitude_compatibility_score`: 35

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__compatibility_top50`: event_f1=0.8936 (policy=compatibility_top50)
- `dtw_aligned_donor__cross_dataset_compatible`: event_f1=0.8936 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__cross_dataset_all`: event_f1=0.8842 (policy=cross_dataset_all)
- `learned_prototype_event_time__strict_filter`: event_f1=0.8842 (policy=cross_dataset_compatible)
- `learned_prototype_event_time__same_dataset_only`: event_f1=0.8817 (policy=same_dataset_only)
- `real_only`: event_f1=0.8352 (policy=None)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.8315 (policy=cross_dataset_compatible)

## Masked-Completion Ranking
- `dtw_aligned_donor__cross_dataset_compatible`: shape_correlation=0.6147, normalized_MAE=0.6635
- `learned_prototype_event_time__cross_dataset_all`: shape_correlation=0.4512, normalized_MAE=1.1477
- `learned_prototype_event_time__cross_dataset_compatible`: shape_correlation=0.4512, normalized_MAE=1.1477
- `learned_prototype_event_time__strict_filter`: shape_correlation=0.4512, normalized_MAE=1.1477
- `learned_prototype_event_time__compatibility_top50`: shape_correlation=0.4184, normalized_MAE=1.0892
- `learned_prototype_event_time__same_dataset_only`: shape_correlation=0.4043, normalized_MAE=1.0777

## Reconstruction vs Downstream Utility
- Pearson correlation (shape_correlation vs event_f1): **0.209**

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 2624
- Strict-filter failures: 35
