# Compatibility Transfer Analysis

## Best Methods
- Best **event_f1**: `learned_prototype_event_time__same_dataset_only` (0.1466, policy=same_dataset_only)
- Best **point F1**: `learned_prototype_event_time__same_dataset_only` (0.0948, policy=same_dataset_only)

## Cross-Dataset Compatible vs Same-Dataset Only
- Cross-dataset compatible transfer **does not help** vs same-dataset-only (Δ=-0.0357).

## Donor Compatibility Summary
- Target timelines: 14
- Source timelines: 21
- Donor pairs kept / rejected: 216 / 992
- No compatible donor (targets): 0
- Mean compatibility (kept): 0.6620
- Mean compatibility (rejected): 0.5807

## Major Rejection Reasons
- `below_top_quantile_0.5`: 992

## Downstream Detection Ranking (event F1)
- `learned_prototype_event_time__same_dataset_only`: event_f1=0.1466 (policy=same_dataset_only)
- `real_only`: event_f1=0.1270 (policy=None)
- `learned_prototype_event_time__cross_dataset_all`: event_f1=0.1207 (policy=cross_dataset_all)
- `learned_prototype_event_time__cross_dataset_compatible`: event_f1=0.1109 (policy=cross_dataset_compatible)

## Strict / Top-Quantile Filtering
- Top-quantile rejections: 992
- Strict-filter failures: 0
