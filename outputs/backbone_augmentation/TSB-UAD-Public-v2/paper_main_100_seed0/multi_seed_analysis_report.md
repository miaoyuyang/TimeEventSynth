# Analysis Report

## Best Augmentation Policy Per Detector
- `autoencoder`: `all_donors_no_filter` (event_f1=0.0280, auprc=nan)
- `iforest`: `adaptive_groupwise_transfer` (event_f1=0.0000, auprc=nan)
- `internal_classifier`: `real_only` (event_f1=0.1101, auprc=nan)
- `lof`: `all_donors_no_filter` (event_f1=0.0708, auprc=nan)
- `ocsvm`: `adaptive_groupwise_transfer` (event_f1=0.0000, auprc=nan)
- `timesnet`: `adaptive_groupwise_transfer` (event_f1=0.0000, auprc=nan)

- Average event-F1 gain of `compatibility_strict` over `real_only`: nan
- Average event-F1 gain of `cross_dataset_compatible` over `cross_dataset_all`: nan
- Detector backbones improved by compatibility-aware augmentation: 0
- Average false-positive reduction: 0.0000

## Improvement Characterization

