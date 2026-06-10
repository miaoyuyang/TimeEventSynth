# Analysis Report

## Best Augmentation Policy Per Detector
- `autoencoder`: `adaptive_groupwise_transfer` (event_f1=0.0483, auprc=nan)
- `cnn`: `random_event_oversampling` (event_f1=0.0411, auprc=nan)
- `iforest`: `adaptive_groupwise_transfer` (event_f1=0.0532, auprc=nan)
- `internal_classifier`: `real_only` (event_f1=0.0591, auprc=nan)
- `lof`: `groupwise_cross_dataset_all` (event_f1=0.0375, auprc=nan)
- `ocsvm`: `random_event_oversampling` (event_f1=0.0025, auprc=nan)
- `timesnet`: `random_event_oversampling` (event_f1=0.0101, auprc=nan)

- Average event-F1 gain of `compatibility_strict` over `real_only`: 0.0106
- Average event-F1 gain of `cross_dataset_compatible` over `cross_dataset_all`: 0.0029
- Detector backbones improved by compatibility-aware augmentation: 0
- Average false-positive reduction: 0.0000

## Improvement Characterization

