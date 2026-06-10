# Analysis Report

## Best Augmentation Policy Per Detector
- `autoencoder`: `random_event_oversampling` (event_f1=0.0883, auprc=nan)
- `cnn`: `compatibility_strict` (event_f1=0.0339, auprc=nan)
- `iforest`: `adaptive_groupwise_transfer` (event_f1=0.0437, auprc=nan)
- `internal_classifier`: `groupwise_cross_dataset_compatible` (event_f1=0.0627, auprc=nan)
- `lof`: `adaptive_groupwise_transfer` (event_f1=0.0967, auprc=nan)
- `ocsvm`: `real_only` (event_f1=0.0079, auprc=nan)
- `timesnet`: `real_only` (event_f1=0.0379, auprc=nan)

- Average event-F1 gain of `compatibility_strict` over `real_only`: 0.0229
- Average event-F1 gain of `cross_dataset_compatible` over `cross_dataset_all`: 0.0107
- Detector backbones improved by compatibility-aware augmentation: 0
- Average false-positive reduction: 0.0000

## Improvement Characterization

