# Analysis Report

## Best Augmentation Policy Per Detector
- `autoencoder`: `groupwise_cross_dataset_compatible` (event_f1=0.0315, auprc=nan)
- `cnn`: `random_event_oversampling` (event_f1=0.0413, auprc=nan)
- `iforest`: `groupwise_cross_dataset_all` (event_f1=0.0567, auprc=nan)
- `internal_classifier`: `real_only` (event_f1=0.1143, auprc=nan)
- `lof`: `groupwise_compatibility_strict` (event_f1=0.0522, auprc=nan)
- `ocsvm`: `random_event_oversampling` (event_f1=0.0025, auprc=nan)
- `timesnet`: `random_event_oversampling` (event_f1=0.0101, auprc=nan)

- Average event-F1 gain of `compatibility_strict` over `real_only`: 0.0115
- Average event-F1 gain of `cross_dataset_compatible` over `cross_dataset_all`: -0.0100
- Detector backbones improved by compatibility-aware augmentation: 0
- Average false-positive reduction: 0.0000

## Improvement Characterization

