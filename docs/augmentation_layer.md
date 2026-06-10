# Detector-Agnostic Augmentation Layer

`TimeEventSynth` is not primarily a new anomaly detector. Its main idea is a
detector-agnostic, compatibility-aware augmentation layer for low-label
time-series anomaly detection. The layer can be attached to existing
TSB-UAD/TSB-AD detector backbones and evaluated as an add-on rather than as a
replacement detector family.

## Motivation

- TSB-UAD and TSB-AD contain heterogeneous timelines drawn from many datasets
  and domains.
- Existing benchmark comparisons mainly ask which detector architecture works
  best across those datasets.
- Our question is different: when can one timeline safely provide anomalous
  supervision for another timeline?
- Naive cross-dataset transfer can hurt because timelines may differ in scale,
  context, duration, frequency content, or event structure.
- Compatibility-aware transfer filters donor timelines before synthesis so only
  more plausible transfers are kept.

## How It Integrates With Existing Detectors

The intended pipeline is:

1. real training timelines
2. compatibility-aware donor selection
3. universal event-time synthesis
4. synthetic anomalous windows
5. detector backbone
6. threshold calibration and/or training augmentation
7. evaluation

The augmentation layer is therefore orthogonal to the detector itself. The
backbone still decides how anomaly scores are produced; the augmentation layer
decides which anomalous supervision is safe enough to transfer.

## Supported Detector Backbones

Current wrappers:

- `internal_classifier`
- `iforest`
- `ocsvm`
- `lof`
- `autoencoder` if enabled in the config

Important distinction:

- supervised backbones may use synthetic windows directly in training
- unsupervised backbones use synthetic windows mainly for threshold calibration

This matters especially for detectors such as Isolation Forest, One-Class SVM,
LOF, matrix-profile-style methods, and reconstruction models, where synthetic
anomalous windows should generally not be mixed into normal training data.

## Augmentation Policies

Current policies:

- `real_only`
- `random_event_oversampling`
- `all_donors_no_filter`
- `same_dataset_only`
- `cross_dataset_all`
- `cross_dataset_compatible`
- `compatibility_top50`
- `compatibility_strict`

These policies let the paper compare:

- no augmentation
- naive within-dataset or cross-dataset transfer
- compatibility-aware transfer with stronger filtering

## Main Experiment Commands

Smoke experiment:

```bash
cd TimeEventSynth
python3.10 -m src.experiments.run_backbone_augmentation \
  --config configs/experiment_backbone_augmentation_smoke.yaml \
  --use-synthetic
```

Full experiment:

```bash
cd TimeEventSynth
python3.10 -m src.experiments.run_backbone_augmentation \
  --config configs/experiment_backbone_augmentation_full.yaml \
  --data ../TSB-UAD/data/benchmark
```

## Main Output Files

Each backbone-augmentation run writes:

- `backbone_comparison_metrics.csv`
- `backbone_comparison_metrics.json`
- `synthetic_audit.csv`
- `rejection_summary.json`
- `compatibility_summary.json`
- `threshold_diagnostics.csv`
- `analysis_report.json`
- `analysis_report.md`

The analysis layer can also write:

- `detector_policy_pivot_event_f1.csv`
- `detector_policy_pivot_point_f1.csv`
- `detector_policy_gain_summary.csv`
- `threshold_calibration_summary.csv` when available

## Expected Paper Claims

Claims should be framed carefully:

- the method is orthogonal to detector architecture
- the method may improve or calibrate existing detectors under low-label
  conditions
- the key comparison is not only against TSB-UAD detector baselines, but also
  against naive augmentation policies
- stronger claims require multi-seed and dataset-balanced evaluation

If results are mixed, the right claim is still useful but narrower: compatibility-aware
transfer can be evaluated as a detector plug-in layer, and its gains should be
judged against naive augmentation rather than assumed.

## Caveats

- absolute TSB-UAD performance may still be low under severe low-label
  conditions
- cross-dataset transfer is often unsafe and many donor timelines should be
  rejected
- better-looking reconstructions or syntheses do not necessarily improve
  downstream detection
- threshold calibration gains and training-time gains may disagree
- results should be reported across multiple seeds, not just a single smoke run
