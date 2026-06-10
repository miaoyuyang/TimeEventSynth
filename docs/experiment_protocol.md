# Experiment protocol

This document defines the first internal paper-ready experiment protocol for TimeEventSynth.

## Configuration

Primary config: `configs/experiment_low_label.yaml` (inherits `configs/tsb_uad.yaml`).

Key fields:

```yaml
dataset:
  path: data/raw/TSB-UAD
  name: TSB-UAD
  max_series: null

split:
  seed: 42
  train_ratio: 0.7
  val_ratio: 0.1
  test_ratio: 0.2

low_label:
  fractions: [0.01, 0.05, 0.1, 0.2]
  default_fraction: 0.2

synthesis:
  methods: [normalized_time_mean_donor, dtw_aligned_donor, learned_prototype_event_time]
  top_k: 3
  donor_source: train_only
  confidence_filter:
    default: no_filter
    strict: {...}

detector:
  name: random_forest_window
  params:
    window_size: 5
    contamination: 0.1

evaluation:
  threshold_selection: point_f1
  event_iou_threshold: 0.1
```

## Commands

### Inspect dataset

```bash
python3 scripts/inspect_dataset.py \
  --data data/raw/TSB-UAD \
  --max-series 20 \
  --dataset-name TSB-UAD
```

### Synthetic smoke ablation

```bash
python3 -m src.experiments.run_ablation \
  --config configs/experiment_low_label.yaml \
  --use-synthetic
```

### Real-data ablation

```bash
python3 -m src.experiments.run_ablation \
  --config configs/experiment_low_label.yaml \
  --data data/raw/TSB-UAD \
  --max-series 20 \
  --dataset-name TSB-UAD
```

### Low-label fraction sweep

Runs the full ablation at each `low_label.fractions` value using **one fixed train/val/test split**.

```bash
python3 -m src.experiments.run_low_label_sweep \
  --config configs/experiment_low_label.yaml \
  --use-synthetic

python3 -m src.experiments.run_low_label_sweep \
  --config configs/experiment_low_label.yaml \
  --data data/raw/TSB-UAD \
  --max-series 20 \
  --dataset-name TSB-UAD
```

Override fractions:

```bash
python3 -m src.experiments.run_low_label_sweep \
  --config configs/experiment_low_label.yaml \
  --use-synthetic \
  --fractions 0.05 0.1 0.2
```

Sweep outputs under `outputs/low_label_sweep/<dataset-or-synthetic>/<timestamp>/`:

- `low_label_sweep_comparison.csv` — method × fraction aggregate table
- `low_label_sweep_per_series.csv`
- `fraction_<f>/` — full per-fraction artifact bundle

## Required artifacts (every experiment run)

Each run writes under `outputs/<experiment>/<dataset-or-synthetic>/<timestamp>/`:

| File | Description |
|------|-------------|
| `config_resolved.yaml` | Fully merged config used for the run |
| `run_metadata.json` | UTC timestamp, experiment name, seed, git commit hash |
| `dataset_stats.json` | Series/event statistics for loaded data |
| `split_ids.json` | Train/val/test series IDs |
| `metrics.json` | Aggregate metrics |
| `per_series_metrics.csv` | Per-series test metrics |
| `synthetic_audit.csv` | Present when synthesis is used |

Ablation runs also write:

- `comparison_metrics.csv`
- `rejection_summary.json`
- `event_pattern_policy.json`

## Evaluation protocol

1. Fit detector on train (possibly augmented).
2. Score validation and test series.
3. Select threshold on validation using `evaluation.threshold_selection`.
4. Report test point metrics and event metrics.
5. Never use test labels for synthesis, donor retrieval, or threshold tuning.

## Reproducibility

- Global seed set via `split.seed` / top-level `seed`.
- Detector uses the same seed through `random_state`.
- Dataset subsampling (`max_series`) uses seeded shuffling.

## Tests

```bash
pytest -q
```

Includes split disjointness, loader normalization, synthesis shape checks, and leakage guards.
