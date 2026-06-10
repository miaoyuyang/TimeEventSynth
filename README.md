# TimeEventSynth

TimeEventSynth is a research codebase for low-label time-series anomaly detection with synthetic event transfer. It focuses on using scarce labeled anomaly events more effectively across heterogeneous time-series datasets.

The project implements an augmentation layer that can be plugged into different anomaly detection backbones. Given a small set of labeled anomaly events, TimeEventSynth selects compatible donor events, synthesizes event windows for target series, and uses the generated windows for training or threshold calibration.

Current focus:

- cross-dataset anomaly event transfer
- compatibility-aware donor selection
- groupwise event-family modeling
- detector-specific augmentation policies
- event-level and false-positive-aware evaluation

## Core Concepts

- An `event` is a contiguous anomaly segment.
- A `timeline` is one complete time series.
- A `donor` timeline provides anomaly structure that may be transferred to a target timeline.
- `Groupwise` transfer clusters events into event families before donor selection.
- `Adaptive` transfer applies different augmentation policies for different detector families.

The main research question is whether carefully selected synthetic anomaly events can improve existing detectors in low-label settings, especially when anomaly labels are sparse and heterogeneous across datasets.

## What Is Implemented

### Data and Event Layer

- loaders for TSB-UAD / TSB-AD style datasets
- series-level train / validation / test splits
- event extraction with half-open intervals `[start, end)`
- dataset-balanced multi-seed evaluation

### Synthesis Layer

- normalized-time event synthesis
- DTW-aligned event synthesis
- prototype-based event-time synthesis
- compatibility-aware donor retrieval
- uncertainty filtering
- donor diversity control
- groupwise event-family transfer
- adaptive groupwise transfer

### Detector Layer

Implemented detector backbones include:

- `internal_classifier`
- `iforest`
- `ocsvm`
- `lof`
- `autoencoder`
- `cnn`
- `timesnet`

Synthetic windows are used differently across detectors. Supervised or trainable models can use them as additional anomaly examples, while score-based unsupervised models mainly use them for calibration and synthetic-positive separation.

## Current Interpretation

The current results suggest that synthetic event transfer is detector-dependent. Some backbones benefit clearly from groupwise or compatibility-aware transfer, while others are more sensitive to false positives or threshold selection.

In particular:

- compatibility-aware transfer is safer than uniform cross-dataset transfer
- groupwise transfer helps avoid treating all anomaly events as one universal pattern
- the best augmentation policy can vary by detector family
- Event-F1 gains should be interpreted together with precision, recall, and false-positive events

## Project Layout

```text
TimeEventSynth/
  configs/        experiment configs
  data/           raw / processed / split artifacts
  docs/           protocol notes and paper-related notes
  outputs/        experiment results
  scripts/        utility scripts
  src/
    alignment/    normalized time, DTW, learned event time, event grouping
    augmentation/ detector-agnostic policy layer
    datasets/     loading, event extraction, splits
    detectors/    backbone wrappers
    evaluation/   metrics, thresholding, reporting
    experiments/  experiment runners
    synthesis/    donor retrieval, synthesis, filtering, audit
  tests/
