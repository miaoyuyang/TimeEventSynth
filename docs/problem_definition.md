# Problem definition

TimeEventSynth studies **time-series event detection under low label availability** using **synthetic event-window augmentation**.

## Objects

- **Timeline**: one full univariate or multivariate time series instance (`series_id`, values, point labels).
- **Event window**: a contiguous anomaly segment `[start, end)` extracted from pointwise binary labels.
- **Donor window**: an event window retrieved from another series and used as a synthesis source.
- **Synthetic window**: a generated anomaly segment inserted into training.

## Task

Given partially labeled train series, train an anomaly/event detector and evaluate on held-out test series using:

- pointwise metrics (AUROC, AUPRC, precision/recall/F1)
- event-level metrics (precision/recall/F1 with IoU matching)

## Research question

Can donor-based event-window synthesis improve rare-event detection when only a small fraction of anomaly segments remain labeled, without leaking test information or poisoning the detector with low-confidence synthetics?

## Non-goals in the current internal experiment

- Crisis-specific text or label semantics
- Online deployment or streaming inference
- Neural end-to-end detectors beyond simple classical baselines
