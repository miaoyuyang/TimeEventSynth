# Method overview

TimeEventSynth augments low-label training data by synthesizing anomaly event windows from retrieved donors.

## Pipeline

1. **Load dataset** from nested CSV/out files into series records.
2. **Split by series** into train / validation / test.
3. **Mask train labels** by randomly hiding a fraction of anomaly segments (train only).
4. **Retrieve donors** from an allowed donor pool (train-only by default).
5. **Synthesize windows** using one of:
   - normalized-time mean donor
   - DTW-aligned donor average
   - learned prototype event-time average
6. **Filter synthetics** using confidence scores (optional strict policy).
7. **Train detector** (default: windowed isolation forest).
8. **Tune threshold on validation**, report test metrics.

## Alignment / synthesis methods

| Method | Idea |
|--------|------|
| `normalized_time_mean_donor` | Resample donors to a fixed event-time grid and average |
| `dtw_aligned_donor` | DTW-warp donors to target length, then average |
| `learned_prototype_event_time` | Fit a prototype event-time axis on donors only, then synthesize |

## Uncertainty filtering

Each synthetic candidate receives:

- donor similarity confidence
- reconstruction consistency confidence
- donor agreement confidence
- amplitude compatibility score

Policies: `no_filter`, `top_quantile`, `min_confidence`, `strict`.

## Leakage controls

- Splits are **series-disjoint**.
- Donor pool defaults to **train series only** (`synthesis.donor_source: train_only`).
- Low-label masking applies to **train labels only**.
- Learned prototype fitting uses **donors only**, not the target window.

## Baselines

- `real_only`
- `random_event_oversampling`
- synthesis variants above
