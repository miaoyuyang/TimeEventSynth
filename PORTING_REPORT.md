# PORTING_REPORT

This report documents method-level ideas reused from the parent
`crisisverse_align` project and how they were adapted into `TimeEventSynth`.

## Reused Components

### 1. Monotone universal timeline parameterization

Source idea:
- `crisisfacts_rarepred/crisisverse_align/src/alignment/monotone_mapping.py`

What it does:
- Represents event-specific monotone mappings with control points and positive
  increments.
- Separates the mapping parameterization from the learning objective.

Decision:
- `rewrite`

Ported location:
- [src/alignment/learned_event_time.py](./src/alignment/learned_event_time.py)

Adaptation:
- Added:
  - `EventTimeMappingResult`
  - `build_control_points`
  - `piecewise_linear_map`
  - `MonotoneEventTimeMap`
- Renamed abstractions:
  - crisis event -> time series
  - universal timeline -> universal event time
  - bin positions -> event-window positions

Why rewritten:
- The original implementation depends on PyTorch and crisis-specific bin/event
  structures.
- `TimeEventSynth` currently uses a lighter prototype + DTW learner, so the
  monotone mapping code is ported as a scaffold for a future stronger learner.

### 2. Donor diversity constraints

Source idea:
- `crisisfacts_rarepred/crisisverse_align/src/completion/diversity.py`

What it does:
- Prevents all synthetic completions from coming from the same donor source.
- Tracks donor concentration and prunes low-diversity selections.

Decision:
- `port`

Ported location:
- [src/synthesis/donor_retrieval.py](./src/synthesis/donor_retrieval.py)

Adaptation:
- Added:
  - `max_donors_per_source_series`
  - `avoid_single_series_dominance`
  - `group_key`
  - `max_donors_per_group`
- Renamed abstractions:
  - donor event -> donor window
  - donor event diversity -> donor series diversity

Why ported:
- This is a reusable method-level safeguard against synthetic poisoning.

### 3. Prior / uncertainty overlay

Source idea:
- `crisisfacts_rarepred/crisisverse_align/src/inference/retrieval_priors.py`

What it does:
- Converts retrieved support plus grounding quality into a prior-like score that
  can influence downstream inference.

Decision:
- `rewrite`

Ported location:
- [src/synthesis/uncertainty_filter.py](./src/synthesis/uncertainty_filter.py)

Adaptation:
- Added `overlay_confidence_prior(...)`
- Renamed abstractions:
  - label prior -> event-pattern prior
  - grounding quality -> synthesis quality
  - donor support score -> synthetic confidence overlay

Why rewritten:
- The original logic depends on strict/weak crisis grounding states and
  bucket-label priors.
- The time-series version uses:
  - donor count
  - synthesis method
  - donor diversity status
  to produce `prior_adjusted_confidence`.

### 4. Event-pattern-specific selection policy

Source idea:
- `crisisfacts_rarepred/crisisverse_align/src/completion/label_synthesis_policy.py`

What it does:
- Decides per rare label whether synthetic examples should be used for training,
  prior-only inference, diagnostics only, or disabled.

Decision:
- `rewrite`

Ported location:
- [src/synthesis/event_pattern_policy.py](./src/synthesis/event_pattern_policy.py)

Adaptation:
- Added:
  - `compute_event_pattern_pool_diagnostics`
  - `select_event_pattern_policy_mode`
  - `build_event_pattern_policy`
- Renamed abstractions:
  - label -> event pattern
  - rare label -> rare event window / rare event pattern
  - weak diagnostic only -> diagnostic only

Why rewritten:
- The crisis version depends on grounding levels like `strict_textual`.
- The time-series version uses confidence and donor diversity diagnostics.

### 5. Method-comparison balance scoring

Source idea:
- `crisisfacts_rarepred/crisisverse_align/src/debug/compare_runs.py`

What it does:
- Compares runs not only by predictive score, but also by diversity / trust
  signals.

Decision:
- `rewrite`

Ported location:
- [src/evaluation/reporting.py](./src/evaluation/reporting.py)
- [scripts/collect_results.py](./scripts/collect_results.py)

Adaptation:
- Added `balance_score(...)`
- Applied to `outputs/summary/all_results.csv`

Why rewritten:
- The crisis version balances accepted completions vs donor concentration.
- The time-series version balances event-level performance vs the size of the
  synthetic pool.

## Discarded Components

The following were intentionally **not** ported:

- TREC-IS downloaders and crisis dataset loaders
- tweet/text processing
- crisis-specific labels and label lexicons
- event IDs and donor-event semantics tied to named disasters
- strict/weak text-support detection based on crisis language evidence
- LLM verifier / rewriter modules
- crisis-specific leakage audits
- crisis-specific paper tables and figures

Reason:
- These are not method-level abstractions for generic time-series event
  synthesis.

## Renamed Abstractions

- crisis event -> time series
- label -> event pattern
- rare label -> rare event window / rare event pattern
- tweet / time bin -> time point / time window
- donor event -> donor window
- label-specific policy -> event-pattern-specific policy
- universal timeline -> universal event time
- grounding prior -> synthesis confidence prior

## Remaining TODOs

- Replace the prototype learner in
  [src/alignment/learned_event_time.py](./src/alignment/learned_event_time.py)
  with a stronger learner using:
  - cross-series consistency loss
  - uncertainty-aware alignment
  - event-pattern-specific mappings
- Use `MonotoneEventTimeMap` inside the learned aligner rather than keeping it
  as a scaffold.
- Add real multi-pattern anomaly benchmarks so
  [src/synthesis/event_pattern_policy.py](./src/synthesis/event_pattern_policy.py)
  is exercised on more than a single anomaly pattern.
- Tune strict uncertainty thresholds on real datasets; the current toy
  synthetic benchmark shows they are very conservative.

## Commands To Reproduce Current Experiments

```bash
cd TimeEventSynth
pytest -q
python3.10 scripts/run_smoke_test.py
python3.10 -m src.experiments.run_ablation --config configs/experiment_low_label.yaml --use-synthetic
python3.10 scripts/collect_results.py
python3.10 scripts/plot_ablation.py
python3.10 scripts/plot_qualitative_examples.py
```
