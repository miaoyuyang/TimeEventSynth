# Multi-Universal-Timeline Design

## Motivation

The current experiments suggest that `cross_dataset_compatible` is more promising
than naive `cross_dataset_all` transfer. This points to a stronger modeling
assumption:

- heterogeneous anomaly/event windows should not all share one global event-time axis
- only matchable event families should be aligned together
- each event family should have its own universal event-time axis

This design changes the core question from:

- "Can every anomaly be aligned on one universal timeline?"

to:

- "Which events are compatible enough to share a transferable universal timeline?"

That framing better matches both the empirical results and the intended paper
story.

## Core idea

We introduce a latent intermediate object:

- `event_group_id`

Each event window belongs to a matchable transfer group. Examples:

- abrupt spike-like anomalies
- sustained level shifts
- periodic degradation patterns
- transient bursts with recovery

For each group `g`, we learn:

- a donor bank `D_g`
- a group-level universal event-time axis `tau_g in [0, 1]`
- a group-specific alignment model `phi_i^g` for each event window in that group

Transfer is then restricted to:

- retrieve donors from the same group
- align within-group only
- synthesize within-group only

## Proposed pipeline

1. Extract event windows from train donor pool and target timelines.
2. Compute event compatibility features.
3. Assign each event to a matchable transfer group.
4. Build donor banks per group.
5. Learn a universal event-time model separately for each group.
6. Retrieve donors only from the matched group.
7. Synthesize target anomalous windows from group-matched donors.
8. Apply uncertainty filtering and donor diversity constraints.
9. Use resulting synthetic windows for:
   - supervised detector training augmentation
   - threshold calibration for unsupervised detectors

## Why this should improve quality

### Better transfer fidelity

Within-group events should have:

- more similar temporal progression
- more compatible amplitude/context patterns
- lower shape variance

That should reduce noisy donor transfer and synthetic poisoning.

### Better alignment

A universal timeline is only meaningful when events are semantically and
dynamically similar. Group-wise alignment makes:

- local event-time progression more coherent
- prototype learning easier
- learned event-time mappings less noisy

### Better efficiency

Grouping should reduce compute at several stages:

- fewer donor-pair comparisons
- smaller donor retrieval candidate sets
- smaller alignment optimization problems
- fewer low-quality synthesis candidates

This makes compatibility grouping both a modeling constraint and an efficiency
mechanism.

## Concrete implementation plan

### 1. Add event grouping module

Suggested file:

- `src/alignment/event_grouping.py`

Proposed functions:

```python
def compute_event_group_features(event_windows, config=None) -> pd.DataFrame:
    ...

def assign_event_groups(event_windows, config=None) -> dict[str, str]:
    ...

def build_group_donor_banks(event_windows, event_group_ids) -> dict[str, list[EventWindow]]:
    ...
```

Initial grouping features can be simple:

- normalized event shape embedding
- event length
- amplitude range / variance
- slope / recovery profile
- context compatibility summary
- donor retrieval embedding already used by current synthesis

Initial assignment methods can be lightweight:

- nearest prototype
- agglomerative clustering
- k-means on event-window embeddings
- thresholded compatibility graph components

TODO:
- replace heuristic grouping with learned event-family assignment later

### 2. Extend donor retrieval with group constraints

Suggested changes:

- `src/synthesis/donor_retrieval.py`

Add options:

- `restrict_to_group: bool = True`
- `target_group_id`
- `donor_group_id`

Behavior:

- cross-dataset retrieval is allowed only within the same event group
- donor diversity is enforced inside each group donor bank

### 3. Add per-group universal event-time learner

Suggested extension:

- `src/alignment/learned_event_time.py`

Current placeholder learner can be refactored into:

```python
class GroupwiseLearnedEventTimeAligner:
    def fit(event_windows, event_group_ids):
        ...

    def synthesize(target_window, donor_windows, target_group_id, ...):
        ...
```

Each group gets:

- its own prototype event shape
- its own donor-to-prototype warps
- its own inverse mapping logic

TODO:
- later add group-specific monotone neural mappings
- later add cross-series consistency loss within-group

### 4. Add compatibility-first policy variants

Suggested policies:

- `groupwise_cross_dataset_all`
- `groupwise_cross_dataset_compatible`
- `groupwise_compatibility_strict`

These differ from current policies by:

- first requiring group match
- then applying compatibility filtering

This creates a cleaner ablation:

1. no grouping
2. grouping only
3. grouping + compatibility filtering
4. grouping + compatibility filtering + learned event time

### 5. Add analysis outputs

New diagnostics should report:

- number of discovered event groups
- group sizes
- cross-dataset coverage per group
- donor diversity per group
- compatibility rejection rates per group
- performance gains by group-aware vs global transfer

Suggested artifacts:

- `event_group_summary.json`
- `groupwise_compatibility_summary.json`
- `groupwise_transfer_gain.csv`

## Minimal first implementation

The first implementation should stay simple:

1. compute event-window embeddings using the current normalized-time donor features
2. cluster train donor events into a small number of groups
3. assign target events to nearest group centroid
4. restrict donor retrieval to the assigned group
5. reuse the existing learned-prototype-event-time aligner within each group

This gives us a low-risk path to test the paper hypothesis before building a
more complex neural alignment model.

## Experiment plan

Recommended comparison for the next paper-facing pass:

- `real_only`
- `cross_dataset_all`
- `cross_dataset_compatible`
- `groupwise_cross_dataset_all`
- `groupwise_cross_dataset_compatible`
- `groupwise_compatibility_strict`

Backbones to prioritize:

- `internal_classifier`
- `cnn`
- `timesnet`
- `iforest`

Primary questions:

1. Does grouping reduce false-positive events?
2. Does grouping improve event precision without destroying recall?
3. Does groupwise compatible transfer beat global compatible transfer?
4. Does groupwise learned event-time help more than global learned event-time?

## Paper-facing claim

The clean claim is:

- compatibility-aware cross-dataset transfer works better when events are first
  partitioned into matchable families
- each family should have its own universal event-time axis
- universal event-time alignment is therefore group-specific rather than global

This is a stronger and more defensible story than claiming one universal
timeline for all anomaly types.
