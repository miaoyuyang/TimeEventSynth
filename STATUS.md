# TimeEventSynth status

Last updated: 2026-05-30

## Paper-readiness for first internal experiment

The project is configured for reproducible internal experiments with standardized artifacts, leakage guards, and documentation.

### Real data: TSB-UAD-Public-v2

Symlink: `data/raw/TSB-UAD-Public-v2 -> ../../../TSB-UAD/data/TSB-UAD-Public-v2` (~29 domains, ~3400 series).

Shared load defaults: `configs/tsb_uad_public_v2.yaml` (medium-length domains, excludes 650k-point ECG/SWaT for dev runs).

| Config | Purpose |
|--------|---------|
| `experiment_cross_dataset_compatibility_tsb_real_smoke.yaml` | Cross-dataset donor-policy smoke on real multi-domain data |
| `experiment_cross_dataset_compatibility_tsb_real_full.yaml` | Larger real-data compatibility transfer run |
| `experiment_low_label.yaml` | Ablation / sweep (inherits Public-v2 filters) |

Legacy 3-series slice remains at `data/raw/TSB-UAD` for quick regression only.

### Implemented

| Area | Status |
|------|--------|
| Series-level train/val/test split | Done |
| Train-only donor pool (default) | Done (`synthesis.donor_source: train_only`) |
| Low-label masking on train only | Done |
| Central seed utility | Done (`src/utils/seeds.py`) |
| Standard artifact bundle | Done (`src/experiments/artifacts.py`) |
| Resolved config + git commit hash | Done (`run_metadata.json`) |
| Dataset stats + split IDs | Done |
| Aggregate + per-series metrics | Done |
| Synthetic audit on synthesis runs | Done |
| Paper config schema | Done (`configs/tsb_uad.yaml`) |
| Docs | Done (`docs/problem_definition.md`, `docs/method_overview.md`, `docs/experiment_protocol.md`) |
| Leakage tests | Done (`tests/test_leakage.py`) |

### Experiment runners

| Runner | Output path | Notes |
|--------|-------------|-------|
| `run_ablation` | `outputs/ablation/<dataset\|synthetic>/<timestamp>/` | Main internal comparison (incl. context-calibrated + filter variants) |
| `run_low_label` | `outputs/low_label/<dataset\|synthetic>/<timestamp>/` | Single-fraction low-label table |
| `run_low_label_sweep` | `outputs/low_label_sweep/<dataset\|synthetic>/<timestamp>/` | Fraction sweep + aggregated CSV |
| `run_masked_completion` | `outputs/masked_completion/<dataset\|synthetic>/<timestamp>/` | Direct synthesis-fidelity task |
| `run_real_only` | `outputs/real_only/<dataset\|synthetic>/<timestamp>/` | Baseline |
| `analyze_low_label_sweep.py` | `<sweep-dir>/analysis_report.{md,json}` | Post-hoc sweep analysis |
| `run_full_experiment_suite.py` | `outputs/experiment_suite/<timestamp>/suite_manifest.json` | Orchestrates all streams |
| `inspect_dataset.py` | `outputs/inspect/<dataset>/dataset_stats.json` | Dataset QA |

### Latest validation

- `pytest -q`: 68 tests passing (loader, synthesis, leakage, masked completion, sweep analysis).
- Context-calibrated synthesis, masked completion, and sweep analysis scripts added 2026-05-30.

### Known limitations (document honestly)

1. **Detector**: default is supervised `random_forest_window`; unsupervised `isolation_forest` remains available but is not label-sensitive.
2. **Retrieval method flag** in smoke helper still ignores `method=` for ranking (embedding is normalized-time cosine).
3. **Real TSB-UAD-Public-v2** is wired via symlink and `configs/tsb_uad_public_v2.yaml`; default loaders exclude very long domains (MITDB, SWaT, etc.) for dev runs.
4. **Strict confidence filter** is conservative on real data (often rejects all synthetics).
5. **Global universal timeline assumption** is probably too coarse for heterogeneous cross-dataset transfer. Current evidence suggests matchable event families may need separate universal event-time axes.

### Updated paper direction

The strongest current signal is not "one global universal timeline improves all transfer."
It is:

- naive cross-dataset transfer is often unsafe
- compatibility-aware cross-dataset transfer is more promising
- matchable event families may require separate universal timelines

That means the next method version should likely move from:

- global universal event time

to:

- multiple universal timelines, one per compatible event family

See:

- `docs/multi_universal_timelines.md`

### Recommended next internal experiment

1. Inspect TSB-UAD-Public-v2 with domain filters (`configs/tsb_uad_public_v2.yaml`).
2. Run cross-dataset compatibility smoke on real data, then ablation with `--max-series 50`.
3. Sweep `low_label.fractions` with `run_low_label_sweep` (fixed split, aggregated CSV).
4. Compare `no_filter` vs `strict` synthesis policies using `synthetic_audit.csv`.
5. Add groupwise compatibility transfer:
   - assign event-family groups
   - retrieve donors within-group only
   - learn universal event time per group

### Quick commands

```bash
cd TimeEventSynth
pytest -q

# Full multi-stream rerun (synthetic benchmark)
python3 scripts/run_full_experiment_suite.py --profiles synthetic

# Real TSB-UAD-Public-v2 (medium domains, max 50 series)
python3 scripts/run_full_experiment_suite.py --profiles tsb --max-series 50

# Cross-dataset compatibility on real data
python3 -m src.experiments.run_compatibility_transfer \
  --config configs/experiment_cross_dataset_compatibility_tsb_real_smoke.yaml \
  --data data/raw/TSB-UAD-Public-v2 \
  --smoke

# Inspect load
python3 scripts/inspect_dataset.py \
  --data data/raw/TSB-UAD-Public-v2 \
  --max-series 30 \
  --dataset-name TSB-UAD-Public-v2
```
