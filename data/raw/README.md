# Raw benchmark data

Place or symlink downloaded TSB-UAD files here.

## TSB-UAD-Public-v2 (recommended)

Download location (sibling repo):

```
../TSB-UAD/data/TSB-UAD-Public-v2/
  MSL/
  NAB/
  SWaT/
  MITDB/
  ...
```

Symlink (from `TimeEventSynth/`):

```bash
ln -sf ../../../TSB-UAD/data/TSB-UAD-Public-v2 data/raw/TSB-UAD-Public-v2
```

Run cross-dataset compatibility smoke on real data:

```bash
python3 -m src.experiments.run_compatibility_transfer \
  --config configs/experiment_cross_dataset_compatibility_tsb_real_smoke.yaml \
  --data data/raw/TSB-UAD-Public-v2
```

Inspect load:

```bash
python3 scripts/inspect_dataset.py \
  --data data/raw/TSB-UAD-Public-v2 \
  --max-series 30 \
  --dataset-name TSB-UAD-Public-v2
```

## Legacy small benchmark slice

```
data/raw/TSB-UAD -> ../../../TSB-UAD/data/benchmark
```

Only a few ECG series; use Public-v2 for paper experiments.
