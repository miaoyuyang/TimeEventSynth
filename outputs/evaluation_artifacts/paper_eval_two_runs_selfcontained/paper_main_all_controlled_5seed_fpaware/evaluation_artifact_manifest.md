# Evaluation Artifact Manifest

- Run dir: `C:\Users\Admin\Desktop\paper_main_all_controlled_5seed_fpaware`
- Artifact dir: `C:\Users\Admin\Desktop\TimeEventSynth\outputs\evaluation_artifacts\paper_eval_two_runs_selfcontained\paper_main_all_controlled_5seed_fpaware`
- Seeds: 5 ([0, 1, 2, 3, 4])
- Detectors: 7 (iforest, lof, ocsvm, autoencoder, cnn, timesnet, internal_classifier)
- Policies: 6 (real_only, random_event_oversampling, groupwise_cross_dataset_all, groupwise_cross_dataset_compatible, groupwise_compatibility_strict, adaptive_groupwise_transfer)
- Mean series per seed: 63.0

## Paper Tables
- `dataset_setup_summary.csv` / `dataset_setup_summary.md`
- `paper_main_full_metrics.csv` / `paper_main_full_metrics.md`
- `paper_policy_ablation_event_f1.csv` / `paper_policy_ablation_event_f1.md`
- `paper_adaptive_timeeventsynth_vs_baselines.csv` / `paper_adaptive_timeeventsynth_vs_baselines.md`
- `paper_best_timeeventsynth_vs_baselines.csv` / `paper_best_timeeventsynth_vs_baselines.md`
- `paper_adaptive_timeeventsynth_vs_random_win_table.csv` / `paper_adaptive_timeeventsynth_vs_random_win_table.md`
- `paper_best_timeeventsynth_vs_random_win_table.csv` / `paper_best_timeeventsynth_vs_random_win_table.md`
- `paper_compatibility_filter_summary.csv` / `paper_compatibility_filter_summary.md`
- `paper_threshold_diagnostics.csv` / `paper_threshold_diagnostics.md`
- `paper_threshold_tradeoff_summary.csv` / `paper_threshold_tradeoff_summary.md`

## Missing Or Not Reconstructable From Saved Outputs
- `threshold_tradeoff_curve.csv or threshold_tradeoff_summary.csv`
