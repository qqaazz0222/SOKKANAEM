# Frozen-candidate new-sequence validation — 2026-09-09

The MLP q50 weights and threshold were fixed before acquiring these sequences. No retraining, retuning, scene replacement or use of the old reserved final occurred.

## Scope and pre-registration

- Newly acquired official TUM Freiburg1 room and Freiburg2 desk. These sequence identifiers were absent from searched local configs, manifests, paper, scripts and text experiment history before preparation.
- This supports separation from the recorded local Q0 calibration and policy-selection pipeline, not certified disjointness from Depth Anything V2 pretraining or every possible historical external experiment.
- Two new sequences are not a broad statistical generalization benchmark. Camera motion and acquisition environment differ from the reused development clips.
- Before prediction: L32 first/last32 associated frames per sequence; L256 windows centered at one-third/two-thirds. RGB/depth paths are disjoint across selected windows. Same256px loader/scorer, Q0 internal518px, DIS128px.
- Source: [official TUM download page](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download). Archive URLs and SHA256, per-frame hashes, manifests and pre-registration are preserved in the experiment directory.
- Depth conversion uses PNG value/5000. Camera-specific scale corrections are already applied by the dataset; no extra1.035/1.031 factor is applied. [Official format and calibration documentation](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats).

## Aggregate results

| Length | Policy | Raw AbsRel | Edge AbsRel | Boundary F1 | Flat-TV change vs Q0 | Screen | Mean ms | Skip |
|---|---|---:|---:|---:|---:|---|---:|---:|
| L32 | dense | 0.283891 | 0.298985 | 0.606404 | +0.000% | pass | 7.550 | 0.00% |
| L32 | output_k2 | 0.284293 | 0.299161 | 0.609204 | +0.012% | pass | 4.388 | 50.00% |
| L32 | mlp_q50 | 0.284206 | 0.298418 | 0.609718 | +0.377% | FAIL | 3.645 | 64.06% |
| L256 | dense | 0.212614 | 0.248874 | 0.583142 | +0.000% | pass | 7.534 | 0.00% |
| L256 | output_k2 | 0.212594 | 0.249759 | 0.586912 | +0.290% | pass | 4.383 | 50.00% |
| L256 | mlp_q50 | 0.212941 | 0.249821 | 0.587015 | +0.363% | pass | 4.260 | 54.98% |

Times are single-run synchronized shared-RTX4090 resident-input measurements, including internal policy/flow/transfers but excluding initial I/O/H2D. They are not repeated timing estimates or edge-device evidence.

## Sequence-specific diagnostics

| Length | Sequence | Policy | Raw AbsRel | Boundary F1 | Flat-TV change | Failed checks |
|---|---|---|---:|---:|---:|---|
| L32 | rgbd_dataset_freiburg1_room | output_k2 | 0.348568 | 0.524096 | -0.507% | overshoot_within_1pct |
| L32 | rgbd_dataset_freiburg2_desk | output_k2 | 0.220019 | 0.694313 | +0.433% | none |
| L32 | rgbd_dataset_freiburg1_room | mlp_q50 | 0.348455 | 0.525745 | -0.716% | overshoot_within_1pct |
| L32 | rgbd_dataset_freiburg2_desk | mlp_q50 | 0.219956 | 0.693690 | +1.262% | flat_tv_within_1pct |
| L256 | rgbd_dataset_freiburg1_room | output_k2 | 0.302901 | 0.467073 | -0.141% | none |
| L256 | rgbd_dataset_freiburg2_desk | output_k2 | 0.122287 | 0.706751 | +0.678% | none |
| L256 | rgbd_dataset_freiburg1_room | mlp_q50 | 0.302791 | 0.468325 | -0.199% | none |
| L256 | rgbd_dataset_freiburg2_desk | mlp_q50 | 0.123092 | 0.705704 | +0.869% | none |

## Interpretation

- Aggregate MLP screen on both lengths: **FAIL**.
- All sequence-specific MLP diagnostic screens: **FAIL**.
- A quality-screen failure is retained even when latency improves. No acceptance threshold is revised after this test.
- Passing tolerances would not establish statistical equivalence, temporal stability or all-scene safety. Failure means the earlier development claim cannot be generalized to these new sequences without qualification.
- Exact Q0 refresh comparisons: 1083; all zero difference. Input/code/checkpoint hashes unchanged.
- This holdout is now consumed. If its results inform future training or threshold selection, it becomes development data for that future model.

## Reproduction

`scripts/qindependent_prepare.py`, `scripts/qindependent_eval.py`, `scripts/report_qindependent.py`.
`work_dirs/qindependent_20260909/preregistered.json`, `prepared.json`, `evaluation/results.json`, `evaluation/integrity.json`.
No native-model final or original submission source was changed.
