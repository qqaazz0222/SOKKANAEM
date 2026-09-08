# Historical VDA evidence audit — 2026-09-07

## Decision

Keep historical VDA evidence visible and separate; do not merge it into the eight-model
frozen final table. Restore the relevant literature and mark R1-M3 **partial**, not resolved.
No new baseline inference or final-condition modification was performed.

The original [Video Depth Anything paper](https://arxiv.org/abs/2501.12375) introduces a
spatial–temporal depth head and long-video inference. [Online VDA](https://arxiv.org/abs/2510.09182)
is a separate online formulation. Their published performance is not our matched measurement.

## Input access / causal-label correction

`scripts/eval_baseline_vda.py` calls `infer_video_depth` with `input_size=256`, FP32 and
`target_fps=8`. The currently available upstream checkout's `video_depth.py` sets
`INFER_LEN=32`, `OVERLAP=10`, stacks a full chunk before forwarding, and subsequently
aligns/blends chunks. `motion_module/motion_module.py` applies temporal attention without
a causal mask (`assert attention_mask is None`). Thus this audited execution path allows
future-frame access within a chunk; it must not be labeled frame-causal. The wrapper's old
causal docstring is not evidence of causal execution. We preserve historical code intact
and record the correction here instead of rewriting a past experiment's source history.

The current checkout was inspected, not proven identical to the historical executing revision.
That missing linkage is a further reason not to assert historical causal behavior. Upstream
source: [official implementation](https://github.com/DepthAnything/Video-Depth-Anything/blob/main/video_depth_anything/video_depth.py).

## Quantitative evidence retained, not promoted

The two historical JSON files named
`work_dirs/baselines/video-depth-anything-small-metric-28-4m-{tum,bonn}-l256.json`
contain metric arrays for 2 TUM and 11 Bonn clips. The arrays lack explicit frame/clip IDs,
executing source commit and weight SHA. Equal array lengths cannot prove matched frame order.
The wrapper applies one median depth scale using all valid clip pixels before scoring;
this is not the new final table's common clip disparity scale–shift gauge.

The old `work_dirs/r2-bootstrap.log` reports these **historical, source-balanced clip means**:

| Condition | AbsRel | Raw temporal delta | OPW | TCE |
|---|---:|---:|---:|---:|
| Reported native v11 K30 | 0.1962 | 0.0692 | 0.0265 | 0.0341 |
| Historical VDA Small metric | 0.1297 | 0.0854 | 0.0219 | 0.0276 |

These values are transcribed for provenance, **not a newly validated matched ranking**.
Their old source-balanced clip estimator differs from the older table's source pixel-pooled
AbsRel and the current four-sequence estimator. The old bootstrap checks array lengths,
not frame identities, and resamples clips within sources without accounting for sequence
correlation. Its win fractions are not null-distribution p-values; intervals including a
ratio of one are not equivalence tests. We do not carry its significance claims forward.

## What is needed for a defensible new comparison

Pin weights and dependency/source versions; record exact RGB/depth IDs, input resolution,
past/future frame access and reset/overlap policy; score all declared clips under common
gauge/masks; retain failures; compare at the sequence level. Online and offline paths must
be identified separately, and matched latency must include their actual buffering/output
boundary. An extension after viewing the existing final set is post-hoc, not a new blind
evaluation. New model selection requires a separately untouched evaluation set.
