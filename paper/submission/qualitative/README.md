# Qualitative replay supplement

Post-hoc results from the already-opened final evaluation. No new training or
model selection. The first L256 clip of each TUM sequence was fixed before
prediction-map generation. All 16 full raw-prediction hashes match the frozen
evaluation. These are actual predictions, not AI-generated illustrative images.

| Video | Sequence | Frames |
|---|---|---|
| [tum_0.mp4](tum_0.mp4) | sitting_xyz | 0–255 |
| [tum_4.mp4](tum_4.mp4) | sitting_rpy | 0–255 |
| [tum_7.mp4](tum_7.mp4) | walking_xyz | 0–255 |
| [tum_10.mp4](tum_10.mp4) | walking_rpy | 0–255 |

Rows in the videos: RGB / GT / SOKKANAEM / dense carry, then MiDaS Small /
output hold / actual SOKKANAEM active mask / presentation information.
Playback is 10 fps (25.6 s per video), **not inference speed or source capture rate**.
Teal masks mean active; pale gray masks mean inactive. Gray GT means invalid.

All depth panels use viridis at a fixed 0–5 m display range. Values above 5 m
saturate visually; raw values and quantitative scores are not clipped.
Each model uses one GT-assisted disparity scale–shift fit over the entire clip,
not per-frame fitting. This is relative-shape visualization, not calibration-free
metric output; fitting uses later GT frames and is not part of causal inference.

[provenance.json](provenance.json) records frame paths/IDs, hashes, fitted
coefficients, framewise errors, clipping fractions and activation telemetry.
Some frames have substantial display saturation, particularly in unobserved
backgrounds; identical colors above 5 m do not establish depth agreement.
The fixed temporal figure's five masks are all active, so it cannot isolate
sparse-cache benefits. The walking examples also show adverse native errors.

Reproduction from the repository root, with the pinned environment:

```sh
python scripts/paper_qualitative.py infer
python scripts/render_paper_qualitative.py
```

Use the separate rendering command above for the validated 16×16 mask layout.
The original inference script is retained byte-for-byte for its audit hash.
Full float32 replay tensors remain under `work_dirs/paper_qualitative_20260908/`;
they are not included in the compact private source archive. Their hashes and
generation code are retained. Dataset/weight rights require review before public sharing.
