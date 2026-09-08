# Qualitative extension protocol — 2026-09-08

This is post-hoc visualization of the already-opened frozen L256 evaluation,
not a new independent test or a model-selection experiment. No training,
threshold change, test-score replacement, or synthetic prediction image.

- Select the first manifest clip of each of the four reserved TUM sequences:
  `tum:0`, `tum:4`, `tum:7`, `tum:10`. Include all four regardless of quality.
- Replay all 256 frames, with state reset only at the clip boundary, using the
  frozen native checkpoint and existing adapters: sparse K30, dense carry,
  MiDaS Small 256, and output hold. Verify each raw prediction checksum against
  its previously recorded final result before publishing visual assets.
- Spatial panel: frame index 127 (zero-based) in all four clips, RGB / GT /
  SOKKANAEM / dense carry / MiDaS / output hold. No outcome-based frame search.
- Temporal panel: first walking-xyz clip (`tum:7`), frames 28–32, including
  scheduled keyframe 30. Add the actual post-fallback native activation mask.
- Supplement: all 256 frames of every selected clip, synchronized comparison,
  10 fps presentation speed (not inference throughput or source capture rate).
- Use the existing single disparity scale–shift fit per whole clip/model,
  never a per-frame fit. This is GT-assisted relative-shape visualization,
  not calibration-free metric prediction or evidence of causal alignment.
- Fix the depth color scale to 0–5 m across every frame/model/clip. Render
  invalid GT as gray. Saturate values beyond this display range only in the
  color image; retain raw predictions and disclose clipping fractions.
- Record input/frame IDs, hashes, alignment coefficients, invalid-fit flags,
  mask activity, framewise valid-GT errors and display clipping fractions.
  Report limitations and adverse behavior; do not infer speed from video playback.

The protocol is written before this extension's prediction-map generation.
The previous aggregate results were already visible when it was chosen.
