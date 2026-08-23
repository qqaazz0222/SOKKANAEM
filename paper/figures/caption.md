# Figure captions

Captions are kept out of the SVGs so the artwork can be submitted as-is and the
journal typesets the caption. The text below is the authoritative wording; the
inline captions in `../draft.md` mirror it.

No section of the draft carries `[UNDER TEST]` any more, so these numbers are
the reported ones. They are still copies: `make_figures.py` holds its own copy
of each table, and a table re-measured without updating that copy leaves the
figure quietly stale — which has happened, and is why the figures are checked
against the draft whenever a table moves.

---

**Figure 1.** Streaming pipeline. A change detector compares consecutive frames
patch-wise and emits a binary activity mask, which reaches the backbone as the
\(\Delta\)-gating signal. Static patches retain their hidden state exactly and
their computation is skipped. Two caches — spatial outputs and temporal hidden
state — are where the reduction in multiply–accumulate operations comes from,
taking a frame from 1.644 to 0.608 GMAC at 15.4% activity. The global motion
compensation branch is used only for moving cameras. Frames above 40% activity
are routed through the dense path instead.

**Figure 2.** Exact \(\Delta\)-gating. The activity mask multiplies the
discretization step, \(\widetilde{\Delta} = M\Delta\). A changed patch takes the
standard selective-SSM update. A static patch takes \(\bar A = I\) and
\(\bar B = 0\), so its hidden state is copied rather than reconstructed:
skipping the computation is not compensated for, it is algebraically identical
to preserving the state. Early exit and token dropping instead substitute zero
or an approximation, and that error accumulates across frames.

**Figure 3.** Activity against accuracy on the real indoor and synthetic
holdouts, sweeping the detector threshold. Accuracy is nearly flat until
roughly 30% activity and then bends. On real footage, cutting computation
thirteenfold costs 13% relative AbsRel, and the default operating point (circled)
costs 2.8%. The per-clip optimal constant-depth control lies far above both
panels and is marked off scale, so the vertical axis can resolve the curve the
panel exists to show.

**Figure 4.** Accuracy against temporal stability across the comparison group,
with marker area scaling as the logarithm of parameter count. Down and to the
left is better on both axes. Our model is the smallest marker in each panel,
lowest on the stability axis and rightmost on accuracy. The stability axis is
logarithmic because t-delta spans two orders of magnitude across the group: the
gap to the next best model is a factor of 1.35 on real footage and 4.2 on
synthetic.

**Figure 5.** Pixel gating against global-motion-compensated feature gating on
real driving footage, swept as curves. The two strategies score change on
different scales, so only the curves are comparable and points at equal
thresholds are not. At matched activity the compensated variant is better on
both axes, and it reaches 14% activity while still beating pixel gating at 51%.

**Figure 6.** Per-frame latency before and after the fused scan kernel, measured
at 22% activity on one RTX 4090 at 256 pixels, batch size one, fp32. Every path
became faster and the ordering inverted: what sparsity was saving was the scan,
and the scan is now nearly free, leaving the sparse path with bookkeeping that
does not scale with activity.

**Figure 7.** Accuracy decays between keyframes. Panel (a) scores by frame index
within a 32-frame clip, with each frame aligned independently. Carried state
does not accumulate accuracy: the dynamic-object source degrades 61% from frame
0 to frame 28, and the recovery at frame 31 is the keyframe firing at frame 30.
Panel (b) sweeps the refresh period, showing that the remedy is already in the
architecture and merely applied too rarely, and that a period below 10 forfeits
the stability lead the model is built for.

---

**Figure 8.** The sawtooth, as pictures. One 32-frame clip of the dynamic-object
holdout; rows are frames 24, 28, 29, 30 and 31; columns are RGB, prediction,
ground truth and relative error, where black is invalid ground truth. Frame 30
is the keyframe: activity goes to 100% and clip AbsRel falls from 0.3740 to
0.2030 in one frame, visible as the error column darkening over the moving
person. The prediction column is also where range compression (Section 6.5)
shows without a histogram — it is uniformly flatter than the ground-truth
column that shares its colour scale.

---

**Figure 9.** Qualitative comparison on three held-out scene types, last frame
of an eight-frame clip. Columns: RGB, the detector's activity mask, our
prediction, Depth Anything V2 Small (24.8M), DPT-Large (343M), and ground
truth. In the mask column, patches the detector selected keep their brightness
and skipped patches are dimmed, with the boundary outlined; the percentage
beside each row label is the detector's mean selection over the clip. Every
model is aligned by its own native rule — per-clip median scaling for ours,
least-squares scale and shift in disparity space for the two relative
baselines (Section 5.4) — and all depth tiles in a row share one colour range
taken from that row's ground truth, so a wrong prediction cannot be rescued by
its own scaling.

Three things are visible here that no table in this paper shows. The detector
fires on the walking person and the trolley and almost nowhere else in the
indoor dynamic scene, while the static scene lights almost nothing: this is the
mechanism working as designed. Our prediction is markedly smoother than the
ground truth has structure, which is the visual signature of the range
compression of Section 6.5 and the ceiling gap of Section 6.4, and it is the
honest counterpart to our position in Table 3 — we are not competitive on
detail with either baseline shown. And the driving row lights 80% of the field,
the transfer failure quantified in Section 5.6: on real ego-motion the pixel
threshold selects nearly everything.

Two caveats on reading the figure. The activity mask is the detector's own
selection; above the 40% dense-fallback threshold the shipped model overrules
it and computes the frame densely, so the compute actually paid is higher than
the number shown (17.9%, 49.9% and 100% for the three rows against the
detector's 9.8%, 34.4% and 79.6%). And KITTI's ground truth is projected LiDAR,
valid on 18.1% of the pixels in that row, which is why the bottom-right tile is
mostly black.

---

## Pending

None. Figure 8 was the last one outstanding.
