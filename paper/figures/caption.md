# Figure captions

Captions are kept out of the SVGs so the artwork can be submitted as-is and the
journal typesets the caption. The text below is the authoritative wording; the
inline captions in `../draft.md` mirror it.

Two of these serve sections tagged `[UNDER TEST]` in the draft — Figures 3 and
7 — so their numbers, and therefore their captions, are expected to change.

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

## Pending

**Figure 8 (qualitative).** Not yet generated: the panels must come from the
final checkpoint. Planned as rows of held-out scenes — static indoor, indoor
with a moving person, driving — against columns of RGB, activity mask overlay,
our prediction, a comparable-size baseline, a large baseline, and ground truth.
The mask column does work no other figure does: it shows the detector firing on
the moving object and nowhere else.
