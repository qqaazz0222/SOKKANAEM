# Figures for paper/draft.md

Status of the figure set, and what is still missing. The draft currently has
three figures; a results-heavy paper of this shape needs roughly seven, and
five of the results that carry the argument are still presented only as tables.

The test applied below is not "would a picture be nice" but **does the claim
live in a shape** — a curve, a distribution, a spatial artefact. Where the claim
is two numbers, the table stays.

## Have

| # | File | Section | Note |
|---|---|---|---|
| 1 | `fig1-pipeline.svg` | 3.1 | Fine as is. |
| 2 | `fig2-delta-gating.svg` | 3.3 | Fine as is. |
| 3 | `qualitative/01_tum_0_active3pct.png` | 5.1 | Placeholder. One source, no baseline column — see F6. |

Ten qualitative panels exist under `qualitative/` (two per source, from
`scripts/viz.py`), each RGB | prediction | ground truth.

## Need

Ordered by how much the paper loses without them.

### F5 — Streaming drift sawtooth `[highest value]`

**Shows.** AbsRel against frame index within a 32-frame clip, TUM and Bonn, with
the keyframe at frame 30 marked. Second panel: refresh period against AbsRel and
t-delta on twin axes, showing the two trade against each other.

**Why not a table.** Section 5.7 is the paper's most distinctive finding and the
one a reader is most likely to disbelieve. It is *literally a sawtooth* — a
nine-column table asks the reader to reconstruct a shape that one line would
show. The keyframe recovery at frame 31 is the whole argument and is nearly
invisible in Table 7.

**Data.** Complete. Tables 7 and 8; regenerate with `scripts/frame_index_probe.py`.
**Effort.** Plotting only. **Caveat.** Section is `[UNDER TEST]`; redraw after the
long-clip run.

### F3 — Activity–accuracy trade-off

**Shows.** Activity on x, AbsRel on y, one curve per domain, constant-depth
control as a horizontal reference. The default operating point marked.

**Why not a table.** This is the paper's central claim — computation falls
thirteenfold for 13% relative error — and a trade-off curve is the natural form
for it. Table 1 also hides that the real curve is nearly flat until ~30%
activity and then bends, which is the part that matters operationally.

**Data.** Complete (Table 1). **Effort.** Plotting only.

### F4 — Where the model sits against the comparison group

**Shows.** Parameters (log x) against AbsRel (y), one point per model, marker
size or colour encoding t-delta. Both domains as two panels.

**Why not a table.** Table 3 has 8 rows x 8 columns across two sub-tables. The
three-way position — last on accuracy, first on stability, smallest by 6-82x —
is a shape, and no reader extracts it from that table on first pass.

**Data.** Complete (Table 3). A Korean-labelled version exists at
`../../reports/20260818/fig3-pareto.svg` and needs English labels plus the four
models added since. **Effort.** Redraw.

### F6 — Qualitative comparison `[replaces F3 placeholder]`

**Shows.** Rows: two real indoor scenes (one static-camera, one with a moving
person) plus one driving scene. Columns: RGB, activity mask overlay, ours, one
comparable-size baseline, one large baseline, ground truth.

**Why not a table.** No table can show that our prediction is smooth where the
ground truth has structure, which is the visual signature of Section 6.5's range
compression and Section 6.4's ceiling gap. The activity-mask column also does
work no other figure does: it shows the detector firing on the moving object and
nowhere else.

**Data.** Ours and GT exist. **Baseline columns need inference runs** — the
baseline scripts currently score but do not save predictions.
**Effort.** Largest of the set. Add a save path to `eval_baseline_da2.py`.

### F7 — Range compression

**Shows.** Predicted against ground-truth disparity distribution on Bonn and
TUM, as overlaid densities or a quantile-quantile plot.

**Why not a table.** Table 9 gives one ratio per source. The claim is that the
whole *distribution* is narrowed, and a Q-Q plot shows immediately whether the
compression is uniform or concentrated in the tails — which the ratio cannot
distinguish and which points at different fixes.

**Data.** Needs a short inference pass to dump per-pixel values (a few minutes).
**Effort.** Small. **Caveat.** Section is `[UNDER TEST]`.

## Optional, appendix

- **GMC against pixel gating, as curves** (Table 6). Section 5.5 argues
  explicitly that "the fair comparison is the activity-accuracy curve" and then
  prints a table. Two curves would make the argument directly. Data complete.
- **Latency before and after the fused kernel** (Section 5.6 table). A Korean
  version exists at `../../reports/20260818/fig2-latency.svg`. The table is
  compact and readable, so this is presentation polish rather than a gap.
- **Ceiling visualisation** (Section 6.4). Ground truth, ground truth pushed
  through the patch-token bottleneck, and our prediction, side by side. Makes
  "two to three times above our own ceiling" concrete. Needs a small render.

## Not needed

- Token-drop ablation (Table 4): two rows, and the surviving difference is in
  temporal metrics, which a still frame cannot show.
- Design-decision ablations (6.3): three independent scalar comparisons.
