# R3 correction and evidence notes — 2026-09-07

The authoritative manuscript is `manuscript.tex` / `manuscript.pdf`; `paper/draft.md` is now
its full synchronized Markdown edition with five figures. The previous English draft is
preserved as `paper/draft_legacy_20260907.md`, and `paper/draft_ko.md` remains historical.
The complete [40-item response matrix](../self-revision/r3/revision-status.md)
supersedes the R1/R2 self-assessments. Those reviews were internal simulations, not journal decisions.

## Priority corrections

| Earlier claim | Correct interpretation | Evidence / disposition |
|---|---|---|
| Table 7d describes the final model | It describes v9 versus v10, not frozen v11 | `work_dirs/r1-ls1024.log`; archived captions/rows corrected |
| L1024 clip/per-frame columns decompose the same error | PointOdyssey score aggregates 20 clips; frame curve aggregates 8 | Same log; neither difference nor ratio is an additive scale/shape decomposition |
| Nearly unchanged endpoints establish stable shape throughout | v10 curve has a 1.0870 intermediate value at frame 448 despite 0.2700/0.2730 endpoints | Same log; continuous stability claim withdrawn |
| Metric deployment avoids scale drift | Units alone do not prevent prediction-scale drift; GT frame fitting can hide it | New `revision_results.tex`, all eight frozen models, no new inference |
| Bootstrap interval crossing 1 confirms equal accuracy | Non-rejection is not equivalence; no equivalence margin/test was specified | Archived R2 status corrected; current uncertainty uses four sequence units |
| Bootstrap win fraction is a significance probability | It is a resampling statistic, not a null-distribution p-value | Legacy script has no frame-ID pairing check and clips share sequences; do not carry old significance into current manuscript |
| Every historical baseline except DA3 is causal | The audited VDA path is 32-frame offline, unmasked temporal attention | `VIDEO_BASELINE_AUDIT.md`; old causal designation withdrawn |
| All main evidence survived integration | VDA, qualitative frames, continuous L1024 and some diagnostics did not retain current-model provenance | Explicitly partial/excluded in R3, not silently marked resolved |

`scripts/table_check.py` checks numeric occurrences in logs; passing it does not establish
one-run provenance, matching frame identities, checkpoint identity or an appropriate estimator.
The new report verifies frozen source/result hashes and exact final manifest coverage first.
The original final contract, predictions, selection and scoring have not been changed.

## What the new scale panel supports

Scale statistics were already stored; R3 adds a **post-hoc descriptive** presentation, not
a preregistered primary endpoint. All eight models and both frozen lengths are retained.
Clip means are averaged within each sequence, then equally across four sequences; failed
gauge fits remain included. Native metric outputs use no GT fit, relative outputs use
their median-gauge inverse-disparity depth. Affine disparity fits are not used here.

K30 L256 scale CV is 0.2185 versus 0.1892 for MiDaS; mean absolute log-scale step is
0.0180 versus 0.0559. These measure different behavior, not a single stability ranking.
Intervals are exploratory, small-sample and unadjusted. L8/L256 partly differ in footage
and reset scheduling, so their difference is not an isolated history-length intervention.
Scale medians can also change with local shape error: this is a diagnostic, not a pure
latent global-scale decomposition. Current 1024 diagnostic frames are four L256 clips,
**not one continuous 1024-frame trajectory**.

## Submission gate

Automatic document/evidence work cannot complete independent mathematical review, author
approval/declarations, novelty/venue-fit judgment, or redistribution-rights confirmation.
A modern matched video-depth comparison and current-weight long-horizon/qualitative evidence
remain scientific strengthening items. The narrowed conditional-analysis manuscript makes
no broad video-depth superiority claim; removing that claim is not completing those experiments.
The final set has been opened. Any new model development needs a new untouched evaluation set;
an extension on the current set must be explicitly labeled post-hoc, not a new blind test.
