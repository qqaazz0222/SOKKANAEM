# Submission gate and contribution audit

Working manuscript, 7 September 2026. **Not ready for submission.** No external submission
or public upload has been performed. The author will add identity and declaration details later.

## Targeted prior-work comparison

| Topic | Primary source | Consequence for this manuscript |
|---|---|---|
| Binary update/copy | [Skip RNN](https://arxiv.org/abs/1708.06834) | Not a first-contribution claim |
| Selective SSM dynamics | [Mamba](https://arxiv.org/abs/2312.00752) | Attribute selection; distinguish exponential transition from first-order input integration |
| Frame-difference sparse computation | [DeltaCNN](https://arxiv.org/abs/2203.03996) | Change-based video reuse is established |
| Token reuse | [Eventful Transformers](https://arxiv.org/abs/2308.13494) | Do not claim first temporal token caching |
| Sparse/spiking SSMs | [SPikE-SSM](https://arxiv.org/abs/2410.17268), [SpikySpace](https://arxiv.org/abs/2601.02411) | Abstract-level comparison only; not a numerical cross-task comparison or exhaustive priority search |
| Depth references | [DPT](https://arxiv.org/abs/2103.13413), [MiDaS](https://arxiv.org/abs/1907.01341), [DA2](https://arxiv.org/abs/2406.09414) | Different model sizes and training; these do not exhaust causal video baselines |
| Fixed-cost limitation | [Amdahl](https://doi.org/10.1145/1465482.1465560) | Additive cost derivation is an application, not a novel universal law |

Provisional contribution: an implementation-specific synthesis of state-copy semantics,
local defect versus history error, cache-age assumptions, spatial-context omission and paid
pipeline costs, supported by same-weight controls and frozen evaluation. Whether this is
sufficiently novel for *Mathematics* remains an editorial and expert judgment.

## Proof and evidence audit

- [x] Separate real-arithmetic preservation, shared-input recurrence and full-network perturbations.
- [x] State uniform-contraction/defect/readout assumptions explicitly; do not infer them from negative A.
- [x] Prove refresh removes same-incoming-state local defect, not accumulated dense-history error.
- [x] Restrict the pixel-MSE cache bound to the first temporal block and applicable gate.
- [x] Keep gauge-fitted shape scores separate from no-fit metric accuracy.
- [x] Check algebra/counterexamples in executable tests and 1024 development diagnostic steps.
- [x] Freeze all eight final comparison paths before reserved predictions; retain K30 afterward.
- [x] Count alignment failures and report n=4 sequence uncertainty without false significance.
- [ ] Independent human mathematical review of every proof and applicability assumption.
- [ ] Author assessment of novelty, venue fit, related-work completeness and English wording.

## Author-only or externally blocked items

- [ ] Names, affiliations, corresponding email, actual contributions and all-author approval.
- [ ] Verified funding/COI statements; applicability of ethics/consent and dataset permissions.
- [ ] Rights to redistribute code, native weights, template assets and derived data; no automatic public release.
- [ ] Public repository/DOI and access arrangements, if required for the actual submission.
- [ ] AI-assistance tool/version details and complete author verification of AI-assisted work.
- [ ] Current journal scope, article type, reporting and submission policies rechecked by authors.
- [x] User confirmed Nano Developer Kit B01 and 5W/10W modes; historical aggregate logs located and audited.
- [ ] Link historical weights/code/config and validate the actual video path; Pi 4 RAM/OS and SSH access only if new runs are needed.

The [official MDPI template](https://mdpi-res.com/data/MDPI_template.zip) was downloaded
and its unmodified class/assets are recorded in `template_source.json`. The Mathematics
instructions page returned access/rate-limit errors during this run; template use is not
a certification of compliance with every current journal requirement.

The complete [R3 40-item mapping](../self-revision/r3/revision-status.md) supersedes the earlier
R1/R2 self-assessments; it is not an external reviewer decision. A matched modern video-depth
baseline, current-weight continuous long-horizon/qualitative/range diagnostics and full-training
replicates remain scientific strengthening items. Their exclusion is not experimental completion.

The conditional model-improvement task was deliberately not run: this manuscript analyzes
the frozen model rather than claiming improved performance. Any development after opening
the reserved data needs a new untouched test. Historical Nano speed/energy evidence exists,
but is not used as final-model real-video superiority evidence; its provenance and measurement
boundary are documented separately in `EDGE_LEGACY_AUDIT.md`.
