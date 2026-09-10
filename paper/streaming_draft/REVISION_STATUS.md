# Fixed-view selective-refresh draft status — 2026-09-10

This is a new-method draft, not a declaration that all native-paper R1/R2/R3 comments are resolved.
The old submission source, native checkpoint and reserved final remain unchanged.

| Required draft task | Status | Evidence / limitation |
|---|---|---|
| Fix method and claims | Done | `frozen_candidate.json`; MLP q50, exact threshold and checkpoint hashes. No SSM-memory or new-backbone claim |
| Fix application scope | Done; validation pending | `FIXED_CAMERA_SCOPE.md`: indoor fixed-view camera, dynamic objects remain in scope. Post-hoc narrowing disclosed; not a real-time performance claim |
| Audit camera motion | Done | `CAMERA_POSE_AUDIT.md`: 11/12 development clips exceed diagnostic motion rule, 1 has insufficient pose coverage. No clip certified fixed-mounted |
| Find public fixed-camera RGB-D | Next candidate reviewed; not admitted | `BEHAVE_VALIDATION_PLAN.md`: registered depth, calibration and raw continuous video documented; license confirmation, acquisition, timestamp/decoder checks and physical-location grouping pending |
| Execute first fixed-camera data admission | Completed; depth not admitted | `FIXED_CAMERA_VALIDATION.md`: 160 paired frames, fixed-mount support, original RGB matches. Original/derived depth-range discrepancy persists; no GT scores or quality-gate decision |
| Fixed-camera RGB execution pilot | Completed, not accuracy validation | 3 repeats, dense/hold K2/flow K2/MLP; MLP mean 3.239 ms but larger Q0 output difference and higher p95 than dense. 624 refresh/1280 repeated-frame exact checks; tests107 |
| Complete-reference admission round | Completed; not admitted | Original URFD RGB/depth 160/160 acquired; RGB pixel-exact; C1 fails, C3/masked C4 pass, unmasked C4 fails. Three alignment attempts do not corroborate registration |
| Review admission overclaims | Corrected in current draft | `FIXED_CAMERA_REVIEW.md` supersedes broad scale-error-exclusion and impossibility claims; historical code/results/report preserved |
| Known-shift instrument calibration | Completed; not cross-modal verification | 27/27 shifts recovered, constant field ambiguous. Does not admit GT or change the model |
| Delivery/audit integrity | Updated | Actual C1/C3/C4 flags, repository-layout metadata bundle, Markdown links checked within bundle, prior delivery hashes compared before/after |
| BEHAVE input preparation | Completed; real-data checks pending | `BEHAVE_INPUT_READINESS.md`: lossless packed-depth decoding, one-to-one timestamp associations, metadata-only acquisition command; no dataset GET or GT scores |
| Current project regression suite | Passed | 299 tests; includes instrument, delivery, BEHAVE input and metadata regressions. Not a GT-quality or submission-approval result |
| Audit unused real scenes | New local sequence evaluation completed; quality transfer fails | Initial roots had no eligible scenes. Subsequently acquired Freiburg1 room/Freiburg2 desk after candidate freeze; 1,152 frames, no retuning. L32 aggregate overshoot +2.285% fails; L256 passes. DA2 pretraining independence unverified. See `INDEPENDENT_VALIDATION.md` |
| Lock numeric evidence | Done | `evidence_table.json`; primary, repeated-timing and shifted results kept distinct |
| Matched qualitative evidence | Done | Four-scene midpoint-nearest skipped frames, fixed central crops, and explicitly post-hoc worst raw-error-increase cases. Not independent validation |
| Method-centered prose and figures | Draft done | New `paper/draft.md`, Figure 1 pipeline, accuracy/speed/ablation tables and failure discussion |
| Native/final preservation | Done | `paper/draft_native_20260909.md`, unchanged original submission and final artifacts |
| New-method mathematical reasoning | Drafted, not independently approved | Fixed nearest-sampling infinity-norm bound and additive cost accounting; neither implies GT quality certification |
| Modern video baseline and temporal metric | Pending | Related work is not a completed numerical comparison. No candidate-specific TCE claim |
| Broad generalization / seeds / long horizon | Pending | Reused four scenes, one seed, one initialization shift. Do not inherit native-model final/seed results |
| Nano B01 / Pi 4 | Pending for new method | Shared RTX4090 timing is not new 5W/10W or energy evidence |
| Author, rights, bibliography, venue | Pending | Author supplies identity/declarations. Full bibliography, current venue rules and independent novelty review remain required |

## How prior comments map to the new draft

- R1-M2 (configuration consistency): new candidate is explicitly fixed; no replacement of the old final contract.
- R1-M3 (video baseline): still pending, not resolved by citing VDA/oVDA.
- R1-M4 / R2-M1 (temporal overclaim): no temporal-stability superiority claimed for this method.
- R1-M5 / R2-m3 (cost versus speed): actual timings, strong K2 control and measurement boundary reported.
- R1-M6 (novelty): contribution narrowed; independent assessment still pending.
- R1-M8 / R2-M2 (statistical reliability): no significance/equivalence or independent-scene claim. Broader evidence pending.
- R1-m12 (qualitative results): actual candidate-linked plots now provided; this does not retroactively reclassify the native review item.
- R1-m15 / R2-m12 (authors and access): remain author-confirmation items.

The existing `paper/self-revision/r3/revision-status.md` applies to an earlier native manuscript and contains stale title/figure references. It must not be cited as current approval of this new draft. This mapping supersedes it only for the selective-refresh draft's task status, without deleting historical comments.

## Next action before stronger claims

The newly acquired moving-camera holdout has been evaluated without retuning and does not support uniform quality preservation. Keep the failure in the draft as a transparently re-scoped stress test, not a removed result. The next action is BEHAVE license confirmation, metadata-led acquisition and fixed-camera admission before frozen-candidate GT scoring; the user has no separately recorded RGB-D data. Any risk-target or scene-cut-policy revision must be a separately logged development experiment and needs another fresh holdout for validation. This draft is suitable for author review, not an automatic submission decision. Historical 98/103/107 counts refer to narrower suites at those stages; the current project command is `python -m pytest -q`, restricted to `tests/` by pyproject configuration. Software tests are not scientific acceptance criteria.
