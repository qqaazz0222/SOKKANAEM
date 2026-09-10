# Fixed-camera evidence review and instrument calibration — 2026-09-10

This addendum supersedes interpretations, not experimental results, in the preserved
[second admission report](FIXED_CAMERA_REFERENCE_ADMISSION.md). Historical code,
protocols, failure records and model weights have not been changed.

## Corrected conclusions

| Issue | Supported conclusion | Unsupported extension removed from the current draft |
|---|---|---|
| Original-versus-derived values | All-frame C1 fails; masked C4 passes, unmasked C4 fails | All reference-value checks passed; every unit/scale error excluded |
| Source-range plus erosion mask | Researcher-defined mask using source quantities | Official publisher validity mask or proof of correct depth |
| Three alignment attempts | They did not corroborate alignment under their respective rules | Three independent proofs that the dataset cannot be checked |
| Fixed-support NMI | The last attempt fixes sample positions but still recomputes depth histogram limits per shift | All estimator comparability issues were removed |
| Boundary optima | Five control sweeps and one corrected sweep hit the boundary; the declared void rule applies | Six control failures; a boundary maximum necessarily invalidates an unregistered control |
| Low median shift response | A setting-dependent descriptive statistic | A general impossibility theorem or validated exclusion of ten sequences |

The first edge test translates a fixed set of depth-edge samples. The later
changing-support critique should not be assigned to it without demonstrating that
specific defect. A deliberately unregistered control can have its best alignment
outside a small search window. Neither observation reverses the recorded failures.

The relief ratio uses horizontal displacement, a pixel median and 32 bins across
the p01–p99 range. Changing to 128 bins alone multiplies that ratio by four. It can
hide local depth boundaries and is not the same range used by the earlier NMI code.
The historical diagnostic also does not explicitly reject all image borders before
negative indexing; this is an implementation risk, not a quantified effect here.

## New known-shift experiment: completed, not GT admission

Entry point: `scripts/check_registration_instrument.py`.
Records: `work_dirs/qregistration_instrument_20260910/protocol.json` and `results.json`.
The protocol and input/code hashes were written before running the sweeps.

- Fixed 32-bin edges throughout each sweep, fixed pixel support/sample counts,
  explicit constant-invalid border handling, no circular translation.
- Search ±8 pixels in both axes; injected (dy, dx) = (0, 0), (3, −2), (−4, 4).
- Eight existing depth frames (1, 21, …, 141) and one synthetic textured image.
- **27/27 exact unique-peak recoveries**, including **24/24 real-depth self-shifts**.
- A constant-field negative control has 289 tied shifts and is correctly labelled
  ambiguous, rather than being assigned a meaningful translation.

This demonstrates that these depth images contain usable *same-modality* shift
information for this instrument. It does not establish cross-modal RGB/depth
alignment, robustness to realistic sensor noise, or validity of the corrected
depth reference. Self-alignment is an intentionally limited positive control.
It neither admits URFD/SBM GT nor replaces an independent scientific review.

## Current decision and next work

No fixed-camera GT accuracy or quality-preserving acceleration claim is added.
The moving-camera quality failure and fixed-camera RGB-only trade-offs remain.
We will not retune the frozen MLP against the failed holdout.

The next data route is [BEHAVE feasibility and acquisition plan](BEHAVE_VALIDATION_PLAN.md).
It provides a more explicit registered-depth/calibration path, but remains a
candidate until licensing, acquisition, decoding, synchronization and location
grouping are resolved. Low relief alone will not exclude a candidate.

This review also separates software consistency from scientific admission:
the delivery audit now reports the actual C1/C3/C4 statuses, rather than claiming
that all value checks passed. Root pytest discovery is restricted to `tests`,
excluding archived and vendored test suites from the project test command.
