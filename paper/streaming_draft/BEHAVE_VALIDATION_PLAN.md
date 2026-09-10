# BEHAVE fixed-view validation candidate — 2026-09-10

Status: primary-source feasibility reviewed; raw data not acquired, not admitted,
no model predictions generated. User confirmation of non-commercial use and
license conditions requested. This document is an acquisition/validation plan,
not a completed benchmark or retroactive change to the URFD acceptance rule.

Subsequent progress: [input readiness report](BEHAVE_INPUT_READINESS.md). Public
source revisions and HTTP metadata have been recorded, and decoding/timestamp
tools have been tested on synthetic fixtures. Actual dataset admission is pending.

## Why this candidate

BEHAVE contains 321 sequences captured with four RGB-D cameras across five
environments. Cameras/views of the same event must not count as independent
locations. Date labels alone do not establish distinct rooms.
[Official project](https://virtualhumans.mpi-inf.mpg.de/behave/).

The publisher offers raw RGB and registered-depth video, timestamp files and
calibration archives. Date02 alone is listed as 5.4 GB RGB plus 46.4 GB depth;
acquisition should therefore select complete matched members when possible,
rather than downloading all dates without a storage/transfer plan.
The license restricts use to non-commercial scientific research, requires blurred
faces in published imagery and restricts redistribution. Cite Bhatnagar et al.,
CVPR 2022. Do not include raw video or calibration data in manuscript deliveries.
[Official download and license](https://virtualhumans.mpi-inf.mpg.de/behave/license.html).

The official repository explicitly identifies depth images as aligned to color
and supplies per-location intrinsics/extrinsics. This is stronger documentation
than an independent intensity-matching heuristic, but is not an assurance that
every sensor pixel is correct. Fixed mounting still needs sequence-specific
source/calibration/background evidence.
[Official reader and structure](https://github.com/xiexh20/behave-dataset).

The reader uses uint16 depth and separate RGB/depth timestamps converted from
microseconds. Use timestamp associations with unique source indices; avoid
counting nearest-frame duplicates as genuine 30 Hz input. Audit first-frame
indexing against an independent decoder, not just the helper's nominal fps.
[Video reader](https://github.com/xiexh20/behave-dataset/blob/main/data/video_reader.py).
The calibration code converts stored depth to metres by dividing by 1000 and
treats zero as invalid. Do not use an 8-bit preview decoder for depth.
[Depth conversion](https://github.com/xiexh20/behave-dataset/blob/main/data/kinect_calib.py).

## Ordered execution and acceptance requirements

1. Confirm permitted use; snapshot official reader revision and source notices.
   Inspect timestamp/calibration/archive manifests before selecting clips.
   No prediction-dependent sample selection.
2. Establish physical-location groups from calibration and source documentation.
   Reserve entire new locations; different dates, cameras and objects in the same
   room are not independent locations. If only one is acquired, label it a pilot.
3. Freeze a separate data protocol before inference: sequence IDs, camera IDs,
   timestamp association tolerance, sampling windows, validity and boundary rules,
   exclusions, checksum records and the existing four model controls.
   The tolerance must be justified by actual timestamp cadence, not chosen to
   make a weak pairing rate pass. Keep ordinary continuous RGB cadence for reuse.
4. Acquire complete paired members with hash/size verification. For a partial TAR
   retrieval, distinguish member completeness from whole-archive completeness.
   Decode depth losslessly; check dtype, resolution, counts, timestamps and units.
   Do not silently inpaint holes or align depth to model predictions.
5. Validate fixed mounting and registration documentation plus targeted,
   model-independent correspondence checks. Report residual uncertainty; any
   new admission standard must be explicit and prospective. Same-modality NMI
   recovery alone cannot admit cross-modal GT.
6. Evaluate frozen dense Q0, hold K2, flow K2 and MLP q50 on identical continuous
   frames. Preserve raw/edge AbsRel, boundary F1, overshoot and flat-TV thresholds;
   show per-location results and foreground/boundary subsets where valid masks
   exist. Report missing region annotations instead of inventing ground truth.
7. Add motion-referenced temporal error and long-horizon drift as separate
   diagnostics. Bootstrap independent locations, not correlated frames, when
   enough locations are available; otherwise avoid population-equivalence claims.
8. Only then update fixed-view accuracy claims. Device-specific sustained 30 Hz,
   p95/deadlines/drops and Nano B01 5W/10W / Pi 4 results remain separate work.

The current candidate checkpoint, threshold, development record and failed
moving-camera holdout stay frozen. Training-source overlap will be checked against
local manifests; foundation-model pretraining independence remains unverified.
