# Public RGB-D data screening for fixed-view depth — 2026-09-10

Status: source/format screening, **not completed model validation**. The user has no separately recorded RGB-D footage and requested public alternatives. The frozen Q0/MLP candidate and threshold remain unchanged. Source pages and actual download endpoints were checked on this date.

Subsequent execution: [data admission and RGB-only pilot](FIXED_CAMERA_VALIDATION.md). The full official Shadows archive and 160 paired frames were later acquired. Initial depth admission failed; a separate RGB-only timing/reference-retention pilot was completed without treating the disputed depth maps as GT. The initial partial-download snapshot below is historical, not the current acquisition inventory.

## Decision

Prioritize **UR Fall Detection (URFD) ceiling-camera recordings** for fixed-mount and depth-format verification. Use **SBM-RGBD's corrected URFD-derived frames** as a registration cross-check, then investigate its other source groups for scene diversity. Princeton is a secondary candidate requiring sequence-specific camera-motion evidence. Do not promote any whole dataset to a certified fixed-camera metric-depth benchmark yet.

| Candidate | Verified source information | Suitability and remaining admission checks |
|---|---|---|
| UR Fall Detection | 70 sequences (30 falls, 40 daily activities). Camera 1 is ceiling-mounted; RGB and PNG16 depth are available. Official per-camera/type scale formulas recover millimeters. | Strongest explicit mounting evidence found. Start with camera 1. Check RGB/depth registration, matching frame IDs, valid-depth encoding and recording continuity. Multiple events/views do not establish independent locations. |
| SBM-RGBD | 33 Kinect RGB-D videos, 640×480, 8- or 16-bit depth; synchronized color/depth and foreground masks. Sources include URFD and Princeton. Official updates include depth scaling/registration fixes. | Relevant illumination, occlusion and intermittent-motion challenges. Must establish per-sequence mounting and metric encoding; reject uncalibrated visualization depth for raw AbsRel. Its `groundtruth` masks are segmentation, not depth maps. |
| Princeton Tracking Benchmark | 100 RGB-D videos; PNG16 depth uses a three-bit rotation encoding, then millimeters; frame metadata contains timestamps and intrinsics. Dataset includes moving-camera examples. | Potential additional scenes only after fixed-view evidence and registration checks. Do not equate a tracking/motion category or static background with a fixed camera. |
| PKU-MMD | RGB and millimeter depth are provided at different resolutions (1920×1080 and 512×424), with multiple camera views. | Reserve candidate: confirm registration/calibration, time pairing, actual mounting and access before pixel-wise depth evaluation. Matching image sizes by resize is not registration. |
| Existing TUM/Bonn clips | Recorded poses were audited separately. No evaluated clip is certified fixed-mounted. | Historical mixed-camera development/stress evidence only. Do not use sequence names to relabel it as the new primary benchmark. |

Primary sources: [URFD official documentation](https://fenix.ur.edu.pl/~mkepski/ds/uf.html), [SBM-RGBD official documentation](https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset.html), [SBM source/sequence list](https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/ListOfSequences.pdf), [Princeton data format](https://tracking.cs.princeton.edu/dataset.html), [Princeton authors' paper](https://www.cv-foundation.org/openaccess/content_iccv_2013/papers/Song_Tracking_Revisited_Using_2013_ICCV_paper.pdf), [PKU-MMD official documentation](https://struct002.github.io/PKUMMD/).

## Encoding and overlap safeguards

URFD documents `depth_mm = C * PNG16 / 65535`, with C=3640 for fall camera 1, C=6000 for fall camera 0 and C=7000 for daily-activity camera 0. Inter-camera recordings are not strictly synchronized. Use one camera at a time; do not apply these original-file scales to a separately rescaled SBM release without verification. Camera 1 is preferred because its mounting is explicitly documented, not because it produced favorable predictions. [URFD documentation](https://fenix.ur.edu.pl/~mkepski/ds/uf.html).

For original Princeton files, decode the unsigned 16-bit rotation (`(v >> 3) | ((v << 13) & 65535)`) before dividing by 1000. This is not the TUM `/5000` convention. The official example marks zero depth invalid. [Princeton format and reader](https://tracking.cs.princeton.edu/dataset.html).

SBM is an aggregation, not an independent source for every sequence. Group a URFD original, both views of its event and its SBM derivative together. Likewise group Princeton originals with SBM derivatives. Raw-file hashes alone cannot identify resized/rescaled duplicate content. Reserve entire location/source groups where possible and report the actual count of distinct verified locations. No model-quality-based sequence filtering is permitted. [SBM provenance list](https://rgbd2017.na.icar.cnr.it/SBM-RGBDdataset/ListOfSequences.pdf).

URFD's page states CC BY-NC-SA 4.0 and non-commercial academic use. This is a source notice, not legal clearance for every derived artifact. Preserve attribution; do not redistribute raw data in the paper_work manuscript delivery. Check separate terms for other collections before release. [URFD source notice](https://fenix.ur.edu.pl/~mkepski/ds/uf.html).

## Access and first probe

HTTP HEAD returned 200 for the official SBM DepthCamouflage archive (999,772,706 bytes), Princeton ValidationSet archive (631,649,593 bytes), and SBM fall01cam1 correction (29,792,821 bytes). The two large archives were **not** downloaded in this screening stage.

`scripts/probe_fixed_camera_data.py` records a bounded format-only probe of the first URFD fall event, ceiling-camera RGB/depth and its official SBM correction. Inputs were chosen before model inference. The completed correction archive passes ZIP CRC checks and contains 160 depth PNGs plus 7 foreground-mask PNGs. First/middle/last depth samples are 480×640 uint16, with maximum values 3357/3372/3423 and nonzero fractions 72.1%/78.2%/80.2%. These numeric ranges alone do not establish metric units or correct alignment.

The URFD RGB/depth transfers were slow and stopped after this bounded probe; incomplete 2 MiB RGB and 3 MiB depth prefixes were preserved, **not accepted as data**. No download remains running. `scripts/summarize_fixed_camera_probe.py` saved completion flags, archive hashes and sample metadata in `work_dirs/qfixed_public_probe_20260910/probe_snapshot.json`, also delivered as `PUBLIC_DATA_PROBE.json`. The correction has no RGB frames, so a usable paired evaluation set has not yet been obtained. Download or format success does not certify registration or model performance. No training, Q0 prediction or policy selection was performed.

The inspected sequence has only 160 frames. Do not pad/repeat it or concatenate separate events to manufacture an L256 test. Use its genuine continuous duration for short-horizon diagnostics and obtain separate longer recordings for the long-horizon requirement.

## Required order before scoring

1. Validate the original/derived encoding and registration using source calibration or documented registration; visually matching edges is a diagnostic, not a substitute for missing calibration. Audit pairing and invalid/saturated sensor pixels.
2. Audit fixed mounting and remaining background motion without looking at model depth errors. Log uncertain cases separately instead of silently certifying them.
3. Check local prior-use records, derivatives and location/event overlap. Preserve at least one fresh location/source group for later model revisions; pretrained-data disjointness remains unknown.
4. Register eligible sequences, full continuous evaluation, frame windows, model hashes, unchanged tolerances and four controls: dense Q0, periodic output hold, K2 flow reuse and frozen MLP refresh. Report static and dynamic regions separately where labels support it.
5. Evaluate once without retuning. Any failure used for subsequent training consumes that holdout. Publish failures alongside aggregate and per-sequence results.
6. Separately measure real-time end-to-end performance on Nano B01 (5 W / 10 W) and Pi 4. A public RGB-D accuracy dataset cannot establish hardware real-time performance by itself.
