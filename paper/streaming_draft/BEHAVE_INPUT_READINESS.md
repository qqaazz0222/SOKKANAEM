# BEHAVE input readiness — 2026-09-10

Status: input code and public-source checks completed; dataset acquisition,
physical-location grouping and GT evaluation have **not** been completed.
Non-commercial scientific-research confirmation remains pending. A generic request
to continue was not recorded as a factual declaration about the research funding
or commercial status. No dataset GET has been performed in this step.

## Completed source inspection

`scripts/inspect_behave_sources.py` recorded public documentation, six code/license
files at exact Git commits, and HTTP HEAD responses in
`work_dirs/qbehave_readiness_20260910/`. Code was inspected, not installed or run.
The BEHAVE repository revision is `85664832b43008d70a9bc5f5ba3bb2aa173bc077`.
Headers are server declarations, not checksums of acquired data.

| Official Date02 resource | Declared bytes | Acquired? |
|---|---:|---|
| RGB video TAR | 5,763,911,680 | No |
| Registered-depth video TAR | 49,815,623,680 | No |
| Timestamp TAR | 5,550,080 | No |
| Calibration ZIP (all dates) | 395,854,758 | No |
| Pre-extracted Date02 ZIP | 17,881,182,670 | No |

All five HEAD requests returned HTTP 200 and advertised byte ranges. Range GET
support and remote archive member integrity have not yet been tested. The
RGB/depth TAR pair totals 55,579,535,360 bytes; begin with timestamp metadata.
The pre-extracted release is not automatically a substitute for native continuous
video. [Official resource list](https://virtualhumans.mpi-inf.mpg.de/behave/license.html).

## Implemented input safeguards

`sokkanaem/behave_io.py` provides:

- Depth decoding directly from native YUV 4:4:4 planes, restoring the parity-folded
  low byte and high byte before converting millimetres to metres. No RGB/gray
  preview conversion, fps-based seek or packed-channel resize is used.
- Sequential source indexing from frame zero, decoder exit-code checks, truncated
  frame rejection and optional decoded-count versus timestamp-count equality.
- Strict finite/increasing timestamps and mutual-nearest, one-to-one RGB/depth
  association. Ties go to earlier indices; tolerance is an explicit argument.
  Unmatched RGB frames are retained, not deleted from the streaming input.

The byte format is documented by the upstream
[videoio implementation](https://github.com/vguzov/videoio/blob/master/videoio/video_uint16.py).
BEHAVE uses separate timestamps for color and depth in its
[official controller](https://github.com/xiexh20/behave-dataset/blob/main/data/video_reader.py).
These checks address file decoding and indexing, not sensor calibration quality.

`scripts/acquire_behave_timestamps.py` is prepared but **not run**. It requires an
explicit non-commercial-research confirmation flag, caps the transfer at 10 MB,
checks HEAD/GET size and ETag consistency, records a local SHA256 and inventories
regular JSON members without extracting archive-controlled paths. A local SHA256
is not a publisher checksum. It does not select clips or authorize GT scoring.

`scripts/check_behave_timestamps.py` audits a local time.json with a protocol-supplied
tolerance. The actual tolerance and validation clips are not selected yet.

## Verification and limitations

The new tests cover all **65,536 uint16 values**, a five-frame lossless synthetic
MP4 round trip with exact frame order, count mismatches, units/invalid zeros,
timestamp duplicates/ties/missing matches and unsafe TAR paths/links. The input
and metadata tests total 17; the complete project suite passes **299 tests**.
Synthetic codec success is not evidence that a real BEHAVE recording has been
decoded correctly. Actual-data cross-checks remain mandatory.

Current model weights, threshold, manuscript result tables and the moving-camera
failure are unchanged. No new accuracy, boundary, temporal or hardware result is
claimed. Next: confirm permitted use, acquire timestamp metadata, establish
location groups and a prospective data protocol, then acquire matched videos.
