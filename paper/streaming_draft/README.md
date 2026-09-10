# Fixed-view selective-refresh working draft

Primary Markdown source: `../draft.md` in both the repository and the paper_work delivery copy.

`manuscript.tex` and `manuscript.pdf` are standalone review outputs, not replacements for the archived MDPI submission source. Compile the TeX from this directory with XeLaTeX. Tables use the available text width through proportional Pandoc column specifications; this single-column review layout does not require a wider two-column float.

Figure 1 is a code-rendered pipeline, Figure 2 uses repeated measured latency, and Figures 3–5 use saved actual predictions. No image-generation model fabricated evaluation outputs. See `visual_provenance.json` for selections and hashes.

Latest update: `FIXED_CAMERA_VALIDATION.md` reports failed depth-data admission for the first fixed-camera event and a separately registered RGB-only execution pilot. Its 3.239 ms MLP mean is not a GT-quality-preserving speed claim; p95 is higher than dense Q0. No fixed-camera GT accuracy is reported. The data-only visualization under `fixed_camera_validation/` is not a model-error figure. The new tests bring the related regression count to 107.

Read `FIXED_CAMERA_SCOPE.md`, `CAMERA_POSE_AUDIT.md`, `PUBLIC_DATASET_SCREENING.md`, `REVISION_STATUS.md` and `INDEPENDENT_VALIDATION.md` before interpreting readiness. The target is now indoor fixed-view cameras observing dynamic scenes. This narrowing occurred after the new moving-camera holdout failed short-clip quality; the failure remains visible. Existing mixed-camera timings are not fixed-camera or real-time deployment validation. Public-data screening prioritizes URFD ceiling-camera data and SBM derivatives, with registration/overlap checks still required. New-model hardware evidence, broader statistical support, complete bibliography and author approval remain pending.
