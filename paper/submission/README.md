# Working submission and reproduction guide

Qualitative extension (2026-09-08): Figures 6–7 and
[four comparison videos](qualitative/README.md) replay the first L256 clip of
each evaluated sequence. All 16 prediction hashes match the frozen scores.
The analysis includes adverse cases and the all-active temporal window;
no model tuning or replacement of final scores was performed.

2026-09-08: the manuscript is now model-centered: **SOKKANAEM: Change-Gated State-Space
Modeling with Cached Readout for Streaming Video Depth**. Architecture, streaming execution
and existing component ablations lead the narrative; conditional analysis supports design.
See `paper/MODEL_FOCUS.md`. This reframing does not claim new training or improved performance.

Use `manuscript.pdf` / `manuscript.tex` as the current integrated manuscript.
`paper/draft.md` is now its full synchronized Markdown edition, with five PNG figures.
`paper/draft_legacy_20260907.md` and `paper/draft_ko.md` are historical drafts,
not current submission sources. The [R3 40-item response](../self-revision/r3/revision-status.md)
and `REVISION_NOTES.md` identify corrections and still-partial items.
**Submission readiness is false.** See `REVIEW_CHECKLIST.md` and `COVER_LETTER.md`.
Author identity, declarations, human review and any actual submission remain pending.

## Evidence map

| Item | Location |
|---|---|
| Current Korean summary / status | `paper/FINAL_EVALUATION.md`, `paper/CLOSEOUT.md` |
| Frozen final protocol | `paper/CLOSEOUT_PLAN.md`, `work_dirs/paper_closeout/final_contract.json` |
| Final access/completion | `work_dirs/paper_closeout/final_opened.json`, `final_complete.json` |
| Full final clip results | `work_dirs/paper_closeout/final_L256.jsonl`, `final_L8.jsonl` |
| Final tables and development traces | `paper/submission/tables/` |
| Theory source and tests | `paper/theory12.tex`, `sokkanaem/theory.py`, `tests/test_theory.py` |
| Development accuracy/efficiency provenance | `paper/STUDY_5_8.md`, `STUDY_9_11.md`, `paper/tables/` |
| Frozen native EMA | original `work_dirs/v11-longclip-spread-s0/latest.pt` and sibling `config.toml` |
| Private device package | `work_dirs/paper_edge_bundle.zip`; instructions `EDGE_RUN.md` |
| New descriptive scale panel | `revision_results.tex`, `tables/revision_scale_*.csv` |
| New SVG/PDF/PNG artwork | `figures/r3_*`; generator `scripts/paper_revision_figures.py` |
| Historical video-baseline boundary | `VIDEO_BASELINE_AUDIT.md` |
| R3 result / figure / Markdown hashes | `work_dirs/paper_revision_r3/{results,figures,draft}.json` |

The reserved test has now been used. Do not tune on its results or call it untouched again.
The older `paper/freeze/final_test.json` is an immutable record of the earlier sealing event,
not the current access status. L256/L8 overlap in footage; there are four sequence units,
not hundreds of independent test replicates. No new edge measurements or training were performed
in this closeout. Historical Nano B01 5W/10W logs do exist; see `EDGE_LEGACY_AUDIT.md` for
their synthetic-input boundary and missing provenance. The earlier blanket denial was corrected.

## Existing environment commands

Run at the repository root using the evaluated environment (Python 3.11, PyTorch 2.8.0+cu128;
exact versions, paths and hashes in the final contract and verification manifest).

```bash
python scripts/paper_freeze.py --verify
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests -q
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python -m pytest tests/test_scan_kernel.py -q
python scripts/paper_revision_report.py
python scripts/paper_revision_figures.py
python scripts/render_paper_draft.py
python scripts/build_paper_submission.py
python scripts/verify_paper_closeout.py
python scripts/package_paper_submission.py
```

The R3 report generator revalidates the prior evidence chain, final coverage and hashes; it does
not infer predictions. The original one-time experimental commands were:

```bash
python scripts/paper_closeout_study.py --phase diagnostics
python scripts/paper_closeout_study.py --phase freeze
python scripts/paper_closeout_study.py --phase final --final-test
```

Do not rerun `freeze` to change the contract. Existing completed rows are retained on resume.
Reproduction on a new machine must keep the original evidence immutable and record new outputs
as a separate reproduction, not overwrite the published/finalization artifacts.

## Dependencies and portability limits

The private source/evidence archive includes code, configs, tables, final/development result
records and native weights. It deliberately excludes original TUM/Bonn media, external baseline
weights, Python installations, caches, credentials and the `.git` directory.
Data and pretrained weights must be obtained from their original providers under applicable terms.
The recorded baseline snapshots and file hashes, not a moving model name, define the comparisons.

Historical manifests and frozen code contain absolute dataset and Hugging Face cache paths,
including `/home/hyunsu/...`. A fresh machine is **not** claimed to work by simply unzipping:
provision the same directory mapping in an isolated container/mount, or create a separately
versioned relocation manifest and update the verifier for that reproduction while preserving
original hashes. Rewriting frozen source and then treating its old contract as valid is wrong.
Full clean-machine retraining/inference reproduction has not been executed in this closeout.

Typesetting uses Tectonic 0.15.0 (`scripts/prepare_paper_typesetter.py`) and Ghostscript 10.04.0,
installed under `work_dirs/paper_tools/`, not in the evaluated ML environment. The official
MDPI class remains unmodified. Its EPS logos are converted to vector PDF for Tectonic;
template/tool hashes and build results are recorded. The cached TeX bundle needs network access
on a first build. To provision the optional local EPS converter:

```bash
conda create --prefix ./work_dirs/paper_tools/ghostscript --yes --override-channels --channel conda-forge ghostscript=10.04.0
```

`PyMuPDF==1.26.4` was installed only into `work_dirs/paper_tools/pdf_python` for local PDF
inspection and vector diagnostic plotting; it is not an inference dependency. Install it
with `python -m pip install --target work_dirs/paper_tools/pdf_python PyMuPDF==1.26.4`
before generating the report on a fresh environment.

The Markdown edition is generated from the LaTeX body, its included tables/proofs and author
details, rather than maintained as a second hand-edited set of results. Its isolated converter
is Pandoc 3.6.1, provided by `pypandoc_binary==1.15`:

```bash
python -m pip install --target work_dirs/paper_tools/document_python pypandoc_binary==1.15
```

SVG charts are authored programmatically from CSV values, then converted by PyMuPDF to PDF
and PNG; no AI-generated prediction maps or invented observations are included. New figures
use zero-based quantitative axes. Timing min/max bars and sequence-bootstrap intervals are
explicitly distinguished. The four new figures plus the existing state diagnostic appear
in both the PDF and `paper/draft.md`.

The private edge bundle additionally copies development RGB for convenience. **Do not publicly
upload either bundle until dataset, code, template and weight redistribution rights are checked.**
No public DOI, license grant, author approval or journal submission is implied by creating files.
