"""Build and verify an allowlisted PRIVATE LOCAL source/evidence archive. No upload."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.paper_closeout_study import verify_files


def main():
    out = Path('work_dirs/paper_closeout')
    report = json.loads((out/'report_artifacts.json').read_text())
    verify_files([report['generator'], *report['inputs'], *report['artifacts']])
    revision_path = Path('work_dirs/paper_revision_r3/results.json')
    revision = json.loads(revision_path.read_text())
    verify_files([revision['generator'], *revision['inputs'], *revision['artifacts'],
                  *revision['legacy_context'], *revision['inspected_external_sources']])
    pdf = json.loads((out/'pdf_build.json').read_text())
    verify_files([pdf['pdf'], pdf['log'], *pdf['source'], *pdf['converted_logos']])
    tests = json.loads((out/'verification.json').read_text())
    verify_files([tests['source'], *tests['tested_files'], *[r['log'] for r in tests['runs']]])
    verify_files([tests['revision']])
    verify_files(tests['document_audits'])
    for name in ('figures', 'draft'):
        audit = json.loads(Path(f'work_dirs/paper_revision_r3/{name}.json').read_text())
        verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])
    assert all(r['returncode'] == 0 for r in tests['runs'])
    complete = json.loads((out/'final_complete.json').read_text())
    verify_files([complete['contract'], *complete['results']])
    status = dict(submission_ready=False, external_submission=False, public_upload=False,
        final_test_used=True, final_lengths={'L256': 13, 'L8': 462}, final_models=8,
        new_training=False, theory_development_frames=1024,
        task13='conditional nonexecution: frozen-model analysis scope',
        task14='Nano B01 5W/10W historical synthetic logs audited; new real-video runner has desktop CPU smoke only',
        task15='integrated review manuscript/PDF and local evidence package built',
        blockers=['author identity/declarations/contributions/approval',
                  'human mathematical and novelty/venue-fit review',
                  'current journal policy and redistribution rights confirmation'],
        self_review='R3: 40 items mapped; partial/excluded/author-pending items remain',
        revision=file_record(revision_path),
        extra_limitations=['no clean-machine full reproduction',
                           'no provenance-matched modern video baseline in the frozen final table',
                           'current-weight continuous L1024/range extensions incomplete; qualitative replay is post-hoc',
                           'historical edge logs lack weight/code linkage, real-video accuracy and end-to-end validation'],
        evidence=[file_record(out/name) for name in ('final_complete.json', 'report_artifacts.json',
                                                    'pdf_build.json', 'verification.json')])
    (out/'status.json').write_text(json.dumps(status, indent=2)+'\n')
    paths = set()
    for base in ('sokkanaem', 'tests', 'configs', 'manifests', 'paper/submission', 'paper/tables',
                 'paper/freeze', 'paper/self-revision'):
        paths.update(p for p in Path(base).rglob('*') if p.is_file() and '__pycache__' not in p.parts
                     and p.suffix not in ('.pyc', '.aux', '.out'))
    scripts = ['paper_closeout_study', 'paper_closeout_report', 'paper_efficiency_study',
        'paper_efficiency_report', 'paper_study', 'paper_study_report', 'paper_freeze', 'paper_theory_check',
        'prepare_efficiency_dependencies', 'prepare_paper_test', 'prepare_paper_typesetter',
        'prepare_paper_edge_bundle', 'paper_edge_input_bench', 'build_paper_submission',
        'verify_paper_closeout', 'package_paper_submission', 'paper_revision_report',
        'paper_revision_figures', 'render_paper_draft', 'paper_qualitative', 'render_paper_qualitative',
        'eval_baseline_vda', 'bootstrap_ci', 'eval_acc', 'train', 'prepare_data', 'edge_bench']
    paths.update(Path('scripts')/(name+'.py') for name in scripts)
    paths.add(revision_path)
    paths.add(Path('paper/QUALITATIVE_PLAN.md'))
    paths.update(Path('work_dirs/paper_qualitative_20260908').glob('*.json'))
    paths.update(Path(f'work_dirs/paper_revision_r3/{name}.json') for name in ('figures', 'draft'))
    # Local historical evidence only; upstream code is hash-recorded, not redistributed here.
    paths.update(Path(r['path']) for r in revision['legacy_context'])
    for base in ('work_dirs/paper_study_5_8', 'work_dirs/paper_study_9_11',
                 'work_dirs/paper_theory_12', 'work_dirs/paper_closeout'):
        paths.update(p for p in Path(base).glob('*') if p.is_file() and p.suffix in ('.json', '.jsonl', '.log')
                     and p.name != 'package_archive.json')
    paths.update(Path(p) for p in ('README.md', 'PROGRESS.md', 'pyproject.toml', 'environment.yml',
        'paper/CLOSEOUT_PLAN.md', 'paper/CLOSEOUT.md', 'paper/FINAL_EVALUATION.md',
        'paper/STUDY_5_8.md', 'paper/STUDY_9_11.md', 'paper/STUDY_9_11_PLAN.md',
        'paper/THEORY_12.md', 'paper/theory12.tex', 'paper/PROTOCOL.md', 'paper/evaluation_protocol.json',
        'paper/WORK_PLAN.md', 'paper/PREPARATION.md', 'paper/MODEL_FOCUS.md', 'paper/draft.md', 'paper/draft_ko.md',
        'paper/draft_legacy_20260907.md',
        'work_dirs/v11-longclip-spread-s0/latest.pt', 'work_dirs/v11-longclip-spread-s0/config.toml',
        'work_dirs/paper_edge_bundle_archive.json', 'measures/nano-b01-5w.log', 'measures/nano-b01-10w.log'))
    if Path('LICENSE').is_file():
        paths.add(Path('LICENSE'))
    files = [file_record(p) for p in sorted(paths)]
    for r, p in zip(files, sorted(paths)):
        r['path'] = str(p)
    archive = Path('work_dirs/paper_submission_review.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for p in sorted(paths):
            z.write(p, str(p))
        z.writestr('PACKAGE_FILES.json', json.dumps(dict(private_local_only=True, files=files,
            excluded='original datasets, external weights, environments, caches, credentials, git',
            clean_machine_reproduction_validated=False), indent=2)+'\n')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in files:
            if hashlib.sha256(z.read(r['path'])).hexdigest() != r['sha256']:
                raise ValueError('archive checksum mismatch: '+r['path'])
    (out/'package_archive.json').write_text(json.dumps(dict(archive=file_record(archive),
        verified_files=len(files), private_local_only=True), indent=2)+'\n')
    print('Verified private archive:', archive, 'files:', len(files), 'bytes:', archive.stat().st_size)


if __name__ == '__main__':
    main()
