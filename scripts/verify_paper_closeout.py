"""Run and record bounded finalization checks without re-running reserved inference."""
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.protocol import file_record
from scripts.paper_closeout_study import prior_check, verify_files


def main():
    out = Path('work_dirs/paper_closeout')
    prior_check()
    for name in ('report_artifacts.json',):
        audit = json.loads((out/name).read_text())
        verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])
    revision = json.loads(Path('work_dirs/paper_revision_r3/results.json').read_text())
    verify_files([revision['generator'], *revision['inputs'], *revision['artifacts'],
                  *revision['legacy_context'], *revision['inspected_external_sources']])
    assert revision['post_hoc'] and not revision['inference_performed']
    document_audits = []
    for name in ('inference', 'render'):
        path = Path(f'work_dirs/paper_qualitative_20260908/{name}.json')
        audit = json.loads(path.read_text())
        verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])
        document_audits.append(file_record(path))
    for name in ('figures', 'draft'):
        path = Path(f'work_dirs/paper_revision_r3/{name}.json')
        audit = json.loads(path.read_text())
        verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])
        document_audits.append(file_record(path))
    video_checks = []
    for path in sorted(Path('paper/submission/qualitative').glob('*.mp4')):
        meta = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_frames,avg_frame_rate,duration', '-of', 'json', str(path)], text=True))
        stream = meta['streams'][0]
        assert int(stream['nb_frames']) == 256 and stream['avg_frame_rate'] == '10/1'
        assert abs(float(stream['duration'])-25.6) < .001
        video_checks.append(dict(file=file_record(path), stream=stream))
    assert len(video_checks) == 4
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    runs = []
    for name, cmd, extra in [
        ('cpu_tests', [sys.executable, '-m', 'pytest', 'tests', '-q'], {'CUDA_VISIBLE_DEVICES': ''}),
        ('cuda_tests', [sys.executable, '-m', 'pytest', 'tests/test_scan_kernel.py', '-q'], {}),
        ('old_freeze', [sys.executable, 'scripts/paper_freeze.py', '--verify'], {})]:
        process = subprocess.run(cmd, env=dict(env, **extra), text=True, capture_output=True)
        dest = out/(name+'.log')
        dest.write_text(process.stdout+process.stderr)
        print(name, process.returncode, process.stdout[-1000:], flush=True)
        runs.append(dict(name=name, command=cmd, returncode=process.returncode, log=file_record(dest)))
        process.check_returncode()
    packages = sorted([dict(name=d.metadata['Name'], version=d.version)
                       for d in importlib.metadata.distributions()], key=lambda d: d['name'].lower())
    (out/'verification.json').write_text(json.dumps(dict(runs=runs,
        source=file_record(__file__), tested_files=[file_record(p) for p in sorted(Path('tests').glob('test_*.py'))],
        packages=packages, revision=file_record('work_dirs/paper_revision_r3/results.json'),
        document_audits=document_audits, qualitative_videos=video_checks,
        edge_execution='new runner: desktop CPU smoke only; historical Nano logs audited separately',
        clean_machine_reproduction=False, author_review=False), indent=2)+'\n')


if __name__ == '__main__':
    main()
