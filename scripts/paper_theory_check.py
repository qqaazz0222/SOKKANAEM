"""Reproduce task-12 algebra/weight/cost checks without training or data inference."""
import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem import from_checkpoint
from sokkanaem.model import TemporalBlock
from sokkanaem.protocol import file_record, runtime_versions
from sokkanaem.theory import diagonal_comparison, periodic_refresh_bound


def verify_records(records):
    for r in records:
        actual = file_record(r['path'])
        if actual['sha256'] != r['sha256']:
            raise ValueError(f"input changed: {r['path']}")
    return len(records)


def source_mean(rows, field):
    return statistics.fmean(statistics.fmean(float(r[field]) for r in rows if r['source'] == s)
                            for s in sorted({r['source'] for r in rows}))


def check_documents():
    """Check links and TeX structure only; this is not a TeX compiler."""
    doc = Path('paper/THEORY_12.md')
    links = re.findall(r'\]\(([^)]+)\)', doc.read_text())
    for target in links:
        if not target.startswith(('http://', 'https://')):
            if not (doc.parent / target.split('#')[0]).exists():
                raise ValueError(f'missing document target: {target}')
    tex = Path('paper/theory12.tex').read_text()
    tex = re.sub(r'(?<!\\)%[^\n]*', '', tex)
    stack = []
    for kind, env in re.findall(r'\\(begin|end)\{([^}]+)\}', tex):
        if kind == 'begin':
            stack.append(env)
        elif not stack or stack.pop() != env:
            raise ValueError(f'mismatched TeX environment: {env}')
    if stack:
        raise ValueError('unclosed TeX environments')
    labels = re.findall(r'\\label\{([^}]+)\}', tex)
    if len(labels) != len(set(labels)):
        raise ValueError('duplicate TeX labels')
    for ref in re.findall(r'\\(?:eqref|ref)\{([^}]+)\}', tex):
        if ref not in labels:
            raise ValueError(f'missing TeX reference: {ref}')
    citations = set(re.findall(r'\\bibitem\{([^}]+)\}', tex))
    for group in re.findall(r'\\cite\{([^}]+)\}', tex):
        if not set(group.split(',')) <= citations:
            raise ValueError(f'missing citation: {group}')
    return dict(local_links_checked=True, tex_environments_and_references_checked=True,
                pdf_compiled=False)


def run():
    baseline = json.loads(Path('paper/freeze/baseline.json').read_text())
    records = [baseline['checkpoint'], baseline['config'], *baseline['code']]
    # Validate complete report chains before using their derived measurements.
    report_manifests = [Path(f'work_dirs/paper_study_{tag}/report_artifacts.json')
                        for tag in ('5_8', '9_11')]
    for path in report_manifests:
        manifest = json.loads(path.read_text())
        records.extend([file_record(path), manifest['generator'], *manifest['inputs'], *manifest['artifacts']])
    for phase in ('quality', 'latency', 'profile', 'seeds'):
        completion = json.loads(Path(f'work_dirs/paper_study_9_11/{phase}_complete.json').read_text())
        records.extend(completion['results'])
    verified = verify_records(records)

    torch.set_num_threads(2)
    model = from_checkpoint(baseline['checkpoint']['path']).double().eval()
    assert sum(p.numel() for p in model.parameters()) == baseline['params']
    assert all(torch.isfinite(p).all() for p in model.parameters())
    temporal = []
    for name, block in model.named_modules():
        if isinstance(block, TemporalBlock):
            lam = block.ssm.A_log.detach().exp()
            temporal.append(dict(block=name, lambda_min=float(lam.min()), lambda_max=float(lam.max()),
                                 inner=block.ssm.d_inner, state=block.ssm.d_state))
    assert all(r['lambda_min'] > 0 for r in temporal)
    pixels = 3 * model.p**2
    period, tau = model.detector.keyframe_every, model.detector.tau_on
    n = (baseline['input_size']//model.p)**2
    state_mib = 4*n*sum(r['inner']*r['state'] for r in temporal) / 2**20
    assert state_mib == 12.

    # Randomized tests check the derived identity and inequality, not global constants.
    max_identity_error, max_bound_violation = 0., 0.
    for seed in range(100):
        rng = np.random.default_rng(seed)
        phi = rng.uniform(.01, .999, (128, 9))
        trial = diagonal_comparison(phi, rng.normal(size=phi.shape),
            rng.integers(0, 2, size=phi.shape), rng.normal(size=9), rng.normal(size=9))
        e = trial['gated'] - trial['dense']
        max_identity_error = max(max_identity_error, float(np.max(np.abs(e[1:]-phi*e[:-1]-trial['residuals']))))
        max_bound_violation = max(max_bound_violation, float(np.max(trial['errors']-trial['bounds'])))
    assert max_identity_error < 1e-12 and max_bound_violation < 1e-12
    periodic = periodic_refresh_bound(.8, .1, 5, 100)
    scalar = diagonal_comparison(np.full((4, 1), .5), np.ones((4, 1)), np.zeros((4, 1)), [0.], [0.])

    with Path('paper/tables/study10_components.csv').open(newline='') as f:
        components = [r for r in csv.DictReader(f) if r['model'] == 'sparse_k30']
    assert len(components) == 4
    input_ms = source_mean(components, 'read_decode_preprocess')
    total_ms = source_mean(components, 'profile_total_ms')
    with Path('paper/tables/study9_operating_points.csv').open(newline='') as f:
        points = {r['model']: r for r in csv.DictReader(f)}
    native_ms, dense_ms = (float(points[m]['mean_ms']) for m in ('sparse_k30', 'dense_carry'))
    assert float(points['dense_carry']['state_mib']) == state_mib
    result = dict(role='analytic_and_synthetic_validation', training_performed=False,
        dataset_inference_performed=False, final_test_used=False,
        global_network_lipschitz_certified=False, learned_error_certificate=False,
        latex_pdf_compiled=False, prior_hash_records_verified=verified,
        environment=runtime_versions(), temporal_blocks=temporal,
        frozen_structure=dict(params=baseline['params'], keyframe=period, tau_on=tau,
            patch_values=pixels, first_block_input_bound_uncapped=(period-1)*np.sqrt(pixels*tau),
            pixel_range_cap=np.sqrt(pixels), temporal_state_fp32_mib=state_mib,
            finite_horizon_keyframes_256=(256+period-1)//period),
        synthetic=dict(trials=100, steps_per_trial=128, state_coordinates=9,
            maximum_identity_residual=max_identity_error, maximum_bound_violation=max_bound_violation,
            constant_input_four_skips_error=float(scalar['errors'][-1]),
            periodic_example=dict(rho=.8, epsilon=.1, period=5,
                bound_after_100_refreshes=float(periodic['after_refresh'][-1]),
                limiting_after_refresh=periodic['limiting_after_refresh'])),
        measured_arithmetic=dict(input_profile_ms=input_ms, total_profile_ms=total_ms,
            fixed_input_fraction=input_ms/total_ms,
            ideal_same_profile_speedup_ceiling_if_all_other_cost_vanishes=total_ms/input_ms,
            sparse_main_ms=native_ms, dense_main_ms=dense_ms,
            sparse_time_reduction_percent=100*(dense_ms-native_ms)/dense_ms,
            warning='Profile-only idealization, not measured speedup or whole-model accuracy guarantee'))
    verify_records(records)
    return result, records


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=Path('work_dirs/paper_theory_12'))
    args = ap.parse_args()
    result, inputs = run()
    target = Path('paper/tables/theory12_checks.json')
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    result['document_checks'] = check_documents()
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    files = ['paper/THEORY_12.md', 'paper/theory12.tex', 'sokkanaem/theory.py', 'tests/test_theory.py', str(target)]
    audit = dict(generator=file_record(__file__), inputs=inputs,
                 artifacts=[file_record(p) for p in files])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'artifacts.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f'Created {target} and {args.out / "artifacts.json"}')


if __name__ == '__main__':
    main()
