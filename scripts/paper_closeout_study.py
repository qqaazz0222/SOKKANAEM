"""Finalization diagnostics and explicitly frozen final-test evaluation.

No training or external submission. Smoke never uses the reserved test.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sokkanaem.data import load_manifest
from sokkanaem.metrics import _flow
from sokkanaem.performance_study import Native, AllActive, adapter, run_clip, POINTS
from sokkanaem.protocol import check_final_test, file_record, runtime_versions
from sokkanaem.study import HFDepth, grid_from_flow, regions_from_flow, score_prediction
from scripts.paper_study import (BASE, HF, snapshot_path, dump, append, records as dev_records,
                                verify_contract)
from scripts.paper_efficiency_study import verify as verify_efficiency

OUT = Path('work_dirs/paper_closeout')
FINAL_MODELS = ('sparse_k30', 'sparse_k5', 'dense_carry', 'dense_reset', 'output_hold',
                'da2_256', 'midas_256', 'dpt_common')


def verify_files(items):
    for r in items:
        if file_record(r['path'])['sha256'] != r['sha256']:
            raise ValueError(f"frozen input changed: {r['path']}")


def prior_check():
    verify_efficiency(json.loads(Path('work_dirs/paper_study_9_11/study.json').read_text()))
    audit = json.loads(Path('work_dirs/paper_theory_12/artifacts.json').read_text())
    verify_files([audit['generator'], *audit['inputs'], *audit['artifacts']])


def freeze():
    prior_check()
    old = json.loads(Path('work_dirs/paper_study_5_8/study.json').read_text())
    eff = json.loads(Path('work_dirs/paper_study_9_11/study.json').read_text())
    files = ['paper/CLOSEOUT_PLAN.md', __file__, 'paper/freeze/baseline.json',
             'paper/freeze/final_test.json', 'paper/freeze/final_test_files.json',
             'manifests/paper_test_tum_L8.json', 'manifests/paper_test_tum_L256.json']
    inputs = [*[file_record(p) for p in files], *old['sources'], old['base'], old['flow'],
              file_record(BASE.with_name('config.toml')), *old['baselines']['dpt']['files'],
              *old['baselines']['da2']['files'], *eff['dependencies'], *eff['sources']]
    final_inventory = json.loads(Path('paper/freeze/final_test_files.json').read_text())
    for frames in final_inventory['sequences'].values():
        for frame in frames:
            for kind in ('rgb', 'depth'):
                inputs.append({'path': str(Path(final_inventory['root'])/frame[kind]),
                               'sha256': frame[f'{kind}_sha256']})
    verify_files(inputs)
    contract = dict(role='final_test', models=list(FINAL_MODELS), lengths=[256, 8],
        precision='FP32 eager TF32 off', selection_frozen_before_final_predictions=True,
        checkpoint=str(BASE), primary='L256 clip scale-shift relative shape; native no-fit separate',
        source_paths=inputs, environment=runtime_versions(), training_performed=False,
        scope='same-scene moving-camera TUM transfer; not scene-disjoint or fixed-camera generalization')
    OUT.mkdir(parents=True, exist_ok=True)
    dump(OUT/'final_contract.json', contract)
    print('Final conditions frozen; no final predictions made by --phase freeze.', flush=True)


def identities(path, final=False):
    check_final_test(path, final)
    doc = json.loads(Path(path).read_text())
    for source, ds in load_manifest(path):
        declared = [c for c in doc['clips'] if c['source'] == source]
        for i, c in enumerate(declared):
            yield {'id': f'{source}:{i}', 'source': source, 'index': i,
                   'sequence': Path(c['pairs'][0][0]).parent.parent.name,
                   'pairs': c['pairs'], 'clip_len': doc['clip_len']}, ds, i


def load_done(path, expected):
    rows = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r['id'] for r in rows}
    if len(done) != len(rows) or not done <= set(expected):
        raise ValueError('duplicate or out-of-manifest completed rows')
    for r in rows:
        if not r.get('no_gt') and set(r['models']) != set(FINAL_MODELS):
            raise ValueError('incomplete model set in saved clip')
    return done


@torch.no_grad()
def evaluate(final=False):
    if final:
        contract = json.loads((OUT/'final_contract.json').read_text())
        if contract['models'] != list(FINAL_MODELS) or contract['lengths'] != [256, 8]:
            raise ValueError('final contract/model mismatch')
        if contract['environment'] != runtime_versions():
            raise ValueError('environment changed after freeze')
        verify_files(contract['source_paths'])
        opened = OUT/'final_opened.json'
        if not opened.exists():
            dump(opened, dict(utc=datetime.now(timezone.utc).isoformat(),
                contract=file_record(OUT/'final_contract.json'), event='final inference authorized and starting',
                historical_seal='paper/freeze/final_test.json remains an immutable historical record'))
    else:
        prior_check()
    models = {name: adapter(POINTS[name]) for name in FINAL_MODELS if name != 'dpt_common'}
    dpt = HFDepth(str(snapshot_path(*HF['dpt'])))
    for length in ([256, 8] if final else [8]):
        path = Path(f'manifests/paper_test_tum_L{length}.json' if final else 'manifests/acc_real_L8.json')
        samples = list(identities(path, final))
        if not final:
            samples = samples[:1]
        dest = OUT/(f'final_L{length}.jsonl' if final else 'smoke_dev.jsonl')
        done = load_done(dest, [r['id'] for r, _, _ in samples])
        for ordinal, (identity, ds, i) in enumerate(samples):
            if identity['id'] in done:
                continue
            frames, gt, valid = ds[i]
            valid = (valid.bool() & torch.isfinite(gt) & (gt > 0)).float()
            row = {**identity, 'models': {}, 'no_gt': not bool(valid.any())}
            if row['no_gt']:
                append(dest, row)
                continue
            fg, gg, vv = frames.cuda(), torch.where(valid.bool(), gt, 0).cuda(), valid.cuda()
            flow = _flow(fg[1:], fg[:-1], chunk=16)
            grid, inb = grid_from_flow(flow)
            regions = regions_from_flow(gg, vv, flow)
            for name in FINAL_MODELS:
                if name == 'dpt_common':
                    pred = dpt.predict(frames, [], False)
                    space, metric, params = dpt.space, False, sum(p.numel() for p in dpt.model.parameters())
                    sizes, active = [dpt.effective['common']], None
                    sha = hashlib.sha256(pred.cpu().contiguous().numpy().tobytes()).hexdigest()
                else:
                    model = models[name]
                    result = run_clip(model, identity['pairs'])
                    pred = result['prediction'].cuda()
                    space, metric, params = model.space, model.metric, model.params
                    sizes = [result['input_size']]
                    vals = [x['active'] for x in result['telemetry'] if x['active'] is not None]
                    active = float(np.mean(vals)) if vals else None
                    sha = result['prediction_sha256']
                if not bool(torch.isfinite(pred).all()):
                    raise ValueError(f'non-finite prediction: {name}, {identity["id"]}')
                row['models'][name] = dict(space=space, params=params, future_frames=False,
                    effective_sizes=sorted({tuple(s) for s in sizes}), active_all=active, prediction_sha256=sha,
                    gauges=score_prediction(fg, pred, gg, vv, space, grid, inb, regions,
                                            long=length == 256, metric=metric))
                del pred
            append(dest, row)
            print(f'{"FINAL" if final else "SMOKE-dev"} L{length} {ordinal+1}/{len(samples)} {identity["sequence"]}', flush=True)
            del fg, gg, vv, flow, grid, inb, regions
        if len(load_done(dest, [r['id'] for r, _, _ in samples])) != len(samples):
            raise ValueError('incomplete manifest coverage')
    if final:
        verify_files(contract['source_paths'])
        dump(OUT/'final_complete.json', dict(final_test_used=True, training_performed=False,
            contract=file_record(OUT/'final_contract.json'),
            results=[file_record(OUT/f'final_L{n}.jsonl') for n in (256, 8)]))


def components(state):
    return [t for key in ('hs', 'sp', 'tc') for t in state[key]]


def rms_difference(left, right):
    a, b = components(left), components(right)
    assert len(a) == len(b) and all(x.shape == y.shape for x, y in zip(a, b))
    return float(torch.sqrt(sum((x.double()-y.double()).square().sum() for x, y in zip(a, b)) /
                            sum(x.numel() for x in a)))


@torch.no_grad()
def diagnostics():
    prior_check()
    chosen, seen = [], set()
    for c, ds, i in dev_records(256):
        if c['sequence'] not in seen:
            chosen.append((c, ds, i))
            seen.add(c['sequence'])
    model = Native(POINTS['sparse_k30'])
    dense = Native(POINTS['sparse_k30'])
    dense.model.detector = AllActive(dense.model.p)
    path = OUT/'theory_dev.jsonl'
    existing = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    done = {r['id'] for r in existing}
    assert len(done) == len(existing) and done <= {c['id'] for c, _, _ in chosen}
    for c, ds, i in chosen:
        if c['id'] in done:
            continue
        frames = ds[i][0].cuda()
        model.reset(); dense.reset()
        age = torch.zeros((1, 256), device='cuda')
        shadow_g = shadow_d = torch.zeros(256, 384, 16, dtype=torch.double, device='cuda')
        bound = 0.
        rows = []
        for t, frame in enumerate(frames):
            frame = frame[None]
            previous = model.state
            tokens = model.model.embed(frame).flatten(2).transpose(1, 2)
            block = model.model.blocks[0]
            normed = block.norm(tokens).reshape(256, model.model.dim)
            x, _, dt, bp, _ = block.ssm._params(normed[:, None], None)
            phi = (dt.double()[..., None] * -block.ssm.A_log.double().exp()).exp()[:, 0]
            q = ((dt.double()*x.double())[..., None]*bp.double()[:, :, None])[:, 0]
            # Counterfactual all-active step from the SAME sparse incoming state.
            all_on = frame.new_ones(1, 256)
            tok, hs, sp, tc, feats = model.model._forward_tokens(tokens, all_on,
                previous['hs'] if previous else None, (16, 16),
                sp=previous['sp'] if previous else [None]*model.model._n_spatial,
                tc=previous['tc'] if previous else [None]*model.model._n_temporal)
            counter = dict(hs=hs, sp=sp, tc=tc)
            pred, info = model.infer(frame)
            ref, _ = dense.infer(frame)
            mask = info['mask']
            age = torch.where(mask.bool(), 0., age+1)
            m = mask.reshape(256, 1, 1).double()
            residual = (1-m)*((1-phi)*shadow_g-q)
            shadow_d = phi*shadow_d+q
            shadow_g = m*(phi*shadow_g+q)+(1-m)*shadow_g
            rho = float(phi.max())
            defect = float(residual.norm())
            bound = rho*bound + defect
            error = float((shadow_g-shadow_d).norm())
            assert error <= bound + 1e-8*(1+bound)
            assert float(age.max()) <= 29
            local = rms_difference(model.state, counter)
            if info['keyframe']:
                assert local < 1e-10
            inactive = mask < .5
            score = F.avg_pool2d((frame-frames[t-1:t]).square().mean(1, keepdim=True), 16).flatten(1) if t else None
            row = dict(frame=t, keyframe=bool(info['keyframe']), **{
                k: v for k, v in model.diagnostics(info).items() if k in ('active', 'dense_fallback')},
                age_max=float(age.max()), age_mean=float(age.mean()),
                inactive_mse_max=float(score[inactive].max()) if t and bool(inactive.any()) else None,
                net_local_defect_rms=local, net_state_error_rms=rms_difference(model.state, dense.state),
                output_dense_mae=float((pred-ref).abs().mean()),
                first_shadow_error_rms=error/np.sqrt(shadow_g.numel()),
                first_shadow_bound_rms=bound/np.sqrt(shadow_g.numel()),
                first_shadow_defect_rms=defect/np.sqrt(shadow_g.numel()), rho=rho,
                fp32_shadow_max_gap=float((model.state['hs'][0].double()-shadow_g).abs().max()))
            if row['inactive_mse_max'] is not None:
                assert row['inactive_mse_max'] <= .05+1e-7
            rows.append(row)
        append(path, dict(id=c['id'], sequence=c['sequence'], source=c['source'], frames=rows))
        print(f'Theory-development complete: {c["sequence"]} ({len(rows)} frames)', flush=True)
    prior_check()
    dump(OUT/'theory_complete.json', dict(role='development_diagnostic', frames=1024,
        final_test_used=False, result=file_record(path),
        source=file_record(__file__), plan=file_record('paper/CLOSEOUT_PLAN.md')))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--phase', choices=['diagnostics', 'smoke', 'freeze', 'final'], required=True)
    ap.add_argument('--final-test', action='store_true')
    args = ap.parse_args()
    if args.phase == 'final' and not args.final_test:
        ap.error('final inference requires --final-test and a frozen contract')
    if args.final_test and args.phase != 'final':
        ap.error('--final-test only accompanies --phase final')
    torch.set_num_threads(2)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    OUT.mkdir(parents=True, exist_ok=True)
    if args.phase == 'freeze':
        freeze()
    elif args.phase == 'diagnostics':
        diagnostics()
    else:
        evaluate(args.phase == 'final')


if __name__ == '__main__':
    main()
