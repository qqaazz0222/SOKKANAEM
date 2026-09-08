"""R3: post-hoc descriptive scale analysis of saved predictions' scores. No inference."""
from collections import defaultdict
import itertools
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_closeout_study import OUT, FINAL_MODELS, prior_check, verify_files
from scripts.paper_closeout_report import read_rows, validate_coverage, csv_write, tex_table, LABELS
from sokkanaem.protocol import file_record

DEST = Path('paper/submission')
FIELDS = ('scale_drift', 'scale_logstd', 'scale_step')


def aggregate_scale(rows, length):
    groups, clip_rows = defaultdict(list), []
    for row in rows:
        if row['no_gt']:
            continue
        for model in FINAL_MODELS:
            item = row['models'][model]
            gauge = 'none' if item['space'] == 'depth' else 'median'
            val = item['gauges'][gauge]
            scores = {key: val['scores'][key] for key in FIELDS}
            if any(not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0 for x in scores.values()):
                raise ValueError('missing/nonfinite/negative scale statistic')
            entry = dict(length=length, clip_id=row['id'], sequence=row['sequence'], model=model,
                         gauge=gauge, failed=int(val['failed']), **scores)
            clip_rows.append(entry)
            groups[model, row['sequence']].append(entry)
    sequences = [dict(length=length, model=model, sequence=sequence, gauge=entries[0]['gauge'],
        clips=len(entries), failed=sum(e['failed'] for e in entries),
        **{k: statistics.fmean(e[k] for e in entries) for k in FIELDS})
        for (model, sequence), entries in sorted(groups.items())]
    summary = []
    for model in FINAL_MODELS:
        entries = [s for s in sequences if s['model'] == model]
        if len(entries) != 4:
            raise ValueError('requires all four declared sequence units')
        draws = np.array(list(itertools.product(range(4), repeat=4)))
        for metric in FIELDS:
            values = np.array([s[metric] for s in entries])
            low, high = np.quantile(values[draws].mean(1), [.025, .975])
            summary.append(dict(length=length, model=model, gauge=entries[0]['gauge'], metric=metric,
                equal_sequence_mean=float(values.mean()), ci_low=float(low), ci_high=float(high),
                sequences=4, clips=sum(s['clips'] for s in entries),
                failed=sum(s['failed'] for s in entries), role='post-hoc descriptive; not a new primary endpoint'))
    return clip_rows, sequences, summary


def main():
    prior_check()
    complete = json.loads((OUT/'final_complete.json').read_text())
    verify_files([complete['contract'], *complete['results']])
    verify_files(json.loads((OUT/'final_contract.json').read_text())['source_paths'])
    all_clips, all_sequences, summary = [], [], []
    for length in (8, 256):
        rows = read_rows(OUT/f'final_L{length}.jsonl')
        validate_coverage(rows, json.loads(Path(f'manifests/paper_test_tum_L{length}.json').read_text()))
        a, b, c = aggregate_scale(rows, length)
        all_clips.extend(a); all_sequences.extend(b); summary.extend(c)
    paths = []
    for name, data in [('revision_scale_clips', all_clips), ('revision_scale_sequences', all_sequences),
                       ('revision_scale_summary', summary)]:
        path = DEST/'tables'/f'{name}.csv'
        csv_write(path, data)
        paths.append(path)
    lookup = {(s['length'], s['model'], s['metric']): s for s in summary}
    table_rows = []
    for model in FINAL_MODELS:
        values = [lookup[length, model, 'scale_drift']['equal_sequence_mean'] for length in (8, 256)]
        s = lookup[256, model, 'scale_drift']
        table_rows.append([LABELS[model], *[f'{v:.4f}' for v in values],
            f'[{s["ci_low"]:.4f}, {s["ci_high"]:.4f}]',
            f'{lookup[256,model,"scale_logstd"]["equal_sequence_mean"]:.4f}',
            f'{lookup[256,model,"scale_step"]["equal_sequence_mean"]:.4f}'])
    tex = ('\\subsection{Post-hoc scale-variation diagnostic}\n'
        'To address the earlier self-review, we summarize statistics already stored during final evaluation. '
        'This analysis was added after test access, not preregistered as a primary endpoint. '
        'It includes all eight frozen models and both lengths without tuning or new inference. '
        'For each valid frame, $s_t=\\operatorname{median}(G_t)/\\operatorname{median}(D_t)$ '
        'uses the same GT-valid pixels. We report the within-clip sample coefficient of variation '
        '$\\operatorname{sd}(s_t)/\\overline{s}$, sample standard deviation of $\\log s_t$, '
        'and mean absolute log difference between consecutive retained valid frames. '
        'The scorer skips frames without valid GT and returns zero if fewer than two remain; '
        'these conventions are unchanged. Native models use no-fit depth; relative models use '
        'their median-gauge inverse-disparity depth, not an affine-fit depth. '
        'Away from numerical clamp activation, a positive constant clip scaling cancels from these statistics; '
        'an affine disparity shift does not. The stored scorer clamps the predicted median at $10^{-6}$, '
        'the scale factor at $10^{-12}$ and the CV denominator at $10^{-6}$.\n')
    tex += tex_table('Post-hoc scale statistics: clip means within each sequence, then four-sequence means. '
        'The CV interval enumerates sequence-bootstrap resamples; it is exploratory and unadjusted. '
        'Full intervals and gauge/failure counts are supplied as CSV.', 'tab:scale',
        'Model & CV L8 & CV L256 & L256 CV interval & Log-SD & Log-step', table_rows, 'lrrrrr')
    tex += ('A smaller scale statistic alone does not establish better depth, and per-frame fitting '
        'can hide scale errors that a metric deployment still incurs. L8 and L256 use partly different '
        'footage and reset schedules, so their difference is not an isolated causal effect of history length. '
        'Local shape changes can also affect the scale medians. '
        'These summaries are not an additive decomposition of AbsRel into scale and shape terms. '
        'Confidence intervals containing a null difference do not establish equivalence.\n')
    path = DEST/'revision_results.tex'
    path.write_text(tex)
    paths.append(path)
    artifact = dict(generator=file_record(__file__), post_hoc=True, inference_performed=False,
        inputs=[file_record('paper/self-revision/r3/plan.md'), file_record(OUT/'final_complete.json'),
                file_record('scripts/paper_closeout_report.py'),
                complete['contract'], *complete['results']], artifacts=[file_record(p) for p in paths])
    artifact['legacy_context'] = [file_record(p) for p in (
        'work_dirs/r1-ls1024.log', 'work_dirs/r1-nofallback.log', 'work_dirs/r2-bootstrap.log',
        'work_dirs/baselines/video-depth-anything-small-metric-28-4m-tum-l256.json',
        'work_dirs/baselines/video-depth-anything-small-metric-28-4m-bonn-l256.json',
        'work_dirs/v11-longclip-spread-s0/scores_L256K30.json',
        'scripts/eval_baseline_vda.py', 'scripts/bootstrap_ci.py', 'scripts/train.py',
        'sokkanaem/data.py', 'sokkanaem/schedule.py',
        'work_dirs/v11-longclip-spread-s0/config.toml')]
    artifact['inspected_external_sources'] = [file_record(
        Path('/home/hyunsu/checkouts/Video-Depth-Anything/video_depth_anything')/p)
        for p in ('video_depth.py', 'motion_module/motion_module.py')]
    artifact['legacy_linkage_verified'] = False
    Path('work_dirs/paper_revision_r3').mkdir(parents=True, exist_ok=True)
    Path('work_dirs/paper_revision_r3/results.json').write_text(json.dumps(artifact, indent=2)+'\n')
    for r in table_rows:
        print(' | '.join(r))


if __name__ == '__main__':
    main()
