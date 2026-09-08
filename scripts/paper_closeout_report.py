"""Validate final evidence and generate the submission tables; never runs inference."""
from collections import defaultdict
import csv
import json
from pathlib import Path
import statistics
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_closeout_study import OUT, FINAL_MODELS, verify_files, prior_check
from scripts.paper_efficiency_report import aggregate_quality
from sokkanaem.metrics import pooled
from sokkanaem.performance_study import paired_sequence_summary
from sokkanaem.protocol import file_record

SUB = Path('paper/submission')
TABLES = SUB/'tables'
LABELS = dict(sparse_k30='Sparse K30', sparse_k5='Sparse K5', dense_carry='Dense carry',
              dense_reset='Dense reset', output_hold='Output hold', da2_256='DA2 Small 252',
              midas_256='MiDaS Small 256', dpt_common='DPT-Large 256')


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines()]


def csv_write(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_coverage(rows, doc):
    expected = {}
    for source in doc['sources']:
        clips = [c for c in doc['clips'] if c['source'] == source]
        expected.update({f'{source}:{i}': c for i, c in enumerate(clips)})
    if len(rows) != len(expected) or {r['id'] for r in rows} != set(expected):
        raise ValueError('incomplete or duplicated final manifest coverage')
    for r in rows:
        c = expected[r['id']]
        if (r['pairs'] != c['pairs'] or r['clip_len'] != doc['clip_len'] or
                r['sequence'] != Path(c['pairs'][0][0]).parent.parent.name or
                r['source'] != c['source']):
            raise ValueError('frame identity mismatch')
        if set(r['models']) != (set() if r['no_gt'] else set(FINAL_MODELS)):
            raise ValueError('model/GT coverage mismatch')


def tex_table(caption, label, header, lines, spec):
    return ('\\begin{table}[H]\n\\caption{'+caption+'}\\label{'+label+'}\n'
            '\\centering\\small\n\\begin{tabular}{'+spec+'}\\toprule\n'+header+
            ' \\\\\n\\midrule\n'+'\n'.join(' & '.join(x)+' \\\\' for x in lines)+
            '\n\\bottomrule\\end{tabular}\n\\end{table}\n')


def main():
    prior_check()
    complete = json.loads((OUT/'final_complete.json').read_text())
    verify_files([complete['contract'], *complete['results']])
    contract = json.loads((OUT/'final_contract.json').read_text())
    verify_files(contract['source_paths'])
    diag = json.loads((OUT/'theory_complete.json').read_text())
    verify_files([diag['result'], diag['source'], diag['plan']])
    TABLES.mkdir(parents=True, exist_ok=True)
    quality, sequences, regions, framefits, paired = [], [], [], [], []
    aggregates, counts = {}, {}
    for length in (256, 8):
        rows = read_rows(OUT/f'final_L{length}.jsonl')
        doc = json.loads(Path(f'manifests/paper_test_tum_L{length}.json').read_text())
        validate_coverage(rows, doc)
        counts[length] = dict(clips=len(rows), frames=len(rows)*length,
                              no_gt=sum(r['no_gt'] for r in rows))
        a = aggregate_quality(rows)
        aggregates[length] = a
        for (model, gauge), values in a.items():
            for mode in ('balanced', 'equal_sequence'):
                quality.append(dict(length=length, model=model, gauge=gauge,
                    aggregation='source_pixel_pooled' if mode == 'balanced' else mode,
                    **values[mode], clips=values['n'], failed=values['failed'],
                    catastrophic=values['catastrophic']))
            for seq, val in values['sequences'].items():
                sequences.append(dict(length=length, model=model, gauge=gauge, sequence=seq, **val))
        groups = defaultdict(list)
        for row in rows:
            for model, item in row['models'].items():
                for gauge, value in item['gauges'].items():
                    groups[(model, gauge)].append(value)
        for (model, gauge), values in groups.items():
            for region in values[0]['regions']:
                px = sum(v['regions'][region]['px'] for v in values)
                regions.append(dict(length=length, model=model, gauge=gauge, region=region,
                    valid_pixels=px, absrel=sum(v['regions'][region]['rel'] for v in values)/px if px else None,
                    delta1=sum(v['regions'][region]['d1'] for v in values)/px if px else None))
            if gauge.endswith('/frame'):
                scores = pooled([v['scores']['_pooled'] for v in values])
                framefits.append(dict(length=length, model=model, gauge=gauge,
                    **{m: scores[m] for m in ('absrel', 'rmse', 'delta1')},
                    failed=sum(v['failed'] for v in values), temporal_metrics='not reported'))
        for reference in ('dense_carry', 'sparse_k5', 'midas_256', 'dpt_common'):
            for metric in ('absrel', 'tce'):
                left, right = a['sparse_k30', 'scaleshift'], a[reference, 'scaleshift']
                names = sorted(left['sequences'])
                assert names == sorted(right['sequences']) and len(names) == 4
                diff = [left['sequences'][s][metric]-right['sequences'][s][metric] for s in names]
                paired.append(dict(length=length, comparison='sparse_k30 - '+reference,
                    gauge='scaleshift', metric=metric, **paired_sequence_summary(diff)))
    for name, data in [('final_quality', quality), ('final_sequences', sequences),
                       ('final_regions', regions), ('final_framefit', framefits), ('final_paired', paired)]:
        csv_write(TABLES/f'{name}.csv', data)
    diagnostics = read_rows(OUT/'theory_dev.jsonl')
    assert len(diagnostics) == 4 and sum(len(r['frames']) for r in diagnostics) == 1024
    frames, summary = [], []
    for r in diagnostics:
        frames += [dict(sequence=r['sequence'], **f) for f in r['frames']]
        keys = [f for f in r['frames'] if f['keyframe'] and f['frame'] > 0]
        assert len(keys) == 8 and all(f['net_local_defect_rms'] == 0 for f in keys)
        assert all(f['first_shadow_error_rms'] <= f['first_shadow_bound_rms']+1e-8 for f in r['frames'])
        summary.append(dict(sequence=r['sequence'], frames=len(r['frames']), refreshes=len(keys),
            refresh_local_defect_max=max(f['net_local_defect_rms'] for f in keys),
            refresh_history_rms_median=statistics.median(f['net_state_error_rms'] for f in keys),
            refresh_history_rms_min=min(f['net_state_error_rms'] for f in keys),
            max_cache_age=max(f['age_max'] for f in r['frames']),
            max_fp32_shadow_gap=max(f['fp32_shadow_max_gap'] for f in r['frames'])))
    csv_write(TABLES/'development_diagnostic_frames.csv', frames)
    csv_write(TABLES/'development_diagnostic_summary.csv', summary)
    sys.path.insert(0, str(Path('work_dirs/paper_tools/pdf_python').resolve()))
    import fitz
    (SUB/'figures').mkdir(exist_ok=True)
    figure = fitz.open()
    page = figure.new_page(width=700, height=490)
    ink, muted = (.133, .208, .302), (.322, .392, .467)
    teal, amber = (.031, .498, .549), (.678, .396, .145)
    for index, r in enumerate(diagnostics):
        fs = r['frames']
        left, top, width, height = 50+350*(index % 2), 50+220*(index//2), 280, 145
        high = max(max(f['net_state_error_rms'], f['net_local_defect_rms']) for f in fs)*1.05
        title = r['sequence'].replace('rgbd_dataset_freiburg3_', 'TUM ').replace('rgbd_bonn_', 'Bonn ')
        page.draw_rect(fitz.Rect(left-40, top-37, left+width+10, top+height+32),
                       color=(.835, .875, .910), fill=(.973, .980, .988), width=.5, radius=.04)
        page.insert_text((left, top-22), title, fontsize=10, fontname='hebo', color=ink)
        page.insert_text((left, top-8), 'Internal state RMS', fontsize=8, color=muted)
        page.draw_line((left, top), (left, top+height), color=muted, width=.5)
        page.draw_line((left, top+height), (left+width, top+height), color=muted, width=.5)
        for tick in (0, 60, 120, 180, 255):
            page.insert_text((left+tick/255*width-4, top+height+12), str(tick), fontsize=8, color=muted)
        for fraction in (0, .5, 1):
            y = top+height*(1-fraction)
            page.draw_line((left, y), (left+width, y), color=(.86, .90, .93), width=.4)
            page.insert_text((left-27, y+2), f'{high*fraction:.2f}', fontsize=8, color=muted)
        for f in fs:
            if f['keyframe']:
                x = left+f['frame']/255*width
                page.draw_line((x, top), (x, top+height), color=(.7, .7, .7), width=.4, dashes='[1 2]')
        for field, color in [('net_state_error_rms', teal),
                             ('net_local_defect_rms', amber)]:
            points = [(left+f['frame']/255*width, top+height*(1-f[field]/high)) for f in fs]
            page.draw_polyline(points, color=color, width=.9)
        page.insert_text((left+70, top+height+27), 'Frame within development clip', fontsize=8, color=muted)
    for x, label, color in [(55, 'Dense-history difference', teal),
                             (300, 'Same-state local defect', amber), (550, 'Keyframe', (.7, .7, .7))]:
        page.draw_line((x, 473), (x+25, 473), color=color, width=1,
                       dashes='[1 2]' if label == 'Keyframe' else None)
        page.insert_text((x+33, 476), label, fontsize=9, color=ink)
    figure.save(SUB/'figures/development_state_diagnostic.pdf', no_new_id=True)
    figure.close()
    macros = '\n'.join('\\newcommand{\\'+macro+'}{'+f'{aggregates[256][model, "scaleshift"]["balanced"]["absrel"]:.4f}'+'}'
        for macro, model in [('FinalNative', 'sparse_k30'), ('FinalDense', 'dense_carry'), ('FinalMidas', 'midas_256')])
    (SUB/'generated_macros.tex').write_text('% Generated; do not hand edit.\n'+macros+'\n')
    tex = '\\subsection{Frozen final transfer evaluation}\n'
    for length in (256, 8):
        a = aggregates[length]
        lines = [[LABELS[m], *[f'{a[m,"scaleshift"]["balanced"][k]:.4f}' for k in ('absrel', 'rmse', 'delta1', 'tce')],
                  str(a[m,'scaleshift']['failed'])+'/'+str(a[m,'scaleshift']['n'])] for m in FINAL_MODELS]
        tex += tex_table(f'Final TUM L{length}: clip scale--shift alignment, source pixel-pooled. '
            'AbsRel, RMSE and TCE lower is better; $\\delta_1$ higher is better. '
            'Fail is the number of clips with any invalid fitted disparity; failures remain in aggregates.',
            f'tab:final{length}', 'Model & AbsRel & RMSE & $\\delta_1$ & TCE & Fail', lines, 'lrrrrr')
    a = aggregates[256]
    tex += ('These panels measure aligned relative shape, not calibration-free metric depth. '
            'The selected K30 model is retained irrespective of its test ranking. '
            'All '+str(counts[256]['clips'])+' L256 and '+str(counts[8]['clips'])+' L8 clips were processed; '
            +str(counts[256]['no_gt']+counts[8]['no_gt'])+' clips had no scorable GT.\n')
    tex += tex_table('Final L256 native K30: alignment gauges must not be conflated.', 'tab:gauges',
        'Gauge & AbsRel & RMSE & $\\delta_1$ & TCE',
        [[g, *[f'{a["sparse_k30",g]["balanced"][m]:.4f}' for m in ('absrel', 'rmse', 'delta1', 'tce')]]
         for g in ('none', 'median', 'scaleshift')], 'lrrrr')
    tex += tex_table('Exploratory equal-sequence L256 AbsRel differences (K30 minus comparator), '
        '$n=4$. Unadjusted percentile sequence-bootstrap intervals; exact sign flips.', 'tab:paired',
        'Comparator & Difference & 95\\% interval & $p$',
        [[LABELS[p['comparison'].split(' - ')[1]], f'{p["difference"]:+.4f}',
          f'[{p["ci_low"]:+.4f}, {p["ci_high"]:+.4f}]', f'{p["sign_flip_p"]:.3f}']
         for p in paired if p['length'] == 256 and p['metric'] == 'absrel'], 'lrrr')
    tex += ('\\subsection{Development diagnostics and pipeline cost}\n'
        'Across 1024 development frames, all 32 post-initial keyframes have zero same-state local defect. '
        'Nevertheless, the median concatenated hidden/cache state RMS differences at those keyframes are '
        +', '.join(f'{r["refresh_history_rms_median"]:.3f}' for r in summary)+
        ' in manifest sequence order. These are internal-state units, not depth errors. '
        'Cache age never exceeds 29. The FP64 first-block shared-input shadow bound holds at all 1024 '
        'steps; this does not certify FP32 end-to-end error. Full sequence names and frame traces are in the supplement.\n'
        'On the four matched development clips, sparse K30 takes 7.496 ms/frame versus 7.521 for dense '
        'carry, a 0.33\\% reduction. MiDaS Small takes 9.439 ms with matched-subset AbsRel '
        '0.1637 versus 0.1630 for K30. That operating-point comparison is not statistical equivalence: '
        'on all 13 development L256 clips, K30/MiDaS AbsRel is 0.2091/0.1749. '
        'The native sparse stream uses 13.501 MiB of persistent state versus 12.000 MiB dense. '
        'Read/decode/preprocessing accounts for 69.4\\% of a separate synchronized profile. '
        'The associated idealized fixed-input ceiling is 1.44 times that profile, not a measured speedup.\n'
        'Existing final-stage seed replicates give source-balanced scale--shift AbsRel '
        '$0.1146\\pm0.0007$ on development L8 and $0.2040\\pm0.0077$ on development L256 '
        '(mean and sample SD, three seeds sharing the earlier training stages).\n')
    tex += ('\\begin{figure}[H]\n\\centering\n'
        '\\includegraphics[width=\\linewidth]{figures/development_state_diagnostic.pdf}\n'
        '\\caption{Development-only extended-state diagnostics. Dotted lines mark keyframes. '
        'The same-state local defect vanishes at refresh, while the difference from the separately '
        'evolved dense history persists. Curves are RMS over concatenated hidden states and caches, '
        'not GT depth error or uniquely allocated memory.}\\label{fig:diagnostic}\n\\end{figure}\n')
    (SUB/'generated_results.tex').write_text('% Generated from checked result files.\n'+tex)
    body = Path('paper/theory12.tex').read_text().split('\\section{Implemented recurrence', 1)[1]
    body = '\\section{Implemented recurrence'+body.split('\\paragraph{Reproducibility and limitations.}', 1)[0]
    body = body.replace('\\section{', '\\subsection{')
    for name in ('proposition', 'theorem', 'remark'):
        body = body.replace('{'+name+'}', '{'+name.title()+'}')
    body = body.replace('$bBN\\sum_{\\ell\\in\\mathrm{temporal}}P_\\ell S_\\ell$ bytes, independent of activity.',
        '\\[ bBN\\sum_{\\ell\\in\\mathrm{temporal}}P_\\ell S_\\ell \\]\nbytes, independent of activity.')
    (SUB/'theory_appendix.tex').write_text('% Derived from immutable paper/theory12.tex.\n'+body)
    template = Path('work_dirs/paper_tools/MDPI_template.zip')
    with zipfile.ZipFile(template) as z:
        for name in z.namelist():
            p = Path(name)
            if p.parts[0] == 'Definitions' and '..' not in p.parts and not p.is_absolute() and not name.endswith('/'):
                target = SUB/p
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))
    (SUB/'template_source.json').write_text(json.dumps(dict(
        url='https://mdpi-res.com/data/MDPI_template.zip', retrieved='2026-09-07',
        archive=file_record(template), files=[file_record(p) for p in sorted((SUB/'Definitions').glob('*'))]), indent=2)+'\n')
    md = '# 최종 평가와 이론–영상 진단\n\n2026-09-07. 모델 선택 후 조건을 봉인하고 평가했다. 새 학습이나 최종 결과에 따른 튜닝은 없다.\n\n'
    md += '| 모델 | L256 AbsRel | L8 AbsRel | L256 실패 클립 |\n|---|---:|---:|---:|\n'
    for m in FINAL_MODELS:
        md += f'| {LABELS[m]} | {aggregates[256][m,"scaleshift"]["balanced"]["absrel"]:.4f} | {aggregates[8][m,"scaleshift"]["balanced"]["absrel"]:.4f} | {aggregates[256][m,"scaleshift"]["failed"]}/13 |\n'
    md += ('\nClip scale–shift 보정 후 TUM pixel-pooled 값이다. 낮을수록 좋으며 보정 없는 metric 정확도가 아니다. '
        'L256 13클립(3328프레임), L8 462클립(3696프레임)은 같은 영상에서 추출되어 서로 독립이 아니다. '
        '새로운 시퀀스지만 같은 촬영 환경이며, 장면 분리 일반화를 검증하지 않았다.\n\n'
        '개발 진단 1024프레임에서 첫 프레임 이후 keyframe 32곳의 같은 상태 국소 결손은 0이었다. '
        '그러나 연속 dense 실행과의 누적 state 차이는 남았다. 첫 block FP64 shadow 상한은 모든 step에서 '
        '충족했으나 전체 FP32 네트워크 정확도의 인증은 아니다.\n\n'
        '[전수 지표](submission/tables/final_quality.csv), [시퀀스별](submission/tables/final_sequences.csv), '
        '[paired 통계](submission/tables/final_paired.csv), [영역별](submission/tables/final_regions.csv), '
        '[진단 원자료](submission/tables/development_diagnostic_frames.csv). '
        'n=4로 최소 양측 sign-flip p=.125이며 탐색적 비교다.\n\n'
        '이 테스트는 이제 미사용 상태가 아니다. 추가 개발 시 새로운 미사용 평가 자료가 필요하다. '
        '기존 봉인 파일은 역사 기록으로 유지하고 `final_opened.json`과 `final_complete.json`으로 개봉/완료를 기록했다.\n')
    Path('paper/FINAL_EVALUATION.md').write_text(md)
    generated = [*TABLES.glob('*.csv'), SUB/'generated_macros.tex', SUB/'generated_results.tex',
                 SUB/'theory_appendix.tex', SUB/'template_source.json', Path('paper/FINAL_EVALUATION.md'),
                 SUB/'figures/development_state_diagnostic.pdf']
    (OUT/'report_artifacts.json').write_text(json.dumps(dict(generator=file_record(__file__),
        inputs=[file_record(OUT/'final_complete.json'), file_record(OUT/'theory_complete.json'),
                file_record('paper/theory12.tex'), file_record(template)],
        artifacts=[file_record(p) for p in sorted(generated)]), indent=2)+'\n')
    print(md, flush=True)


if __name__ == '__main__':
    main()
