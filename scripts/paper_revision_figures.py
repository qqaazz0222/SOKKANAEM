"""Generate R3 vector/PNG figures from verified saved CSVs; no inferred images/data."""
import csv
import html
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paper_closeout_study import verify_files
from scripts.paper_closeout_report import LABELS, FINAL_MODELS
from sokkanaem.protocol import file_record

DEST = Path('paper/submission/figures')
INK, MUTED, BLUE, GRAY, ORANGE = '#22354d', '#526477', '#087f8c', '#a5bdcf', '#ad6525'


def text(x, y, value, size=18, color=INK, anchor='start', bold=False):
    return (f'<text font-family="Helvetica" x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'text-anchor="{anchor}" font-weight="{700 if bold else 400}">'
            f'{html.escape(str(value))}</text>')


def line(x1, y1, x2, y2, color='#dbe3eb', width=1.5):
    return f'<path d="M{x1},{y1} L{x2},{y2}" stroke="{color}" stroke-width="{width}" fill="none"/>'


def rect(x, y, w, h, fill, stroke='none', radius=0):
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
            f'fill="{fill}" stroke="{stroke}"/>')


def canvas(title, subtitle, body, height=675):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="{height-100}" '
            f'viewBox="0 90 1400 {height-100}">'
            + rect(0, 0, 1400, height, '#ffffff')
            + ''.join(body) + '</svg>')


def bars(x, title, labels, values, limit, ticks, fmt, intervals=None, spacing=49):
    """Zero-based horizontal axes. Error bars, when supplied, are explicitly captioned."""
    out = [rect(x-20, 98, 670, 465 if len(values) == 8 else 370, '#f8fafc', '#d5dfe8', 12),
           text(x, 128, title, 20, bold=True)]
    left, width, top = x+205, 375, 163
    bottom = top + spacing*(len(values)-1)+20
    for tick in ticks:
        px = left + width*tick/limit
        out.extend([line(px, top-18, px, bottom), text(px, bottom+28, f'{tick:g}', 15, MUTED, 'middle')])
    for i, (label, value) in enumerate(zip(labels, values)):
        y = top + spacing*i
        color = BLUE if label == 'Sparse K30' else GRAY
        out.extend([text(left-14, y+5, label, 17, anchor='end', bold=label == 'Sparse K30'),
                    rect(left, y-11, width*value/limit, 22, color)])
        if intervals:
            lo, hi = intervals[i]
            a, b = left+width*lo/limit, left+width*hi/limit
            out.extend([line(a, y, b, y, INK, 2), line(a, y-5, a, y+5, INK, 2),
                        line(b, y-5, b, y+5, INK, 2)])
        out.append(text(left+width+13, y+5, fmt(value), 17, bold=label == 'Sparse K30'))
    return out


def architecture():
    """Editable model schematic: data, control, and persistent storage are distinct."""
    body = []
    navy, teal, amber = '#22354d', '#087f8c', '#ad6525'
    def label(x, y, value, size=18, color=INK, anchor='start', bold=False):
        body.append(text(x, y, value, size, color, anchor, bold))
    def card(x, y, w, h, fill, border='#d5dfe8'):
        body.append(rect(x, y, w, h, fill, border, 12))
    def arrow(points, color=navy, dashed=False, both=False):
        path = ' '.join(('M' if i == 0 else 'L') + f'{x},{y}' for i, (x,y) in enumerate(points))
        if dashed:
            # Explicit segments survive the SVG-to-PDF renderer's dash limitations.
            for (x1, y1), (x2, y2) in zip(points, points[1:]):
                length = ((x2-x1)**2 + (y2-y1)**2)**.5
                for start in range(0, int(length), 13):
                    end = min(start+7, length)
                    body.append(line(x1+(x2-x1)*start/length, y1+(y2-y1)*start/length,
                                     x1+(x2-x1)*end/length, y1+(y2-y1)*end/length, color, 2.2))
        else:
            body.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2.2" stroke-linejoin="round"/>')
        def head(a, b):
            dx, dy = b[0]-a[0], b[1]-a[1]
            norm = (dx*dx+dy*dy)**.5
            ux, uy = dx/norm, dy/norm
            x, y = b
            body.append(f'<path d="M{x},{y} L{x-9*ux+4*uy},{y-9*uy-4*ux} L{x-9*ux-4*uy},{y-9*uy+4*ux} Z" fill="{color}"/>')
        head(points[-2], points[-1])
        if both:
            head(points[1], points[0])

    # One clearly directed prediction path; four blocks are expanded explicitly.
    card(25, 115, 155, 125, '#f4f7fa')
    for i in range(3):
        body.append(rect(51+i*9, 134+i*7, 45, 37, ['#ccd9e5','#a5bdcf','#779bb6'][i], '#ffffff', 4))
    label(102, 203, 'RGB frame', 20, navy, 'middle', True)
    label(102, 225, '256 × 256', 16, MUTED, 'middle')
    card(230, 115, 195, 125, '#f4f7fa')
    label(327, 152, 'Patch embedding', 20, navy, 'middle', True)
    label(327, 183, '16 × 16 patches', 17, MUTED, 'middle')
    label(327, 211, '192-d tokens', 17, MUTED, 'middle')
    card(480, 70, 500, 205, '#f8fafc')
    label(505, 101, 'Alternating state-space blocks', 20, navy, bold=True)
    for x, name, detail, fill, accent in [
        (505, 'T₁', 'Temporal', '#e4f3f2', teal),
        (625, 'S₁', 'Spatial', '#edf0f8', '#626c9c'),
        (745, 'T₂', 'Temporal', '#e4f3f2', teal),
        (865, 'S₂', 'Spatial', '#edf0f8', '#626c9c')]:
        card(x, 125, 90, 91, fill)
        label(x+45, 163, name, 27, accent, 'middle', True)
        label(x+45, 194, detail, 15, accent, 'middle')
    for a,b in [(180,230),(425,480),(595,625),(715,745),(835,865),(980,1030),(1235,1280)]:
        arrow([(a,177),(b,177)])
    label(730, 249, 'Selected updates · retained temporal memory', 16, MUTED, 'middle')
    card(1030, 115, 205, 125, '#f4f7fa')
    label(1132, 152, 'Dense decoder', 20, navy, 'middle', True)
    label(1132, 183, 'DPT-style · 64 bins', 17, MUTED, 'middle')
    label(1132, 211, 'All pixels, every frame', 16, MUTED, 'middle')
    # Stylized tensor icon, not a measured depth prediction.
    for i, shade in enumerate(['#dcecf0','#a8d3da','#6aafbe','#31869d']):
        body.append(rect(1285+i*19, 140, 19, 65, shade, radius=0))
    label(1323, 231, 'Depth', 19, navy, 'middle', True)

    # Control mask is dashed and does not look like a feature tensor.
    card(25, 355, 400, 145, '#eef8f7', '#b9dcd7')
    label(48, 387, 'Change detector + refresh', 21, teal, bold=True)
    label(48, 417, 'Patch MSE → hysteresis → dilation', 18, navy)
    label(48, 446, 'Periodic keyframe / high-activity fallback', 17, MUTED)
    label(48, 477, 'Output: active patch mask', 18, teal, bold=True)
    arrow([(102,240),(102,355)], teal)
    label(115, 298, 'Previous frame +', 16, MUTED)
    label(115, 320, 'detector history', 16, MUTED)
    arrow([(425,427),(450,427),(450,295),(490,295),(490,275)], teal, dashed=True)

    card(505, 355, 210, 145, '#e4f3f2', '#b9dcd7')
    label(610, 389, 'Temporal states', 20, teal, 'middle', True)
    label(610, 422, 'Active: update', 18, navy, 'middle')
    label(610, 451, 'Inactive: copy', 18, navy, 'middle')
    label(610, 479, 'Exact state retention', 15, teal, 'middle')
    card(755, 355, 225, 145, '#fff4e9', '#e8cbae')
    label(867, 389, 'T / S output caches', 20, amber, 'middle', True)
    label(867, 422, 'Active: recompute', 18, navy, 'middle')
    label(867, 451, 'Inactive: reuse', 18, navy, 'middle')
    label(867, 479, 'Approximate readout', 15, amber, 'middle')
    arrow([(610,275),(610,355)], teal, both=True)
    arrow([(867,275),(867,355)], amber, both=True)
    label(738, 328, 'Read / write across frames', 16, MUTED, 'middle')

    # Compact symbol key instead of theory panels or an in-image title.
    arrow([(1040,377),(1090,377)])
    label(1103,383,'Feature flow',17,navy)
    arrow([(1040,422),(1090,422)],teal,dashed=True)
    label(1103,428,'Activation control',17,teal)
    arrow([(1040,467),(1090,467)],amber,both=True)
    label(1103,473,'Persistent storage',17,amber)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="475" '
            'viewBox="0 45 1400 475">'
            + rect(0, 0, 1400, 530, '#ffffff')
            + ''.join(body) + '</svg>')


def main():
    revision = json.loads(Path('work_dirs/paper_revision_r3/results.json').read_text())
    report = json.loads(Path('work_dirs/paper_closeout/report_artifacts.json').read_text())
    for a in (revision, report):
        verify_files([a['generator'], *a['inputs'], *a['artifacts']])
    def read(path):
        with Path(path).open() as f:
            return list(csv.DictReader(f))
    quality_path = Path('paper/submission/tables/final_quality.csv')
    scale_path = Path('paper/submission/tables/revision_scale_summary.csv')
    cost_path = Path('paper/tables/study9_operating_points.csv')
    quality, scale, cost = read(quality_path), read(scale_path), read(cost_path)
    order = list(FINAL_MODELS)
    labels = [LABELS[m] for m in order]
    accuracy = []
    for x, length, count in ((40, 8, 462), (725, 256, 13)):
        rows = {r['model']: r for r in quality if r['length'] == str(length)
                and r['gauge'] == 'scaleshift' and r['aggregation'] == 'source_pixel_pooled'}
        assert set(rows) == set(order)
        accuracy.extend(bars(x, f'L{length} | {count} clips', labels,
            [float(rows[m]['absrel']) for m in order], .25, [0, .05, .1, .15, .2, .25], lambda n: f'{n:.4f}'))
    accuracy.append(text(700, 600, 'Clip scale–shift AbsRel · lower is better', 18, MUTED, 'middle'))
    scale_body = []
    for x, metric, title, limit, ticks in ((40, 'scale_drift', 'L256 scale CV', .45, [0, .15, .3, .45]),
            (725, 'scale_step', 'L256 mean absolute log-scale step', .1, [0, .025, .05, .075, .1])):
        rows = {r['model']: r for r in scale if r['length'] == '256' and r['metric'] == metric}
        assert set(rows) == set(order)
        intervals = [(float(rows[m]['ci_low']), float(rows[m]['ci_high'])) for m in order]
        assert all(0 <= lo <= hi <= limit for lo, hi in intervals)
        scale_body.extend(bars(x, title, labels, [float(rows[m]['equal_sequence_mean']) for m in order],
            limit, ticks, lambda n: f'{n:.4f}', intervals=intervals))
    scale_body.append(text(700, 600, 'Lower is better · whiskers: exploratory 95% sequence-bootstrap intervals (n = 4)', 18, MUTED, 'middle'))
    cost_order = ['sparse_k30', 'dense_carry', 'output_hold', 'midas_256', 'da2_256']
    cost_labels = ['Sparse K30', 'Dense carry', 'Output hold', 'MiDaS Small 256', 'DA2 Small 252']
    rows = {r['model']: r for r in cost}
    times = [float(rows[m]['mean_ms']) for m in cost_order]
    spans = [(float(rows[m]['repeat_min_ms']), float(rows[m]['repeat_max_ms'])) for m in cost_order]
    cost_body = bars(40, 'PNG-to-depth latency (ms / frame)', cost_labels, times, 12, [0, 3, 6, 9, 12],
                     lambda n: f'{n:.3f}', intervals=spans, spacing=63)
    cost_body += bars(725, 'Persistent stream state + caches (MiB)', cost_labels,
                     [float(rows[m]['state_mib']) for m in cost_order], 16, [0, 4, 8, 12, 16],
                     lambda n: f'{n:.3f}', spacing=63)
    cost_body.append(text(700, 505, 'RTX 4090 · FP32 · whiskers: min–max of five repeat means', 18, MUTED, 'middle'))
    specs = {'r3_architecture': architecture(),
             'r3_accuracy': canvas('Depth accuracy of SOKKANAEM and the frozen comparison paths',
                 'Eight fixed paths | four reserved TUM sequences | aligned shape, not calibration-free metric depth', accuracy, height=625),
             'r3_cost': canvas('Measured pipeline latency and persistent stream memory',
                 'Same development timing boundary and footage for every plotted operating point', cost_body, height=530),
             'r3_scale': canvas('Scale stability: within-clip variation and short-step jitter',
                 'Post-hoc descriptive analysis of stored final scores | no new inference or tuning', scale_body, height=625)}
    sys.path.insert(0, str(Path('work_dirs/paper_tools/pdf_python').resolve()))
    import fitz
    DEST.mkdir(parents=True, exist_ok=True)
    artifacts = []
    for name, svg in specs.items():
        path = DEST/(name+'.svg')
        path.write_text(svg)
        svg_doc = fitz.open(stream=svg.encode(), filetype='svg')
        doc = fitz.open(stream=svg_doc.convert_to_pdf(), filetype='pdf')
        doc.save(DEST/(name+'.pdf'), no_new_id=True)
        doc[0].get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(DEST/(name+'.png'))
        doc.close(); svg_doc.close()
        artifacts.extend(file_record(DEST/(name+suffix)) for suffix in ('.svg', '.pdf', '.png'))
    diag = DEST/'development_state_diagnostic.pdf'
    doc = fitz.open(diag)
    doc[0].get_pixmap(matrix=fitz.Matrix(2, 2)).save(DEST/'development_state_diagnostic.png')
    doc.close()
    artifacts.append(file_record(DEST/'development_state_diagnostic.png'))
    result = dict(generator=file_record(__file__), inference_performed=False,
        inputs=[file_record(p) for p in (quality_path, scale_path, cost_path, diag,
                Path('work_dirs/paper_revision_r3/results.json'))], artifacts=artifacts,
        rendering=dict(pymupdf=fitz.VersionBind, png_scale=1.5, chart_axes_start_at_zero=True))
    Path('work_dirs/paper_revision_r3/figures.json').write_text(json.dumps(result, indent=2)+'\n')
    print('Generated four new SVG/PDF/PNG figures and one diagnostic PNG; all input hashes recorded.')


if __name__ == '__main__':
    main()
