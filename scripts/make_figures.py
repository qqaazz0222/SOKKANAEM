"""Generate the paper's quantitative figures from the measured numbers.

Hand-placing SVG coordinates is how labels end up on top of data points and
connectors end up pointing at nothing. Everything here is computed from a data
table through one axes helper, so a changed number moves its label with it.
That matters more than usual for this paper: two of the sections these figures
serve are tagged [UNDER TEST], so the numbers are expected to move.

Style targets an MDPI submission: Arial, thin rules, muted fills, no gradients
or shadows, and enough contrast to survive greyscale printing.

    python scripts/make_figures.py --out paper/figures
"""
import argparse
import math
import os

# ---------------------------------------------------------------- style ----
FONT = "Arial, Helvetica, sans-serif"
FG, AXIS, GRID, MUTED = "#1a1a1a", "#333333", "#e8e8e8", "#666666"
# colour-blind safe, and distinguishable once printed in greyscale by their
# very different lightness
NAVY, VERM, TEAL, GRAY, PURPLE, OCHRE = (
    "#1f4e79", "#b8442c", "#2e8b7a", "#8c8c8c", "#6b4fa0", "#c88a1e")


def esc(s):
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def txt(x, y, s, size=12, fill=FG, anchor="start", weight="normal",
        style="normal", rotate=None):
    tr = f' transform="rotate({rotate} {x} {y})"' if rotate is not None else ""
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}" '
            f'font-style="{style}"{tr}>{esc(s)}</text>')


class Ax:
    """One panel. Data coordinates in, pixel coordinates out."""

    def __init__(self, x0, y0, w, h, xlim, ylim, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h = x0, y0, w, h
        self.xlim, self.ylim, self.xlog, self.ylog = xlim, ylim, xlog, ylog

    def _n(self, v, lim, log):
        a, b = (math.log10(lim[0]), math.log10(lim[1])) if log else lim
        v = math.log10(v) if log else v
        return (v - a) / (b - a)

    def X(self, v):
        return self.x0 + self._n(v, self.xlim, self.xlog) * self.w

    def Y(self, v):
        return self.y0 + self.h - self._n(v, self.ylim, self.ylog) * self.h

    def frame(self, xticks, yticks, xlabel, ylabel, xfmt="{:g}", yfmt="{:g}",
              title=None, grid=True):
        o = []
        if grid:
            for t in yticks:
                o.append(f'<line x1="{self.x0}" y1="{self.Y(t):.1f}" '
                         f'x2="{self.x0+self.w}" y2="{self.Y(t):.1f}" '
                         f'stroke="{GRID}" stroke-width="0.8"/>')
        # only left and bottom rules: MDPI figures are rarely boxed
        o.append(f'<line x1="{self.x0}" y1="{self.y0+self.h}" '
                 f'x2="{self.x0+self.w}" y2="{self.y0+self.h}" '
                 f'stroke="{AXIS}" stroke-width="1"/>')
        o.append(f'<line x1="{self.x0}" y1="{self.y0}" x2="{self.x0}" '
                 f'y2="{self.y0+self.h}" stroke="{AXIS}" stroke-width="1"/>')
        for t in xticks:
            x = self.X(t)
            o.append(f'<line x1="{x:.1f}" y1="{self.y0+self.h}" x2="{x:.1f}" '
                     f'y2="{self.y0+self.h+4}" stroke="{AXIS}" stroke-width="1"/>')
            o.append(txt(x, self.y0 + self.h + 17, xfmt.format(t), 11, MUTED,
                         "middle"))
        for t in yticks:
            y = self.Y(t)
            o.append(f'<line x1="{self.x0-4}" y1="{y:.1f}" x2="{self.x0}" '
                     f'y2="{y:.1f}" stroke="{AXIS}" stroke-width="1"/>')
            o.append(txt(self.x0 - 8, y + 4, yfmt.format(t), 11, MUTED, "end"))
        o.append(txt(self.x0 + self.w / 2, self.y0 + self.h + 36, xlabel, 12))
        o[-1] = txt(self.x0 + self.w / 2, self.y0 + self.h + 36, xlabel, 12,
                    FG, "middle")
        o.append(txt(self.x0 - 46, self.y0 + self.h / 2, ylabel, 12, FG,
                     "middle", rotate=-90))
        if title:
            o.append(txt(self.x0, self.y0 - 12, title, 12.5, FG, "start",
                         "bold"))
        return "\n".join(o)

    def line(self, pts, colour, width=1.8, dash=None, marker=3.2,
             fill_marker=True):
        d = f' stroke-dasharray="{dash}"' if dash else ""
        p = " ".join(f"{self.X(a):.1f},{self.Y(b):.1f}" for a, b in pts)
        o = [f'<polyline points="{p}" fill="none" stroke="{colour}" '
             f'stroke-width="{width}"{d}/>']
        for a, b in pts:
            fill = colour if fill_marker else "#ffffff"
            o.append(f'<circle cx="{self.X(a):.1f}" cy="{self.Y(b):.1f}" '
                     f'r="{marker}" fill="{fill}" stroke="{colour}" '
                     f'stroke-width="1.4"/>')
        return "\n".join(o)

    def hline(self, v, colour, label=None, dash="5 4"):
        y = self.Y(v)
        o = [f'<line x1="{self.x0}" y1="{y:.1f}" x2="{self.x0+self.w}" '
             f'y2="{y:.1f}" stroke="{colour}" stroke-width="1.2" '
             f'stroke-dasharray="{dash}"/>']
        if label:
            o.append(txt(self.x0 + self.w - 3, y - 5, label, 10.5, colour,
                         "end"))
        return "\n".join(o)

    def vline(self, v, colour, label=None, dash="4 3"):
        x = self.X(v)
        o = [f'<line x1="{x:.1f}" y1="{self.y0}" x2="{x:.1f}" '
             f'y2="{self.y0+self.h}" stroke="{colour}" stroke-width="1.1" '
             f'stroke-dasharray="{dash}"/>']
        if label:
            o.append(txt(x - 4, self.y0 + 12, label, 10.5, colour, "end"))
        return "\n".join(o)


def wrap(x, y, text, width, size=11, fill=MUTED, lead=16):
    """Caption text broken to fit `width` px. Hand-counted line breaks drift
    the moment a number changes, and a clipped caption is worse than none."""
    cw = size * 0.52                      # mean Arial advance, latin text
    per = max(int(width / cw), 10)
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        cand = (cur + " " + w_).strip()
        if len(cand) > per and cur:
            lines.append(cur)
            cur = w_
        else:
            cur = cand
    if cur:
        lines.append(cur)
    return "\n".join(txt(x, y + i * lead, ln, size, fill)
                      for i, ln in enumerate(lines))


def place_labels(items, size=10.5, pad=3.0):
    """Greedy non-overlapping label placement.

    items: (cx, cy, r, text, colour, bold). Each label tries positions around
    its marker in order of preference and takes the first that hits neither a
    placed label nor any marker. Fixed per-point offsets were what put two
    model names on top of each other in the first draft of this figure.
    """
    boxes, out = [], []
    markers = [(cx, cy, r) for cx, cy, r, *_ in items]

    def hits(bx):
        x1, y1, x2, y2 = bx
        for ox1, oy1, ox2, oy2 in boxes:
            if x1 < ox2 + pad and ox1 < x2 + pad and y1 < oy2 + pad and oy1 < y2 + pad:
                return True
        for mx, my, mr in markers:
            if (x1 < mx + mr and mx - mr < x2
                    and y1 < my + mr and my - mr < y2):
                return True
        return False

    for cx, cy, r, text, colour, bold in items:
        w = len(text) * size * 0.54
        h = size * 1.05
        best = None
        for dy, anchor in ((-(r + 7), "middle"), (r + h + 3, "middle")):
            for dx in (0, -w * 0.6, w * 0.6):
                bx = (cx + dx - w / 2, cy + dy - h, cx + dx + w / 2, cy + dy)
                if not hits(bx):
                    best = (cx + dx, cy + dy, anchor, bx)
                    break
            if best:
                break
        if best is None:                   # everything blocked: go right
            best = (cx + r + 5 + w / 2, cy + h / 3, "middle",
                    (cx + r + 5, cy - h / 2, cx + r + 5 + w, cy + h / 2))
        x, y, anchor, bx = best
        boxes.append(bx)
        out.append(txt(x, y, text, size, colour, anchor,
                       "bold" if bold else "normal"))
    return "\n".join(out)


def svg(w, h, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
            f'width="{w}" height="{h}" font-family="{FONT}">\n'
            f'<rect width="{w}" height="{h}" fill="#ffffff"/>\n{body}\n</svg>\n')


def legend(x, y, items, size=11, dx=0, dy=16):
    """items: (label, colour, kind) with kind in {line, dash, marker}."""
    o = []
    for i, (lab, col, kind) in enumerate(items):
        cx, cy = x + i * dx, y + i * dy
        if kind == "dash":
            o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+22}" y2="{cy}" '
                     f'stroke="{col}" stroke-width="1.8" stroke-dasharray="5 4"/>')
        elif kind == "marker":
            o.append(f'<circle cx="{cx+11}" cy="{cy}" r="4" fill="{col}"/>')
        else:
            o.append(f'<line x1="{cx}" y1="{cy}" x2="{cx+22}" y2="{cy}" '
                     f'stroke="{col}" stroke-width="1.8"/>')
            o.append(f'<circle cx="{cx+11}" cy="{cy}" r="3.2" fill="{col}"/>')
        o.append(txt(cx + 28, cy + 4, lab, size, FG))
    return "\n".join(o)


# ------------------------------------------------------------ measured -----
# Table 1 (Section 5.1): activity sweep, dataset-balanced mean, 100 clips/source
SWEEP_REAL = [(100.0, 0.1290), (94.9, 0.1288), (85.0, 0.1287), (63.6, 0.1282),
               (22.0, 0.1302), (4.6, 0.1373)]
SWEEP_SYN = [(100.0, 0.4299), (62.8, 0.4305), (58.1, 0.4315), (51.2, 0.4313),
              (40.9, 0.4317), (32.1, 0.4332)]
CONST_REAL, CONST_SYN = 0.2761, 0.6303
# index of the default operating point within each sweep (tau_on = 0.05)
DEFAULT_I = 4

# Table 3 (Section 5.3): params (M), AbsRel, t-delta
CMP_REAL = [
    ("DA V1-S", 24.8, 0.0650, 0.0892, GRAY),
    ("DPT-L", 343.0, 0.0875, 0.1162, OCHRE),
    ("DA V2-B", 97.5, 0.0877, 0.3814, TEAL),
    ("ZoeDepth", 345.0, 0.0992, 0.0866, PURPLE),
    ("VDA-S", 28.4, 0.1000, 0.0829, TEAL),
    ("DA3-B", 120.0, 0.1130, 0.0825, VERM),
    ("Ours", 4.19, 0.1263, 0.0750, NAVY),   # final checkpoint (Table 3b)
    ("DA V2-S", 24.8, 0.2068, 1.0015, GRAY),
]
CMP_SYN = [
    ("ZoeDepth", 345.0, 0.3604, 0.9522, PURPLE),
    ("DA3-B", 120.0, 0.3618, 1.0128, VERM),
    ("DA V2-B", 97.5, 0.3701, 5.9644, TEAL),
    ("Ours", 4.19, 0.3791, 0.2242, NAVY),
    ("DA V2-S", 24.8, 0.3818, 5.0124, GRAY),
    ("DA V1-S", 24.8, 0.4210, 4.5525, GRAY),
    ("DPT-L", 343.0, 0.5029, 8.5888, OCHRE),
]
# label offsets, chosen once so no annotation sits on a marker or another label
CMP_OFF = {"DA V1-S": (0, -14), "ZoeDepth": (0, -14), "DPT-L": (0, 18),
           "DA V2-B": (0, -14), "DA3-B": (0, 18), "DA V2-S": (0, -14),
           "VDA-S": (0, 18), "Ours": (0, -16)}

# Table 7 (Section 5.7): AbsRel by frame index, 32-frame clips
DRIFT_TUM = [(0, 0.1353), (4, 0.1206), (8, 0.1249), (12, 0.1510), (16, 0.1521),
             (20, 0.1415), (24, 0.1585), (28, 0.1697), (31, 0.1323)]
DRIFT_BONN = [(0, 0.1167), (4, 0.1239), (8, 0.1316), (12, 0.1383), (16, 0.1485),
              (20, 0.1538), (24, 0.1582), (28, 0.1668), (31, 0.1340)]
# Table 8: keyframe period -> (activity, AbsRel, t-delta)
KEYFRAME = [(5, 39.4, 0.1337, 0.0976), (10, 29.4, 0.1370, 0.0793),
            (15, 26.0, 0.1400, 0.0753), (30, 22.7, 0.1487, 0.0682),
            (60, 19.6, 0.1510, 0.0570)]

# Table 6 (Section 5.5): gating strategy -> (activity, AbsRel, delta1)
GATE_PIX = [(100.0, 0.3083, 0.5142), (92.7, 0.3093, 0.5089),
            (51.1, 0.3357, 0.4749)]
GATE_GMC = [(87.1, 0.3065, 0.5170), (43.8, 0.3084, 0.5342),
            (14.1, 0.3178, 0.5341)]

# Section 5.6: per-frame latency (ms) before/after the fused kernel, 22% active
LATENCY = [("Dense, eager", 11.38, 1.98), ("Dense, compiled", 4.70, 1.29),
           ("Sparse, eager", 4.87, 2.40), ("Sparse + bucket", 5.39, 2.55),
           ("Sparse + bucket\n+ compiled", 2.99, 2.04)]


def fig_tradeoff():
    """Activity against accuracy, both domains (Table 1).

    The constant-depth control is an order of magnitude worse than any
    operating point, so plotting it in range flattens the curve into the axis
    and hides the only thing the panel is for. It is marked off-scale instead.
    """
    W, H = 900, 340
    o = []
    for i, (data, const, lim, ticks, name) in enumerate([
            (SWEEP_REAL, CONST_REAL, (0.126, 0.140),
             [0.128, 0.132, 0.136, 0.140], "(a) Real indoor holdout"),
            (SWEEP_SYN, CONST_SYN, (0.4280, 0.4360),
             [0.429, 0.431, 0.433, 0.435], "(b) Synthetic holdout")]):
        ax = Ax(84 + i * 460, 52, 340, 230, (0, 100), lim)
        o.append(ax.frame([0, 25, 50, 75, 100], ticks, "Active patches (%)",
                          "AbsRel", "{:g}", "{:.3f}", name))
        o.append(ax.line(sorted(data), NAVY))
        dx, dy = data[DEFAULT_I]              # default operating point
        o.append(f'<circle cx="{ax.X(dx):.1f}" cy="{ax.Y(dy):.1f}" r="7.5" '
                 f'fill="none" stroke="{VERM}" stroke-width="1.8"/>')
        o.append(txt(ax.X(dx) - 11, ax.Y(dy) + 22, "default", 10.5, VERM,
                     "end"))
        # control, off scale: an arrow at the top edge rather than a rescale
        o.append(f'<line x1="{ax.x0+ax.w-70}" y1="{ax.y0+6}" '
                 f'x2="{ax.x0+ax.w-70}" y2="{ax.y0-8}" stroke="{GRAY}" '
                 f'stroke-width="1.2" marker-end="url(#up)"/>')
        o.append(txt(ax.x0 + ax.w - 62, ax.y0 - 2,
                     f"control {const:.3f}", 10.5, GRAY))
    head = ('<defs><marker id="up" markerWidth="7" markerHeight="7" refX="3.5" '
            f'refY="6" orient="auto"><path d="M0,6 L3.5,0 L7,6 z" fill="{GRAY}"/>'
            '</marker></defs>')
    o.insert(0, head)
    return svg(W, H, "\n".join(o))


def fig_comparison():
    """Accuracy against stability, marker area by parameter count (Table 3)."""
    W, H = 900, 346
    o = []
    for i, (data, xlim, xticks, ylim, yt, name) in enumerate([
            (CMP_REAL, (0.055, 0.245), [0.08, 0.12, 0.16, 0.20, 0.24],
             (0.05, 1.6), [0.05, 0.1, 0.2, 0.5, 1.0],
             "(a) Real indoor holdout"),
            (CMP_SYN, (0.335, 0.525), [0.36, 0.40, 0.44, 0.48, 0.52],
             (0.15, 13.0), [0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
             "(b) Synthetic holdout")]):
        ax = Ax(84 + i * 460, 50, 340, 240, xlim, ylim, ylog=True)
        o.append(ax.frame(xticks, yt, "AbsRel  (lower is better)",
                          "t-delta  (lower is better)", "{:.2f}", "{:g}", name))
        marks = []
        for lab, par, ar, td, col in data:
            r = 4.0 + 9.0 * math.sqrt(
                math.log10(par / 3.5) / math.log10(345 / 3.5))
            ours = lab == "Ours"
            o.append(f'<circle cx="{ax.X(ar):.1f}" cy="{ax.Y(td):.1f}" '
                     f'r="{r:.1f}" fill="{col}" '
                     f'fill-opacity="{0.45 if ours else 0.28}" '
                     f'stroke="{col}" stroke-width="{2.2 if ours else 1.3}"/>')
            marks.append((ax.X(ar), ax.Y(td), r,
                          f"{lab} ({par:g}M)" if ours else lab, col, ours))
        o.append(place_labels(marks))
    return svg(W, H, "\n".join(o))


def fig_drift():
    """The sawtooth, and what the refresh period costs (Tables 7 and 8)."""
    W, H = 900, 350
    o = []
    ax = Ax(78, 46, 350, 250, (0, 32), (0.110, 0.180))
    o.append(ax.frame([0, 8, 16, 24, 31], [0.12, 0.14, 0.16, 0.18],
                      "Frame index within clip", "AbsRel", "{:g}", "{:.2f}",
                      "(a) Accuracy decays between keyframes"))
    o.append(ax.vline(30, VERM, "keyframe"))
    o.append(ax.line(DRIFT_BONN, VERM))
    o.append(ax.line(DRIFT_TUM, NAVY))
    o.append(legend(96, 66, [("Bonn (dynamic objects)", VERM, "line"),
                             ("TUM (static camera)", NAVY, "line")]))

    ax2 = Ax(538, 46, 300, 250, (0, 65), (0.130, 0.155))
    o.append(ax2.frame([5, 15, 30, 45, 60], [0.13, 0.14, 0.15],
                       "Keyframe refresh period (frames)", "AbsRel",
                       "{:g}", "{:.2f}",
                       "(b) Refreshing more often trades against stability"))
    o.append(ax2.line([(k, a) for k, _, a, _ in KEYFRAME], NAVY))
    # second axis for t-delta, drawn on the right so the two never cross labels
    tlim = (0.05, 0.10)
    ty = lambda v: ax2.y0 + ax2.h - (v - tlim[0]) / (tlim[1] - tlim[0]) * ax2.h
    o.append(f'<line x1="{ax2.x0+ax2.w}" y1="{ax2.y0}" '
             f'x2="{ax2.x0+ax2.w}" y2="{ax2.y0+ax2.h}" stroke="{TEAL}" '
             f'stroke-width="1"/>')
    for t in (0.06, 0.07, 0.08, 0.09, 0.10):
        o.append(f'<line x1="{ax2.x0+ax2.w}" y1="{ty(t):.1f}" '
                 f'x2="{ax2.x0+ax2.w+4}" y2="{ty(t):.1f}" stroke="{TEAL}" '
                 f'stroke-width="1"/>')
        o.append(txt(ax2.x0 + ax2.w + 8, ty(t) + 4, f"{t:.2f}", 11, TEAL))
    o.append(txt(ax2.x0 + ax2.w + 46, ax2.y0 + ax2.h / 2, "t-delta", 12, TEAL,
                 "middle", rotate=90))
    pts = " ".join(f"{ax2.X(k):.1f},{ty(t):.1f}" for k, _, _, t in KEYFRAME)
    o.append(f'<polyline points="{pts}" fill="none" stroke="{TEAL}" '
             f'stroke-width="1.8" stroke-dasharray="5 4"/>')
    for k, _, _, t in KEYFRAME:
        o.append(f'<circle cx="{ax2.X(k):.1f}" cy="{ty(t):.1f}" r="3.2" '
                 f'fill="#ffffff" stroke="{TEAL}" stroke-width="1.4"/>')
    # legend low and left: the AbsRel curve runs through the old position
    o.append(legend(560, 286, [("AbsRel", NAVY, "line"),
                               ("t-delta", TEAL, "dash")], dx=110, dy=0))

    return svg(W, H, "\n".join(o))


def fig_gating():
    """Pixel against GMC feature gating, as curves (Table 6)."""
    W, H = 900, 330
    o = []
    for i, (key, ylab, lim, ticks, name) in enumerate([
            (1, "AbsRel", (0.30, 0.34), [0.30, 0.31, 0.32, 0.33, 0.34],
             "(a) Accuracy"),
            (2, "delta-1", (0.46, 0.55), [0.46, 0.49, 0.52, 0.55],
             "(b) Ratio accuracy")]):
        ax = Ax(78 + i * 460, 44, 350, 230, (0, 105), lim)
        o.append(ax.frame([0, 25, 50, 75, 100], ticks, "Active patches (%)",
                          ylab, "{:g}", "{:.2f}", name))
        o.append(ax.line([(a, r[key]) for a, *r_ in [(p[0], p) for p in GATE_PIX]
                          for r in [r_[0]]], GRAY, dash="5 4"))
        o.append(ax.line([(p[0], p[key]) for p in GATE_GMC], NAVY))
    o.append(legend(96, 62, [("GMC + feature gating", NAVY, "line"),
                             ("Pixel gating", GRAY, "dash")]))
    return svg(W, H, "\n".join(o))


def fig_latency():
    """Per-frame latency before and after the fused scan kernel."""
    W, H = 820, 350
    x0, y0, bw, rowh = 250, 56, 480, 46
    xmax = 12.0
    o = [txt(x0, 26, "Per-frame latency at 22% activity (ms, lower is better)",
             12.5, FG, "start", "bold")]
    for i, (name, before, after) in enumerate(LATENCY):
        y = y0 + i * rowh
        for j, line in enumerate(name.split("\n")):
            o.append(txt(x0 - 12, y + 14 + j * 13, line, 11, FG, "end"))
        o.append(f'<rect x="{x0}" y="{y}" width="{before/xmax*bw:.1f}" '
                 f'height="13" fill="#ffffff" stroke="{GRAY}" stroke-width="1.1"/>')
        o.append(txt(x0 + before / xmax * bw + 6, y + 11, f"{before:.2f}", 10.5,
                     MUTED))
        best = name == "Dense, compiled"
        o.append(f'<rect x="{x0}" y="{y+16}" width="{after/xmax*bw:.1f}" '
                 f'height="13" fill="{NAVY if best else "#5b83a8"}"/>')
        o.append(txt(x0 + after / xmax * bw + 6, y + 27,
                     f"{after:.2f}" + ("  (fastest)" if best else ""), 10.5,
                     NAVY, weight="bold" if best else "normal"))
    ay = y0 + len(LATENCY) * rowh
    o.append(f'<line x1="{x0}" y1="{ay}" x2="{x0+bw}" y2="{ay}" '
             f'stroke="{AXIS}" stroke-width="1"/>')
    for t in (0, 2, 4, 6, 8, 10, 12):
        x = x0 + t / xmax * bw
        o.append(f'<line x1="{x:.1f}" y1="{ay}" x2="{x:.1f}" y2="{ay+4}" '
                 f'stroke="{AXIS}" stroke-width="1"/>')
        o.append(txt(x, ay + 17, t, 11, MUTED, "middle"))
    o.append(txt(x0 + bw / 2, ay + 36, "milliseconds per frame", 12, FG,
                 "middle"))
    o.append(legend(x0, y0 - 12, [("chunked scan", GRAY, "marker"),
                                  ("fused kernel", NAVY, "marker")],
                    dx=150, dy=0))
    return svg(W, H, "\n".join(o))


FIGURES = {
    "tradeoff.svg": fig_tradeoff,
    "comparison.svg": fig_comparison,
    "drift.svg": fig_drift,
    "gating.svg": fig_gating,
    "latency.svg": fig_latency,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name, fn in FIGURES.items():
        path = os.path.join(args.out, name)
        with open(path, "w") as f:
            f.write(fn())
        print(f"wrote {path}")




# ------------------------------------------------------------ diagrams -----
# Boxes are laid out on a grid and connectors are derived from box edges, so a
# connector cannot point into empty space and two boxes cannot overlap: both
# were faults in the hand-placed first version of these diagrams.

class Box:
    def __init__(self, x, y, w, h, title, lines=(), fill="#ffffff",
                 stroke=AXIS, dash=None, title_colour=None):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.title, self.lines = title, list(lines)
        self.fill, self.stroke, self.dash = fill, stroke, dash
        self.title_colour = title_colour or stroke

    # edge anchors, so connectors attach to the border and never to the centre
    def left(self):   return (self.x, self.y + self.h / 2)
    def right(self):  return (self.x + self.w, self.y + self.h / 2)
    def top(self):    return (self.x + self.w / 2, self.y)
    def bottom(self): return (self.x + self.w / 2, self.y + self.h)

    def draw(self):
        d = f' stroke-dasharray="{self.dash}"' if self.dash else ""
        o = [f'<rect x="{self.x}" y="{self.y}" width="{self.w}" '
             f'height="{self.h}" rx="2" fill="{self.fill}" '
             f'stroke="{self.stroke}" stroke-width="1.2"{d}/>']
        cx = self.x + self.w / 2
        ty = self.y + 19
        o.append(txt(cx, ty, self.title, 11.5, self.title_colour, "middle",
                     "bold"))
        for i, ln in enumerate(self.lines):
            o.append(txt(cx, ty + 15 + i * 13, ln, 10, MUTED, "middle"))
        return "\n".join(o)


def arrow(p, q, colour=AXIS, dash=None, label=None, mid=None):
    """Orthogonal connector p -> q. `mid` forces the elbow's x or y so a link
    routes around boxes instead of through them."""
    d = f' stroke-dasharray="{dash}"' if dash else ""
    (x1, y1), (x2, y2) = p, q
    if y1 == y2 or x1 == x2:
        path = f"M{x1},{y1} L{x2},{y2}"
    elif mid is not None and mid[0] == "x":
        path = f"M{x1},{y1} H{mid[1]} V{y2} H{x2}"
    elif mid is not None and mid[0] == "y":
        path = f"M{x1},{y1} V{mid[1]} H{x2} V{y2}"
    else:
        path = f"M{x1},{y1} H{(x1+x2)/2} V{y2} H{x2}"
    o = [f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="1.3"'
         f'{d} marker-end="url(#arw)"/>']
    if label:
        o.append(txt((x1 + x2) / 2, min(y1, y2) - 6, label, 10, MUTED,
                     "middle"))
    return "\n".join(o)


ARROWDEF = ('<defs><marker id="arw" markerWidth="8" markerHeight="8" '
            f'refX="7" refY="3" orient="auto"><path d="M0,0 L7,3 L0,6 z" '
            f'fill="{AXIS}"/></marker></defs>')


def fig_pipeline():
    """Figure 1: the streaming pipeline."""
    W, H = 900, 360
    row, bh = 170, 66
    b = {
        "in":   Box(40, row, 104, bh, "Input frames", ["t - 1,  t"], "#f7f7f7"),
        "det":  Box(196, row, 132, bh, "Change detector",
                    ["patch difference,", "hysteresis, dilation"],
                    "#eef3f8", NAVY, title_colour=NAVY),
        "bb":   Box(386, 138, 190, 130, "State-space backbone",
                    ["temporal / spatial blocks", "", "Δ-gating:  Δ̃ = M · Δ",
                     "", "M = 0 preserves state"], "#eef5f2", TEAL,
                    title_colour=TEAL),
        "dec":  Box(628, row, 116, bh, "Dense decoder",
                    ["64-bin head", "+ regression"], "#f7f7f7"),
        "out":  Box(796, row, 78, bh, "Depth", ["frame t"], "#2b2b2b",
                    "#2b2b2b", title_colour="#ffffff"),
        "gmc":  Box(180, 42, 164, 62, "Global motion comp.",
                    ["low-res tracking", "+ robust homography"], "#fdf6ec",
                    OCHRE, dash="5 3", title_colour=OCHRE),
        "csp":  Box(382, 42, 92, 62, "Spatial cache", ["patch outputs"],
                    "#f4f4f4"),
        "ctm":  Box(492, 42, 96, 62, "Temporal cache", ["hidden state"],
                    "#f4f4f4"),
    }
    o = [ARROWDEF]
    # main chain: every hop attaches to facing edges
    o.append(arrow(b["in"].right(), b["det"].left()))
    o.append(arrow(b["det"].right(), (b["bb"].x, b["bb"].y + 65)))
    o.append(arrow(b["bb"].right(), b["dec"].left()))
    o.append(arrow(b["dec"].right(), b["out"].left()))
    # optional branch: leaves the input link, returns into the detector's top
    o.append(arrow((162, row + bh / 2), b["gmc"].left(), OCHRE, "5 3",
                   mid=("y", 73)))
    o.append(arrow(b["gmc"].bottom(), b["det"].top(), OCHRE, "5 3"))
    # caches sit above the backbone and connect straight down, no crossings
    o.append(arrow(b["csp"].bottom(), (b["csp"].x + 46, b["bb"].y), GRAY))
    o.append(arrow(b["ctm"].bottom(), (b["ctm"].x + 48, b["bb"].y), GRAY))
    # activity mask: down from the detector, along, and up into the backbone
    o.append(arrow(b["det"].bottom(), (b["bb"].x + 60, b["bb"].y + b["bb"].h),
                   NAVY, mid=("y", 322)))
    for k in b:
        o.append(b[k].draw())
    o.append(txt(262, 336, "activity mask M", 10.5, NAVY, "middle"))
    o.append(txt(600, 88, "1.644 → 0.608 GMAC", 10, MUTED))
    o.append(txt(600, 100, "at 15.4% activity", 10, MUTED))
    o.append(txt(40, 60, "used only for", 10, OCHRE))
    o.append(txt(40, 73, "moving cameras", 10, OCHRE))
    return svg(W, H, "\n".join(o))


def fig_deltagate():
    """Figure 2: what the mask does to the discretization step."""
    W, H = 900, 290
    o = [ARROWDEF]
    fr1 = Box(40, 60, 96, 68, "Frame t - 1", [], "#fafafa", "#999999",
              title_colour=MUTED)
    fr2 = Box(40, 158, 96, 68, "Frame t", [], "#fafafa", "#999999",
              title_colour=MUTED)
    det = Box(196, 110, 116, 66, "Change detector", ["patch-wise"], "#eef3f8",
              NAVY, title_colour=NAVY)
    gat = Box(500, 110, 150, 64, "Δ-gating", ["", "Δ̃ = M · Δ"], "#eef5f2",
              TEAL, title_colour=TEAL)
    act = Box(706, 52, 158, 74, "M = 1  changed",
              ["Ā = exp(ΔA),  B̄ ≠ 0",
               "h(t) = Ā h(t-1) + B̄ x(t)", "computed"], "#f7f7f7")
    sta = Box(706, 176, 158, 74, "M = 0  static",
              ["Ā = I,   B̄ = 0",
               "h(t) = h(t-1)  exactly", "computation skipped"], "#eef5f2",
              TEAL, title_colour=TEAL)
    o.append(arrow(fr1.right(), (172, 143), mid=("x", 160)))
    o.append(arrow(fr2.right(), (172, 143), mid=("x", 160)))
    o.append(arrow((172, 143), det.left()))
    o.append(arrow(det.right(), (352, 143)))
    o.append(arrow((470, 143), gat.left()))
    o.append(arrow(gat.right(), act.left(), mid=("x", 678)))
    o.append(arrow(gat.right(), sta.left(), mid=("x", 678)))
    # the mask itself, drawn as a small grid between detector and gate
    gx, gy, c = 356, 116, 22
    for r in range(3):
        for col in range(4):
            on = (r, col) in {(1, 1), (1, 2)}
            o.append(f'<rect x="{gx+col*c}" y="{gy+r*c}" width="{c}" '
                     f'height="{c}" fill="{NAVY if on else "#ffffff"}" '
                     f'stroke="#9db6cc" stroke-width="0.9"/>')
            o.append(txt(gx + col * c + c / 2, gy + r * c + c / 2 + 4,
                         "1" if on else "0", 10,
                         "#ffffff" if on else "#b0b0b0", "middle"))
    o.append(txt(gx + 2 * c, gy - 8, "activity mask M", 10.5, NAVY, "middle"))
    o.append(txt(575, 190, "multiplies a parameter", 10, MUTED, "middle"))
    o.append(txt(575, 202, "the model already had", 10, MUTED, "middle"))
    for bx in (fr1, fr2, det, gat, act, sta):
        o.append(bx.draw())
    # a moving object, to make "changed patch" concrete
    o.append(f'<circle cx="76" cy="100" r="11" fill="#f2c9b8" stroke="{OCHRE}" '
             f'stroke-width="1.1"/>')
    o.append(f'<circle cx="104" cy="198" r="11" fill="{OCHRE}" '
             f'stroke="#8a5a12" stroke-width="1.1"/>')
    o.append(txt(88, 248, "background fixed,", 10, MUTED, "middle"))
    o.append(txt(88, 260, "object moves", 10, MUTED, "middle"))
    return svg(W, H, "\n".join(o))


FIGURES["pipeline.svg"] = fig_pipeline
FIGURES["delta-gating.svg"] = fig_deltagate


if __name__ == "__main__":
    main()
