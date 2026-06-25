#!/usr/bin/env python3
"""Render results/metrics.csv to SVG charts using ONLY the Python standard
library (no matplotlib / numpy / gnuplot needed -- an SVG is just text).

Reads:  results/metrics.csv   (columns: episode,return,abs_delta)
Writes: results/reward.svg    (per-episode return vs episode, clipped)
        results/delta.svg     (mean |TD error| vs episode)

Usage:  python3 tools/plot_svg.py [csv_path] [out_dir]
"""
import csv
import sys
import os

W, H = 900, 360
ML, MR, MT, MB = 70, 20, 40, 50           # margins
PW, PH = W - ML - MR, H - MT - MB          # plot area


def _bucket(xs, ys, n_out=300):
    """Average ys into <= n_out buckets so the polyline stays small."""
    n = len(xs)
    if n <= n_out:
        return xs, ys
    out_x, out_y = [], []
    for b in range(n_out):
        lo = b * n // n_out
        hi = max(lo + 1, (b + 1) * n // n_out)
        seg = ys[lo:hi]
        out_x.append(xs[lo])
        out_y.append(sum(seg) / len(seg))
    return out_x, out_y


def _ticks(lo, hi, n=5):
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def render(xs, ys, title, ylabel, ymin, ymax, color, path):
    xs, ys = _bucket(xs, ys)
    xmin, xmax = xs[0], xs[-1]
    if ymax == ymin:
        ymax = ymin + 1
    if xmax == xmin:
        xmax = xmin + 1

    def px(x): return ML + (x - xmin) / (xmax - xmin) * PW
    def py(y):
        y = max(ymin, min(ymax, y))
        return MT + (ymax - y) / (ymax - ymin) * PH

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="monospace" font-size="13">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2:.0f}" y="22" text-anchor="middle" '
             f'font-size="15" font-weight="bold">{title}</text>')
    # axes
    s.append(f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" '
             f'fill="none" stroke="#888"/>')
    # y gridlines + labels
    for t in _ticks(ymin, ymax):
        y = py(t)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" '
                 f'stroke="#eee"/>')
        s.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{t:.0f}</text>')
    # x gridlines + labels
    for t in _ticks(xmin, xmax):
        x = px(t)
        s.append(f'<line x1="{x:.1f}" y1="{MT}" x2="{x:.1f}" y2="{MT+PH}" '
                 f'stroke="#eee"/>')
        s.append(f'<text x="{x:.1f}" y="{MT+PH+18}" text-anchor="middle">{t:.0f}</text>')
    # axis titles
    s.append(f'<text x="{ML+PW/2:.0f}" y="{H-10}" text-anchor="middle">episode</text>')
    s.append(f'<text x="16" y="{MT+PH/2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {MT+PH/2:.0f})">{ylabel}</text>')
    # data polyline
    pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(xs, ys))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
             f'stroke-width="1.6"/>')
    s.append('</svg>')

    with open(path, "w") as f:
        f.write("\n".join(s))
    print(f"  wrote {path}")


def render_path(path_file, out):
    """Draw the grid, wind, mines, and BOTH the learned path and the
    BFS-optimal path from results/path.txt as an SVG."""
    rows = cols = 0
    start = goal = end = optend = None
    mines = set()
    wind = {}
    steps = []          # learned greedy path: (row, col, action)
    opt = []            # BFS optimal path:    (row, col, action)
    with open(path_file) as f:
        for line in f:
            t = line.split()
            if not t:
                continue
            k = t[0]
            if k == "grid":
                rows, cols = int(t[1]), int(t[2])
            elif k == "start":
                start = (int(t[1]), int(t[2]))
            elif k == "goal":
                goal = (int(t[1]), int(t[2]))
            elif k in ("mine", "cliff"):            # hazard cell
                mines.add((int(t[1]), int(t[2])))
            elif k == "wind":
                wind[int(t[1])] = int(t[2])
            elif k == "step":
                steps.append((int(t[1]), int(t[2]), int(t[3])))
            elif k == "end":
                end = tuple(int(x) for x in t[1:6])
            elif k == "opt":
                opt.append((int(t[1]), int(t[2]), int(t[3])))
            elif k == "optend":
                optend = tuple(int(x) for x in t[1:6])

    # adaptive cell size so big grids stay a sensible canvas
    cs = max(26, min(64, 760 // max(rows, cols)))
    mx, band, my = 24, 40, 0
    my = 30 + band                     # title + wind band above the grid
    width = cols * cs + 2 * mx
    height = my + rows * cs + 64       # + legend/summary band below
    def cx(c): return mx + c * cs + cs / 2
    def cy(r): return my + r * cs + cs / 2
    # action -> (dx, dy) in screen coords (y points down); 0=up 1=right 2=down 3=left
    DIR = {0: (0, -1), 1: (1, 0), 2: (0, 1), 3: (-1, 0)}

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
             f'height="{height}" font-family="monospace" font-size="13">')
    s.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    s.append('<defs>'
             '<marker id="ga" markerWidth="8" markerHeight="8" refX="6" refY="3" '
             'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#1a8a3a"/></marker>'
             '<marker id="wa" markerWidth="8" markerHeight="8" refX="6" refY="3" '
             'orient="auto"><path d="M0,0 L6,3 L0,6 Z" fill="#2b6cb0"/></marker>'
             '</defs>')
    s.append(f'<text x="{width/2:.0f}" y="20" text-anchor="middle" font-size="15" '
             f'font-weight="bold">Mines-with-Wind: learned vs optimal path</text>')

    # --- wind band above the grid: an up-arrow per unit of strength ---------
    for c in range(cols):
        w = wind.get(c, 0)
        if w <= 0:
            continue
        for j in range(w):
            ax = cx(c) + (j - (w - 1) / 2) * 9
            s.append(f'<line x1="{ax:.1f}" y1="{my-6:.0f}" x2="{ax:.1f}" '
                     f'y2="{my-band+10:.0f}" stroke="#2b6cb0" stroke-width="2" '
                     f'marker-end="url(#wa)"/>')

    # --- cells (mine tint, wind-column tint, S/G colours) -------------------
    for r in range(rows):
        for c in range(cols):
            x, y = mx + c * cs, my + r * cs
            fill, lab = "white", ""
            if wind.get(c, 0) > 0:
                fill = "#eef4fb"                    # faint blue = windy column
            if (r, c) in mines:
                fill, lab = "#f4a3a3", "*"
            if start == (r, c):
                fill, lab = "#a8d5a2", "S"
            if goal == (r, c):
                fill, lab = "#f6c66b", "G"
            s.append(f'<rect x="{x}" y="{y}" width="{cs}" height="{cs}" '
                     f'fill="{fill}" stroke="#bbb"/>')
            if lab:
                s.append(f'<text x="{cx(c):.0f}" y="{cy(r)+cs*0.18:.0f}" '
                         f'text-anchor="middle" font-size="{cs*0.5:.0f}" '
                         f'font-weight="bold" fill="#333">{lab}</text>')

    def polyline(path_steps, last, stroke, w, opacity, dash=""):
        centres = [(cx(c), cy(r)) for (r, c, _) in path_steps]
        if last:
            centres.append((cx(last[1]), cy(last[0])))
        if len(centres) < 2:
            return
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in centres)
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{stroke}" '
                 f'stroke-width="{w}" opacity="{opacity}" stroke-linecap="round" '
                 f'stroke-linejoin="round"{d}/>')

    # optimal path: wide translucent blue "corridor" underneath
    polyline(opt, optend, "#2b6cb0", cs * 0.30, 0.30)
    # learned path: green centre-line + a green arrow for the move in each cell
    polyline(steps, end, "#1a8a3a", 3, 0.9)
    for (r, c, a) in steps:
        dx, dy = DIR[a]
        k = cs * 0.22
        s.append(f'<line x1="{cx(c)-dx*k:.1f}" y1="{cy(r)-dy*k:.1f}" '
                 f'x2="{cx(c)+dx*k:.1f}" y2="{cy(r)+dy*k:.1f}" '
                 f'stroke="#1a8a3a" stroke-width="2.2" marker-end="url(#ga)"/>')

    # --- legend / summary ---------------------------------------------------
    ly = my + rows * cs + 22
    g_txt = (f"learned (Q): {end[2]} steps, return {end[3]}"
             if end else "learned (Q)")
    o_txt = (f"optimal (BFS): {optend[2]} steps, return {optend[3]}"
             if optend else "optimal (BFS)")
    s.append(f'<line x1="{mx}" y1="{ly-4:.0f}" x2="{mx+22}" y2="{ly-4:.0f}" '
             f'stroke="#1a8a3a" stroke-width="3"/>')
    s.append(f'<text x="{mx+28}" y="{ly}" font-size="12" fill="#222">{g_txt}</text>')
    s.append(f'<line x1="{mx}" y1="{ly+16:.0f}" x2="{mx+22}" y2="{ly+16:.0f}" '
             f'stroke="#2b6cb0" stroke-width="7" opacity="0.35"/>')
    s.append(f'<text x="{mx+28}" y="{ly+20}" font-size="12" fill="#222">'
             f'{o_txt}   (blue ↑ = wind, * = mine)</text>')
    s.append('</svg>')
    with open(out, "w") as f:
        f.write("\n".join(s))
    print(f"  wrote {out}")


def render_phone(csv_path, out):
    """Plot the phone DVFS trace: workload demand vs the frequency the agent
    chose, over time, with jank frames flagged. Multi-series, stdlib only."""
    fr, demand, qf, of, jank = [], [], [], [], []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            fr.append(int(row["frame"]));      demand.append(float(row["demand_mhz"]))
            qf.append(float(row["q_mhz"]));    of.append(float(row["opt_mhz"]))
            jank.append(int(row["jank"]))
    if not fr:
        return
    xmin, xmax = fr[0], fr[-1]
    ymin, ymax = 1100, 2750
    def px(x): return ML + (x - xmin) / max(1, (xmax - xmin)) * PW
    def py(y): return MT + (ymax - max(ymin, min(ymax, y))) / (ymax - ymin) * PH

    s = []
    s.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="monospace" font-size="13">')
    s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
    s.append(f'<text x="{W/2:.0f}" y="22" text-anchor="middle" font-size="15" '
             f'font-weight="bold">Phone DVFS: workload demand vs chosen CPU frequency</text>')
    s.append(f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#888"/>')
    for t in _ticks(ymin, ymax):
        y = py(t)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{t:.0f}</text>')
    s.append(f'<text x="{ML+PW/2:.0f}" y="{H-10}" text-anchor="middle">frame</text>')
    s.append(f'<text x="16" y="{MT+PH/2:.0f}" text-anchor="middle" '
             f'transform="rotate(-90 16 {MT+PH/2:.0f})">MHz</text>')

    def line(series, color, w, dash=""):
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(fr, series))
        d = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" '
                 f'stroke-width="{w}" opacity="0.9"{d}/>')
    line(demand, "#bbb", 1.4)                 # workload requirement (gray)
    line(of, "#2b6cb0", 1.4, "5 3")           # oracle optimal (blue dashed)
    line(qf, "#1a8a3a", 1.8)                  # agent's choice (green)
    for x, j in zip(fr, jank):                # jank markers
        if j:
            s.append(f'<circle cx="{px(x):.1f}" cy="{MT+8:.0f}" r="2.4" fill="#d62728"/>')

    lx = ML + 8
    for i, (c, t) in enumerate([("#1a8a3a", "chosen freq (Q-learned)"),
                                ("#2b6cb0", "optimal (oracle)"),
                                ("#bbb", "demand"),
                                ("#d62728", "jank")]):
        y = MT + 14 + i * 16
        if c == "#d62728":
            s.append(f'<circle cx="{lx+8:.0f}" cy="{y-4:.0f}" r="3" fill="{c}"/>')
        else:
            s.append(f'<line x1="{lx}" y1="{y-4}" x2="{lx+16}" y2="{y-4}" stroke="{c}" stroke-width="2.5"/>')
        s.append(f'<text x="{lx+22}" y="{y}" font-size="12" fill="#222">{t}</text>')
    s.append('</svg>')
    with open(out, "w") as f:
        f.write("\n".join(s))
    print(f"  wrote {out}")


# RL family -> colour, for the contest plots
_FAMILY = {
    "Q-learning": ("#2b6cb0", "off-policy"), "Double-Q": ("#2b6cb0", "off-policy"),
    "Q(lambda)": ("#2b6cb0", "off-policy"),
    "SARSA": ("#1a8a3a", "on-policy"), "SARSA(lambda)": ("#1a8a3a", "on-policy"),
    "Expected-SARSA": ("#1a8a3a", "on-policy"), "n-step-SARSA": ("#1a8a3a", "on-policy"),
    "Dyna-Q": ("#dd7711", "planning"), "Prio-Sweeping": ("#dd7711", "planning"),
    "Monte-Carlo": ("#888888", "monte-carlo"), "Actor-Critic": ("#8a2be2", "actor-critic"),
}
def _fam(name): return _FAMILY.get(name, ("#444", "other"))


def render_contest_scores(csv_path, out):
    """Horizontal bar chart: steady-state online return per algorithm (longer =
    higher = safer-while-exploring), annotated with greedy eval and cliff falls."""
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        return
    n = len(rows)
    rh, top, left, right = 32, 80, 150, 40
    width, height = 940, top + n*rh + 30
    vals = [int(r["online_ret"]) for r in rows]
    vmin = min(vals + [-40]) - 2; vmax = 0
    bx0, bx1 = left, width - right - 220
    def bx(v): return bx0 + (v - vmin) / (vmax - vmin) * (bx1 - bx0)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
         f'font-family="monospace" font-size="13">',
         f'<rect width="{width}" height="{height}" fill="white"/>',
         f'<text x="{width/2:.0f}" y="26" text-anchor="middle" font-size="16" '
         f'font-weight="bold">RL contest - deterministic cliff, persistent 10% exploration</text>',
         f'<text x="{width/2:.0f}" y="46" text-anchor="middle" font-size="12" fill="#555">'
         f'bar = online return while exploring (higher/longer = better); ranked best-first</text>']
    # family legend
    seen = {}
    lx = left
    for nm,(c,f) in _FAMILY.items():
        if f in seen: continue
        seen[f] = c
    for i,(f,c) in enumerate(seen.items()):
        x = left + i*150
        s.append(f'<rect x="{x}" y="58" width="12" height="12" fill="{c}"/>')
        s.append(f'<text x="{x+16}" y="68" font-size="11" fill="#333">{f}</text>')
    for i, r in enumerate(rows):
        y = top + i*rh
        name = r["algo"]; col, _ = _fam(name)
        ov = int(r["online_ret"])
        s.append(f'<text x="{left-8}" y="{y+rh/2+4:.0f}" text-anchor="end" '
                 f'font-size="12">{r["rank"]}. {name}</text>')
        s.append(f'<rect x="{bx0:.0f}" y="{y+5:.0f}" width="{bx(ov)-bx0:.1f}" height="{rh-12}" '
                 f'fill="{col}" opacity="0.85"/>')
        s.append(f'<text x="{bx(ov)+8:.0f}" y="{y+rh/2+4:.0f}" font-size="12" fill="#222">'
                 f'online {ov}  |  eval {r["eval_return"]}  |  falls {r["train_falls"]}</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"  wrote {out}")


def render_contest_curves(csv_path, out):
    """Learning curves: greedy eval return vs training step, one line per algo,
    coloured by family. Shows convergence speed / sample efficiency."""
    series = {}
    for r in csv.DictReader(open(csv_path)):
        series.setdefault(r["algo"], []).append((int(r["step"]), int(r["eval_return"])))
    if not series:
        return
    ymin, ymax = -100, 0
    xmax = max(p[0] for pts in series.values() for p in pts) or 1
    def px(x): return ML + x / xmax * PW
    def py(y): return MT + (ymax - max(ymin, min(ymax, y))) / (ymax - ymin) * PH
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="monospace" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2:.0f}" y="22" text-anchor="middle" font-size="15" font-weight="bold">'
         f'Contest learning curves (greedy eval return vs training step)</text>',
         f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#888"/>']
    for t in _ticks(ymin, ymax):
        y = py(t)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{t:.0f}</text>')
    s.append(f'<text x="{ML+PW/2:.0f}" y="{H-10}" text-anchor="middle">training step</text>')
    for name, pts in series.items():
        col, _ = _fam(name)
        poly = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        s.append(f'<polyline points="{poly}" fill="none" stroke="{col}" '
                 f'stroke-width="1.6" opacity="0.8"/>')
        lx, ly = pts[-1]
        s.append(f'<text x="{px(lx)-2:.0f}" y="{py(ly)-2:.0f}" text-anchor="end" '
                 f'font-size="9" fill="{col}">{name}</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"  wrote {out}")


def render_swirl(csv_path, out):
    """Swirl-escape trace: swirl strength, thrust fired, and fuel remaining over
    time, with 'caught' moments flagged. Shows burst-to-escape + fuel draining."""
    t, sw, th, fu, ca = [], [], [], [], []
    smax, fmax = 5, 100
    for r in csv.DictReader(open(csv_path)):
        t.append(int(r["t"])); sw.append(int(r["swirl"])); th.append(int(r["thrust"]))
        fu.append(int(r["fuel"])); ca.append(int(r["caught"]))
    if not t:
        return
    xmin, xmax = t[0], t[-1]
    ymin, ymax = 0, smax + 0.5
    def px(x): return ML + (x - xmin) / max(1, xmax - xmin) * PW
    def py(y): return MT + (ymax - max(ymin, min(ymax, y))) / (ymax - ymin) * PH
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="monospace" font-size="13">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2:.0f}" y="22" text-anchor="middle" font-size="15" font-weight="bold">'
         f'Swirl escape: thrust bursts vs swirls, with fuel draining</text>',
         f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#888"/>']
    for k in range(smax + 1):
        y = py(k)
        s.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eee"/>')
        s.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{k}</text>')
    s.append(f'<text x="{ML+PW/2:.0f}" y="{H-10}" text-anchor="middle">time step</text>')
    # swirl strength as faint grey bars
    for x, v in zip(t, sw):
        if v > 0:
            s.append(f'<rect x="{px(x)-1.4:.1f}" y="{py(v):.1f}" width="2.8" '
                     f'height="{py(0)-py(v):.1f}" fill="#ccc"/>')
    # thrust (green) and fuel (blue, scaled to the 0..SMAX axis) as polylines
    s.append('<polyline points="' + " ".join(f"{px(x):.1f},{py(v):.1f}" for x,v in zip(t,th)) +
             '" fill="none" stroke="#1a8a3a" stroke-width="1.8"/>')
    s.append('<polyline points="' + " ".join(f"{px(x):.1f},{py(v*smax/fmax):.1f}" for x,v in zip(t,fu)) +
             '" fill="none" stroke="#2b6cb0" stroke-width="1.6" stroke-dasharray="4 3"/>')
    for x, c in zip(t, ca):
        if c:
            s.append(f'<circle cx="{px(x):.1f}" cy="{MT+8:.0f}" r="2.6" fill="#d62728"/>')
    for i,(c,lab) in enumerate([("#ccc","swirl strength"),("#1a8a3a","thrust fired"),
                                ("#2b6cb0","fuel (scaled)"),("#d62728","caught")]):
        y = MT + 14 + i*16
        s.append(f'<rect x="{ML+10}" y="{y-9}" width="12" height="10" fill="{c}"/>')
        s.append(f'<text x="{ML+26}" y="{y}" font-size="12" fill="#222">{lab}</text>')
    s.append('</svg>')
    open(out, "w").write("\n".join(s))
    print(f"  wrote {out}")


def render_viz(meta_csv, paths_csv, curves_csv, out_dir):
    """Multi-panel: each agent's learned path vs the BFS-optimal path, plus
    overlaid reward and Bellman-delta curves."""
    R = Cn = 0; start = goal = None; mines = set(); opt = []
    for row in csv.DictReader(open(meta_csv)):
        k = row["key"]; a = int(row["a"]); b = int(row["b"])
        if k == "grid": R, Cn = a, b
        elif k == "start": start = (a, b)
        elif k == "goal": goal = (a, b)
        elif k == "mine": mines.add((a, b))
        elif k == "opt": opt.append((a, b))
    paths = {}
    for row in csv.DictReader(open(paths_csv)):
        paths.setdefault(row["agent"], []).append((int(row["row"]), int(row["col"])))
    curves = {}
    for row in csv.DictReader(open(curves_csv)):
        curves.setdefault(row["agent"], []).append(
            (int(row["episode"]), int(row["reward"]), int(row["delta"])))

    # ---- panel grid of paths ----
    agents = list(paths.keys()); n = len(agents)
    cs = 16; cols = 3; prows = (n + cols - 1)//cols
    gw, gh = Cn*cs, R*cs; pad = 14; th = 24
    pw, ph = gw + 2*pad, gh + th + pad
    W, H = cols*pw, prows*ph + 30
    def cx(c, ox): return ox + c*cs + cs/2
    def cy(r, oy): return oy + r*cs + cs/2
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
         f'font-family="monospace" font-size="11"><rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{W/2:.0f}" y="18" text-anchor="middle" font-size="14" font-weight="bold">'
         f'Learned path (colour) vs BFS-optimal (grey) — {R}x{Cn} maze</text>']
    for i, ag in enumerate(agents):
        ox = (i % cols)*pw + pad; oy = (i//cols)*ph + 30
        col, _ = _fam(_canon(ag))
        # grid cells
        for r in range(R):
            for c in range(Cn):
                fill = "white"
                if (r, c) in mines: fill = "#f4a3a3"
                if start == (r, c): fill = "#a8d5a2"
                if goal == (r, c): fill = "#f6c66b"
                s.append(f'<rect x="{ox+c*cs}" y="{oy+r*cs}" width="{cs}" height="{cs}" '
                         f'fill="{fill}" stroke="#ddd"/>')
        # optimal path (grey, thick, under)
        if opt:
            pts = " ".join(f"{cx(c,ox):.1f},{cy(r,oy):.1f}" for r, c in opt)
            s.append(f'<polyline points="{pts}" fill="none" stroke="#999" stroke-width="4" opacity="0.5"/>')
        # agent path (coloured)
        ap = paths[ag]
        pts = " ".join(f"{cx(c,ox):.1f},{cy(r,oy):.1f}" for r, c in ap)
        s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.8"/>')
        reached = ap[-1] == goal
        s.append(f'<text x="{ox}" y="{oy+gh+16:.0f}" font-size="11" fill="{col}">'
                 f'{ag}: {len(ap)-1} steps {"OK" if reached else "(stuck)"}</text>')
    s.append('</svg>')
    open(os.path.join(out_dir, "viz_paths.svg"), "w").write("\n".join(s))
    print("  wrote " + os.path.join(out_dir, "viz_paths.svg"))

    # ---- overlaid reward & delta curves ----
    def curve_plot(idx, title, ylab, ymin, ymax, fname):
        svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W2}" height="{H2}" '
               f'font-family="monospace" font-size="12"><rect width="{W2}" height="{H2}" fill="white"/>',
               f'<text x="{W2/2:.0f}" y="20" text-anchor="middle" font-size="14" font-weight="bold">{title}</text>',
               f'<rect x="{ML}" y="{MT}" width="{PW}" height="{PH}" fill="none" stroke="#888"/>']
        xmax = max((p[0] for pts in curves.values() for p in pts), default=1) or 1
        def px(x): return ML + x/xmax*PW
        def py(y): return MT + (ymax - max(ymin, min(ymax, y)))/(ymax-ymin)*PH
        for t in _ticks(ymin, ymax):
            y = py(t); svg.append(f'<line x1="{ML}" y1="{y:.1f}" x2="{ML+PW}" y2="{y:.1f}" stroke="#eee"/>')
            svg.append(f'<text x="{ML-8}" y="{y+4:.1f}" text-anchor="end">{t:.0f}</text>')
        svg.append(f'<text x="{ML+PW/2:.0f}" y="{H2-8}" text-anchor="middle">episode</text>')
        for i, ag in enumerate(agents):
            col, _ = _fam(_canon(ag)); pts = curves.get(ag, [])
            if not pts: continue
            poly = " ".join(f"{px(e):.1f},{py(row[idx]):.1f}" for row in pts for e in [row[0]])
            svg.append(f'<polyline points="{poly}" fill="none" stroke="{col}" stroke-width="1.4" opacity="0.85"/>')
            ly = MT + 14 + i*13
            svg.append(f'<text x="{ML+PW-150}" y="{ly}" font-size="10" fill="{col}">{ag}</text>')
        svg.append('</svg>')
        open(os.path.join(out_dir, fname), "w").write("\n".join(svg))
        print("  wrote " + os.path.join(out_dir, fname))
    curve_plot(1, "Reward vs episode (per agent, clipped)", "reward", -150, 0, "viz_rewards.svg")
    dmax = max((v[2] for pts in curves.values() for v in pts), default=1) or 1
    curve_plot(2, "Bellman delta vs episode (per agent)", "|delta|", 0, dmax, "viz_delta.svg")


def _canon(name):
    m = {"qlearn":"Q-learning","sarsa":"SARSA","expsarsa":"Expected-SARSA","doubleq":"Double-Q",
         "qlambda":"Q(lambda)","sarsalambda":"SARSA(lambda)","dynaq":"Dyna-Q","mc":"Monte-Carlo",
         "nstep":"n-step-SARSA","psweep":"Prio-Sweeping","actorcritic":"Actor-Critic"}
    return m.get(name, name)

W2, H2 = 760, 380


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "results/metrics.csv"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(csv_path) or "."
    os.makedirs(out_dir, exist_ok=True)

    data_dir = os.path.dirname(csv_path) or "."

    # Gridworld convergence (only if metrics.csv is present).
    if os.path.exists(csv_path):
        ep, ret, dlt = [], [], []
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                ep.append(int(row["episode"]))
                ret.append(float(row["return"]))
                dlt.append(float(row["abs_delta"]))
        render(ep, ret, "Reward vs episode (per-episode return, clipped at -120)",
               "reward", -120, 0, "#1f77b4", os.path.join(out_dir, "reward.svg"))
        render(ep, dlt, "Bellman delta vs episode (mean |TD error|)",
               "|delta|", 0, max(dlt) if dlt else 1, "#d62728",
               os.path.join(out_dir, "delta.svg"))

    # Best path grid (gridworld).
    path_txt = os.path.join(data_dir, "path.txt")
    if os.path.exists(path_txt):
        render_path(path_txt, os.path.join(out_dir, "path.svg"))

    # Phone DVFS trace.
    phone_csv = os.path.join(data_dir, "phone_trace.csv")
    if os.path.exists(phone_csv):
        render_phone(phone_csv, os.path.join(out_dir, "phone.svg"))

    # Swirl-escape trace.
    swirl_csv = os.path.join(data_dir, "swirl_trace.csv")
    if os.path.exists(swirl_csv):
        render_swirl(swirl_csv, os.path.join(out_dir, "swirl.svg"))

    # Path visualizer (agent paths vs BFS-optimal + reward/delta curves).
    vmeta = os.path.join(data_dir, "viz_meta.csv")
    if os.path.exists(vmeta):
        render_viz(vmeta, os.path.join(data_dir, "viz_paths.csv"),
                   os.path.join(data_dir, "viz_curves.csv"), out_dir)

    # RL contest scoreboard + learning curves.
    scores = os.path.join(data_dir, "contest_scores.csv")
    if os.path.exists(scores):
        render_contest_scores(scores, os.path.join(out_dir, "contest_scores.svg"))
    curves = os.path.join(data_dir, "contest_curves.csv")
    if os.path.exists(curves):
        render_contest_curves(curves, os.path.join(out_dir, "contest_curves.svg"))


if __name__ == "__main__":
    main()
