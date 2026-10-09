"""Shared config, style, plotting helpers, significance tests and the master-folder loader."""
import io
import pickle
import re
import string
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import AutoLocator, FuncFormatter, MaxNLocator, ScalarFormatter
from scipy.stats import mannwhitneyu, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
FIGS = ROOT / 'figures'
COLORS = {}
FRAME_BUDGET_MS = 1000 / 30
DOF_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']
STATS = []          # every test run while drawing the current figure, written next to it
DESCR = []          # descriptives (n, mean, sd, ...) of every group drawn in the current figure, written next to it
SHOW_BARS = True    # whether significance() draws its bracket/asterisks (set per-figure by compose)
BOX_IN = 1.5        # axes width per box (inches): every boxplot panel is resized to n_boxes * BOX_IN after layout
STYLE = {}          # config `style`: print sizes (pt) at the journal width, scaled per figure by compose()
MAX_TICKS = 4       # automatic tick locators get at most this many ticks


# ---------------------------------------------------------------- config + style
def load_config(path=ROOT / 'config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_style(cfg):
    """Fonts, colours and line widths from config `style`. Sizes there are what the printed figure shows at
    `figure_width_mm` (180 mm = journal double column); compose() rescales them per figure (print_rc)."""
    s = cfg['style']
    STYLE.clear()
    STYLE.update(s)
    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': s['font_family'], 'axes.titleweight': 'bold',
                         'axes.labelweight': 'bold', 'legend.frameon': False, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    plt.rcParams.update(print_rc(1.0))
    COLORS.update(s['palette'])
    COLORS['_panel_label_size'] = s['panel_label_size']


def print_rc(scale):
    """rcParams for a figure drawn in its own (design) inches that will be printed `scale` times smaller: every size
    is the config print size / scale, so text and lines come out at the config sizes once the figure is shrunk."""
    s, k = STYLE, 1 / scale
    lw, alw = s['line_width'] * k, s['axes_line_width'] * k
    return {'font.size': s['font_size'] * k, 'axes.titlesize': s['title_size'] * k, 'axes.labelsize': s['label_size'] * k,
            'xtick.labelsize': s['tick_size'] * k, 'ytick.labelsize': s['tick_size'] * k,
            'legend.fontsize': s['legend_size'] * k, 'lines.linewidth': lw, 'axes.linewidth': alw,
            'xtick.major.width': alw, 'ytick.major.width': alw, 'xtick.minor.width': alw * 0.6, 'ytick.minor.width': alw * 0.6,
            'xtick.major.size': 3.5 * k, 'ytick.major.size': 3.5 * k, 'xtick.minor.size': 2 * k, 'ytick.minor.size': 2 * k,
            'xtick.major.pad': 2.5 * k, 'ytick.major.pad': 2.5 * k, 'axes.labelpad': 3 * k, 'axes.titlepad': 4 * k,
            'lines.markersize': np.sqrt(s['marker_size']) * k, 'grid.linewidth': 0.5 * k, 'patch.linewidth': alw * 0.8,
            'boxplot.boxprops.linewidth': alw, 'boxplot.whiskerprops.linewidth': alw, 'boxplot.capprops.linewidth': alw,
            'boxplot.medianprops.linewidth': lw, 'hatch.linewidth': 0.5 * k, 'savefig.dpi': s['dpi']}


def marker_area(scale=1.0):
    """Scatter marker area (pt^2, design units) for the config marker size, times `scale` in diameter."""
    return (plt.rcParams['lines.markersize'] * scale) ** 2


def _tidy(fig):
    """Figure-wide conventions: no subplot titles (except axes flagged ax.keep_title: camera names, frame times) and
    at most MAX_TICKS ticks on every automatically ticked linear axis."""
    for ax in fig.axes:
        if not getattr(ax, 'keep_title', False):
            ax.set_title('')
        for axis, scale in ((ax.xaxis, ax.get_xscale()), (ax.yaxis, ax.get_yscale())):
            if scale == 'linear' and isinstance(axis.get_major_locator(), AutoLocator):
                axis.set_major_locator(MaxNLocator(nbins=MAX_TICKS - 1, steps=[1, 1.5, 2, 2.5, 3, 4, 5, 10]))
            # thousands separated by commas (1,000), unless the axis carries its own power-of-ten offset
            if scale == 'linear' and not getattr(axis, 'keep_formatter', False) and isinstance(axis.get_major_formatter(), ScalarFormatter):
                locs = axis.get_majorticklocs()
                if len(locs) and np.nanmax(np.abs(locs)) >= 1000:
                    axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:,.0f}' if abs(v - round(v)) < 1e-9 else f'{v:,.6g}'))


def power_offset(ax, power=3):
    """Counts axis shown as multiples of 10^power: the factor sits diagonally at the top of the y axis (not in the label)."""
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((power, power))
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.keep_formatter = True
    ax.yaxis.get_offset_text().set_rotation(45)
    ax.yaxis.get_offset_text().set_ha('left')
    ax.yaxis.get_offset_text().set_va('bottom')


def place_letters(fig, axd, labels):
    """Panel letters: the top of the letter flush with the top of its panel, its left edge flush with the left edge of the
    panel (axis label included). Panels whose axes top lies on one row share that row's highest panel top."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    boxes_ = {k: (axd[k].get_tightbbox(r, for_layout_only=True), axd[k].get_window_extent(r)) for k in labels}
    for k, (bb, ab) in boxes_.items():
        row = [o[0].y1 for o in boxes_.values() if abs(o[1].y1 - ab.y1) < 0.05 * H]
        t = fig.text(bb.x0 / W, max(row) / H, labels[k], ha='left', va='top', fontweight='bold',
                     fontsize=STYLE['panel_label_size'] * plt.rcParams['font.size'] / STYLE['font_size'])
        lab = axd[k].yaxis.label
        if lab.get_text():                        # a y label as tall as its panel would run into the letter: slide it down
            lb, tb = lab.get_window_extent(r), t.get_window_extent(r)
            if lb.x0 < tb.x1 and lb.y1 > tb.y0:
                shift = (lb.y1 - tb.y0 + 0.3 * tb.height) / axd[k].get_window_extent(r).height
                lab.set_y(lab.get_position()[1] - shift)
    return boxes_


def audit_text(fig, name):
    """Standards check on every piece of text in the figure: sentence case (first letter capital, no other title-cased
    words, no full stop), thousands separators, a unit separated from its number by one space. Prints what it finds."""
    allowed = {'Freebie', 'Pearson', 'Euclidean', 'Monkey', 'Run', 'Camera', 'BRAND', 'Hz', 'Label', 'Index', 'MRS', 'PIP', 'DIP', 'MCP'}
    texts = []
    for ax in fig.axes:
        texts += [ax.get_xlabel(), ax.get_ylabel(), ax.get_title()]
        texts += [t.get_text() for t in ax.get_xticklabels() + ax.get_yticklabels() if not t.get_text().lstrip('-−').replace('.', '').replace(',', '').isdigit()]
        texts += [t.get_text() for t in ax.texts]
        if ax.get_legend():
            texts += [t.get_text() for t in ax.get_legend().get_texts()]
    bad = []
    for t in dict.fromkeys(x for x in texts if x.strip()):
        for line in t.split('\n'):
            words = [w for w in re.findall(r"[A-Za-z][A-Za-z\-']+", line) if len(w) > 3 and not w.isupper() and w not in allowed]
            caps = [w for w in words[1:] if w[0].isupper()]
            first = re.search(r'[A-Za-z]', line)
            numeric = line.strip()[:1].isdigit() or line.strip().startswith('$')
            if (first and first.group().islower() and line is t.split('\n')[0] and not numeric) or line.rstrip().endswith('.') or caps or re.search(r'\d(ms|s|px|Hz|mm|fps|%)\b', line) and not re.search(r'\d%', line):
                bad.append(line)
    for line in dict.fromkeys(bad):
        print(f'  [text standard] {name}: {line!r}')


def _save(fig, name):
    """SVG scaled to exactly figure_width_mm wide (vector, every size shrinks together) and a PNG of the same
    print size at the config dpi."""
    width_in = STYLE['figure_width_mm'] / 25.4
    buf = io.StringIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    svg = buf.getvalue()
    m = re.search(r'<svg[^>]*?width="([\d.]+)pt" height="([\d.]+)pt"', svg)
    w_pt, h_pt = float(m.group(1)), float(m.group(2))
    w_mm = STYLE['figure_width_mm']
    svg = svg.replace(f'width="{m.group(1)}pt" height="{m.group(2)}pt"',
                      f'width="{w_mm:.2f}mm" height="{w_mm * h_pt / w_pt:.2f}mm"', 1)
    (FIGS / f'{name}.svg').write_text(svg)
    fig.savefig(FIGS / f'{name}.png', bbox_inches='tight', dpi=STYLE['dpi'] * width_in / (w_pt / 72))


def compose(layout, panels, data, size, name, labels=None, stats_bars=True, height_ratios=None, width_ratios=None,
            post_layout=None):
    """
    Build one figure from a subplot_mosaic layout. `panels` maps mosaic key -> panel
    function(ax, data). `labels` maps key -> panel letter (default: A, B, ... in `panels`
    order). Rearranging a figure only means editing `layout`; to make some rows/columns
    bigger or smaller than others (equal by default), pass `height_ratios`/`width_ratios`
    (one number per row/column of `layout`). `stats_bars` (from config.yaml's `stats_bars`)
    controls whether significance() draws its bracket/asterisks on this figure; the
    underlying tests are always run and always written to the stats csv. `post_layout(fig, axd)` runs after the
    layout is frozen and the box panels are sized, to place panels by hand.

    `size` is the design size (inches); the saved figure is shrunk to config style.figure_width_mm, and text / lines
    are drawn at the config print sizes divided by that shrink (print_rc).
    """
    with plt.rc_context(print_rc(STYLE['figure_width_mm'] / 25.4 / size[0])):
        return _compose(layout, panels, data, size, name, labels, stats_bars, height_ratios, width_ratios, post_layout)


def _compose(layout, panels, data, size, name, labels, stats_bars, height_ratios, width_ratios, post_layout):
    global SHOW_BARS
    SHOW_BARS = stats_bars
    fig = plt.figure(figsize=size, layout='constrained')
    axd = fig.subplot_mosaic(layout, height_ratios=height_ratios, width_ratios=width_ratios)
    STATS.clear()
    DESCR.clear()
    for key, fn in panels.items():
        n_before, d_before = len(STATS), len(DESCR)
        fn(axd[key], data)
        for row in STATS[n_before:] + DESCR[d_before:]:
            row['panel'] = key
    _tidy(fig)
    # identical box widths: freeze the constrained layout, then give every boxplot panel (tagged with
    # ax.n_boxes by boxes() / the panel) a width of n_boxes * BOX_IN, keeping its left edge and height
    fig.canvas.draw()
    fig.set_layout_engine('none')
    for ax in fig.axes:
        if getattr(ax, 'n_boxes', None):
            x0, y0, _, h = ax.get_position().bounds
            ax.set_position([x0, y0, ax.n_boxes * BOX_IN / fig.get_figwidth(), h])
    if post_layout:
        post_layout(fig, axd)
    labels = dict(zip(panels, string.ascii_uppercase)) if labels is None else labels   # {} = no panel letters
    place_letters(fig, axd, labels)
    audit_text(fig, name)
    FIGS.mkdir(exist_ok=True)
    _save(fig, name)
    print(f'saved {(FIGS / name).relative_to(ROOT)}.svg/.png')
    for rows, suffix in ((STATS, 'stats'), (DESCR, 'descriptives')):
        out = FIGS / f'{name}_{suffix}.csv'
        if rows:
            df = pd.DataFrame(rows)
            df[['panel', *df.columns.drop('panel')]].to_csv(out, index=False)
        else:
            out.unlink(missing_ok=True)      # nothing tested / described this time: drop a stale file
    return fig


def _summary(values):
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {'n': 0}
    return {'n': len(v), 'mean': v.mean(), 'sd': v.std(ddof=1) if len(v) > 1 else np.nan, 'sem': v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
            'median': np.median(v), 'q1': np.percentile(v, 25), 'q3': np.percentile(v, 75), 'min': v.min(), 'max': v.max()}


def describe(ax, label, values, unit=None, **extra):
    """Record one group's descriptives for the figure's <name>_descriptives.csv (and figures/all_stats.csv).
    unit: what one value is (trial, run, session, frame). extra: e.g. n_sessions / n_trials / run-level summaries."""
    DESCR.append({'measure': (ax.get_ylabel() or ax.get_xlabel() or ax.get_title()).replace('\n', ' '),
                  'group': str(label).replace('\n', ' '), 'unit': unit, **_summary(values), **extra})


# the paper's figures, in order, for figures/all_stats.csv (name -> config key)
ALL_STATS_FIGS = {'fig2_validation_heldout': '2', 'fig3_latency_war_machine': '3', 'fig3_latency_war_machine_merged': '3 (merged fps)',
                  'fig4_hand_control': '4',
                  'fig5_kalman_filter': '5', 'figS1_latency_old_pipeline': 'S1',
                  'figS2_hand_control_old_system': 'S2',
                  'figS3_human_hand_control': 'S3', 'figS4_kalman_filter_old_system': 'S4',
                  'validation_all_frames': 'validation (all frames)', 'human_hand_control_0930': 'human 09-30'}


def write_all_stats():
    """figures/all_stats.csv: every figure's descriptives (row_type 'group') and significance tests (row_type 'test')
    from the <name>_descriptives.csv / <name>_stats.csv files on disk, so figures drawn in earlier runs are included."""
    parts = []
    for name, key in ALL_STATS_FIGS.items():
        for suffix, kind in (('descriptives', 'group'), ('stats', 'test')):
            f = FIGS / f'{name}_{suffix}.csv'
            if f.exists():
                df = pd.read_csv(f)
                if kind == 'test':
                    df = df.rename(columns={'ylabel': 'measure'})
                    df['significant'] = df['p_holm'] < 0.05
                parts.append(df.assign(figure=key, figure_file=name, row_type=kind))
    if not parts:
        return
    out = pd.concat(parts, ignore_index=True)
    lvl = lambda sfx: [f'{k}_{sfx}' for k in ('mean', 'sd', 'sem', 'median', 'q1', 'q3', 'min', 'max')]
    first = ['figure', 'figure_file', 'panel', 'row_type', 'measure', 'group', 'unit', 'n', 'n_sessions', 'n_runs', 'n_trials',
             'n_frames', 'mean', 'sd', 'sem', 'median', 'q1', 'q3', 'min', 'max', *lvl('run'), *lvl('trial'), 'trial_value', 'pct_over_budget', 'pct_processor_skipped', 'pct_camera_missing',
             'group_a', 'group_b', 'test', 'tested_on', 'n_a', 'n_b', 'mean_a', 'mean_b', 'median_a', 'median_b',
             'statistic', 'p', 'p_holm', 'significant']
    cols = [c for c in first if c in out] + [c for c in out if c not in first]
    out[cols].to_csv(FIGS / 'all_stats.csv', index=False)
    print(f'saved figures/all_stats.csv ({len(out)} rows)')


# ---------------------------------------------------------------- shared panels
def boxes(ax, groups, colors, ylabel=None, title=None, test=None, log=False, test_groups=None, n_labels=None,
          unit=None, describe_groups=True):
    """
    groups: list of (label, values). Box + jittered points; n_labels: an optional second tick-label line per group.
    test: None, 'independent' (Mann-Whitney U) or 'paired' (Wilcoxon; values are Series indexed by session).
    test_groups: the (label, values) to test instead of the plotted ones (e.g. run means while every trial is drawn).
    unit: what one value is (trial, run, session, frame), for the descriptives csv; describe_groups=False when the
    caller records richer descriptives itself (metric_panel).
    log: log-scale y-axis, for ratio metrics (e.g. normalized MSE) that can span orders of
    magnitude when a validation target has near-zero variance; only strictly positive values
    are plottable on it, so non-positive values are dropped the same way non-finite ones are.
    """
    data = [np.asarray(v, dtype=float) for _, v in groups]
    data = [d[np.isfinite(d) & (d > 0 if log else True)] for d in data]
    ticks = [f'{l}\n{n}' for (l, _), n in zip(groups, n_labels)] if n_labels else [l for l, _ in groups]
    bp = ax.boxplot(data, tick_labels=ticks, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': '#111111'})
    ax.n_boxes = len(groups)
    if len(groups) > 3:      # more than ~3 tick labels crowd/overlap at a normal panel width
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    rng = np.random.default_rng(0)
    for i, (patch, d, c) in enumerate(zip(bp['boxes'], data, colors)):
        patch.set(facecolor=c, alpha=0.3, edgecolor='#444444')
        few = len(d) <= 50          # one point per run/session: bigger, opaque
        ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=marker_area(1.6 if few else 0.9), color=c,
                   alpha=0.85 if few else 0.4, zorder=3, linewidths=0.4 * plt.rcParams['axes.linewidth'] if few else 0,
                   edgecolors='white')
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', color='#ececec')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if describe_groups:
        for (label, _), d in zip(groups, data):
            describe(ax, label, d, unit)
    if test:
        significance(ax, test_groups or groups, paired=test == 'paired', shown=groups,
                     tested_on=getattr(ax, 'tested_on', unit))


# ---------------------------------------------------------------- statistics
def holm(p):
    """Holm-Bonferroni adjusted p-values."""
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    adj = np.minimum(np.maximum.accumulate((len(p) - np.arange(len(p))) * p[order]), 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out


def stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''


def significance(ax, groups, paired=False, shown=None, tested_on=None):
    """
    Two-sided test on every pair of groups (Wilcoxon signed-rank on matching sessions when
    paired, else Mann-Whitney U), Holm-corrected within the panel. Every result is recorded
    in STATS; significant pairs get a bar with asterisks above the boxes, unless the
    figure's `stats_bars` config is off (see compose()), in which case only STATS is written.
    shown: the plotted (label, values) when they differ from the tested ones; the bars go above those.
    tested_on: what one tested value is (run means, trials, ...), recorded in the stats csv.
    """
    rows = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            (la, a), (lb, b) = groups[i], groups[j]
            if paired:
                # inner join on the shared index (e.g. session_id) keeps only matched pairs;
                # both columns must be finite together, or a lone inf/nan desyncs the pairing
                ab = pd.concat([pd.Series(a, dtype=float), pd.Series(b, dtype=float)], axis=1, join='inner')
                ab = ab[np.isfinite(ab).all(axis=1)]
                a, b = ab.iloc[:, 0].to_numpy(), ab.iloc[:, 1].to_numpy()
            else:
                a, b = np.asarray(a, float), np.asarray(b, float)
                a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if min(len(a), len(b)) < 2:
                continue            # nothing to test (e.g. a metric not computed yet)
            res = wilcoxon(a, b) if paired else mannwhitneyu(a, b)
            rows.append({'i': i, 'j': j, 'group_a': la.replace('\n', ' '), 'group_b': lb.replace('\n', ' '),
                         'test': 'Wilcoxon signed-rank' if paired else 'Mann-Whitney U',
                         'n_a': len(a), 'n_b': len(b), 'mean_a': np.mean(a), 'mean_b': np.mean(b),
                         'median_a': np.median(a), 'median_b': np.median(b),
                         'statistic': res.statistic, 'p': res.pvalue})
    if not rows:
        return
    for row, p_adj in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = p_adj
        STATS.append({'ylabel': (ax.get_ylabel() or ax.get_title()).replace('\n', ' '), 'tested_on': tested_on,
                      **{k: v for k, v in row.items() if k not in ('i', 'j')}})

    if not SHOW_BARS:
        return

    # bars: shortest spans lowest, stacked above the highest point of the panel. Positions are
    # computed in log10 space on a log axis (an additive step there is a multiplicative one in
    # data units), else directly in data units.
    is_log = ax.get_yscale() == 'log'
    to_space = (lambda v: np.log10(v)) if is_log else (lambda v: v)
    from_space = (lambda v: 10 ** v) if is_log else (lambda v: v)
    lo, hi = ax.get_ylim()
    lo_s, hi_s = to_space(lo), to_space(hi)
    tops = []
    for _, v in shown or groups:
        v = np.asarray(v, float)
        v = v[np.isfinite(v) & (v > 0)] if is_log else v[np.isfinite(v)]
        if len(v):
            tops.append(np.max(v))
    top_s = min(max(to_space(t) for t in tops), hi_s)   # fliers may lie above the axis
    step_s = 0.11 * (hi_s - lo_s)
    level = 0
    for r in sorted((r for r in rows if r['p_holm'] < 0.05), key=lambda r: (r['j'] - r['i'], r['i'])):
        y_s = top_s + step_s * (level + 0.6)
        y, y_tick = from_space(y_s), from_space(y_s - step_s * 0.25)
        ax.plot([r['i'] + 1, r['i'] + 1, r['j'] + 1, r['j'] + 1], [y_tick, y, y, y_tick],
                color='#111111', lw=plt.rcParams['axes.linewidth'], clip_on=False)
        ax.text((r['i'] + r['j']) / 2 + 1, y, stars(r['p_holm']), ha='center', va='bottom',
                fontsize=plt.rcParams['font.size'] * 1.2, fontweight='bold')
        level += 1
    if level:
        ax.set_ylim(lo, from_space(max(hi_s, top_s + step_s * (level + 0.9))))


def target_trace(ax, trials, dof_col, series, ylabel):
    """Target windows (grey hit / red miss) with traces. series: list of (t, y, label, color, lw)."""
    t0 = trials['start_time'].iloc[0]
    for _, tr in trials.iterrows():
        r = tr['target_radius']
        ax.add_patch(plt.Rectangle((tr['start_time'] - t0, tr[dof_col] - r), tr['stop_time'] - tr['start_time'],
                                   2 * r, facecolor=COLORS['target_hit'] if tr['success'] else COLORS['target_miss'],
                                   edgecolor='#6f6f6f', ls='--', lw=1, alpha=0.7, zorder=1))
    for t, y, label, color, lw in series:
        ax.plot(t - t0, y, color=color, lw=lw, label=label, zorder=3)
    ax.set_xlim(0, trials['stop_time'].iloc[-1] - t0)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    # above the axes, not inside it -- inside overlaps the traces/target boxes once the panel is short
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01), ncol=min(len(series), 2), borderaxespad=0, columnspacing=1.2,
              handlelength=1.6)


# ---------------------------------------------------------------- the master folder
@cache
def _read_master(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def master_dir(cfg):
    p = Path(cfg['master']).expanduser()
    return p if p.is_absolute() else ROOT / p


def load_master(cfg):
    """<master>/master.pkl, read once per process. Every decision (excluded runs, chosen Freebie signals, clock
    offsets, ...) was made when the master was built and is stored in it."""
    return _read_master(str(master_dir(cfg) / 'master.pkl'))


def open_nwb_path(cfg, path):
    """Any NWB file as neuraldecoding reads it (Dataset of type 'nwb'); close with ds.close()."""
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(cfg['neuraldecoding_path']).expanduser()))
    from neuraldecoding.dataset import Dataset
    ds = Dataset(OmegaConf.create({'dataset_type': 'nwb', 'autoload': False, 'save_path': None,
                                   'dataset_parameters': {'nwb_file': str(path)}}))
    ds.load_data()
    return ds


def open_nwb(cfg, run_id):
    """A master run's trimmed NWB as neuraldecoding reads it; close with ds.close()."""
    return open_nwb_path(cfg, master_dir(cfg) / load_master(cfg)['runs'][run_id]['nwb'])


def trials_frame(nwb, dofs):
    """The run's trials with the target on each active DOF as target_<dof>."""
    tr = nwb.trials.to_dataframe()
    for j, d in enumerate(dofs):
        tr[f'target_{d}'] = [t[j] for t in tr['target']]
    return tr


def trial_window(trials, first_trial, n_trials, keep):
    """n_trials consecutive trials from first_trial; auto (None) = a third of the way through `keep`."""
    if first_trial is None:
        ok = trials.index[keep]
        first_trial = int(trials.index.get_loc(ok[len(ok) // 3]))
    return trials.iloc[first_trial:first_trial + n_trials]


ONLINE_METRICS = {'success_rate': 'Success rate', 'bitrate_bits_per_s': 'Bit rate (bits/s)',
                  'targets_per_s': 'Targets per second', 'time_to_target_s': 'Time to target (s)',
                  'orbiting_time_s': 'Orbiting time (s)', 'path_efficiency': 'Path efficiency'}

# every online metric's column in run_trial_metrics.csv. bitrate_bits_per_s and the ttt/orbit/path
# metrics are real per-trial values; success_rate and targets_per_s have no natural per-trial value (one
# is a single 0/1, the other only defined over a whole run) so theirs is a running cumulative rate instead
# (see metrics.run_metrics) -- successes-so-far/trials-so-far, targets-so-far/elapsed-time-so-far.
ONLINE_TRIAL_COLS = {'success_rate': 'success_rate_running', 'bitrate_bits_per_s': 'bitrate_bits_per_s',
                     'targets_per_s': 'targets_per_s_running', 'time_to_target_s': 'time_to_target_s',
                     'orbiting_time_s': 'orbiting_time_s', 'path_efficiency': 'path_efficiency'}

FEATURE_CUTOFF = '20260227'    # first day pip_y was used to drive online Freebie runs; index_angle before it


def online_feature(sessions):
    """The Freebie signal driving an online run, from its (comma-joined) session id(s)."""
    return 'pip_y' if sessions.split(',')[0] >= FEATURE_CUTOFF else 'index_angle'


def split_groups(runs, split_feature=False, split_dof=False):
    """
    Bend vs Freebie groups from a run_metrics-shaped dataframe (needs freebie, sessions, n_dof columns),
    optionally split further: split_feature separates Freebie runs by which signal (index_angle / pip_y,
    see online_feature) drove them; split_dof separates every group into 1 DOF / 2 DOF. Returns
    [(label, runs df, color), ...] for metric_panel / boxes.
    """
    bend, free = runs[~runs['freebie']], runs[runs['freebie']]
    if split_feature:
        feature = free['sessions'].map(online_feature)
        groups = [('Bend sensor', bend, COLORS['bend']),
                  ('Freebie\n(index angle)', free[feature == 'index_angle'], COLORS['index_angle']),
                  ('Freebie\n(PIP y)', free[feature == 'pip_y'], COLORS['pip_y'])]
    else:
        groups = [('Bend sensor', bend, COLORS['bend']), ('Freebie', free, COLORS['freebie'])]
    if not split_dof:
        return groups
    return [(f'{label} {n}DOF', g[g['n_dof'] == n], c) for label, g, c in groups for n in (1, 2)]


def _n_sessions(g):
    """Recording sessions behind a group of runs: the master's session ids, or the day of a War Machine run id."""
    if 'sessions' in g:
        return len({x for v in g['sessions'] for x in str(v).split(',')})
    ids = g['run_id'].astype(str)
    return ids.str[:10].nunique() if len(ids) and ids.str.match(r'\d{4}-\d{2}-\d{2}_').all() else np.nan


def session_values(g, metric):
    """One value per session for a group of runs: a run's sub-recordings (its comma-joined session ids) are one
    session, named by the first; runs that share a session are averaged. Trials and runs of a session are repeated
    measures, so this is the independent unit for the descriptives and tests of the old-system figures."""
    key = g['sessions'].astype(str).str.split(',').str[0]
    return g.groupby(key)[metric].mean()


def metric_panel(ax, groups, metric, trials=None, stats_trials=None, show_trials=True, by_session=False):
    """
    groups: list of (label, runs df, color), each df a subset of run_metrics.csv (see split_groups).
    One box per group. `trials`: run_trial_metrics.csv, to plot every trial's value (all dots) instead
    of one session/run-average point per run, for the metrics that have a per-trial column
    (ONLINE_TRIAL_COLS); groups without a per-trial column always use per-run means.
    stats_trials: the same trial table, to instead test on one mean per run (no pseudo-replication) and label
    the box shows every trial (show_trials) or the run means.
    by_session (needs stats_trials): descriptives and tests on one value per session (session_values) instead of
    one per run; the plotted dots are unchanged.
    Every group's n sessions / runs / trials and its run-level and trial-level summaries go to the descriptives csv.
    """
    col = ONLINE_TRIAL_COLS.get(metric)
    runs = [(label, g[metric]) for label, g, _ in groups]
    colors = [c for *_, c in groups]
    table = stats_trials if stats_trials is not None else trials
    per = [table[table['run_id'].isin(g['run_id'])][col] if table is not None and col is not None else pd.Series(dtype=float)
           for _, g, _ in groups]
    if stats_trials is not None:
        n_labels = None            # n runs / trials go to the descriptives csv and the legend, not the axis
        shown = [(label, t) for (label, *_), t in zip(groups, per)] if show_trials else runs
        tested = [(label, session_values(g, metric)) for label, g, _ in groups] if by_session else runs
        ax.tested_on = 'session means' if by_session else 'run means'
        boxes(ax, shown, colors, ylabel=ONLINE_METRICS[metric], test='independent', test_groups=tested, n_labels=n_labels,
              describe_groups=False)
        plotted = 'session' if by_session else 'trial' if show_trials else 'run'
    elif trials is not None and col is not None:
        ax.tested_on = 'trials'
        boxes(ax, [(label, t) for (label, *_), t in zip(groups, per)], colors, ylabel=ONLINE_METRICS[metric],
              test='independent', describe_groups=False)
        plotted = 'trial'
    else:
        ax.tested_on = 'run means'
        boxes(ax, runs, colors, ylabel=ONLINE_METRICS[metric], test='independent', describe_groups=False)
        plotted = 'run'
    for (label, g, _), t in zip(groups, per):
        r, tr = _summary(g[metric]), _summary(t)
        sess = session_values(g, metric) if plotted == 'session' else None
        describe(ax, label, sess if plotted == 'session' else t if plotted == 'trial' else g[metric], unit=plotted,
                 n_sessions=len(sess.dropna()) if plotted == 'session' else _n_sessions(g),
                 trial_value='running rate up to the trial' if col and col.endswith('_running') else 'per trial',
                 n_runs=r.get('n', 0), n_trials=tr.get('n', 0),
                 **{f'{k}_run': v for k, v in r.items() if k != 'n'}, **{f'{k}_trial': v for k, v in tr.items() if k != 'n'})
