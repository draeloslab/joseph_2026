"""Shared config, style, plotting helpers, significance tests and the master-folder loader."""
import pickle
import string
import sys
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
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


# ---------------------------------------------------------------- config + style
def load_config(path=ROOT / 'config.yaml'):
    with open(path) as f:
        return yaml.safe_load(f)


def apply_style(cfg):
    s = cfg['style']
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': s['font_family'],
        'font.size': s['font_size'], 'axes.titlesize': s['title_size'],
        'axes.titleweight': 'bold', 'axes.labelsize': s['label_size'],
        'axes.labelweight': 'bold', 'xtick.labelsize': s['tick_size'],
        'ytick.labelsize': s['tick_size'], 'legend.fontsize': s['legend_size'],
        'legend.frameon': False, 'lines.linewidth': s['line_width'],
        'axes.linewidth': s['axes_line_width'], 'xtick.major.width': s['axes_line_width'],
        'ytick.major.width': s['axes_line_width'], 'lines.markersize': np.sqrt(s['marker_size']),
        'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': s['dpi'], 'svg.fonttype': 'none', 'pdf.fonttype': 42,
    })
    COLORS.update(s['palette'])
    COLORS['_panel_label_size'] = s['panel_label_size']


def compose(layout, panels, data, size, name, labels=None, stats_bars=True, height_ratios=None, width_ratios=None):
    """
    Build one figure from a subplot_mosaic layout. `panels` maps mosaic key -> panel
    function(ax, data). `labels` maps key -> panel letter (default: A, B, ... in `panels`
    order). Rearranging a figure only means editing `layout`; to make some rows/columns
    bigger or smaller than others (equal by default), pass `height_ratios`/`width_ratios`
    (one number per row/column of `layout`). `stats_bars` (from config.yaml's `stats_bars`)
    controls whether significance() draws its bracket/asterisks on this figure; the
    underlying tests are always run and always written to the stats csv.
    """
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
    # identical box widths: freeze the constrained layout, then give every boxplot panel (tagged with
    # ax.n_boxes by boxes() / the panel) a width of n_boxes * BOX_IN, keeping its left edge and height
    fig.canvas.draw()
    fig.set_layout_engine('none')
    for ax in fig.axes:
        if getattr(ax, 'n_boxes', None):
            x0, y0, _, h = ax.get_position().bounds
            ax.set_position([x0, y0, ax.n_boxes * BOX_IN / fig.get_figwidth(), h])
    labels = dict(zip(panels, string.ascii_uppercase)) if labels is None else labels   # {} = no panel letters
    for key, letter in labels.items():
        axd[key].text(-0.02, 1.02, letter, transform=axd[key].transAxes, ha='right', va='bottom',
                      fontsize=COLORS['_panel_label_size'], fontweight='bold')
    FIGS.mkdir(exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(FIGS / f'{name}.{ext}', bbox_inches='tight')
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
ALL_STATS_FIGS = {'fig1_validation': '1', 'fig1_validation_heldout': '1_heldout', 'fig2_war_machine': '2',
                  'fig2_war_machine_merged': '2 (merged fps)', 'fig2_war_machine_dropped_frames': '2 (dropped frames)',
                  'fig2_supplement_latency': '2_supplement', 'fig2_supplement_dropped_frames': '2_supplement (dropped frames)',
                  'fig3_war_machine': '3', 'fig3_supplement_hand_control': '3_supplement', 'fig4_war_machine': '4',
                  'fig4_supplement_kf_decode': '4_supplement', 'fig6_human': '6'}


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
             'n_frames', 'mean', 'sd', 'sem', 'median', 'q1', 'q3', 'min', 'max', *lvl('run'), *lvl('trial'), 'trial_value',
             'group_a', 'group_b', 'test', 'tested_on', 'n_a', 'n_b', 'mean_a', 'mean_b', 'median_a', 'median_b',
             'statistic', 'p', 'p_holm', 'significant']
    cols = [c for c in first if c in out] + [c for c in out if c not in first]
    out[cols].to_csv(FIGS / 'all_stats.csv', index=False)
    print(f'saved figures/all_stats.csv ({len(out)} rows)')


# ---------------------------------------------------------------- shared panels
def boxes(ax, groups, colors, ylabel=None, title=None, test=None, log=False, test_groups=None, n_labels=None,
          unit=None, describe_groups=True):
    """
    groups: list of (label, values). Box + jittered points, n in the tick label (or n_labels: one string per group).
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
    n_labels = n_labels or [f'(n={len(d)})' for d in data]
    bp = ax.boxplot(data, tick_labels=[f'{l}\n{n}' for (l, _), n in zip(groups, n_labels)],
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': '#111111', 'lw': 2})
    ax.n_boxes = len(groups)
    if len(groups) > 3:      # more than ~3 tick labels crowd/overlap at a normal panel width
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    rng = np.random.default_rng(0)
    for i, (patch, d, c) in enumerate(zip(bp['boxes'], data, colors)):
        patch.set(facecolor=c, alpha=0.3, edgecolor='#444444')
        few = len(d) <= 50          # one point per run/session: draw them big enough to see
        ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=30 if few else 6, color=c, alpha=0.8 if few else 0.35, zorder=3,
                   linewidths=0.5 if few else 0, edgecolors='white')
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
    step_s = 0.07 * (hi_s - lo_s)
    level = 0
    for r in sorted((r for r in rows if r['p_holm'] < 0.05), key=lambda r: (r['j'] - r['i'], r['i'])):
        y_s = top_s + step_s * (level + 0.6)
        y, y_tick = from_space(y_s), from_space(y_s - step_s * 0.25)
        ax.plot([r['i'] + 1, r['i'] + 1, r['j'] + 1, r['j'] + 1], [y_tick, y, y, y_tick],
                color='#111111', lw=1.5, clip_on=False)
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
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1, 0.15), mode='expand', ncol=len(series), borderaxespad=0)


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
                  'targets_per_s': 'Targets / s', 'time_to_target_s': 'Time to target (s)',
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


def metric_panel(ax, groups, metric, trials=None, stats_trials=None, show_trials=True):
    """
    groups: list of (label, runs df, color), each df a subset of run_metrics.csv (see split_groups).
    One box per group. `trials`: run_trial_metrics.csv, to plot every trial's value (all dots) instead
    of one session/run-average point per run, for the metrics that have a per-trial column
    (ONLINE_TRIAL_COLS); groups without a per-trial column always use per-run means.
    stats_trials: the same trial table, to instead test on one mean per run (no pseudo-replication) and label
    each box with its n runs and n trials; the box shows every trial (show_trials) or the run means.
    Every group's n sessions / runs / trials and its run-level and trial-level summaries go to the descriptives csv.
    """
    col = ONLINE_TRIAL_COLS.get(metric)
    runs = [(label, g[metric]) for label, g, _ in groups]
    colors = [c for *_, c in groups]
    table = stats_trials if stats_trials is not None else trials
    per = [table[table['run_id'].isin(g['run_id'])][col] if table is not None and col is not None else pd.Series(dtype=float)
           for _, g, _ in groups]
    if stats_trials is not None:
        n_labels = [f'{np.isfinite(g[metric]).sum()} runs\n{np.isfinite(t).sum()} trials' for (_, g, _), t in zip(groups, per)]
        shown = [(label, t) for (label, *_), t in zip(groups, per)] if show_trials else runs
        ax.tested_on = 'run means'
        boxes(ax, shown, colors, ylabel=ONLINE_METRICS[metric], test='independent', test_groups=runs, n_labels=n_labels,
              describe_groups=False)
        plotted = 'trial' if show_trials else 'run'
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
        describe(ax, label, t if plotted == 'trial' else g[metric], unit=plotted, n_sessions=_n_sessions(g),
                 trial_value='running rate up to the trial' if col and col.endswith('_running') else 'per trial',
                 n_runs=r.get('n', 0), n_trials=tr.get('n', 0),
                 **{f'{k}_run': v for k, v in r.items() if k != 'n'}, **{f'{k}_trial': v for k, v in tr.items() if k != 'n'})
