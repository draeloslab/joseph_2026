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
SHOW_BARS = True    # whether significance() draws its bracket/asterisks (set per-figure by compose)


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
    for key, fn in panels.items():
        n_before = len(STATS)
        fn(axd[key], data)
        for row in STATS[n_before:]:
            row['panel'] = key
    labels = labels or dict(zip(panels, string.ascii_uppercase))
    for key, letter in labels.items():
        axd[key].text(-0.02, 1.02, letter, transform=axd[key].transAxes, ha='right', va='bottom',
                      fontsize=COLORS['_panel_label_size'], fontweight='bold')
    FIGS.mkdir(exist_ok=True)
    for ext in ('svg', 'png'):
        fig.savefig(FIGS / f'{name}.{ext}', bbox_inches='tight')
    print(f'saved figures/{name}.svg/.png')
    if STATS:
        stats = pd.DataFrame(STATS)
        stats[['panel', *stats.columns.drop('panel')]].to_csv(FIGS / f'{name}_stats.csv', index=False)
    return fig


# ---------------------------------------------------------------- shared panels
def boxes(ax, groups, colors, ylabel=None, title=None, test=None, log=False):
    """
    groups: list of (label, values). Box + jittered points, n in the tick label.
    test: None, 'independent' (Mann-Whitney U) or 'paired' (Wilcoxon; values are Series indexed by session).
    log: log-scale y-axis, for ratio metrics (e.g. normalized MSE) that can span orders of
    magnitude when a validation target has near-zero variance; only strictly positive values
    are plottable on it, so non-positive values are dropped the same way non-finite ones are.
    """
    data = [np.asarray(v, dtype=float) for _, v in groups]
    data = [d[np.isfinite(d) & (d > 0 if log else True)] for d in data]
    bp = ax.boxplot(data, tick_labels=[f'{l}\n(n={len(d)})' for (l, _), d in zip(groups, data)],
                    widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': '#111111', 'lw': 2})
    if len(groups) > 3:      # more than ~3 tick labels crowd/overlap at a normal panel width
        plt.setp(ax.get_xticklabels(), rotation=20, ha='right', rotation_mode='anchor')
    rng = np.random.default_rng(0)
    for i, (patch, d, c) in enumerate(zip(bp['boxes'], data, colors)):
        patch.set(facecolor=c, alpha=0.3, edgecolor='#444444')
        ax.scatter(rng.normal(i + 1, 0.06, len(d)), d, s=6, color=c, alpha=0.35, zorder=3, linewidths=0)
    if log:
        ax.set_yscale('log')
    ax.grid(axis='y', color='#ececec')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if test:
        significance(ax, groups, paired=test == 'paired')


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


def significance(ax, groups, paired=False):
    """
    Two-sided test on every pair of groups (Wilcoxon signed-rank on matching sessions when
    paired, else Mann-Whitney U), Holm-corrected within the panel. Every result is recorded
    in STATS; significant pairs get a bar with asterisks above the boxes, unless the
    figure's `stats_bars` config is off (see compose()), in which case only STATS is written.
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
                         'n_a': len(a), 'n_b': len(b), 'median_a': np.median(a), 'median_b': np.median(b),
                         'statistic': res.statistic, 'p': res.pvalue})
    if not rows:
        return
    for row, p_adj in zip(rows, holm([r['p'] for r in rows])):
        row['p_holm'] = p_adj
        STATS.append({'ylabel': (ax.get_ylabel() or ax.get_title()).replace('\n', ' '),
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
    for _, v in groups:
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


def open_nwb(cfg, run_id):
    """A run's trimmed NWB as neuraldecoding reads it (Dataset of type 'nwb'); close with ds.close()."""
    from omegaconf import OmegaConf
    sys.path.insert(0, str(Path(cfg['neuraldecoding_path']).expanduser()))
    from neuraldecoding.dataset import Dataset
    ds = Dataset(OmegaConf.create({'dataset_type': 'nwb', 'autoload': False, 'save_path': None,
                                   'dataset_parameters': {'nwb_file': str(master_dir(cfg) / load_master(cfg)['runs'][run_id]['nwb'])}}))
    ds.load_data()
    return ds


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


def metric_panel(ax, groups, metric, trials=None):
    """
    groups: list of (label, runs df, color), each df a subset of run_metrics.csv (see split_groups).
    One box per group. `trials`: run_trial_metrics.csv, to plot every trial's value (all dots) instead
    of one session/run-average point per run, for the metrics that have a per-trial column
    (ONLINE_TRIAL_COLS); groups without a per-trial column always use per-run means.
    """
    col = ONLINE_TRIAL_COLS.get(metric)
    if trials is not None and col is not None:
        data = [(label, trials[trials['run_id'].isin(g['run_id'])][col]) for label, g, _ in groups]
    else:
        data = [(label, g[metric]) for label, g, _ in groups]
    boxes(ax, data, [c for *_, c in groups], ylabel=ONLINE_METRICS[metric], test='independent')
