"""Figure S4: Kalman filter decoding from EMG on the earlier system, trained on Freebie signals vs bend sensors."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import COLORS, DATA, ONLINE_METRICS, boxes, compose, metric_panel, split_groups

DECODERS = {'freebie': 'Freebie KF', 'bend': 'KF trained on bend sensor'}
# (DOF, signal) of each offline-KF box (data/offline_kf.csv); the bend-sensor KF of a session is trained on the same
# stretch of data as its Freebie KFs
OFFLINE = [('Index', 'Bend sensor'), ('Index', 'Freebie index angle'), ('Index', 'Freebie PIP y'),
           ('MRS', 'Bend sensor'), ('MRS', 'Freebie MRS x')]
SIGNAL_COLOR = {'Bend sensor': 'bend', 'Freebie index angle': 'index_angle', 'Freebie PIP y': 'pip_y', 'Freebie MRS x': 'mrs_x'}

OFFLINE_METRICS = {'nmse_valid': 'Normalized MSE', 'correlation_valid': 'Correlation (r)'}

# A = Freebie KF, B = bend KF (index + MRS stacked in one axis each), D = offline KF (index nmse | r over MRS nmse | r),
# C = online metrics. A / B take C's first column, the D panels its second and third (spread() lines them up).
m = list(ONLINE_METRICS)
_D = [f'{mt}_{dof}' for mt in OFFLINE_METRICS for dof in ('Index', 'MRS')]     # nmse_Index, nmse_MRS, corr_Index, corr_MRS
LAYOUT = [['freebie'] * 3 + [_D[0]] * 3 + [_D[2]] * 3,
          ['bend'] * 3 + [_D[1]] * 3 + [_D[3]] * 3,
          [k for k in m[:3] for _ in range(3)],
          [k for k in m[3:] for _ in range(3)]]
WIDTH_RATIOS = None
SIZE = (21, 19)
HEIGHT_RATIOS = [1.1, 1.1, 1, 1]
TRACE_DOFS = [(1, 'Index', 'index_angle', 1.45), (3, 'MRS', 'mrs_x', 0.0)]   # (dof, label, freebie colour, y offset)


def load(cfg):
    opts = cfg['online_metrics']
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    closed = runs[runs['mode'] == 'closed_loop']
    return {**{key: (dict(np.load(DATA / f'figS4_example_{key}_trace.npz')),
                     pd.read_csv(DATA / f'figS4_example_{key}_trials.csv')) for key in DECODERS},
            'offline': pd.read_csv(DATA / 'offline_kf.csv'),
            'groups': split_groups(closed, opts['split_feature'], opts['split_dof']),
            'trials': pd.read_csv(DATA / 'run_trial_metrics.csv'), 'show_trials': opts['all_trials']}


# ---------------------------------------------------------------- panels
def panel_decode(ax, d, key):
    """Both DOFs of one decoder in one axis, index stacked above MRS, each over its own target boxes."""
    tr, trials = d[key]
    t0 = trials['start_time'].iloc[0]
    ticks, labels = [], []
    for dof, name, freebie_color, off in TRACE_DOFS:
        color = COLORS[freebie_color] if key == 'freebie' else COLORS['bend']
        for _, row in trials.iterrows():
            r = row['target_radius']
            ax.add_patch(plt.Rectangle((row['start_time'] - t0, row[f'target_{dof}'] - r + off),
                                       row['stop_time'] - row['start_time'], 2 * r,
                                       facecolor=COLORS['target_hit'] if row['success'] else COLORS['target_miss'],
                                       edgecolor='#6f6f6f', ls='--', lw=1, alpha=0.7, zorder=1))
        ax.plot(tr['t'] - t0, tr[f'decode_{dof}'] + off, color=color, lw=2.5, zorder=3)
        ax.text(0.005, off + 1.12, name, transform=ax.get_yaxis_transform(), ha='left', va='bottom', zorder=4,
                color=color, fontweight='bold', fontsize=plt.rcParams['axes.labelsize'])
        ticks += [off, off + 0.5, off + 1]
        labels += ['0', '0.5', '1']
    ax.set_yticks(ticks, labels)
    ax.set_ylim(-0.15, TRACE_DOFS[0][3] + 1.4)
    ax.set_xlim(0, trials['stop_time'].iloc[-1] - t0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (a.u.)')
    ax.set_title(DECODERS[key])


def panel_offline(ax, d, dof, metric):
    off = d['offline'][d['offline']['dof'] == dof]
    signals = [s for dof_, s in OFFLINE if dof_ == dof]
    by_signal = {s: g.set_index('session_id')[metric] for s, g in off.groupby('signal')}
    # normalized MSE is a variance ratio: near-zero-variance validation targets in a handful of
    # sessions push it into the hundreds, so it needs a log axis to stay readable
    boxes(ax, [(s.replace('Freebie ', 'Freebie\n'), by_signal[s]) for s in signals],
          [COLORS[SIGNAL_COLOR[s]] for s in signals], ylabel=OFFLINE_METRICS[metric],
          title=f'Offline KF, {dof} DOF', test='paired', log=metric == 'nmse_valid', unit='session')


def panel_metric(ax, d, metric):
    # box + dots: every trial (config online_metrics.all_trials) or the run means; descriptives and tests on one mean per session
    metric_panel(ax, d['groups'], metric, stats_trials=d['trials'], show_trials=d['show_trials'], by_session=True)


PANELS = {**{k: partial(panel_decode, key=k) for k in DECODERS},
          **{f'{m}_{dof}': partial(panel_offline, dof=dof, metric=m) for m in OFFLINE_METRICS for dof in ('Index', 'MRS')},
          **{m: partial(panel_metric, metric=m) for m in ONLINE_METRICS}}
LABELS = {'freebie': 'A', 'bend': 'B', 'nmse_valid_Index': 'D', 'success_rate': 'C'}


ROW_GAP_IN = 1.5       # gap between neighbouring panels (inches, design size): room for the y label and ticks


def spread(fig, axd):
    """C's three columns span the figure with equal gaps; A / B sit over C's first column, and the offline-KF panels (D)
    over its second and third at the same box width as C (the 2-box MRS panels start at their column's left edge)."""
    pos = lambda k: axd[k].get_position()
    left, right = min(pos(k).x0 for k in m), max(pos(k).x1 for k in m)
    gap = ROW_GAP_IN / fig.get_figwidth()
    unit = (right - left - 2 * gap) / 9
    col = [left + i * (3 * unit + gap) for i in range(3)]
    for row in (m[:3], m[3:]):
        for k, x in zip(row, col):
            p = pos(k)
            axd[k].set_position([x, p.y0, 3 * unit, p.height])
    for k in ('freebie', 'bend'):
        p = pos(k)
        axd[k].set_position([col[0], p.y0, 3 * unit, p.height])
    for k, x in zip(_D, (col[1], col[1], col[2], col[2])):     # nmse_Index, nmse_MRS | corr_Index, corr_MRS
        p = pos(k)
        axd[k].set_position([x, p.y0, axd[k].n_boxes * unit, p.height])


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(cfg), SIZE, 'figS4_kalman_filter_old_system', labels=LABELS,
                   stats_bars=cfg['stats_bars']['S4'], height_ratios=HEIGHT_RATIOS,
                   width_ratios=WIDTH_RATIOS, post_layout=spread)
