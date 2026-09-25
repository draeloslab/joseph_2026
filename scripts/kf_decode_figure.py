"""Figure 4: Kalman filter decoding from EMG, trained on Freebie signals vs bend sensors."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import freebie
from metrics import export_run_metrics
from utils import (COLORS, DATA, ONLINE_METRICS, boxes, compose, load_master, metric_panel, open_nwb, split_groups,
                   trial_window, trials_frame)

DECODERS = {'freebie': 'Freebie KF', 'bend': 'KF trained on bend sensor'}
# (DOF, signal label, freebie_kf.csv conditions, output); 'bend (same window)' is trained on the
# same stretch of data as the Freebie KF of that session
OFFLINE = [('Index', 'Bend sensor', ['bend (same window)'], 'index_pos'),
           ('Index', 'Freebie index angle', ['index_angle', 'index_angle+mrs_x'], 'index_pos'),
           ('Index', 'Freebie PIP y', ['pip_y', 'pip_y+mrs_x'], 'index_pos'),
           ('MRS', 'Bend sensor', ['bend (same window)'], 'ring_pos'),
           ('MRS', 'Freebie MRS x', ['index_angle+mrs_x'], 'ring_pos')]
SIGNAL_COLOR = {'Bend sensor': 'bend', 'Freebie index angle': 'index_angle', 'Freebie PIP y': 'pip_y', 'Freebie MRS x': 'mrs_x'}

OFFLINE_METRICS = {'nmse_valid': 'Normalized MSE\n(MSE / target variance)', 'correlation_valid': 'Correlation (r)'}

# A = Freebie KF, B = bend KF (index + MRS stacked in one axis each), C = online metrics (2 per row), D = offline KF
# (2x2) under A and B. Columns are in box units: D's index panels have 3 boxes and its MRS panels 2, so a 0.5-box
# column lets A and B split that 5-box block evenly.
m = list(ONLINE_METRICS)
_D = [f'{mt}_{dof}' for mt in OFFLINE_METRICS for dof in ('Index', 'MRS')]
LAYOUT = [['freebie', 'bend', 'bend', m[0], m[1]],
          [_D[0], _D[0], _D[1], m[2], m[3]],
          [_D[2], _D[2], _D[3], m[4], m[5]]]
WIDTH_RATIOS = [2.5, 0.5, 2, 3, 3]
SIZE = (22, 17)
HEIGHT_RATIOS = [1, 1, 1]
TRACE_DOFS = [(1, 'Index', 'index_angle', 1.45), (3, 'MRS', 'mrs_x', 0.0)]   # (dof, label, freebie colour, y offset)


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    export_run_metrics(cfg)

    kf = freebie.offline_kf_rows(cfg)
    rows = []
    for dof, signal, conditions, output in OFFLINE:
        sel = kf[kf['condition'].isin(conditions) & (kf['output'] == output)]
        rows.append(sel.assign(dof=dof, signal=signal)[['session_id', 'n_dof', 'dof', 'signal', 'mse_valid',
                                                        *OFFLINE_METRICS]])
    pd.concat(rows).to_csv(DATA / 'offline_kf.csv', index=False)

    # example closed-loop trials: the decoder's output on each active DOF against the target windows
    for key in DECODERS:
        ex = cfg['examples'][f'{key}_decoder_trace']
        ds = open_nwb(cfg, ex['run'])
        dec = ds.dataset.processing['behavior']['decode']
        t, data = dec.timestamps[:], dec.data[:]
        dofs = load_master(cfg)['runs'][ex['run']]['dofs']
        trials = trials_frame(ds.dataset, dofs)
        feature = int(np.median(trials['DecodeFeature']))             # decoder output columns 0-4 TCFR, 5-9 SBP
        trials = trial_window(trials, ex['first_trial'], ex['n_trials'], ~trials['blank'] & trials['decoded'])
        win = (t >= trials['start_time'].iloc[0]) & (t <= trials['stop_time'].iloc[-1])
        np.savez(DATA / f'fig4_example_{key}_trace.npz', t=t[win], run=ex['run'],
                 **{f'decode_{d}': data[win, d + 5 * feature] for d in dofs})
        trials[['start_time', 'stop_time', 'target_radius', 'success',
                *[f'target_{d}' for d in dofs]]].to_csv(DATA / f'fig4_example_{key}_trials.csv', index=False)
        ds.close()


def load(cfg):
    opts = cfg['online_metrics']
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    closed = runs[runs['mode'] == 'closed_loop']
    return {**{key: (dict(np.load(DATA / f'fig4_example_{key}_trace.npz')),
                     pd.read_csv(DATA / f'fig4_example_{key}_trials.csv')) for key in DECODERS},
            'offline': pd.read_csv(DATA / 'offline_kf.csv'),
            'groups': split_groups(closed, opts['split_feature'], opts['split_dof']),
            'trials': pd.read_csv(DATA / 'run_trial_metrics.csv') if opts['all_trials'] else None}


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
    ax.set_ylabel('Decoded position (a.u.)')
    ax.set_title(DECODERS[key])


def panel_offline(ax, d, dof, metric):
    off = d['offline'][d['offline']['dof'] == dof]
    signals = [s for dof_, s, *_ in OFFLINE if dof_ == dof]
    by_signal = {s: g.set_index('session_id')[metric] for s, g in off.groupby('signal')}
    # normalized MSE is a variance ratio: near-zero-variance validation targets in a handful of
    # sessions push it into the hundreds, so it needs a log axis to stay readable
    boxes(ax, [(s.replace('Freebie ', 'Freebie\n'), by_signal[s]) for s in signals],
          [COLORS[SIGNAL_COLOR[s]] for s in signals], ylabel=OFFLINE_METRICS[metric],
          title=f'Offline KF, {dof} DOF', test='paired', log=metric == 'nmse_valid')


def panel_metric(ax, d, metric):
    metric_panel(ax, d['groups'], metric, trials=d['trials'])


PANELS = {**{k: partial(panel_decode, key=k) for k in DECODERS},
          **{f'{m}_{dof}': partial(panel_offline, dof=dof, metric=m) for m in OFFLINE_METRICS for dof in ('Index', 'MRS')},
          **{m: partial(panel_metric, metric=m) for m in ONLINE_METRICS}}
LABELS = {'freebie': 'A', 'bend': 'B', 'success_rate': 'C', 'nmse_valid_Index': 'D'}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(cfg), SIZE, 'fig4_kf_decode', labels=LABELS,
                   stats_bars=cfg['stats_bars'][4], height_ratios=HEIGHT_RATIOS,
                   width_ratios=WIDTH_RATIOS)
