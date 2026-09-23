"""Figure 4: Kalman filter decoding from EMG, trained on Freebie signals vs bend sensors."""
from functools import partial

import numpy as np
import pandas as pd

import freebie
from metrics import export_run_metrics
from utils import (COLORS, DATA, ONLINE_METRICS, boxes, compose, load_master, metric_panel, open_nwb, split_groups,
                   target_trace, trial_window, trials_frame)

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

LAYOUT = [['freebie_1'] * 3 + ['nmse_valid_Index', 'nmse_valid_MRS', 'success_rate', 'bitrate_bits_per_s'],
          ['freebie_3'] * 3 + ['correlation_valid_Index', 'correlation_valid_MRS', 'targets_per_s', 'time_to_target_s'],
          ['bend_1'] * 3 + ['.', '.', 'orbiting_time_s', 'path_efficiency'],
          ['bend_3'] * 3 + ['.'] * 4]
SIZE = (30, 14)
HEIGHT_RATIOS = [1, 1, 1, 1]


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
def panel_decode(ax, d, key, dof):
    tr, trials = d[key]
    color = COLORS['freebie'] if key == 'freebie' else COLORS['bend']
    target_trace(ax, trials, f'target_{dof}', [(tr['t'], tr[f'decode_{dof}'], 'Decoder output', color, 2.5)],
                 f"{'Index' if dof == 1 else 'MRS'}\nposition (a.u.)")
    if dof == 1:
        ax.set_title(f"{DECODERS[key]}", pad=34)   # clear the legend row above the axes


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


PANELS = {**{f'{k}_{dof}': partial(panel_decode, key=k, dof=dof) for k in DECODERS for dof in (1, 3)},
          **{f'{m}_{dof}': partial(panel_offline, dof=dof, metric=m) for m in OFFLINE_METRICS for dof in ('Index', 'MRS')},
          **{m: partial(panel_metric, metric=m) for m in ONLINE_METRICS}}
LABELS = {'freebie_1': 'A', 'bend_1': 'B', 'nmse_valid_Index': 'C', 'success_rate': 'D'}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(cfg), SIZE, 'fig4_kf_decode', labels=LABELS,
                   stats_bars=cfg['stats_bars'][4], height_ratios=HEIGHT_RATIOS)
