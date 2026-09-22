"""Figure 3: Freebie tracking vs bend sensors during hand control."""
from functools import partial

import numpy as np
import pandas as pd

import freebie
from metrics import export_run_metrics
from utils import (COLORS, DATA, ONLINE_METRICS, boxes, compose, load_master, metric_panel, open_nwb, split_groups,
                   target_trace, trial_window, trials_frame)

SIGNALS = {1: [('index_angle', 'Freebie index angle'), ('pip_y', 'Freebie PIP y')], 3: [('mrs_x', 'Freebie MRS x')]}

LAYOUT = [['trace_1'] * 4,
          ['trace_3'] * 4,
          ['corr', 'success_rate', 'bitrate_bits_per_s', 'targets_per_s'],
          ['.', 'time_to_target_s', 'orbiting_time_s', 'path_efficiency']]
SIZE = (20, 18)


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    export_run_metrics(cfg)
    master = load_master(cfg)
    corr = freebie.correlations(master)
    corr.dropna(how='all', subset=corr.columns[1:]).to_csv(DATA / 'freebie_bend_corr.csv', index=False)

    # example trace: bend sensor (1 kHz) vs the Freebie signals at their own frame times, on the xPC clock
    ex = cfg['examples']['hand_control_trace']
    s = master['sessions'][ex['session']]
    run = master['runs'][s['run_id']]
    ds = open_nwb(cfg, s['run_id'])
    fa = ds.dataset.processing['behavior']['finger_angles']
    t, bend = fa.timestamps[:], fa.data[:]
    cam = freebie.xpc_signals(s, t)
    trials = trials_frame(ds.dataset, run['dofs'])
    lo, hi = max(c[0][0] for c in cam.values()), min(c[0][-1] for c in cam.values())
    keep = ~trials['blank'] & trials['success'] & (trials['start_time'] >= lo) & (trials['stop_time'] <= hi)
    trials = trial_window(trials, ex['first_trial'], ex['n_trials'], keep)
    t0, t1 = trials['start_time'].iloc[0], trials['stop_time'].iloc[-1]
    win = (t >= t0) & (t <= t1)
    out = {'t': t[win], 'session': ex['session'], **{f'bend_{d}': bend[win, d] for d in run['dofs']}}
    for name, (tc, v) in cam.items():
        w = (tc >= t0) & (tc <= t1)
        out[f'{name}_t'], out[name] = tc[w], v[w]
    np.savez(DATA / 'fig3_example_trace.npz', **out)
    trials[['start_time', 'stop_time', 'target_radius', 'success',
            *[f'target_{d}' for d in run['dofs']]]].to_csv(DATA / 'fig3_example_trials.csv', index=False)
    ds.close()


def load(cfg):
    opts = cfg['online_metrics']
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    hand = runs[runs['mode'] == 'hand_control']
    return {'trace': dict(np.load(DATA / 'fig3_example_trace.npz')),
            'ex_trials': pd.read_csv(DATA / 'fig3_example_trials.csv'),
            'corr': pd.read_csv(DATA / 'freebie_bend_corr.csv'),
            'groups': split_groups(hand, opts['split_feature'], opts['split_dof']),
            'trials': pd.read_csv(DATA / 'run_trial_metrics.csv') if opts['all_trials'] else None}


# ---------------------------------------------------------------- panels
def panel_trace(ax, d, dof):
    tr = d['trace']
    lines = [(tr['t'], tr[f'bend_{dof}'], 'Bend sensor', COLORS['bend'], 2.5)]
    lines += [(tr[f'{key}_t'], tr[key], label, COLORS[key], 2) for key, label in SIGNALS[dof]]
    target_trace(ax, d['ex_trials'], f'target_{dof}', lines, f"{'Index' if dof == 1 else 'MRS'}\nposition (a.u.)")
    if dof == 1:
        ax.set_title(f"Hand control example ({tr['session']})")


def panel_corr(ax, d):
    c = d['corr'].set_index('session_id').abs()
    boxes(ax, [('Index angle\nvs index', c['corr_index_angle']), ('PIP y\nvs index', c['corr_pip_y']),
               ('MRS x\nvs MRS', c['corr_mrs_x'])],
          [COLORS['index_angle'], COLORS['pip_y'], COLORS['mrs_x']],
          ylabel='|Pearson r| (per session)', title='Freebie vs bend sensor', test='paired')
    ax.set_ylim(bottom=0)


def panel_metric(ax, d, metric):
    metric_panel(ax, d['groups'], metric, trials=d['trials'])


PANELS = {'trace_1': partial(panel_trace, dof=1), 'trace_3': partial(panel_trace, dof=3), 'corr': panel_corr,
          **{m: partial(panel_metric, metric=m) for m in ONLINE_METRICS}}
LABELS = {'trace_1': 'A', 'corr': 'B', 'success_rate': 'C'}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(cfg), SIZE, 'fig3_hand_control', labels=LABELS,
                   stats_bars=cfg['stats_bars'][3])
