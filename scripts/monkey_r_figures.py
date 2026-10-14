"""Figures 3-5, Monkey R (index DOF): 3 vision-pipeline latency, 4 hand control (bend sensor vs Freebie), 5 Kalman
filter (online decoders trained on each sensor, and offline KFs). Drawn from data/monkey_r/."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
from utils import COLORS, DATA, ONLINE_METRICS, compose

OUT = DATA / 'monkey_r'
C_GAP = 0.75                   # fig 4: gap between the C panels relative to the default
C_SHRINK = 0.8                 # fig 4: width of the online-metric (C) panels relative to the default 2 boxes x BOX_IN
LABEL = {'bend': 'Bend sensor', 'freebie': 'Freebie'}
STAGE_LABELS = [('frame_acq', 'Frame acquisition'), ('inference', 'Inference'), ('sender', 'Sender'), ('e2e', 'End-to-end')]
FPS_COLOR = {30: '#757272', 60: '#2a78d6'}


def _color(group):
    return COLORS[group.replace('kf_', '')]


def _metric_panels(runs, trials, keys):
    """One panel per online metric: every trial drawn, tested on run means."""
    groups = [(LABEL[k.replace('kf_', '')], runs[runs['group'] == k], _color(k)) for k in keys]
    return {m: partial(lambda ax, d, m: utils.metric_panel(ax, groups, m, stats_trials=trials), m=m) for m in ONLINE_METRICS}


def _boxes_and_trace(ax, trials, series, ylabel, legend=False):
    """Target windows (grey hit / red miss) under traces with a line style each, so a dashed trace stays visible on top
    of a solid one. legend: False or 'lower right' (inside the axes, on a white backing)."""
    t0, t1 = trials['start_time'].iloc[0], trials['stop_time'].iloc[-1]
    for _, tr in trials.iterrows():
        rad = tr['target_radius']
        ax.add_patch(plt.Rectangle((tr['start_time'] - t0, tr['target'] - rad), tr['stop_time'] - tr['start_time'], 2 * rad,
                                   facecolor=COLORS['target_hit'] if tr['success'] else COLORS['target_miss'],
                                   edgecolor='#6f6f6f', ls='--', lw=1, alpha=0.7, zorder=1))
    for t, y, label, color, lw, ls in series:
        ax.plot(t - t0, y, color=color, lw=lw, ls=ls, label=label, zorder=3)
    ax.set_xlim(0, t1 - t0)
    ax.set_ylim(-0.1, 1.1)
    ax.set_ylabel(ylabel)
    ax.set_xlabel('Time (s)')
    if legend:
        ax.legend(loc='lower right', ncol=1, frameon=True, facecolor='white', edgecolor='none', framealpha=0.9,
                  borderaxespad=0.3, handlelength=1.6)


def _load_example(name):
    return dict(np.load(OUT / f'{name}.npz')), pd.read_csv(OUT / f'{name}_trials.csv')


# ================================================================= fig 3: vision-pipeline latency
def make_fig3(cfg):
    """A: end-to-end latency of every frame, one histogram per camera rate; B: each run's mean time per stage."""
    runs = pd.read_csv(OUT / 'latency_runs.csv')
    frames = np.load(OUT / 'latency_e2e_frames.npz')
    drops = pd.read_csv(OUT / 'latency_drops.csv')
    fps = sorted(runs['fps'].unique())
    groups = [(f'{f} fps runs', runs.loc[runs['fps'] == f, 'run_id'], f) for f in fps]

    def hist(ax, d):
        # 60 fps first, so the grey 30 fps histogram is drawn on top; the legend only names the two colours
        for z, (label, ids, f) in enumerate(sorted(groups, key=lambda g: -g[2])):
            e = np.concatenate([frames[i] for i in ids])
            ax.hist(e, bins=np.arange(15, 20.025, 0.05), color=FPS_COLOR[f], alpha=0.75, zorder=2 + z,
                    label=label.replace(' runs', ''))
            dr = drops[drops['run_id'].isin(ids)]
            utils.describe(ax, label, e, unit='frame', n_runs=len(ids), n_sessions=ids.str[:10].nunique(),
                           pct_over_budget=100 * np.mean(e > 1000 / f), budget_ms=1000 / f, median_ms=np.median(e),
                           camera_frames=dr['camera_frames'].sum(), processor_skipped=dr['processor_skipped'].sum(),
                           pct_processor_skipped=100 * dr['processor_skipped'].sum() / dr['camera_frames'].sum(),
                           processor_bursts=dr['processor_bursts'].sum(), processor_longest_burst=dr['processor_longest_burst'].max(),
                           camera_missing=dr['camera_missing'].sum(),
                           pct_camera_missing=100 * dr['camera_missing'].sum() / dr['camera_frames'].sum(),
                           camera_bursts=dr['camera_bursts'].sum(), camera_longest_burst=dr['camera_longest_burst'].max())
        ax.set_xlim(15, 20)
        ax.set_xticks(np.arange(15, 20.1, 1)); ax.set_xticks(np.arange(15, 20.1, 0.25), minor=True)
        ax.set_xlabel('End-to-end latency (ms)'); ax.set_ylabel('Frames')
        utils.power_offset(ax, 3)             # diagonal 10^3 at the top of the y axis
        ax.legend(loc='upper right')

    def stages(ax, d):
        utils.boxes(ax, [(label, runs[k]) for k, label in STAGE_LABELS], [COLORS[k] for k, _ in STAGE_LABELS],
                    ylabel='Processing time (ms)', unit='run')
        for f in fps:                          # frame budgets (1,000 / fps): 33.3 ms at 30 fps, 16.7 ms at 60 fps
            ax.axhline(1000 / f, color=COLORS['budget'], ls='--')
        ax.set_ylim(0, max(1000 / min(fps), runs['e2e'].max()) * 1.12)

    return compose([['hist', 'stages']], {'hist': hist, 'stages': stages}, {}, (18, 6.5), 'fig3_latency',
                   labels={'hist': 'A', 'stages': 'B'}, stats_bars=cfg['stats_bars'][3])


# ================================================================= fig 4: hand control
def make_fig4(cfg):
    runs, trials = pd.read_csv(OUT / 'run_metrics.csv'), pd.read_csv(OUT / 'trial_metrics.csv')
    corr = pd.read_csv(OUT / 'sensor_corr.csv')
    ex, ex_trials = _load_example('fig4_example')
    m = list(ONLINE_METRICS)
    layout = [['trace'] * 3 + ['corr'] * 3,
              [k for k in m[:3] for _ in range(2)], [k for k in m[3:] for _ in range(2)]]

    def trace(ax, d):
        _boxes_and_trace(ax, ex_trials, [(ex['manip_t'], ex['manip'], 'Bend sensor (drove cursor)', COLORS['bend'], 2.5, '-'),
                                         (ex['vision_t'], ex['vision'], 'Freebie (recorded)', COLORS['freebie'], 2, '--')],
                         'Index position (a.u.)', legend='lower right')

    def corrp(ax, d):
        utils.boxes(ax, [('Freebie vs\nbend sensor', corr['r'].abs())], [COLORS['freebie']], ylabel='|Pearson r| (per run)', unit='run')
        ax.set_ylim(0, 1.05)

    def widths(fig, axd):
        # C narrower (C_SHRINK) and closer together (C_GAP); B widened to end where C ends, its box as wide as one of C's
        W = fig.get_figwidth()
        c = [[axd[k] for k in m[:3]], [axd[k] for k in m[3:]]]
        x0, gap = c[0][0].get_position().x0, (c[0][1].get_position().x0 - c[0][0].get_position().x1) * C_GAP
        w = c[0][0].get_position().width * C_SHRINK
        for row in c:
            for i, ax in enumerate(row):
                p = ax.get_position()
                ax.set_position([x0 + i * (w + gap), p.y0, w, p.height])
        right = x0 + 2 * (w + gap) + w
        b = axd['corr'].get_position()
        axd['corr'].set_position([b.x0, b.y0, right - b.x0, b.height])
        lo, hi = c[0][0].get_xlim()
        half = (right - b.x0) * W / (w * W / (hi - lo)) / 2          # B width / inches per x unit in C / 2
        axd['corr'].set_xlim(1 - half, 1 + half)

    panels = {'trace': trace, 'corr': corrp, **_metric_panels(runs, trials, ['bend', 'freebie'])}
    return compose(layout, panels, {}, (12, 10.6 * 2.95 / 3.95), 'fig4_hand_control',
                   labels={'trace': 'A', 'corr': 'B', 'success_rate': 'C'},
                   stats_bars=cfg['stats_bars'][4], height_ratios=[0.95, 1, 1], post_layout=widths)


# ================================================================= fig 5: Kalman filter
def make_fig5(cfg):
    runs, trials = pd.read_csv(OUT / 'run_metrics.csv'), pd.read_csv(OUT / 'trial_metrics.csv')
    off = pd.read_csv(OUT / 'offline_kf.csv')
    m = list(ONLINE_METRICS)
    layout = [['freebie'] * 3 + ['bend'] * 3,
              [k for k in m[:3] for _ in range(2)], [k for k in m[3:] for _ in range(2)],
              ['nmse_valid'] * 2 + ['correlation_valid'] * 2 + ['.'] * 2]
    ylab = {'nmse_valid': 'Normalized MSE', 'correlation_valid': 'Correlation (r)'}

    def decode(ax, d, key):
        ex, ex_trials = _load_example(f'fig5_example_{key}')
        _boxes_and_trace(ax, ex_trials, [(ex['cursor_t'], ex['cursor'], 'Decoded position', COLORS[key], 2.5, '-')],
                         'Position (a.u.)')

    def offline(ax, d, metric):
        by = {s: g.set_index('run_id')[metric] for s, g in off.groupby('signal')}
        utils.boxes(ax, [(LABEL[s], by[s]) for s in ('bend', 'freebie')], [COLORS['bend'], COLORS['freebie']],
                    ylabel=ylab[metric], title='Offline KF, index', test='paired', log=metric == 'nmse_valid',
                    unit='run')

    panels = {'freebie': partial(decode, key='freebie'), 'bend': partial(decode, key='bend'),
              **_metric_panels(runs, trials, ['kf_bend', 'kf_freebie']),
              **{k: partial(offline, metric=k) for k in ('nmse_valid', 'correlation_valid')}}
    return compose(layout, panels, {}, (12, 10.6), 'fig5_kalman_filter',
                   labels={'freebie': 'A', 'bend': 'B', 'success_rate': 'C', 'nmse_valid': 'D'},
                   stats_bars=cfg['stats_bars'][5], height_ratios=[0.95, 1, 1, 1])
