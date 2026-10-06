"""
Figures 2-4 from War Machine (BRAND rig, index DOF); the bend-vs-Freebie sensor check is notebooks/war_machine_sensors/. Same code as
the paper's figures: metrics.run_metrics for the online metrics, kf.train_kf for the offline KFs, utils.compose /
boxes / metric_panel / trial_window for the panels. analyze_* reads <master>/war_machine.pkl (scripts/war_machine.py; fig 2's
dropped frames also the improv logs, config war_machine.predictions)
and writes data/war_machine/; make_* draws from data/war_machine/ only.
"""
import tempfile
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

import frame_drops
import kf
import utils
import war_machine as wm
from metrics import run_metrics
from utils import COLORS, DATA, ONLINE_METRICS, compose

OUT = DATA / 'war_machine'
KF_CACHE = DATA / 'cache' / 'war_machine_offline_kf.csv'
INDEX_DOF = 1                  # kf.py's DOF_NAMES[1] = index; War Machine only drives index
LABEL = {'bend': 'Bend sensor', 'freebie': 'Freebie'}
STAGE_LABELS = [('frame_acq', 'Frame acquisition'), ('inference', 'Inference'), ('sender', 'Sender'), ('e2e', 'End-to-end')]
FPS_COLOR = {30: '#757272', 60: '#2a78d6'}
DATE = datetime(2026, 9, 30, tzinfo=timezone.utc)


def _color(group):
    return COLORS[group.replace('kf_', '')]


def _window(r):
    tr = wm.in_mode(r)
    return tr['start_time'].iloc[0], tr['stop_time'].iloc[-1]


def _both_sensors_hand(cfg):
    """Hand-control runs that recorded both sensors (every run on these days)."""
    return [rid for rid in wm.run_ids(cfg) if (r := wm.run(cfg, rid))['group'] in ('bend', 'freebie')
            and 'manip' in r['streams'] and 'vision' in r['streams']]


# ================================================================= online metrics (shared by figs 3 and 4)
def _metrics_nwb(r, path):
    """The run's in-mode trials + `cursor` (what the task scored: the sensor in hand control, the decoder in closed
    loop) as finger_angles -- the layout metrics.run_metrics reads."""
    t, x = r['streams']['cursor']
    t0 = t[0]
    nwb = NWBFile(session_description='war machine', identifier=path.stem, session_start_time=DATE)
    for c in ['target', 'target_radius', 'hold_time', 'success', 'in_mode', 'DecodeFeature']:
        nwb.add_trial_column(name=c, description=c)
    for _, row in wm.in_mode(r).iterrows():
        nwb.add_trial(start_time=row.start_time - t0, stop_time=row.stop_time - t0, target=np.array([row.target]),
                      target_radius=row.target_radius, hold_time=row.hold_time, success=row.success, in_mode=True,
                      DecodeFeature=0)
    nwb.create_processing_module('behavior', 'cursor').add(
        TimeSeries(name='finger_angles', data=x[:, None].astype('float64'), timestamps=t - t0, unit='a.u.'))
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwb)


def export_online_metrics(cfg):
    """data/war_machine/run_metrics.csv + trial_metrics.csv: metrics.run_metrics on every run (same columns as the
    paper's run_metrics.csv / run_trial_metrics.csv) plus `group` (bend, freebie, kf_bend, kf_freebie)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trial_rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in wm.run_ids(cfg):
            r = wm.run(cfg, rid)
            path = Path(tmp) / f'{rid}.nwb'
            _metrics_nwb(r, path)
            with NWBHDF5IO(str(path), 'r') as io:
                agg, tr = run_metrics(io.read(), closed=False, dofs=[0])
            rows.append({'run_id': rid, 'group': r['group'], 'mode': r['mode'], 'decoder': r['decoder'],
                         'trained_on': r['trained_on'], **agg})
            trial_rows.append(tr.assign(run_id=rid, group=r['group']))
    pd.DataFrame(rows).to_csv(OUT / 'run_metrics.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(OUT / 'trial_metrics.csv', index=False)


def _metric_panels(cfg, runs, trials, keys):
    labels = {'bend': 'Bend sensor', 'freebie': 'Freebie', 'kf_bend': 'Bend sensor', 'kf_freebie': 'Freebie'}
    groups = [(labels[k], runs[runs['group'] == k], _color(k)) for k in keys]
    show = cfg['war_machine']['all_trials']        # box + dots: every trial, or one mean per run; stats on run means either way
    return {m: partial(lambda ax, d, m: utils.metric_panel(ax, groups, m, stats_trials=trials, show_trials=show), m=m)
            for m in ONLINE_METRICS}


def _boxes_and_trace(ax, trials, series, ylabel, legend=True):
    """utils.target_trace with a line style per series, so a dashed trace stays visible on top of a solid one."""
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
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.01, 1, 0.15), mode='expand', ncol=len(series), borderaxespad=0)


def _save_example(cfg, key, streams, name):
    """Trials + the given streams over an example window (config war_machine.<key>) -> data/war_machine/<name>.*"""
    ex = cfg['war_machine'][key]
    r = wm.run(cfg, ex['run'])
    tr = wm.in_mode(r)
    tr = utils.trial_window(tr, ex['first_trial'], ex['n_trials'], tr['success'].to_numpy())
    t0, t1 = tr['start_time'].iloc[0], tr['stop_time'].iloc[-1]
    out = {'run': ex['run']}
    for s in streams:
        t, x = r['streams'][s]
        w = (t >= t0) & (t <= t1)
        out[f'{s}_t'], out[s] = t[w], x[w]
    np.savez(OUT / f'{name}.npz', **out)
    tr.to_csv(OUT / f'{name}_trials.csv', index=False)


def _load_example(name):
    return dict(np.load(OUT / f'{name}.npz')), pd.read_csv(OUT / f'{name}_trials.csv')


# ================================================================= fig 2: vision-pipeline latency
def _frame_logs(cfg, day):
    """Per-frame ids of the live single-camera improv sessions that day (same sessions as war_machine._vision_logs)."""
    folder = Path(cfg['war_machine']['predictions']).expanduser() / day.replace('-', '')
    return [frame_drops.session_log(s, 'cam2', 'front_left') for s in (sorted(folder.iterdir()) if folder.exists() else [])
            if (s / 'true_e2e_cam2_frame_nums.npy').exists() and (s / 'TIScapture_front_left.npy').exists()]


def analyze_fig2(cfg):
    """Per-frame stage times during each run (frames inside the run's task window), its camera rate, and its dropped
    frames (read from the improv logs, config war_machine.predictions; see frame_drops.py)."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, frames, drop_rows, drop_events, logs = [], {}, [], [], {}
    for rid in wm.run_ids(cfg):
        r = wm.run(cfg, rid)
        lat = r['latency']
        if not len(lat['e2e']):
            continue
        rows.append({'run_id': rid, 'fps': lat['fps'], 'frames': len(lat['e2e']),
                     **{k: lat[k].mean() for k, _ in STAGE_LABELS}, 'e2e_p95': np.percentile(lat['e2e'], 95)})
        frames[rid] = lat['e2e']
        if r['day'] not in logs:
            logs[r['day']] = _frame_logs(cfg, r['day'])
        ev, summary = frame_drops.drops(logs[r['day']], *r['window_unix'])
        drop_rows.append({'run_id': rid, 'fps': lat['fps'], **summary})
        drop_events.append(ev.assign(run_id=rid))
    pd.DataFrame(rows).to_csv(OUT / 'latency_runs.csv', index=False)
    np.savez(OUT / 'latency_e2e_frames.npz', **frames)
    drops = pd.DataFrame(drop_rows)
    drops.to_csv(OUT / 'dropped_frames_runs.csv', index=False)
    pd.concat([e for e in drop_events if len(e)] or drop_events[:1], ignore_index=True).to_csv(OUT / 'dropped_frames_events.csv', index=False)
    frame_drops.print_summary(drops, 'run_id')


def make_fig2(cfg):
    """config war_machine.fig2_merge_fps: true pools every run into one histogram with only the 60 fps budget."""
    runs = pd.read_csv(OUT / 'latency_runs.csv')
    frames = np.load(OUT / 'latency_e2e_frames.npz')
    merge = cfg['war_machine'].get('fig2_merge_fps', False)
    fps = [60] if merge else sorted(runs['fps'].unique())
    groups = [(f"All runs (n={len(runs)}", runs['run_id'], 60)] if merge else \
             [(f'{f} fps runs (n={(runs["fps"] == f).sum()}', runs.loc[runs['fps'] == f, 'run_id'], f) for f in fps]

    def hist(ax, d):
        for label, ids, f in groups:
            e = np.concatenate([frames[i] for i in ids])
            budget = 1000 / f
            ax.hist(e, bins=np.arange(12, 36.05, 0.1), color=FPS_COLOR[f], alpha=0.6,
                    label=f'{label}, {len(e):,} frames), median {np.median(e):.1f} ms')
            ax.axvline(budget, color=FPS_COLOR[f] if not merge else COLORS['budget'], ls='--', lw=2,
                       label=f'{f} FPS budget ({budget:.1f} ms): {np.mean(e > budget):.1%} over')
            utils.describe(ax, label.split(' (')[0], e, unit='frame', n_runs=len(ids),
                           n_sessions=ids.str[:10].nunique(), pct_over_budget=100 * np.mean(e > budget),
                           budget_ms=budget)
        ax.set_xlim(12, 36)
        ax.set_xlabel('End-to-end latency (ms)'); ax.set_ylabel('Frames'); ax.set_title('End-to-end latency distribution')
        ax.legend(loc='upper right')

    def stages(ax, d):
        utils.boxes(ax, [(label, runs[k]) for k, label in STAGE_LABELS], [COLORS[k] for k, _ in STAGE_LABELS],
                    ylabel='Processing time (ms)', title=f'Per stage: run means (n={len(runs)})', unit='run')
        for f in fps:
            ax.axhline(1000 / f, color=COLORS['budget'], ls='--' if f == 30 or merge else ':')
            ax.text(0.99, 1000 / f, f'{f} FPS budget ({1000 / f:.1f} ms)', transform=ax.get_yaxis_transform(),
                    ha='right', va='bottom', color=COLORS['budget'], fontweight='bold')
        ax.set_ylim(0, max(1000 / min(fps), runs['e2e'].max()) * 1.12)

    name = 'fig2_war_machine_merged' if merge else 'fig2_war_machine'
    fig = compose([['hist', 'stages']], {'hist': hist, 'stages': stages}, {}, (18, 6.5), name,
                  labels={'hist': 'A', 'stages': 'B'}, stats_bars=cfg['stats_bars'][2])
    make_fig2_drops(cfg)
    return fig


def make_fig2_drops(cfg):
    """figures/fig2_war_machine_dropped_frames: where in each run frames were dropped (data/war_machine/dropped_frames_*.csv)."""
    runs, ev = pd.read_csv(OUT / 'dropped_frames_runs.csv'), pd.read_csv(OUT / 'dropped_frames_events.csv')
    tot = lambda k: 100 * runs[f'{k}_dropped'].sum() / runs[f'{k}_frames'].sum()
    return frame_drops.make_figure(runs, ev, 'run_id', 'fig2_war_machine_dropped_frames',
                                   f'War Machine dropped frames (all runs: pipeline {tot("pipeline"):.2f}%, camera {tot("camera"):.2f}%)')


# ================================================================= fig 3: hand control
def sensor_correlations(cfg):
    """|r| bend sensor (`manipulandum`) vs Freebie (`vision_dofs`) per hand-control run, over its task window, bend
    interpolated onto the Freebie frame times, zero lag."""
    rows = []
    for rid in _both_sensors_hand(cfg):
        r = wm.run(cfg, rid)
        rows.append({'run_id': rid, 'group': r['group'],
                     'r': wm.corr(r['streams']['vision'], r['streams']['manip'], *_window(r))})
    return pd.DataFrame(rows)


def analyze_fig3(cfg):
    export_online_metrics(cfg)
    sensor_correlations(cfg).to_csv(OUT / 'sensor_corr.csv', index=False)
    _save_example(cfg, 'hand_control_trace', ['manip', 'vision'], 'fig3_example')


def make_fig3(cfg):
    runs, trials = pd.read_csv(OUT / 'run_metrics.csv'), pd.read_csv(OUT / 'trial_metrics.csv')
    corr = pd.read_csv(OUT / 'sensor_corr.csv')
    ex, ex_trials = _load_example('fig3_example')
    m = list(ONLINE_METRICS)
    layout = [['trace'] * 4 + ['corr'] * 2, ['trace'] * 4 + ['corr'] * 2,
              [k for k in m[:3] for _ in range(2)], [k for k in m[3:] for _ in range(2)]]

    def trace(ax, d):
        _boxes_and_trace(ax, ex_trials, [(ex['manip_t'], ex['manip'], 'Bend sensor (drove the cursor)', COLORS['bend'], 2.5, '-'),
                                         (ex['vision_t'], ex['vision'], 'Freebie (recorded alongside)', COLORS['freebie'], 2, '--')],
                         'Index position (a.u.)')
        ax.set_title('Hand control example', pad=34)

    def corrp(ax, d):
        utils.boxes(ax, [('Freebie vs\nbend sensor', corr['r'].abs())], [COLORS['freebie']], ylabel='|Pearson r| (per run)',
                    n_labels=[f'{len(corr)} runs'], unit='run')
        ax.set_ylim(0, 1.05)

    panels = {'trace': trace, 'corr': corrp, **_metric_panels(cfg, runs, trials, ['bend', 'freebie'])}
    return compose(layout, panels, {}, (16, 13), 'fig3_war_machine', labels={'trace': 'A', 'corr': 'B', 'success_rate': 'C'},
                   stats_bars=cfg['stats_bars'][3], height_ratios=[0.6, 0.6, 1, 1])


# ================================================================= fig 4: Kalman filter
def _kf_nwb(r, signal, path):
    """Neural (decoder channels, pre-binned) + one sensor at its own sample times, both cut to the run's task window,
    + the in-mode trials: what kf_config's Dataset2DictBlock reads."""
    lo, hi = _window(r)
    nt, nx = r['neural']; bt, bx = r['streams'][signal]
    nk, bk = (nt >= lo) & (nt <= hi), (bt >= lo) & (bt <= hi)
    beh = np.full((bk.sum(), 5), np.nan); beh[:, INDEX_DOF] = bx[bk]
    nwb = NWBFile(session_description='war machine KF', identifier=path.stem, session_start_time=DATE)
    nwb.add_trial_column(name='TargetPos', description='target, NaN off the index DOF')
    for _, row in wm.in_mode(r).iterrows():
        tp = np.full(5, np.nan); tp[INDEX_DOF] = row.target
        nwb.add_trial(start_time=row.start_time - lo, stop_time=row.stop_time - lo, TargetPos=tp)
    nwb.create_processing_module('ecephys', 'decoder channels').add(
        TimeSeries(name='EMG', data=nx[nk], timestamps=nt[nk] - lo, unit='uV'))
    nwb.create_processing_module('behavior', signal).add(
        TimeSeries(name='finger_angles', data=beh, timestamps=bt[bk] - lo, unit='a.u.'))
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwb)


def _train(cfg, r, signal, tmp):
    path = Path(tmp) / f"{r['run_id']}_{signal}.nwb"
    _kf_nwb(r, signal, path)
    chans = kf.GOOD_CHANS_EMG
    kf.GOOD_CHANS_EMG = list(range(r['neural'][1].shape[1]))   # the master holds only the decoder's channels
    try:
        ds = utils.open_nwb_path(cfg, path)
        m = kf.train_kf(ds, [INDEX_DOF]).loc['index_pos']
        ds.close()
    finally:
        kf.GOOD_CHANS_EMG = chans
    return m


def offline_kf(cfg):
    """Per hand-control run, two KFs on the same neural data and window: one fit to the bend sensor, one to Freebie
    (the paired 'bend (same window)' design of the paper's Fig 4). Cached in data/cache/ while the master is unchanged;
    cfg['retrain'] forces retraining."""
    built = wm.load(cfg)['meta']['built']
    if KF_CACHE.exists() and not cfg.get('retrain'):
        cached = pd.read_csv(KF_CACHE)
        if (cached['master_built'] == built).all() and set(cached['run_id']) == set(_both_sensors_hand(cfg)):
            return cached
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in _both_sensors_hand(cfg):
            print('training War Machine offline KFs:', rid, flush=True)
            r = wm.run(cfg, rid, neural=True)
            for signal, label in (('manip', 'bend'), ('vision', 'freebie')):
                m = _train(cfg, r, signal, tmp)
                rows.append({'run_id': rid, 'run_group': r['group'], 'n_trials': len(wm.in_mode(r)), 'signal': label,
                             'nmse_valid': m['nmse_valid'], 'correlation_valid': m['correlation_valid'],
                             'mse_valid': m['mse_valid']})
    KF_CACHE.parent.mkdir(exist_ok=True)
    out = pd.DataFrame(rows).assign(master_built=built)
    out.to_csv(KF_CACHE, index=False)
    return out


def kf_closed_loop_check(cfg):
    """Per closed-loop run: success by target height, how far the cursor reached while the hand flexed, drift while
    the hand was still, cursor speed per unit hand speed (flexion / extension); and per training run, how the labels
    the decoder learned compare with the hand (the bend sensor). The online cursor is 0.5 + the integrated KF
    velocity (beta_velocity 1 in rhodes_1d_universal.yaml), so a flexion-velocity error accumulates."""
    dt, rows = 0.032, []
    peak = lambda s, a, b: np.interp(np.arange(a, b, dt), *s).max()
    for rid in wm.run_ids(cfg):
        r = wm.run(cfg, rid)
        if not r['decoder'] or not r['trained_on']:
            continue
        t = wm.in_mode(r); s = r['streams']; hi = t[t['target'] > 0.75]
        g = np.concatenate([np.arange(a, b, dt) for a, b in zip(t.start_time, t.stop_time)])
        hand, cur = np.interp(g, *s['manip']), np.interp(g, *s['cursor'])
        vh, vc = np.gradient(hand, dt), np.gradient(cur, dt)
        still, f, e = (np.abs(vh) < 0.05) & (cur > 0.02) & (cur < 0.98), vh > 0.3, vh < -0.3
        by = t.groupby(pd.cut(t['target'], [0, .25, .75, 1.0001], labels=['low', 'mid', 'high']), observed=True)['success'].mean()
        tr_run = wm.run(cfg, r['trained_on']); lo, hi_t = _window(tr_run)
        lab, hnd = tr_run['streams']['behavior'], tr_run['streams']['manip']
        def speeds(sig):
            ts, v = sig; w = (ts >= lo) & (ts <= hi_t); ts, v = ts[w], v[w]
            sp = np.diff(v) / np.diff(ts); sp = sp[np.diff(ts) < 0.1]
            return np.percentile(sp[sp > 0], 95), np.percentile(-sp[sp < 0], 95), np.mean(v > 0.75)
        lf, le, lhi = speeds(lab); hf, he, hhi = speeds(hnd)
        rows.append({'run_id': rid, 'decoder': r['decoder'], 'labels': r['group'][3:], 'trained_on': r['trained_on'],
                     **{f'success_{k}_targets': v for k, v in by.items()},
                     'cursor_p99': np.percentile(np.interp(g, *s['cursor']), 99),
                     'high_trials_hand_peak': np.median([peak(s['manip'], a, b) for a, b in zip(hi.start_time, hi.stop_time)]),
                     'high_trials_cursor_peak': np.median([peak(s['cursor'], a, b) for a, b in zip(hi.start_time, hi.stop_time)]),
                     'drift_hand_still_per_s': np.median(vc[still]),
                     'gain_flexion': np.sum(vc[f] * vh[f]) / np.sum(vh[f] ** 2),
                     'gain_extension': np.sum(vc[e] * vh[e]) / np.sum(vh[e] ** 2),
                     'labels_time_above_0.75': lhi, 'hand_time_above_0.75': hhi,
                     'labels_flex_speed_p95': lf, 'labels_ext_speed_p95': le, 'hand_flex_speed_p95': hf, 'hand_ext_speed_p95': he})
    return pd.DataFrame(rows)


def analyze_fig4(cfg):
    export_online_metrics(cfg)
    off = offline_kf(cfg)
    off.drop(columns='master_built').to_csv(OUT / 'offline_kf.csv', index=False)
    kf_closed_loop_check(cfg).to_csv(OUT / 'kf_closed_loop_check.csv', index=False)
    for key in ('freebie', 'bend'):
        _save_example(cfg, f'{key}_decoder_trace', ['cursor'], f'fig4_example_{key}')


def make_fig4(cfg):
    wmc = cfg['war_machine']
    runs, trials = pd.read_csv(OUT / 'run_metrics.csv'), pd.read_csv(OUT / 'trial_metrics.csv')
    runs = runs[~runs['run_id'].isin(wmc['exclude_closed_loop'])]
    off = pd.read_csv(OUT / 'offline_kf.csv')
    off = off[off['n_trials'] >= wmc['offline_kf_min_trials']]
    m = list(ONLINE_METRICS)
    layout = [['freebie'] * 3 + ['bend'] * 3,
              [k for k in m[:3] for _ in range(2)], [k for k in m[3:] for _ in range(2)],
              ['nmse_valid'] * 2 + ['correlation_valid'] * 2 + ['.'] * 2]
    title = {'freebie': 'Freebie KF', 'bend': 'KF trained on bend sensor'}
    ylab = {'nmse_valid': 'Normalized MSE\n(MSE / target variance)', 'correlation_valid': 'Correlation (r)'}

    def decode(ax, d, key):
        ex, ex_trials = _load_example(f'fig4_example_{key}')
        _boxes_and_trace(ax, ex_trials, [(ex['cursor_t'], ex['cursor'], 'Decoded position', COLORS[key], 2.5, '-')],
                         'Decoded position (a.u.)', legend=False)
        ax.set_title(title[key])

    def offline(ax, d, metric):
        by = {s: g.set_index('run_id')[metric] for s, g in off.groupby('signal')}
        utils.boxes(ax, [(LABEL[s], by[s]) for s in ('bend', 'freebie')], [COLORS['bend'], COLORS['freebie']],
                    ylabel=ylab[metric], title='Offline KF, index', test='paired', log=metric == 'nmse_valid',
                    n_labels=[f'{len(by[s])} runs' for s in ('bend', 'freebie')], unit='run')

    panels = {'freebie': partial(decode, key='freebie'), 'bend': partial(decode, key='bend'),
              **_metric_panels(cfg, runs, trials, ['kf_bend', 'kf_freebie']),
              **{k: partial(offline, metric=k) for k in ('nmse_valid', 'correlation_valid')}}
    return compose(layout, panels, {}, (16, 19), 'fig4_war_machine',
                   labels={'freebie': 'A', 'bend': 'B', 'success_rate': 'C', 'nmse_valid': 'D'},
                   stats_bars=cfg['stats_bars'][4], height_ratios=[0.7, 1, 1, 1])
