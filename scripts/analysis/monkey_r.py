"""
Monkey R (BRAND rig, index DOF), from raw/monkey_r/: vision-pipeline latency and dropped frames (fig 3), online metrics,
bend-vs-Freebie correlations and example traces (fig 4), offline Kalman filters and decoder examples (fig 5), and the
online metrics of the hand-control runs without the manipulandum (fig S5).

raw/monkey_r/runs.csv       one row per run: figure ('3-5' or 'S5'), group (bend / freebie / kf_bend / kf_freebie), mode,
                            decoder and the hand-control run it was trained on, camera rate, task window (Unix s), n_logs
raw/monkey_r/trials/        each run's trials: start / stop time, success, screen on, target, radius, hold time, mode
raw/monkey_r/runs/          each run's signals ({manip, vision, cursor}_{t,x}: bend sensor, Freebie, the signal that drove
                            the cursor), per-frame stage times (latency_*), neural_{t,x} (8 decoder channels, 4 ms mean |x|,
                            hand-control runs only), and per vision session the camera frame starts and processor frame
                            numbers inside the task window (log<k>_camera_starts, log<k>_processor_frames)
"""
import tempfile
from datetime import datetime, timezone
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from utils import DATA, RAW
from . import kf
from .common import csv_trials, open_nwb_path, shortest_run_of_successes, trial_window
from .metrics import run_metrics

SRC = RAW / 'monkey_r'
OUT = DATA / 'monkey_r'
INDEX_DOF = 1                  # kf.py's DOF_NAMES[1] = index; Monkey R only drives index
STAGES = ['frame_acq', 'inference', 'sender', 'e2e']
DATE = datetime(2026, 9, 30, tzinfo=timezone.utc)


# ---------------------------------------------------------------- the package
@cache
def table():
    return pd.read_csv(SRC / 'runs.csv', dtype=str, keep_default_na=False).set_index('run_id')


def run_ids(figure):
    return [rid for rid, row in table().iterrows() if row['figure'] == figure]


def run(run_id, neural=False):
    """One run: trials, signals {name: (t, x)}, stage times, task window, and (neural=True) neural = (t, x[n, channels])."""
    row, z = table().loc[run_id], np.load(SRC / 'runs' / f'{run_id}.npz')
    r = {'run_id': run_id, 'trials': csv_trials(SRC / 'trials' / f'{run_id}.csv'), 'mode': row['mode'], 'group': row['group'],
         'decoder': row['decoder'] or None, 'trained_on': row['trained_on'] or None,
         'streams': {n: (z[f'{n}_t'], z[f'{n}_x']) for n in ('manip', 'vision', 'cursor') if f'{n}_t' in z},
         'window_unix': (float(row['window_start']), float(row['window_end'])),
         'latency': {k: z[f'latency_{k}'] for k in STAGES if f'latency_{k}' in z}, 'fps': int(row['fps']) if row['fps'] else None}
    if neural:
        r['neural'] = (z['neural_t'], z['neural_x'])
    return r


def in_mode(r):
    """The run's trials that count: screen on and in the run's own mode."""
    tr = r['trials']
    return tr[tr['screen_on'] & (tr['mode'] == r['mode'])].reset_index(drop=True)


def _window(r):
    tr = in_mode(r)
    return tr['start_time'].iloc[0], tr['stop_time'].iloc[-1]


def corr(a, b, lo, hi):
    """Pearson r of stream b interpolated onto stream a's sample times, inside [lo, hi] and both streams' spans."""
    ta, xa = a
    tb, xb = b
    lo, hi = max(ta[0], tb[0], lo), min(ta[-1], tb[-1], hi)
    ok = (ta >= lo) & (ta <= hi)
    return np.corrcoef(xa[ok], np.interp(ta[ok], tb, xb))[0, 1]


def _both_sensors_hand():
    """Hand-control runs that recorded both sensors."""
    return [rid for rid in run_ids('3-5') if (r := run(rid))['group'] in ('bend', 'freebie')
            and 'manip' in r['streams'] and 'vision' in r['streams']]


# ---------------------------------------------------------------- online metrics
def _metrics_nwb(r, path):
    """The run's in-mode trials + `cursor` (what the task scored: the sensor in hand control, the decoder in closed
    loop) as finger_angles: the layout metrics.run_metrics reads."""
    t, x = r['streams']['cursor']
    t0 = t[0]
    nwb = NWBFile(session_description='Monkey R run', identifier=path.stem, session_start_time=DATE)
    for c in ['target', 'target_radius', 'hold_time', 'success', 'in_mode', 'DecodeFeature']:
        nwb.add_trial_column(name=c, description=c)
    for _, row in in_mode(r).iterrows():
        nwb.add_trial(start_time=row.start_time - t0, stop_time=row.stop_time - t0, target=np.array([row.target]),
                      target_radius=row.target_radius, hold_time=row.hold_time, success=row.success, in_mode=True,
                      DecodeFeature=0)
    nwb.create_processing_module('behavior', 'cursor').add(
        TimeSeries(name='finger_angles', data=x[:, None].astype('float64'), timestamps=t - t0, unit='a.u.'))
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwb)


def export_online_metrics(run_ids_, prefix=''):
    """data/monkey_r/<prefix>run_metrics.csv + <prefix>trial_metrics.csv: metrics.run_metrics on every run."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, trial_rows = [], []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in run_ids_:
            r = run(rid)
            path = Path(tmp) / f'{rid}.nwb'
            _metrics_nwb(r, path)
            with NWBHDF5IO(str(path), 'r') as io:
                agg, tr = run_metrics(io.read(), closed=False, dofs=[0])
            rows.append({'run_id': rid, 'group': r['group'], 'mode': r['mode'], 'decoder': r['decoder'],
                         'trained_on': r['trained_on'], **agg})
            trial_rows.append(tr.assign(run_id=rid, group=r['group']))
    pd.DataFrame(rows).to_csv(OUT / f'{prefix}run_metrics.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(OUT / f'{prefix}trial_metrics.csv', index=False)


@cache
def export_fig345_metrics():
    export_online_metrics(run_ids('3-5'))


def _save_example(ex, streams, name):
    """Trials + the given signals over an example window of a run -> data/monkey_r/<name>.npz and <name>_trials.csv."""
    r = run(ex['run'])
    tr = in_mode(r)
    if ex.get('select') == 'shortest_success':
        tr = shortest_run_of_successes(tr, ex['n_trials'])
    else:
        tr = trial_window(tr, None, ex['n_trials'], tr['success'].to_numpy())
    t0, t1 = tr['start_time'].iloc[0], tr['stop_time'].iloc[-1]
    out = {'run': ex['run']}
    for s in streams:
        t, x = r['streams'][s]
        w = (t >= t0) & (t <= t1)
        out[f'{s}_t'], out[s] = t[w], x[w]
    np.savez(OUT / f'{name}.npz', **out)
    tr.to_csv(OUT / f'{name}_trials.csv', index=False)


# ---------------------------------------------------------------- fig 3: vision-pipeline latency
def frame_drops():
    """data/monkey_r/latency_drops.csv: per run and vision session, the camera frames inside the task window, the frames
    the camera failed to deliver (a gap > 1.5 frame periods between frame starts) and the frames the processor skipped
    (gaps in its frame numbers), each with the number of bursts and the longest one (frames)."""
    rows = []
    for rid, row in table().iterrows():
        if row['figure'] != '3-5':
            continue
        z, fps = np.load(SRC / 'runs' / f'{rid}.npz'), int(row['fps']) if row['fps'] else None
        for k in range(int(row['n_logs'])):
            start = z[f'log{k}_camera_starts']
            dt = np.diff(start)
            late = dt > 1.5 * np.median(dt)
            missing = np.maximum(np.round(dt / np.median(dt)) - 1, 0)[late]
            gaps = np.diff(z[f'log{k}_processor_frames'])
            skipped = gaps[gaps > 1] - 1
            rows.append({'run_id': rid, 'fps': fps, 'camera_frames': len(start),
                         'camera_missing': int(missing.sum()), 'camera_bursts': int(late.sum()),
                         'camera_longest_burst': int(missing.max()) if len(missing) else 0,
                         'processor_skipped': int(skipped.sum()), 'processor_bursts': len(skipped),
                         'processor_longest_burst': int(skipped.max()) if len(skipped) else 0})
    pd.DataFrame(rows).to_csv(OUT / 'latency_drops.csv', index=False)


def analyze_fig3(cfg):
    """Per-frame stage times during each run's task window, its camera rate, and the dropped frames."""
    OUT.mkdir(parents=True, exist_ok=True)
    rows, frames = [], {}
    for rid in run_ids('3-5'):
        r = run(rid)
        lat = r['latency']
        if not len(lat['e2e']):
            continue
        rows.append({'run_id': rid, 'fps': r['fps'], 'frames': len(lat['e2e']), **{k: lat[k].mean() for k in STAGES},
                     'e2e_p95': np.percentile(lat['e2e'], 95)})
        frames[rid] = lat['e2e']
    pd.DataFrame(rows).to_csv(OUT / 'latency_runs.csv', index=False)
    np.savez(OUT / 'latency_e2e_frames.npz', **frames)
    frame_drops()


# ---------------------------------------------------------------- fig 4: hand control
def analyze_fig4(cfg):
    """Online metrics, |r| between the bend sensor and Freebie per hand-control run, and an example trace."""
    OUT.mkdir(parents=True, exist_ok=True)
    export_fig345_metrics()
    pd.DataFrame([{'run_id': rid, 'group': r['group'],
                   'r': corr(r['streams']['vision'], r['streams']['manip'], *_window(r))}
                  for rid in _both_sensors_hand() if (r := run(rid))]).to_csv(OUT / 'sensor_corr.csv', index=False)
    _save_example(cfg['analysis']['examples']['fig4'], ['manip', 'vision'], 'fig4_example')


# ---------------------------------------------------------------- fig 5: Kalman filter
def _kf_nwb(r, signal, path):
    """Neural (decoder channels, pre-binned) + one sensor at its own sample times, both cut to the run's task window, + the
    in-mode trials: what kf.py's Dataset2DictBlock reads."""
    lo, hi = _window(r)
    nt, nx = r['neural']
    bt, bx = r['streams'][signal]
    nk, bk = (nt >= lo) & (nt <= hi), (bt >= lo) & (bt <= hi)
    beh = np.full((bk.sum(), 5), np.nan)
    beh[:, INDEX_DOF] = bx[bk]
    nwb = NWBFile(session_description='Monkey R KF', identifier=path.stem, session_start_time=DATE)
    nwb.add_trial_column(name='TargetPos', description='target, NaN off the index DOF')
    for _, row in in_mode(r).iterrows():
        if row.start_time < lo or row.stop_time > hi:
            continue
        tp = np.full(5, np.nan)
        tp[INDEX_DOF] = row.target
        nwb.add_trial(start_time=row.start_time - lo, stop_time=row.stop_time - lo, TargetPos=tp)
    nwb.create_processing_module('ecephys', 'decoder channels').add(
        TimeSeries(name='EMG', data=nx[nk], timestamps=nt[nk] - lo, unit='uV'))
    nwb.create_processing_module('behavior', signal).add(
        TimeSeries(name='finger_angles', data=beh, timestamps=bt[bk] - lo, unit='a.u.'))
    with NWBHDF5IO(str(path), 'w') as io:
        io.write(nwb)


def _train(r, signal, tmp):
    path = Path(tmp) / f"{r['run_id']}_{signal}.nwb"
    _kf_nwb(r, signal, path)
    chans = kf.GOOD_CHANS_EMG
    kf.GOOD_CHANS_EMG = list(range(r['neural'][1].shape[1]))   # the package holds only the decoder's channels
    try:
        ds = open_nwb_path(path)
        m = kf.train_kf(ds, [INDEX_DOF]).loc['index_pos']
        ds.close()
    finally:
        kf.GOOD_CHANS_EMG = chans
    return m


def offline_kf():
    """Per hand-control run, two KFs on the same neural data and window: one fit to the bend sensor, one to Freebie."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in _both_sensors_hand():
            print('training Monkey R offline KFs:', rid, flush=True)
            r = run(rid, neural=True)
            for signal, label in (('manip', 'bend'), ('vision', 'freebie')):
                m = _train(r, signal, tmp)
                rows.append({'run_id': rid, 'run_group': r['group'], 'n_trials': len(in_mode(r)), 'signal': label,
                             'nmse_valid': m['nmse_valid'], 'correlation_valid': m['correlation_valid'],
                             'mse_valid': m['mse_valid']})
    return pd.DataFrame(rows)


def analyze_fig5(cfg):
    """Online metrics of the closed-loop and hand-control runs, the offline KFs, and one decoder example per training signal."""
    OUT.mkdir(parents=True, exist_ok=True)
    export_fig345_metrics()
    offline_kf().to_csv(OUT / 'offline_kf.csv', index=False)
    for key in ('freebie', 'bend'):
        _save_example(cfg['analysis']['examples'][f'fig5_{key}'], ['cursor'], f'fig5_example_{key}')


# ---------------------------------------------------------------- fig S5: without the manipulandum
def analyze_s5(cfg):
    """Online metrics of the bend-sensor run and the two Freebie runs without the manipulandum."""
    export_online_metrics(run_ids('S5'), prefix='figS5_')
