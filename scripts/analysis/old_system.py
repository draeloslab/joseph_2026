"""
Monkey N (earlier system), from raw/monkey_n/: online metrics of every run, Freebie vs bend-sensor correlations and
offline Kalman filters per session, the example traces, and the pooled pipeline latency. Feeds figs S2, S3 and S4.

raw/monkey_n/runs.csv       one row per xPC run: sessions, mode, active DOFs, group, Freebie or bend, excluded, NWB file
raw/monkey_n/nwb/           a trimmed NWB per run: trials table, finger_angles, decode (closed loop), EMG (runs behind a Freebie KF)
raw/monkey_n/sessions.csv   one row per Freebie session used here, its run
raw/monkey_n/sessions/      per session (names in the npz keys): Freebie signals (t, v), the xPC clock offset, the run window
                            on the Freebie clock, the bend packets inside it (fpos_t, fpos_bend, fpos_dofs), per-frame latency
"""
from functools import cache

import numpy as np
import pandas as pd

from utils import DATA, RAW
from . import kf
from .common import open_nwb_path, trial_window, trials_frame
from .metrics import run_metrics

SRC = RAW / 'monkey_n'
BEND_DOF = {'index_angle': 1, 'pip_y': 1, 'mrs_x': 3}      # the bend-sensor DOF each Freebie signal is compared with
TRACE_PCT = (20, 80)       # fig S3 example trace: Freebie scaling (tight, so the plateaus land in the target boxes); the KFs keep 1-99
ACTORS = ['frame_acq', 'inference', 'sender', 'e2e']
OFFLINE_METRICS = ['nmse_valid', 'correlation_valid']
# (DOF, signal label, KF conditions, output) of each offline-KF box of fig S4; 'bend (same window)' is trained on the
# same stretch of data as the Freebie KF of that session
OFFLINE = [('Index', 'Bend sensor', ['bend (same window)'], 'index_pos'),
           ('Index', 'Freebie index angle', ['index_angle', 'index_angle+mrs_x'], 'index_pos'),
           ('Index', 'Freebie PIP y', ['pip_y', 'pip_y+mrs_x'], 'index_pos'),
           ('MRS', 'Bend sensor', ['bend (same window)'], 'ring_pos'),
           ('MRS', 'Freebie MRS x', ['index_angle+mrs_x'], 'ring_pos')]


# ---------------------------------------------------------------- the package
@cache
def runs():
    d = pd.read_csv(SRC / 'runs.csv', dtype=str, keep_default_na=False)
    return {r.run_id: {'sessions': r.sessions.split(','), 'mode': r.mode, 'dofs': [int(x) for x in r.dofs.split()],
                       'group': r.group, 'freebie': r.freebie == 'True', 'excluded': r.excluded == 'True', 'nwb': SRC / r.nwb}
            for r in d.itertuples()}


@cache
def sessions():
    d = pd.read_csv(SRC / 'sessions.csv', dtype=str, keep_default_na=False)
    out = {}
    for r in d.itertuples():
        z = np.load(SRC / 'sessions' / f'{r.session_id}.npz')
        get = lambda k: z[k] if k in z else None
        out[r.session_id] = {
            'run_id': r.run_id, 'freebie': r.freebie == 'True', 'freebie_kf': r.freebie_kf == 'True',
            'signals': {str(n): (z[f'signal_{n}_t'], z[f'signal_{n}_v']) for n in z['signal_names']} if 'signal_names' in z else {},
            'xpc_offset_s': float(z['xpc_offset_s']) if 'xpc_offset_s' in z else None,
            'run_window_s': tuple(z['run_window_s']) if 'run_window_s' in z else None,
            'fpos_t': get('fpos_t'), 'fpos_bend': get('fpos_bend'),
            'fpos_dofs': [int(x) for x in z['fpos_dofs']] if 'fpos_dofs' in z else None,
            'e2e_latency_s': get('e2e_latency_s'),
            'actors': {k: z[f'actors_{k}'] for k in ACTORS} if 'actors_e2e' in z else None}
    return out


def open_nwb(run_id):
    return open_nwb_path(runs()[run_id]['nwb'])


# ---------------------------------------------------------------- online metrics (figs S3, S4)
@cache
def export_run_metrics():
    """data/run_metrics.csv: one row per run (means over its trials); data/run_trial_metrics.csv: one row per in-mode trial.
    The excluded run is left out of both."""
    rows, trial_rows = [], []
    for run_id, r in runs().items():
        if r['excluded']:
            print(f'excluding {run_id} from online metrics')
            continue
        ds = open_nwb(run_id)
        agg, trials = run_metrics(ds.dataset, r['mode'] == 'closed_loop', r['dofs'])
        rows.append({'run_id': run_id, 'sessions': ','.join(r['sessions']), 'group': r['group'], 'freebie': r['freebie'],
                     'mode': r['mode'], 'n_dof': len(r['dofs']), **agg})
        trial_rows.append(trials.assign(run_id=run_id, freebie=r['freebie'], mode=r['mode'], n_dof=len(r['dofs'])))
        ds.close()
    pd.DataFrame(rows).to_csv(DATA / 'run_metrics.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(DATA / 'run_trial_metrics.csv', index=False)


# ---------------------------------------------------------------- Freebie signals against the bend sensors
def unit_range(x, lo=1, hi=99):
    """Robust 0-1 scaling so pixel / degree signals share the bend sensors' 0-1 units."""
    p_lo, p_hi = np.nanpercentile(x, [lo, hi])
    return (x - p_lo) / (p_hi - p_lo)


def xpc_signals(s, t_x, pct=(1, 99)):
    """The session's Freebie signals on xPC time inside the run, 0-1 scaled (pct = the percentiles mapped to 0 and 1):
    {name: (t, v)}."""
    out = {}
    for name, (t, v) in s['signals'].items():
        tx = t - s['xpc_offset_s']
        ok = np.isfinite(v) & (tx >= t_x[0]) & (tx <= t_x[-1])
        out[name] = (tx[ok], unit_range(v[ok], *pct))
    return out


def correlations():
    """Pearson r of each Freebie signal with the bend value at the Freebie frame times, over the frames inside the xPC
    run, per bend-sensor session with a known run window."""
    rows = []
    for sid, s in sessions().items():
        if s['freebie'] or s['run_window_s'] is None:
            continue
        tr, (lo, hi), row = s['fpos_t'], s['run_window_s'], {'session_id': sid}
        for name, (t, v) in s['signals'].items():
            ok = np.isfinite(v) & (t >= max(tr[0], lo)) & (t <= min(tr[-1], hi))
            bend = s['fpos_bend'][:, s['fpos_dofs'].index(BEND_DOF[name])]
            row[f'corr_{name}'] = np.corrcoef(v[ok], np.interp(t[ok], tr, bend))[0, 1]
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=['session_id', 'corr_index_angle', 'corr_pip_y', 'corr_mrs_x'])


def freebie_kfs(sid):
    """KFs on the run's EMG with the behaviour replaced by Freebie signals at their own frame times, next to a bend
    baseline on the same frames. The MRS signal is interpolated onto the index-signal frames."""
    s = sessions()[sid]
    run = runs()[s['run_id']]
    ds = open_nwb(s['run_id'])
    fa = ds.dataset.processing['behavior']['finger_angles']
    t_x, bend = fa.timestamps[:], fa.data[:]
    dofs = run['dofs'][:1] if len(run['dofs']) == 2 and 'mrs_x' not in s['signals'] else run['dofs']
    need = ['index_angle', 'pip_y'] + (['mrs_x'] if len(dofs) == 2 else [])
    tg = s['signals']['pip_y'][0]                                  # index-signal frame times
    vals = {n: np.interp(tg, *s['signals'][n], left=np.nan, right=np.nan) for n in need if n in s['signals']}
    tx = tg - s['xpc_offset_s']                                    # xPC seconds
    ok = np.all([np.isfinite(v) for v in vals.values()], axis=0) & (tx >= t_x[0]) & (tx <= t_x[-1])
    span, tx = tx[ok][-1] - tx[ok][0], tx[ok]

    def behavior(cols):                                            # (n, 5) like finger_angles, DOF columns filled
        b = np.full((len(tx), 5), np.nan)
        for col, dof in zip(cols, dofs):
            b[:, dof] = col
        return b

    conditions = [('bend (same window)', behavior([np.interp(tx, t_x, bend[:, d]) for d in dofs]))]
    for lead in ('index_angle', 'pip_y'):
        if lead in vals:
            names = [lead] + (['mrs_x'] if len(dofs) == 2 else [])
            conditions.append(('+'.join(names), behavior([unit_range(vals[n][ok]) for n in names])))
    rows = []
    for label, b in conditions:
        for output, m in kf.train_kf(ds, dofs, behavior=b, behavior_ts_ms=tx * 1000).iterrows():
            rows.append({'session_id': sid, 'condition': label, 'output': output, 'n_dof': len(dofs), 'window_s': span,
                         'mse_valid': m['mse_valid'], 'nmse_valid': m['nmse_valid'], 'correlation_valid': m['correlation_valid']})
    ds.close()
    return rows


def offline_kf_rows():
    rows = []
    for sid, s in sessions().items():
        if s['freebie_kf']:
            print(f'training Freebie KFs: {sid}', flush=True)
            rows += freebie_kfs(sid)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- one analysis per figure
def analyze_s2(cfg):
    """Pooled end-to-end latency of the earlier sessions and the mean processing time of each stage per session."""
    skip = cfg['analysis']['latency_skip_first_frames']
    ss = sessions()
    pooled = [s['e2e_latency_s'][skip:] * 1000 for s in ss.values() if s['e2e_latency_s'] is not None]
    np.savez(DATA / 'e2e_pooled.npz', e2e_ms=np.concatenate(pooled), n_sessions=len(pooled))
    rows = [{'session': sid, 'n_frames': len(s['actors']['e2e']), **{k: s['actors'][k][skip:].mean() for k in ACTORS}}
            for sid, s in ss.items() if s['actors'] is not None and len(s['actors']['e2e']) >= 100]
    pd.DataFrame(rows).to_csv(DATA / 'latency_stage_summary.csv', index=False)


def analyze_s3(cfg):
    """Online metrics of the hand-control runs, Freebie-vs-bend correlations, and one example trace."""
    export_run_metrics()
    corr = correlations()
    corr.dropna(how='all', subset=corr.columns[1:]).to_csv(DATA / 'freebie_bend_corr.csv', index=False)

    ex = cfg['analysis']['examples']['figS3']
    s = sessions()[ex['session']]
    dofs = runs()[s['run_id']]['dofs']
    ds = open_nwb(s['run_id'])
    fa = ds.dataset.processing['behavior']['finger_angles']
    t, bend = fa.timestamps[:], fa.data[:]
    cam = xpc_signals(s, t, pct=TRACE_PCT)
    trials = trials_frame(ds.dataset, dofs)
    lo, hi = max(c[0][0] for c in cam.values()), min(c[0][-1] for c in cam.values())
    keep = ~trials['blank'] & trials['success'] & (trials['start_time'] >= lo) & (trials['stop_time'] <= hi)
    trials = trial_window(trials, None, ex['n_trials'], keep)
    t0, t1 = trials['start_time'].iloc[0], trials['stop_time'].iloc[-1]
    win = (t >= t0) & (t <= t1)
    out = {'t': t[win], 'session': ex['session'], **{f'bend_{d}': bend[win, d] for d in dofs}}
    for name, (tc, v) in cam.items():
        w = (tc >= t0) & (tc <= t1)
        out[f'{name}_t'], out[name] = tc[w], v[w]
    np.savez(DATA / 'figS3_example_trace.npz', **out)
    trials[['start_time', 'stop_time', 'target_radius', 'success', *[f'target_{d}' for d in dofs]]].to_csv(
        DATA / 'figS3_example_trials.csv', index=False)
    ds.close()


def analyze_s4(cfg):
    """Online metrics of the closed-loop runs, the offline KFs, and one example decoder trace per training signal."""
    export_run_metrics()
    kfs = offline_kf_rows()
    rows = []
    for dof, signal, conditions, output in OFFLINE:
        sel = kfs[kfs['condition'].isin(conditions) & (kfs['output'] == output)]
        rows.append(sel.assign(dof=dof, signal=signal)[['session_id', 'n_dof', 'dof', 'signal', 'mse_valid', *OFFLINE_METRICS]])
    pd.concat(rows).to_csv(DATA / 'offline_kf.csv', index=False)

    for key in ('freebie', 'bend'):
        ex = cfg['analysis']['examples'][f'figS4_{key}']
        ds = open_nwb(ex['run'])
        dec = ds.dataset.processing['behavior']['decode']
        t, data = dec.timestamps[:], dec.data[:]
        dofs = runs()[ex['run']]['dofs']
        trials = trials_frame(ds.dataset, dofs)
        feature = int(np.median(trials['DecodeFeature']))             # decoder output columns 0-4 TCFR, 5-9 SBP
        trials = trial_window(trials, None, ex['n_trials'], ~trials['blank'] & trials['decoded'])
        win = (t >= trials['start_time'].iloc[0]) & (t <= trials['stop_time'].iloc[-1])
        np.savez(DATA / f'figS4_example_{key}_trace.npz', t=t[win], run=ex['run'],
                 **{f'decode_{d}': data[win, d + 5 * feature] for d in dofs})
        trials[['start_time', 'stop_time', 'target_radius', 'success', *[f'target_{d}' for d in dofs]]].to_csv(
            DATA / f'figS4_example_{key}_trials.csv', index=False)
        ds.close()
