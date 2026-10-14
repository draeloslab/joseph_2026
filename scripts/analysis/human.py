"""
Figure S1, from raw/human/: human 2-4 DOF hand control with 3D hand tracking (four cameras, triangulated), one run per DOF count.

raw/human/<run>.npz       the run's trials (start_t, result_t, success, screen_on, targets, target_radius, hold_time, target_dofs),
                          the cursor (cursor_t, cursor_x, cursor_names) and the vision packets (vision_t, vision_frame: the
                          vision frame number each packet came from)
raw/human/improv_logs.npz the vision pipeline's per-frame logs of the session: each camera's frame starts (TISstarts_<camera>),
                          and per sent packet its send time, frame number and logged end-to-end latency
"""
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from utils import DATA, RAW
from .metrics import run_metrics

SRC = RAW / 'human'
OUT = DATA / 'figS1_human'


def trials(z):
    """One row per completed trial (trial_info followed by a trial_success before the next trial_info)."""
    start, res = z['start_t'], z['result_t']
    rows = []
    for i, (a, b) in enumerate(zip(start, np.r_[start[1:], np.inf])):
        k = np.flatnonzero((res > a) & (res <= b))
        if len(k) and np.isfinite(z['targets'][i]).all():       # a run can open on a smaller DOF set: those trials are out
            rows.append(dict(start_time=a, stop_time=res[k[0]], success=bool(z['success'][k[0]]),
                             screen_on=bool(z['screen_on'][i]), targets=z['targets'][i],
                             target_radius=float(z['target_radius'][i]), hold_time=float(z['hold_time'][i])))
    return pd.DataFrame(rows)


def cursor(z):
    """The cursor in the trials' DOF order."""
    names = list(z['cursor_names'])
    return z['cursor_t'], z['cursor_x'][:, [names.index(n) for n in z['target_dofs']]]


def task_metrics(z, tr):
    t, x = cursor(z)
    t0 = t[0]
    nwb = NWBFile(session_description='human run', identifier='run', session_start_time=datetime(2026, 10, 6, tzinfo=timezone.utc))
    for c in ['target', 'target_radius', 'hold_time', 'success', 'in_mode', 'DecodeFeature']:
        nwb.add_trial_column(name=c, description=c)
    for _, r in tr.iterrows():
        nwb.add_trial(start_time=r.start_time - t0, stop_time=r.stop_time - t0, target=np.asarray(r.targets, float),
                      target_radius=r.target_radius, hold_time=r.hold_time, success=r.success, in_mode=r.screen_on,
                      DecodeFeature=0)
    nwb.create_processing_module('behavior', 'cursor').add(
        TimeSeries(name='finger_angles', data=x.astype('float64'), timestamps=t - t0, unit='a.u.'))
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / 'run.nwb'
        with NWBHDF5IO(str(p), 'w') as io:
            io.write(nwb)
        with NWBHDF5IO(str(p), 'r') as io:
            return run_metrics(io.read(), closed=False, dofs=list(range(x.shape[1])))


def camera_e2e(logs, tol_ms=0.1):
    """{frame number: [send time - frame start of each camera's frame in that step] (ms)}, one value per camera. Only
    packets where the oldest camera reproduces the logged end-to-end latency (to tol_ms) are kept, i.e. where the frame
    numbers index the cameras' frame starts correctly."""
    fn = logs['batch3d_frame_nums']
    row = {int(f): i for i, f in enumerate(fn.max(axis=1))}
    starts = [logs[f'TISstarts_{c}'] for c in logs['cameras']]
    ts, e2e = logs['sender_timestamps'], logs['sender_joint_e2e']
    out = {}
    for k, f in enumerate(logs['sender_joint_frame_nums']):
        i = row.get(int(f))
        if i is None:
            continue
        st = np.array([s_[int(fn[i, j])] if 0 <= fn[i, j] < len(s_) else np.nan for j, s_ in enumerate(starts)])
        lat = (ts[k] - st) * 1000
        if np.isfinite(lat).all() and abs(lat.max() - e2e[k] * 1000) <= tol_ms:
            out[int(f)] = lat
    return out


def analyze(cfg):
    """Per run: task metrics per trial (bit rate ...) and the end-to-end latency of every vision frame inside a trial,
    per camera -> data/figS1_human/{runs,trials}.csv, e2e_cam_frames.npz."""
    OUT.mkdir(parents=True, exist_ok=True)
    logs = np.load(SRC / 'improv_logs.npz')
    ce = camera_e2e(logs)
    runs, trial_rows, cam_frames = [], [], {}
    for run, n_dof in cfg['analysis']['human_runs'].items():
        z = dict(np.load(SRC / f'{run}.npz'))
        tr = trials(z)
        agg, per = task_metrics(z, tr)
        runs.append({'run_id': run, 'n_dof': n_dof, 'dofs': ' '.join(map(str, z['target_dofs'])), **agg})
        trial_rows.append(per.assign(run_id=run, n_dof=n_dof))
        vt, vf = z['vision_t'], z['vision_frame']
        sel = np.zeros(len(vt), bool)
        for _, r in tr.iterrows():
            sel |= (vt >= r.start_time) & (vt <= r.stop_time)
        cam_frames[run] = np.array([ce[int(f)] for f in vf[sel] if int(f) in ce])      # (kept frames, cameras) ms
        print(f'{run}: per-camera end-to-end kept for {len(cam_frames[run])} of {int(sel.sum())} frames in trials')
    np.savez(OUT / 'e2e_cam_frames.npz', **cam_frames)
    pd.DataFrame(runs).to_csv(OUT / 'runs.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(OUT / 'trials.csv', index=False)
