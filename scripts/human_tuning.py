"""
Human 3D hand tracking on the BRAND triangulation set-up (2026-10-06 on, improv hand3d_bare -> BRAND
rhodes_2d_vision_bare): per run, the task metrics (metrics.run_metrics), how the control signal behaved (tracking
dropouts, jitter, the lag the vision_dofs smoothing added) and each pipeline stage's latency. Feeds fig 6 and the
parameter comparison (data/human_1006/runs.csv).

Runs and their settings are config `human_bare`. Per run, scripts/human_rdb.py extracts the .rdb (server copy never
touched) into data/human_1006/<run>.npz; the improv prediction folder of the run is matched on improv frame number
(the vision PC and BRAND clocks differ by ~1.7 s, so not on time).
"""
import glob
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from metrics import run_metrics
from utils import DATA, ROOT

OUT = DATA / 'human_1006'
GRID_DT = 0.01        # 100 Hz analysis grid
# improv per-frame stages (s -> ms): label, file, indexed by 'step' (batch3d rows) or 'send' (sender rows)
IMPROV_STAGES = [('Camera', 'TISlatencies', 'cam'), ('Store get', 'batch3d_lat_store_get', 'step'),
                 ('Inference', 'batch3d_lat_inference', 'step'), ('Triangulation', 'batch3d_lat_triangulate', 'step'),
                 ('Joint angles', 'batch3d_lat_angles', 'step'), ('Sender', 'sender_step_latencies', 'send'),
                 ('Vision end-to-end', 'sender_joint_e2e', 'send')]


def extract(cfg, force=False, key='human_bare'):
    hb = cfg[key]
    OUT.mkdir(parents=True, exist_ok=True)
    for run in hb['runs']:
        out = OUT / f'{run}.npz'
        if out.exists() and not force:
            continue
        rdb = Path(hb['day_dir']) / run / f'{run}.rdb'
        for attempt in range(4):       # the redis env occasionally dies at startup
            if subprocess.run([str(Path(cfg['human']['rdb_python']).expanduser()), str(ROOT / 'scripts' / 'human_rdb.py'),
                               str(rdb), str(out)], capture_output=True).returncode == 0:
                break
        else:
            raise RuntimeError(f'{rdb}: extraction failed')


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


def cursor(z, tr):
    """The cursor in the trials' DOF order."""
    names = list(z['cursor_names'])
    return z['cursor_t'], z['cursor_x'][:, [names.index(n) for n in z['target_dofs']]]


def task_metrics(z, tr):
    t, x = cursor(z, tr)
    t0 = t[0]
    nwb = NWBFile(session_description='human bare', identifier='run', session_start_time=datetime(2026, 10, 6, tzinfo=timezone.utc))
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


def _lag_ms(t_raw, raw, t_cur, cur, max_ms=250, step_ms=2):
    """Lag (ms) of the cursor behind the raw angle: the shift L minimising mean (cursor(t + L) - raw(t))^2 over the
    raw packets (the cursor is held between its samples, as the task sees it)."""
    best, best_err = 0, np.inf
    for L in np.arange(0, max_ms + step_ms, step_ms) / 1000:
        idx = np.clip(np.searchsorted(t_cur, t_raw + L, side='right') - 1, 0, len(t_cur) - 1)
        err = np.nanmean((cur[idx] - raw) ** 2)
        if err < best_err:
            best, best_err = L, err
    return best * 1000


def _jitter(t, x, still_t):
    """RMS of x minus its own 100 ms centred moving average, over samples inside still periods (normalised units)."""
    n = max(3, int(round(0.1 / np.median(np.diff(t)))) | 1)
    ma = np.convolve(x, np.ones(n) / n, 'same')
    keep = np.interp(t, still_t[0], still_t[1].astype(float)) > 0.5
    keep[:n] = keep[-n:] = False
    return np.sqrt(np.mean((x - ma)[keep] ** 2)) if keep.any() else np.nan


def signal_quality(cfg, z, tr):
    """Over the task window, on the index DOF: tracking dropouts, jitter of the raw angle and of the cursor while the
    hand is still, and the lag the BRAND smoothing added."""
    lo, hi = tr['start_time'].iloc[0], tr['stop_time'].iloc[-1]
    rng = cfg['human_bare']['angle_range']['index']
    vt, vx = z['vision_t'], z['vision_x'][:, 0]
    w = (vt >= lo) & (vt <= hi)
    vt, vx = vt[w], vx[w]
    raw = np.clip((vx - rng[0]) / (rng[1] - rng[0]), 0, 1)
    ok = np.isfinite(raw)
    gaps = np.diff(vt)
    ct, cx = cursor(z, tr)
    wc = (ct >= lo) & (ct <= hi)
    ct, c0 = ct[wc], cx[wc, 0]
    # still: the cursor moved < 0.05 (5 % of the range) over the surrounding 300 ms
    g = np.arange(lo, hi, GRID_DT)
    cg = np.interp(g, ct, c0)
    k = int(0.3 / GRID_DT)
    span = pd.Series(cg).rolling(k, center=True).max() - pd.Series(cg).rolling(k, center=True).min()
    still = (g, (span < 0.05).to_numpy())
    return {'vision_rate_hz': len(vt) / (hi - lo), 'tracking_lost_pct': 100 * np.mean(~ok),
            'vision_gaps_over_100ms': int(np.sum(gaps > 0.1)), 'longest_vision_gap_ms': 1000 * gaps.max(),
            'still_pct': 100 * still[1].mean(), 'raw_jitter_rms': _jitter(vt[ok], raw[ok], still),
            'cursor_jitter_rms': _jitter(ct, c0, still), 'smoothing_lag_ms': _lag_ms(vt[ok], raw[ok], ct, c0)}


def improv_latency(cfg, run, z):
    """Per-frame stage times (ms) of the improv frames the run received (matched on frame number)."""
    folder = Path(cfg['human_bare']['predictions']).expanduser() / cfg['human_bare']['runs'][run]['session']
    L = lambda n: np.load(folder / f'{n}.npy')
    lo, hi = np.nanmin(z['vision_frame']), np.nanmax(z['vision_frame'])
    send = (L('sender_joint_frame_nums') >= lo) & (L('sender_joint_frame_nums') <= hi)
    fn = L('batch3d_frame_nums').max(axis=1)
    step = (fn >= lo) & (fn <= hi)
    t_lo, t_hi = L('sender_timestamps')[send][[0, -1]]
    out = {}
    for label, name, kind in IMPROV_STAGES:
        if kind == 'cam':
            cams = sorted(f.stem.split('TISlatencies_')[1] for f in folder.glob('TISlatencies_*.npy'))
            v = np.concatenate([L(f'TISlatencies_{c}')[(L(f'TISstarts_{c}') >= t_lo) & (L(f'TISstarts_{c}') <= t_hi)]
                                for c in cams])
        else:
            v = L(name)[step if kind == 'step' else send]
        out[label] = v * 1000
    return out


def brand_latency(z):
    """BRAND side (one clock): vision_dofs output -> age of the newest vision_data packet; cursor -> its vision_dofs."""
    vt, dt, ct = z['vision_t'], z['dofs_t'], z['cursor_t']
    i = np.searchsorted(vt, dt, side='right') - 1
    age = (dt - vt[np.clip(i, 0, None)]) * 1000
    j = np.searchsorted(dt, ct, side='right') - 1
    mux = (ct - dt[np.clip(j, 0, None)]) * 1000
    return {'vision -> vision_dofs': age[(i >= 0) & (age < 100)], 'vision_dofs -> cursor': mux[(j >= 0) & (mux < 100)]}


def analyze(cfg):
    extract(cfg)
    rows, trial_rows, lat_rows = [], [], []
    for run, info in cfg['human_bare']['runs'].items():
        z = dict(np.load(OUT / f'{run}.npz'))
        tr = trials(z)
        agg, per = task_metrics(z, tr)
        q = signal_quality(cfg, z, tr)
        lat = improv_latency(cfg, run, z) | brand_latency(z)
        rows.append({'run_id': run, 'setting': info['setting'], 'duration_s': tr['stop_time'].iloc[-1] - tr['start_time'].iloc[0],
                     **agg, **q, **{f'{k} (ms)': np.median(v) for k, v in lat.items()}})
        trial_rows.append(per.assign(run_id=run, setting=info['setting']))
        lat_rows += [pd.DataFrame({'run_id': run, 'stage': k, 'latency_ms': v}) for k, v in lat.items()]
    pd.DataFrame(rows).to_csv(OUT / 'runs.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(OUT / 'trials.csv', index=False)
    pd.concat(lat_rows, ignore_index=True).to_csv(OUT / 'latency.csv', index=False)
    ex = cfg['human_bare']['example']
    z = dict(np.load(OUT / f"{ex['run']}.npz"))
    tr = trials(z)
    ok, n = tr['success'].to_numpy(), ex['n_trials']
    if ex['first_trial'] is not None:
        first = ex['first_trial']
    else:       # auto: the n consecutive successful trials with the shortest total time (falls back to a third in)
        starts = [i for i in range(len(tr) - n + 1) if ok[i:i + n].all()]
        dur = lambda i: tr['stop_time'].iloc[i + n - 1] - tr['start_time'].iloc[i]
        first = min(starts, key=dur) if starts else int(np.flatnonzero(ok)[ok.sum() // 3])
    tr = tr.iloc[first:first + n]
    t, x = cursor(z, tr)
    w = (t >= tr['start_time'].iloc[0]) & (t <= tr['stop_time'].iloc[-1])
    np.savez(OUT / 'example.npz', t=t[w], x=x[w], dofs=z['target_dofs'], run=ex['run'])
    frame = tr[['start_time', 'stop_time', 'target_radius', 'success']].copy()
    for j in range(len(z['target_dofs'])):
        frame[f'target_{j}'] = [tg[j] for tg in tr['targets']]
    frame.to_csv(OUT / 'example_trials.csv', index=False)


# ---------------------------------------------------------------- 2-4 DOF runs (config human_dof, 2026-10-07)
# improv stages of one vision frame; they sum to the vision total (camera callback -> UDP send):
#   Camera = slowest camera of the step; Gather wait = the step waiting for frames; Send wait = what is left between the
#   step's end and the send (queue put + hand-over), Sender = the UDP send itself
DOF_STAGES = ['Camera', 'Gather wait', 'Store get', 'Inference', 'Triangulation', 'Joint angles', 'Send wait', 'Sender', 'Vision total']


def frame_stages(folder):
    """{frame number: [DOF_STAGES in ms]} for every frame improv sent (matched step <-> sender row on frame number)."""
    folder = Path(folder)
    L = lambda n: np.load(folder / f'{n}.npy', allow_pickle=True)
    fn = L('batch3d_frame_nums')
    row = {int(f): i for i, f in enumerate(fn.max(axis=1))}
    cams = sorted(f.stem.split('TISlatencies_')[1] for f in folder.glob('TISlatencies_*.npy'))
    cam = np.stack([L(f'TISlatencies_{c}') for c in cams]) if len({len(L(f'TISlatencies_{c}')) for c in cams}) == 1 else None
    lat = {k: L(f'batch3d_lat_{k}') * 1000 for k in ('store_get', 'inference', 'triangulate', 'angles', 'queue_wait')}
    tcam = [L(f'TISlatencies_{c}') * 1000 for c in cams]
    e2e, snd = L('sender_joint_e2e') * 1000, L('sender_step_latencies') * 1000
    out = {}
    for k, f in enumerate(L('sender_joint_frame_nums')):
        i = row.get(int(f))
        if i is None:
            continue
        c = max(t[min(int(fn[i, j]), len(t) - 1)] for j, t in enumerate(tcam))
        parts = [c, lat['queue_wait'][i], lat['store_get'][i], lat['inference'][i], lat['triangulate'][i], lat['angles'][i]]
        out[int(f)] = parts + [e2e[k] - sum(parts) - snd[k], snd[k], e2e[k]]
    return out


def camera_e2e(folder, tol_ms=0.1):
    """{frame number: [send time - camera_start of each camera's frame in that step] (ms)}, like fig 3's single-camera
    end-to-end (send_time - camera_start), one value per camera. send_time = sender_timestamps (one per sent joint row),
    each camera's camera_start = its TISstarts at the step's frame number. Only rows where the oldest camera
    reproduces the logged sender_joint_e2e (to tol_ms) are kept, i.e. where the frame numbers index TISstarts correctly."""
    folder = Path(folder)
    L = lambda n: np.load(folder / f'{n}.npy', allow_pickle=True)
    fn = L('batch3d_frame_nums')
    row = {int(f): i for i, f in enumerate(fn.max(axis=1))}
    cams = sorted(f.stem.split('TISstarts_')[1] for f in folder.glob('TISstarts_*.npy'))
    starts = [L(f'TISstarts_{c}') for c in cams]
    ts, e2e = L('sender_timestamps'), L('sender_joint_e2e')
    out, n_rows = {}, 0
    for k, f in enumerate(L('sender_joint_frame_nums')):
        i = row.get(int(f))
        if i is None:
            continue
        n_rows += 1
        st = np.array([s_[int(fn[i, j])] if 0 <= fn[i, j] < len(s_) else np.nan for j, s_ in enumerate(starts)])
        lat = (ts[k] - st) * 1000
        if np.isfinite(lat).all() and abs(lat.max() - e2e[k] * 1000) <= tol_ms:
            out[int(f)] = lat
    out['_kept'] = (len(out), n_rows)
    return out


def analyze_dof(cfg):
    """Per run of config human_dof: per-trial task metrics (bit rate ...) and per-trial mean stage latencies
    (every vision frame inside the trial, matched on frame number) -> data/human_dof/{runs,trials,latency_trials}.csv."""
    hd = cfg['human_dof']
    out = DATA / 'human_dof'
    out.mkdir(parents=True, exist_ok=True)
    runs, trial_rows, lat_rows, e2e_frames, cam_frames = [], [], [], {}, {}
    tmp_cfg = {'human': cfg['human'], 'human_bare': {'day_dir': hd['day_dir'], 'runs': hd['runs']}}
    global OUT
    old, OUT = OUT, out
    try:
        extract(tmp_cfg)
    finally:
        OUT = old
    frames = {}
    for run, info in hd['runs'].items():
        z = dict(np.load(out / f'{run}.npz'))
        tr = trials(z).iloc[info.get('first_trial', 0):].reset_index(drop=True)
        agg, per = task_metrics(z, tr)
        runs.append({'run_id': run, 'n_dof': info['n_dof'], 'dofs': ' '.join(map(str, z['target_dofs'])), **agg})
        per = per.assign(run_id=run, n_dof=info['n_dof'])
        trial_rows.append(per)
        folder = Path(hd['predictions']).expanduser() / info['session']
        if info['session'] not in frames:
            frames[info['session']] = frame_stages(folder)
        st = frames[info['session']]
        if ('cam', info['session']) not in frames:
            frames[('cam', info['session'])] = camera_e2e(folder)
        ce = frames[('cam', info['session'])]
        vt, vf = z['vision_t'], z['vision_frame']
        frame_e2e = []
        for k, r in tr.iterrows():
            w = (vt >= r.start_time) & (vt <= r.stop_time)
            rows = np.array([st[int(f)] for f in vf[w] if int(f) in st])
            if len(rows):
                frame_e2e.append(rows[:, -1])
                lat_rows.append({'run_id': run, 'n_dof': info['n_dof'], 'trial': k, 'success': r.success, 'n_frames': len(rows),
                                 **dict(zip(DOF_STAGES, rows.mean(axis=0)))})
        e2e_frames[run] = np.concatenate(frame_e2e)
        sel = np.zeros(len(vt), bool)
        for _, r in tr.iterrows():
            sel |= (vt >= r.start_time) & (vt <= r.stop_time)
        per_cam = np.array([ce[int(f)] for f in vf[sel] if int(f) in ce])
        cam_frames[run] = per_cam                                  # (kept frames, cameras) ms
        print(f'{run}: per-camera end-to-end kept for {len(per_cam)} of {int(sel.sum())} frames in trials (validated against the logged e2e)')
    np.savez(out / 'e2e_frames.npz', **e2e_frames)       # vision total (ms) of every frame inside a trial, per run
    np.savez(out / 'e2e_cam_frames.npz', **cam_frames)   # per camera: send time - that camera's start (ms), frames x cameras
    pd.DataFrame(runs).to_csv(out / 'runs.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(out / 'trials.csv', index=False)
    pd.DataFrame(lat_rows).to_csv(out / 'latency_trials.csv', index=False)
