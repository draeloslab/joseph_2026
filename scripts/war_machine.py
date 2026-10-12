"""
War Machine (Rhodes rig, BRAND) master: <master>/war_machine.pkl plus <master>/war_machine/<run_id>_neural.npz,
built from the server's Run-XXX.rdb files (via war_machine_rdb.py) and the vision pipeline's per-frame logs
(~/predictions). Which runs go in, and why the others are left out, is config.yaml `war_machine`. The figure scripts
only read this master, like the paper's master.pkl.

    python scripts/war_machine.py              # build the War Machine master (resumes an interrupted build)
    python scripts/war_machine.py --rebuild    # re-extract every run from the server
    python scripts/war_machine.py --key war_machine_s5   # fig S5's master (2026-10-09, no manipulandum), see sub_config

A sibling of master.pkl rather than a key inside it: master.pkl is rewritten from scratch by
~/freebie/script/build_master.py.

What drove the cursor on each trial comes from the control mux's own log (`control_state`) at the trial start,
not the session notes (they have 09-30 Run-006, 010, 012 backwards; the `behavior` stream confirms the mux):
cursor_source 'decoder' = closed loop with the decoder loaded at that moment, else behavior_source 'manipulandum' =
bend hand control, 'vision' = Freebie hand control. A run's mode is the mode of most of its screen-on trials; only
trials in that mode count for it (drops the KF runs' hand warm-up trials and trials around a source switch).

Neural: only the online decoder's channels (config war_machine.decoder_channels, from kf_rhodes.yaml), pre-binned to
4 ms mean |x|. kf.py's own 32 ms MAV over those equals the MAV of the raw 2 kHz samples (equal windows; checked on
09-30 Run-005: NMSE 0.358 raw vs 0.359 pre-binned) and its binning, which masks the whole array per bin, finishes in
seconds instead of hours.
"""
import datetime
import pickle
import subprocess
import sys
import tempfile
from collections import Counter
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from utils import load_config, master_dir

PRE_BIN_MS = 4
HAND_MODES = {'manipulandum': 'bend', 'vision': 'freebie'}
STREAMS = ['manip', 'manip_raw', 'vision', 'vision_raw', 'behavior', 'cursor', 'decoded']
# the paper's fig 2 stages, from the improv logs of the single camera on this rig (cam2, TIS 'front_left'); end-to-end
# = camera frame start -> UDP send to BRAND (improv experiments/FES/actors/sender_udp.py), as the UART sender did
LATENCY_STAGES = [('frame_acq', 'TISstarts_front_left', 'TISlatencies_front_left'),
                  ('inference', 'proc_timestamps_cam2', 'proc_total_cam2'),
                  ('sender', 'sender_timestamps', 'sender_step_latencies'),
                  ('e2e', 'true_e2e_cam2_timestamps', 'true_e2e_cam2')]


def sub_config(cfg, key):
    """cfg with cfg[key] laid over cfg['war_machine'] (server, envs and channels inherited; runs, master_name, ...
    replaced), for a second War Machine master next to the figs 3-5 one."""
    return {**cfg, 'war_machine': {**cfg['war_machine'], 'kf_training_runs': {}, **cfg[key]}}


def _master_path(cfg):
    return master_dir(cfg) / f"{cfg['war_machine'].get('master_name', 'war_machine')}.pkl"


def run_ids(cfg):
    return [f'{day}_{run}' for day, runs in cfg['war_machine']['runs'].items() for run in runs]


# ================================================================= build
def _extract(cfg, rdb, out):
    wmc = cfg['war_machine']
    cmd = [str(Path(wmc['rdb_python']).expanduser()), str(HERE / 'war_machine_rdb.py'), str(rdb), str(out),
           *map(str, wmc['decoder_channels'])]
    for attempt in range(6):        # the redis env's interpreter / redis-server occasionally dies at startup; retry
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return np.load(out)
        print(f'    extraction failed, retrying ({attempt + 1})', flush=True)
    raise RuntimeError(f'{rdb}: extraction kept failing')


def _at(times, values, t):
    i = np.searchsorted(times, t, side='right') - 1
    return values[i] if i >= 0 else None


def _trials(z):
    """One row per completed trial (trial_info followed by a trial_success before the next trial_info; restarts leave
    starts with no result, dropped), with its mode: 'bend' | 'freebie' | 'kf:<decoder file>'."""
    start, res = z['start_t'], z['result_t']
    rows = []
    for i, (a, b) in enumerate(zip(start, np.r_[start[1:], np.inf])):
        k = np.flatnonzero((res > a) & (res <= b))
        if not len(k):
            continue
        cursor_src, behavior_src = _at(z['mux_t'], z['mux_cursor'], a), _at(z['mux_t'], z['mux_behavior'], a)
        if cursor_src == 'decoder':
            model = _at(z['decoder_t'], z['decoder_model'], a) if 'decoder_t' in z else None
            mode = f'kf:{model}'
        else:
            mode = HAND_MODES.get(str(behavior_src), str(behavior_src))
        rows.append(dict(start_time=a, stop_time=res[k[0]], success=bool(z['success'][k[0]]), screen_on=bool(z['screen_on'][i]),
                         target=float(z['target'][i]), target_radius=float(z['target_radius'][i]),
                         hold_time=float(z['hold_time'][i]), mode=mode))
    return pd.DataFrame(rows)


def _prebin(t, x, win_ms=PRE_BIN_MS):
    w = win_ms / 1000
    edges = t[0] + np.arange(int(np.ceil((t[-1] - t[0]) / w)) + 1) * w
    idx = np.searchsorted(t, edges)
    keep = np.flatnonzero(np.diff(idx) > 0)
    out = np.stack([np.abs(x[idx[i]:idx[i + 1]]).mean(axis=0) for i in keep])
    return 0.5 * (edges[keep] + edges[keep + 1]), out.astype('float32')


def _vision_logs(cfg, day):
    """Per-frame stage times (ms) of the live single-camera improv sessions that day (the 3-camera 'batch3d' sessions
    are other recordings; sessions without TIS logs replayed a video). First frame of each dropped (GPU warm-up)."""
    folder = Path(cfg['war_machine']['predictions']).expanduser() / day.replace('-', '')
    logs = []
    for s in sorted(folder.iterdir()) if folder.exists() else []:
        if not ((s / 'true_e2e_cam2.npy').exists() and (s / 'TISstarts_front_left.npy').exists()):
            continue
        L = lambda n: np.load(s / f'{n}.npy')
        d = {}
        for key, tname, vname in LATENCY_STAGES:
            t, v = L(tname)[1:], L(vname)[1:] * 1000
            if key == 'sender':
                fresh = L('sender_fresh_cam2').astype(bool)[1:]
                t, v = t[fresh], v[fresh]
            d[key] = (t, v)
        logs.append(d)
    return logs


def _latency(logs, window):
    """Frames whose time falls inside the run's task window (both clocks Unix), and the camera rate during it."""
    inside = lambda t: (t >= window[0]) & (t <= window[1])
    lat = {key: np.concatenate([d[key][1][inside(d[key][0])] for d in logs]) if logs else np.array([])
           for key, *_ in LATENCY_STAGES}
    starts = np.concatenate([d['frame_acq'][0][inside(d['frame_acq'][0])] for d in logs]) if logs else np.array([])
    lat['fps'] = int(round(1 / np.median(np.diff(starts)) / 30) * 30) if len(starts) > 1 else None
    return lat


def _build_run(cfg, run_id, logs, tmp, out_dir):
    wmc = cfg['war_machine']
    day, run = run_id.split('_')
    z = _extract(cfg, Path(wmc['server']).expanduser() / day / run / f'{run}.rdb', Path(tmp) / 'run.npz')
    tr = _trials(z)
    nt, nx = _prebin(z['neural_t'], z['neural_x'])
    np.savez_compressed(out_dir / f'{run_id}_neural.npz', t=nt, x=nx)
    if day not in logs:
        logs[day] = _vision_logs(cfg, day)
    return {'day': day, 'run': run, 'trials': tr, 'mode': Counter(tr.loc[tr['screen_on'], 'mode']).most_common(1)[0][0],
            'streams': {s: (z[f'{s}_t'], z[f'{s}_x'].astype('float32')) for s in STREAMS if f'{s}_t' in z},
            'window_unix': tuple(z['window_unix']), 'latency': _latency(logs[day], z['window_unix']),
            'neural': f'war_machine/{run_id}_neural.npz'}


def build(cfg, rebuild=False):
    """Each run is saved to <master>/war_machine/<run_id>.pkl as soon as it is extracted, so an interrupted build
    resumes where it stopped; rebuild=True re-extracts every run."""
    wmc = cfg['war_machine']
    out_dir = master_dir(cfg) / 'war_machine'
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, logs = {}, {}
    with tempfile.TemporaryDirectory() as tmp:
        for run_id in run_ids(cfg):
            rec_path = out_dir / f'{run_id}.pkl'
            if rec_path.exists() and not rebuild:
                runs[run_id] = _read(str(rec_path))
                continue
            print('extracting', run_id, flush=True)
            runs[run_id] = _build_run(cfg, run_id, logs, tmp, out_dir)
            with open(rec_path, 'wb') as f:
                pickle.dump(runs[run_id], f, protocol=4)
    # decoder runs: label = the mode of the hand-control run the decoder was trained on (config kf_training_runs)
    for run_id, r in runs.items():
        r['decoder'] = r['mode'][3:] if r['mode'].startswith('kf:') else None
        trained_on = (wmc['kf_training_runs'].get(r['day']) or {}).get(r['decoder']) if r['decoder'] else None
        r['trained_on'] = f"{r['day']}_{trained_on}" if trained_on else None
        r['group'] = r['mode'] if not r['decoder'] else (f"kf_{runs[r['trained_on']]['mode']}" if r['trained_on'] in runs else 'kf_unknown')
    master = {'meta': {'built': datetime.datetime.now().isoformat(timespec='seconds'),
                       'decoder_channels': list(wmc['decoder_channels']), 'neural_pre_bin_ms': PRE_BIN_MS},
              'runs': runs}
    with open(_master_path(cfg), 'wb') as f:
        pickle.dump(master, f, protocol=4)
    print('wrote', _master_path(cfg))
    return master


# ================================================================= read
@cache
def _read(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def load(cfg):
    """<master>/war_machine.pkl (or config master_name), read once per process."""
    return _read(str(_master_path(cfg)))


def run(cfg, run_id, neural=False):
    """One run: its master record, plus 'neural' = (t, x[n, channels]) when asked for (tens of MB)."""
    r = dict(load(cfg)['runs'][run_id], run_id=run_id)
    if neural:
        z = np.load(master_dir(cfg) / r['neural'])
        r['neural'] = (z['t'], z['x'])
    return r


def in_mode(r):
    """The run's trials that count: screen on and in the run's own mode."""
    tr = r['trials']
    return tr[tr['screen_on'] & (tr['mode'] == r['mode'])].reset_index(drop=True)


def corr(a, b, lo=None, hi=None):
    """Pearson r of stream b interpolated onto stream a's sample times, inside [lo, hi] and both streams' spans."""
    ta, xa = a; tb, xb = b
    lo = max(ta[0], tb[0], -np.inf if lo is None else lo)
    hi = min(ta[-1], tb[-1], np.inf if hi is None else hi)
    ok = (ta >= lo) & (ta <= hi)
    return np.corrcoef(xa[ok], np.interp(ta[ok], tb, xb))[0, 1]


if __name__ == '__main__':
    cfg = load_config()
    if '--key' in sys.argv:
        cfg = sub_config(cfg, sys.argv[sys.argv.index('--key') + 1])
    build(cfg, rebuild='--rebuild' in sys.argv)
