"""
Figure 6: 3D human hand tracking, 1-4 DOF tasks. Online metrics (same six panels as fig 3) with one box per
DOF count, plus a latency figure (per-actor boxes like fig 2B) from every latency we can find for the runs:
the prediction folder of each NWB run, the NWB itself, and the run's Redis dump (.rdb) if there is one.

Runs, DOF labels and paths are under `human:` in config.yaml. `plot_dofs` (or generate_figs.py --dofs 1 2) picks
which DOF tasks are drawn; analyze() always processes every run, so switching needs no rerun.
"""
import json
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functools import partial
from pynwb import NWBFile, NWBHDF5IO, TimeSeries

from metrics import run_metrics
from utils import COLORS, DATA, ONLINE_METRICS, ROOT, boxes, compose, metric_panel

DOF_COLORS = {1: '#2a78d6', 2: '#1baf7a', 3: '#eb6834', 4: '#b455e0'}
TITLE = '3D Human Hand Tracking'
LATENCY_SOURCES = {'prediction': '#2a78d6', 'nwb': '#1baf7a', 'rdb': '#eb6834'}

m = list(ONLINE_METRICS)


def layout_for(cfg):
    n = len(plot_dofs(cfg))        # one grid column per box, so box widths match however many DOF groups are drawn
    cols = 3 * n                   # 3 metric panels per row, each n boxes wide; one trace panel per DOF task, 3 wide
    return [['title'] * cols, [f'trace_{d}' for d in plot_dofs(cfg) for _ in range(3)],
            [k for k in m[:3] for _ in range(n)], [k for k in m[3:] for _ in range(n)]]


def _path(p):
    p = Path(p).expanduser()
    return p if p.is_absolute() else ROOT / p


def run_files(cfg):
    """[(run, n_dof, nwb path, rdb path)] for every run in config human.runs."""
    day = Path(cfg['human']['day_dir'])
    return [(r, n, day / r / f'{r}.nwb', day / r / f'{r}.rdb') for r, n in cfg['human']['runs'].items()]


def to_metrics_nwb(cfg, run, src, dst):
    """
    The human NWB (acquisition/continuous + trials with targets/target_names/screen_on) rewritten in the layout
    metrics.run_metrics reads: finger_angles restricted to the DOFs the trials target, and target / in_mode /
    DecodeFeature columns. Returns the number of DOFs actually targeted.
    """
    with NWBHDF5IO(str(src), 'r') as io:
        nwb = io.read()
        tr = read_trials(cfg, run, nwb)
        cont = nwb.acquisition['continuous']
        names = cont.description.split(';')     # DOF names, in column order
        t, x = cont.timestamps[:], cont.data[:]
    targeted = full_dofs(tr)
    x = x[:, [names.index(n) for n in targeted]]
    tr = tr[tr['target_names'].map(len) == len(targeted)]    # a run can mix DOF sets (Run-015: index-only + 3 DOF); keep the full-DOF trials
    out = NWBFile(session_description='human run', identifier=Path(src).stem, session_start_time=datetime(2026, 9, 30, tzinfo=timezone.utc))
    for c in ['target', 'target_radius', 'hold_time', 'success', 'in_mode', 'DecodeFeature']:
        out.add_trial_column(name=c, description=c)
    for _, r in tr.iterrows():
        out.add_trial(start_time=r.start_time, stop_time=r.stop_time, target=np.asarray(r.targets, float),
                      target_radius=float(r.target_radius), hold_time=float(r.hold_time), success=bool(r.success),
                      in_mode=bool(r.screen_on), DecodeFeature=0)
    out.create_processing_module('behavior', 'b').add(TimeSeries(name='finger_angles', data=x.astype('float64'),
                                                                  timestamps=t, unit='a.u.'))
    with NWBHDF5IO(str(dst), 'w') as io:
        io.write(out)
    return len(targeted)


# ---------------------------------------------------------------- trials (NWB + the RDB's full target sets)
def read_trials(cfg, run, nwb):
    """
    The NWB trials table, with target_names / targets replaced by the RDB's trial_info where the two disagree: the
    NWB export of Run-015 (3 DOF) kept only the index target of trials that had index + middle + rp targets. Trials
    are matched on start time (same clock); needs data/human_rdb_trials_<run>.json from analyze_latency.
    """
    tr = nwb.trials.to_dataframe()
    f = DATA / f'human_rdb_trials_{run}.json'
    if not f.exists():
        return tr
    rdb = json.loads(f.read_text())
    starts = np.array([r['start'] for r in rdb])
    names, targets = [], []
    for st, old_names in zip(tr['start_time'], tr['target_names']):
        j = int(np.argmin(np.abs(starts - st)))
        if abs(starts[j] - st) < 0.1 and len(rdb[j]['targets']) > len(old_names):
            names.append(list(rdb[j]['targets'])); targets.append(np.array(list(rdb[j]['targets'].values())))
        else:
            names.append(list(old_names)); targets.append(np.asarray(_row(tr, st)))
    tr = tr.copy()
    tr['target_names'], tr['targets'] = names, targets
    return tr


def full_dofs(tr):
    """The run's DOF names: the trials' longest target set."""
    return list(max(tr['target_names'], key=len))


def _row(tr, st):
    return tr.loc[tr['start_time'] == st, 'targets'].iloc[0]


# ---------------------------------------------------------------- example trajectories
N_EXAMPLE_TRIALS = 4
MAX_EXAMPLE_TRIAL_S = 12     # example window = consecutive successful trials that are each this short, if any exist
DOF_NAMES_SHOWN = {'rp': 'ring/pinky'}


def export_example(cfg, run, n_dof, src):
    """Hand-control trajectory of a few consecutive successful trials (a third of the way in) + their targets."""
    with NWBHDF5IO(str(src), 'r') as io:
        nwb = io.read()
        tr = read_trials(cfg, run, nwb)
        cont = nwb.acquisition['continuous']
        names = cont.description.split(';')
        t, x = cont.timestamps[:], cont.data[:]
    targeted = full_dofs(tr)
    x = x[:, [names.index(n) for n in targeted]]
    tr = tr[tr['target_names'].map(len) == len(targeted)].reset_index(drop=True)
    ok = (tr['success'] & (tr['screen_on'] == 1)).to_numpy()
    short = ((tr['stop_time'] - tr['start_time']) <= MAX_EXAMPLE_TRIAL_S).to_numpy()
    good = ok & short
    starts = [i for i in np.flatnonzero(good) if good[i:i + N_EXAMPLE_TRIALS].all() and i + N_EXAMPLE_TRIALS <= len(tr)]
    first = int(starts[len(starts) // 3] if starts else np.flatnonzero(ok)[ok.sum() // 3])
    tr = tr.iloc[first:first + N_EXAMPLE_TRIALS]
    win = (t >= tr['start_time'].iloc[0]) & (t <= tr['stop_time'].iloc[-1])
    frame = tr[['start_time', 'stop_time', 'target_radius', 'success']].copy()
    for j in range(len(targeted)):
        frame[f'target_{j}'] = [tg[j] for tg in tr['targets']]
    frame.to_csv(DATA / f'human_example_{run}_trials.csv', index=False)
    np.savez(DATA / f'human_example_{run}_trace.npz', t=t[win], x=x[win], names=np.array(targeted))


# ---------------------------------------------------------------- online metrics
def analyze(cfg):
    """Every run in config human.runs (plot_dofs is applied only when drawing, so changing it needs no rerun)."""
    analyze_latency(cfg)        # first: it also saves each run's full RDB target sets that read_trials() uses
    rows, trial_rows = [], []
    tmp = Path(tempfile.mkdtemp())
    for run, n_dof, nwb_path, _ in run_files(cfg):
        n_targeted = to_metrics_nwb(cfg, run, nwb_path, tmp / f'{run}.nwb')
        export_example(cfg, run, n_dof, nwb_path)
        if n_targeted != n_dof:
            print(f'WARNING {run}: labelled {n_dof} DOF but its trials target {n_targeted} DOF -- metrics use those')
        with NWBHDF5IO(str(tmp / f'{run}.nwb'), 'r') as io:
            agg, trials = run_metrics(io.read(), closed=False, dofs=list(range(n_targeted)))
        rows.append({'run_id': run, 'n_dof': n_dof, 'n_dof_targeted': n_targeted, **agg})
        trial_rows.append(trials.assign(run_id=run, n_dof=n_dof))
    shutil.rmtree(tmp)
    pd.DataFrame(rows).to_csv(DATA / 'human_run_metrics.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(DATA / 'human_trial_metrics.csv', index=False)


def plot_dofs(cfg):
    return list(cfg['human']['plot_dofs'])


def load(cfg):
    runs, trials = pd.read_csv(DATA / 'human_run_metrics.csv'), pd.read_csv(DATA / 'human_trial_metrics.csv')
    keep = plot_dofs(cfg)
    by_dof = {n: r for r, n in cfg['human']['runs'].items()}
    ex = {n: (dict(np.load(DATA / f'human_example_{by_dof[n]}_trace.npz')), pd.read_csv(DATA / f'human_example_{by_dof[n]}_trials.csv'))
          for n in keep}
    return {'runs': runs[runs['n_dof'].isin(keep)], 'trials': trials[trials['n_dof'].isin(keep)], 'dofs': keep, 'examples': ex}


def panel_metric(ax, d, metric):
    groups = [(f'{n} DOF', d['runs'][d['runs']['n_dof'] == n], c) for n, c in DOF_COLORS.items() if n in d['dofs']]
    metric_panel(ax, groups, metric, trials=d['trials'])


STEP = 1.45    # vertical spacing between stacked DOFs in a trace panel


def panel_title(ax, d):
    ax.axis('off')
    ax.text(0.5, 0.5, TITLE, ha='center', va='center', fontsize=22, fontweight='bold', transform=ax.transAxes)


def panel_trace(ax, d, n):
    """Hand-control trajectory (black) through the target boxes (grey hit / red miss), one stacked row per DOF."""
    tr, trials = d['examples'][n]
    t0 = trials['start_time'].iloc[0]
    names = [DOF_NAMES_SHOWN.get(x, x) for x in tr['names']]
    for j, name in enumerate(names):
        off = j * STEP
        for _, row in trials.iterrows():
            r = row['target_radius']
            ax.add_patch(plt.Rectangle((row['start_time'] - t0, off + row[f'target_{j}'] - r), row['stop_time'] - row['start_time'],
                                       2 * r, facecolor=COLORS['target_hit'] if row['success'] else COLORS['target_miss'],
                                       edgecolor='#6f6f6f', ls='--', lw=1, alpha=0.7, zorder=1))
        ax.plot(tr['t'] - t0, off + tr['x'][:, j], color=DOF_COLORS[n], lw=2.2, zorder=3)
        ax.text(0.005, off + 1.05, name, transform=ax.get_yaxis_transform(), ha='left', va='bottom', color=DOF_COLORS[n],
                fontweight='bold', fontsize=plt.rcParams['axes.labelsize'])
    ax.set_yticks([v for j in range(len(names)) for v in (j * STEP, j * STEP + 0.5, j * STEP + 1)],
                  [l for _ in names for l in ('0', '0.5', '1')])
    ax.set_ylim(-0.15, (len(names) - 1) * STEP + 1.4)
    ax.set_xlim(0, trials['stop_time'].iloc[-1] - t0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (a.u.)')
    ax.set_title(f'{n} DOF hand control')


PANELS = {'title': panel_title, **{k: (lambda ax, d, k=k: panel_metric(ax, d, k)) for k in m}}


def make_figure(cfg):
    n = len(plot_dofs(cfg))
    panels = PANELS | {f'trace_{d}': partial(panel_trace, n=d) for d in plot_dofs(cfg)}
    first = plot_dofs(cfg)[0]
    return compose(layout_for(cfg), panels, load(cfg), (6 * n + 2, 17), 'fig6_human',
                   labels={f'trace_{first}': 'A', 'success_rate': 'B'}, stats_bars=cfg['stats_bars'][6],
                   height_ratios=[0.12, 1.3, 1, 1])


# ---------------------------------------------------------------- latency collection
def _ms(name, v):
    """Latency values in ms: `_s`/`sec` names, or anything whose median is < 1 (a fraction of a second), are seconds."""
    v = np.asarray(v, float).ravel()
    v = v[np.isfinite(v)]
    if len(v) and (re.search(r'(_s|_sec|seconds?)$', name) or np.median(v) < 1):
        v = v * 1000
    return v


def _is_latency(name):
    return 'latenc' in name.lower()


# vision pipeline latency files in a prediction folder: label -> (file, timestamp file). Values are seconds; the
# timestamp file (unix s, same length) is what lets one session-long folder be cut into the human runs' windows.
VISION = {
    'Camera top (capture)': ('TISlatencies_top', 'TISstarts_top'),
    'Camera front-right (capture)': ('TISlatencies_front_right', 'TISstarts_front_right'),
    'Vision: store get': ('batch3d_lat_store_get', 'batch3d_timestamps'),
    'Vision: inference': ('batch3d_lat_inference', 'batch3d_timestamps'),
    'Vision: triangulate': ('batch3d_lat_triangulate', 'batch3d_timestamps'),
    'Vision: angles': ('batch3d_lat_angles', 'batch3d_timestamps'),
    'Vision: queue put': ('batch3d_lat_queue_put', 'batch3d_timestamps'),
    'Vision: total': ('batch3d_lat_total', 'batch3d_timestamps'),
    'Sender step': ('sender_step_latencies', 'sender_timestamps'),
    'Vision end-to-end': ('sender_joint_e2e', 'sender_timestamps'),
}


def run_window(nwb_path):
    """(start, stop) of a run in unix seconds: NWB session start + the span of its continuous stream."""
    with NWBHDF5IO(str(nwb_path), 'r') as io:
        nwb = io.read()
        t = nwb.acquisition['continuous'].timestamps
        return nwb.session_start_time.timestamp(), nwb.session_start_time.timestamp() + (t[-1] - t[0])


def from_prediction_dir(folder, window):
    """Vision latencies (ms) of the prediction folder's frames that fall inside the run's unix-time window."""
    out = {}
    for label, (lat_f, ts_f) in VISION.items():
        try:
            lat, ts = np.load(Path(folder) / f'{lat_f}.npy'), np.load(Path(folder) / f'{ts_f}.npy')
        except FileNotFoundError as e:
            print(f'  missing {e.filename}')
            continue
        if len(lat) != len(ts):
            print(f'  {label}: {len(lat)} latencies vs {len(ts)} timestamps, skipped')
            continue
        inside = (ts >= window[0]) & (ts <= window[1])
        out[label] = lat[inside & np.isfinite(lat)] * 1000
    return out


def from_nwb(path):
    """What the human NWB has: start -> cue delay of every trial (ms). It has no latency-named series."""
    with NWBHDF5IO(str(path), 'r') as io:
        tr = io.read().trials.to_dataframe()
    return {'trial start -> cue': (tr['cue_time'] - tr['start_time']).to_numpy() * 1000}


def from_rdb(cfg, rdb):
    """Latencies derived from the node timestamps in the run's Redis dump (see human_rdb_latency.py)."""
    out = Path(tempfile.mkdtemp()) / 'lat.csv'
    for attempt in range(4):        # the redis-server / import occasionally dies at startup; a retry has always worked
        try:
            return _rdb_once(cfg, rdb, out)
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            print('  rdb extraction failed, retrying')


def _rdb_once(cfg, rdb, out):
    subprocess.run([str(Path(cfg['human']['rdb_python']).expanduser()), str(ROOT / 'scripts' / 'human_rdb_latency.py'),
                    str(rdb), str(out)], check=True,
                   env={k: v for k, v in os.environ.items() if not k.startswith('PYTHON') and k != 'LD_LIBRARY_PATH'})   # keep this env's libs out of the other one
    df = pd.read_csv(out)
    shutil.copy(str(out) + '.trials.json', DATA / f'human_rdb_trials_{Path(rdb).stem}.json')
    return {a: g['latency_ms'].to_numpy() for a, g in df.groupby('actor')}


def analyze_latency(cfg):
    rows = []
    for run, n_dof, nwb_path, rdb_path in run_files(cfg):
        found = {'nwb': from_nwb(nwb_path)}
        pred = cfg['human']['prediction_dir']
        if pred:
            found['prediction'] = from_prediction_dir(_path(pred), run_window(nwb_path))
        if rdb_path.exists():
            try:
                found['rdb'] = from_rdb(cfg, rdb_path)
            except Exception as e:
                print(f'could not read {rdb_path}: {e}')
        for source, series in found.items():
            for label, v in series.items():
                rows.append(pd.DataFrame({'run_id': run, 'n_dof': n_dof, 'source': source, 'actor': label, 'latency_ms': v}))
                print(f'  {run} [{source}] {label}: {len(v)} values, median {np.median(v):.1f} ms')
    pd.concat(rows, ignore_index=True).to_csv(DATA / 'human_latencies.csv', index=False)


def make_latency_figure(cfg):
    """One box per latency series (pooled over runs, every value a dot), coloured by where it came from."""
    lat = pd.read_csv(DATA / 'human_latencies.csv')
    lat = lat[lat['n_dof'].isin(plot_dofs(cfg))]
    keys = lat.groupby(['source', 'actor']).size().index.tolist()
    rng = np.random.default_rng(0)      # long series (10^4-10^5 frames): draw a random 3000 of each, keeps the SVG small
    groups = [(a, (lambda v: rng.choice(v, min(len(v), 3000), replace=False))(lat[(lat['source'] == s) & (lat['actor'] == a)]['latency_ms'].to_numpy()))
              for s, a in keys]

    def panel(ax, _):
        boxes(ax, groups, [LATENCY_SOURCES[s] for s, _ in keys], ylabel='Latency (ms)',
              title=f'{TITLE}: latency per actor\n({lat["run_id"].nunique()} runs: {", ".join(f"{n} DOF" for n in plot_dofs(cfg))})')
        ax.axhline(1000 / 30, color=COLORS['budget'], ls='--')
        ax.text(0.99, 1000 / 30, '30 FPS budget', transform=ax.get_yaxis_transform(), ha='right', va='bottom',
                color=COLORS['budget'], fontweight='bold')
        handles = [ax.scatter([], [], color=c, label=s) for s, c in LATENCY_SOURCES.items() if s in {k[0] for k in keys}]
        ax.legend(handles=handles, title='Source')

    return compose([['lat']], {'lat': panel}, None, (max(7, 1.5 * len(groups) + 3.5), 6.5), 'fig6_human_latency',
                   labels={}, stats_bars=False)
