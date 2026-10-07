"""
Figure 6: 3D human hand tracking on the BRAND triangulation set-up (2026-10-06, config human_bare, analysis in
human_tuning.py): pipeline-stage latency, bit rate per run, a 2-DOF hand-control example. make_figure_0930 (key
6_0930) is the earlier 2026-09-30 version: 1-4 DOF tasks on the hand3d node, online metrics per DOF count. The latency
figure and the vision diagnostics of the 09-30 runs are notebooks/human_vision/.

Runs, DOF labels and paths are under `human:` in config.yaml. `plot_dofs` (or generate_figs.py --dofs 1 2) picks
which DOF tasks are drawn; analyze() always processes every run, so switching needs no rerun.
"""
import json
import os
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
    are matched on start time (same clock); needs data/human_rdb_trials_<run>.json from save_rdb_trials.
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
    save_rdb_trials(cfg)        # first: each run's full RDB target sets, which read_trials() uses
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
    runs['sessions'] = Path(cfg['human']['day_dir']).name       # one session (day) for every run, for n_sessions
    return {'runs': runs[runs['n_dof'].isin(keep)], 'trials': trials[trials['n_dof'].isin(keep)], 'dofs': keep, 'examples': ex}


def panel_metric(ax, d, metric):
    groups = [(f'{n} DOF', d['runs'][d['runs']['n_dof'] == n], c) for n, c in DOF_COLORS.items() if n in d['dofs']]
    metric_panel(ax, groups, metric, stats_trials=d['trials'])     # every trial shown, tested on run means


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


def make_figure_0930(cfg):
    """The 2026-09-30 version (1-4 DOF tasks, hand3d node): online metrics per DOF count. Key 6_0930."""
    n = len(plot_dofs(cfg))
    panels = PANELS | {f'trace_{d}': partial(panel_trace, n=d) for d in plot_dofs(cfg)}
    first = plot_dofs(cfg)[0]
    return compose(layout_for(cfg), panels, load(cfg), (6 * n + 2, 17), 'human_hand_control_0930',
                   labels={f'trace_{first}': 'A', 'success_rate': 'B'}, stats_bars=cfg['stats_bars']['human_0930'],
                   height_ratios=[0.12, 1.3, 1, 1])


# ---------------------------------------------------------------- fig 6: BRAND triangulation set-up (2026-10-06)
# stage label (human_tuning) -> (tick label, colour): improv stages blue, BRAND stages green, end-to-end grey
STAGES = {'Camera': ('Camera', '#0a11c4'), 'Store get': ('Store\nget', '#2f5fd0'), 'Inference': ('Infer-\nence', '#519bc6'),
          'Triangulation': ('Triangu-\nlation', '#2ccded'), 'Joint angles': ('Joint\nangles', '#8fd6ea'),
          'Sender': ('Sender', '#b9e3f0'), 'Vision end-to-end': ('Vision\ntotal', '#757272'),
          'vision -> vision_dofs': ('BRAND\nDOF node', '#1baf7a'), 'vision_dofs -> cursor': ('BRAND\nmux', '#7fd3a8')}
RUN_COLOR = '#2a78d6'
DOF_SHOWN = {'index': 'Index', 'mrs': 'Middle, ring and pinky'}


def load_bare(cfg):
    import human_tuning as ht
    runs = cfg['human_bare']['figure_runs']
    lat = pd.read_csv(ht.OUT / 'latency.csv')
    tr = pd.read_csv(ht.OUT / 'trials.csv')
    return {'runs': runs, 'settings': cfg['human_bare']['runs'], 'lat': lat[lat['run_id'].isin(runs)],
            'trials': tr[tr['run_id'].isin(runs)], 'ex': dict(np.load(ht.OUT / 'example.npz')),
            'ex_trials': pd.read_csv(ht.OUT / 'example_trials.csv')}


def panel_stage_latency(ax, d):
    """One box per pipeline stage: each run's mean (like fig 2B), camera -> cursor."""
    means = d['lat'].groupby(['stage', 'run_id'])['latency_ms'].mean()
    boxes(ax, [(STAGES[k][0], means[k]) for k in STAGES], [c for _, c in STAGES.values()], ylabel='Latency (ms)', unit='run')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.axhline(1000 / 60, color=COLORS['budget'], ls='--')
    ax.text(0.99, 1000 / 60, '60 FPS budget', transform=ax.get_yaxis_transform(), ha='right', va='bottom',
            color=COLORS['budget'], fontweight='bold')
    ax.set_ylim(0, None)


def panel_bitrate(ax, d):
    """Bit rate of every successful trial, one box per run (its smoothing setting)."""
    short = lambda txt: txt.replace('One-Euro + glide', '1€ + glide').replace('2 hands, ', '2 hands\n')
    groups = [(f"Run {int(r[-3:])}\n" + short(d['settings'][r]['setting']).replace(', ', '\n'),
               d['trials'].loc[d['trials']['run_id'] == r, 'bitrate_bits_per_s']) for r in d['runs']]
    boxes(ax, groups, [RUN_COLOR] * len(groups), ylabel='Bit rate (bits/s)', unit='trial')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylim(0, None)


def panel_trace_2dof(ax, d):
    """2-DOF hand control: cursor (blue) through the target boxes (grey hit / red miss), one stacked row per DOF."""
    ex, trials = d['ex'], d['ex_trials']
    t0 = trials['start_time'].iloc[0]
    names = [DOF_SHOWN.get(str(x), str(x)) for x in ex['dofs']]
    for j, name in enumerate(names):
        off = j * STEP
        for _, row in trials.iterrows():
            r = row['target_radius']
            ax.add_patch(plt.Rectangle((row['start_time'] - t0, off + row[f'target_{j}'] - r), row['stop_time'] - row['start_time'],
                                       2 * r, facecolor=COLORS['target_hit'] if row['success'] else COLORS['target_miss'],
                                       edgecolor='#6f6f6f', ls='--', lw=0.6 * plt.rcParams['axes.linewidth'], alpha=0.7, zorder=1))
        ax.plot(ex['t'] - t0, off + ex['x'][:, j], color=RUN_COLOR, zorder=3)
        ax.text(0.005, off + 1.04, name, transform=ax.get_yaxis_transform(), ha='left', va='bottom', color=RUN_COLOR,
                fontweight='bold')
    ax.set_yticks([v for j in range(len(names)) for v in (j * STEP, j * STEP + 1)], [l for _ in names for l in ('0', '1')])
    ax.set_ylim(-0.15, (len(names) - 1) * STEP + 1.3)
    ax.set_xlim(0, trials['stop_time'].iloc[-1] - t0)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Position (a.u.)')


def make_figure(cfg):
    """Fig 6: the BRAND triangulation set-up (config human_bare): stage latency, bit rate per run, a 2-DOF example."""
    layout = [['lat', 'lat', 'lat'], ['bit', 'bit', 'trace']]
    panels = {'lat': panel_stage_latency, 'bit': panel_bitrate, 'trace': panel_trace_2dof}

    def place(fig, axd):
        # the trace fills the space right of the bit-rate boxes
        b, t, l = axd['bit'].get_position(), axd['trace'].get_position(), axd['lat'].get_position()
        x0 = b.x1 + 0.07
        axd['trace'].set_position([x0, t.y0, l.x0 + max(l.width, 0.85) - x0, t.height * 0.92])   # clear A's tick labels

    return compose(layout, panels, load_bare(cfg), (18, 10), 'figS3_human_2dof_hand_control', labels={'lat': 'A', 'bit': 'B', 'trace': 'C'},
                   stats_bars=cfg['stats_bars']['S3'], post_layout=place)


# ---------------------------------------------------------------- the RDB's full target sets
def rdb_latency_csv(cfg, rdb):
    """Run human_rdb_latency.py on one run's Redis dump: saves data/human_rdb_trials_<run>.json (the full target sets
    read_trials() needs) and returns the path of its per-actor latency csv (used by notebooks/human_vision/)."""
    out = Path(tempfile.mkdtemp()) / 'lat.csv'
    for attempt in range(4):        # the redis-server / import occasionally dies at startup; a retry has always worked
        try:
            subprocess.run([str(Path(cfg['human']['rdb_python']).expanduser()), str(ROOT / 'scripts' / 'human_rdb_latency.py'),
                            str(rdb), str(out)], check=True,
                           env={k: v for k, v in os.environ.items() if not k.startswith('PYTHON') and k != 'LD_LIBRARY_PATH'})   # keep this env's libs out of the other one
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            print('  rdb extraction failed, retrying')
    shutil.copy(str(out) + '.trials.json', DATA / f'human_rdb_trials_{Path(rdb).stem}.json')
    return out


def save_rdb_trials(cfg):
    for run, _, _, rdb_path in run_files(cfg):
        if rdb_path.exists():
            try:
                rdb_latency_csv(cfg, rdb_path)
            except Exception as e:
                print(f'could not read {rdb_path}: {e}')
