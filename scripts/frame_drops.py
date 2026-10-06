"""
Dropped frames of the vision pipeline, from its per-frame logs (~/predictions/<YYYYMMDD>/<session>/), for fig 2
(War Machine) and 2_supplement (the earlier sessions). Two kinds:
  - pipeline: frames the camera reader numbered that never reached the output (gaps in true_e2e_<cam>_frame_nums,
    e.g. the processor still busy with the previous frame).
  - camera:   frames the camera never delivered (gaps of more than one frame period in TIScapture_<camera>, or in
              TISstarts_<camera> in the earlier sessions, which did not save capture times).
Each drop event is placed in time (s from the window start) and carries the running % dropped of the frames up to it.
Stretches of the window with no camera frame logged at all for more than OFFLINE_S (the pipeline was not running, e.g.
restarted between two prediction folders) are not drops; they are reported separately as offline spans.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import compose, describe

KIND_COLOR = {'pipeline': '#d62728', 'camera': '#2a78d6'}
OFFLINE_S = 1.0
WARMUP_FRAMES = 5      # output frames skipped at the start of every prediction folder (GPU warm-up jumps)


def session_log(folder, cam, tis_cam):
    """{'pipeline': (unix t, frame ids), 'camera': (unix t, frame ids), 'period': s} of one prediction folder.
    Camera ids are synthesised from the capture times (one per frame period), so a gap of k periods is k - 1 drops."""
    folder = Path(folder)
    L = lambda n: np.load(folder / f'{n}.npy')
    cap = L(f'TIScapture_{tis_cam}') if (folder / f'TIScapture_{tis_cam}.npy').exists() else L(f'TISstarts_{tis_cam}')
    period = float(np.median(np.diff(cap)))
    cam_ids = np.r_[0, np.cumsum(np.maximum(np.round(np.diff(cap) / period), 1))].astype(int)
    # the first output frames are the GPU warm-up: their frame-number jumps are not drops
    w = WARMUP_FRAMES
    return {'pipeline': (L(f'true_e2e_{cam}_timestamps')[w:], L(f'true_e2e_{cam}_frame_nums')[w:].astype(int)),
            'camera': (cap, cam_ids), 'period': period}


def _events(t, ids, lo, hi):
    w = (t >= lo) & (t <= hi)
    t, ids = t[w], ids[w]
    if len(ids) < 2:
        return pd.DataFrame(), 0
    step = np.diff(ids)
    k = np.flatnonzero(step > 1)
    n = step[k] - 1
    so_far = ids[k + 1] - ids[0] + 1
    ev = pd.DataFrame({'t_s': t[k + 1] - lo, 'n_dropped': n, 'frames_so_far': so_far,
                       'cum_dropped': np.cumsum(n), 'cum_pct': 100 * np.cumsum(n) / so_far})
    return ev, int(ids[-1] - ids[0] + 1)


def drops(logs, lo, hi):
    """Drop events + summary of the logs' frames inside [lo, hi] (unix s). logs: session_log() dicts (a run can
    overlap more than one prediction folder)."""
    events, summary = [], {'duration_s': hi - lo}
    for kind in ('pipeline', 'camera'):
        total, dropped = 0, 0
        for log in sorted(logs, key=lambda log: log[kind][0][0]):      # running totals carry across folders
            ev, n = _events(*log[kind], lo, hi)
            if len(ev):
                ev['frames_so_far'] += total
                ev['cum_dropped'] += dropped
                ev['cum_pct'] = 100 * ev['cum_dropped'] / ev['frames_so_far']
                events.append(ev.assign(kind=kind, frac_of_run=ev['t_s'] / (hi - lo)))
                dropped += int(ev['n_dropped'].sum())
            total += n
        summary |= {f'{kind}_frames': total, f'{kind}_dropped': dropped,
                    f'{kind}_pct': 100 * dropped / total if total else np.nan,
                    f'{kind}_events': sum(len(e) for e in events if e['kind'].iloc[0] == kind)}
    t = np.sort(np.concatenate([log['camera'][0] for log in logs])) if logs else np.array([])
    t = np.r_[lo, t[(t >= lo) & (t <= hi)], hi]
    k = np.flatnonzero(np.diff(t) > OFFLINE_S)
    spans = [(t[i] - lo, t[i + 1] - lo) for i in k]
    summary |= {'offline_s': sum(b - a for a, b in spans),
                'offline_spans_s': ' '.join(f'{a:.1f}-{b:.1f}' for a, b in spans)}
    events = [e for e in events if len(e)]
    ev = pd.concat(events, ignore_index=True) if events else pd.DataFrame(columns=['t_s', 'n_dropped', 'frames_so_far', 'cum_dropped', 'cum_pct', 'kind', 'frac_of_run'])
    if len(ev):
        big = ev.loc[ev['n_dropped'].idxmax()]
        summary |= {'largest_drop_frames': int(big['n_dropped']), 'largest_drop_at_s': big['t_s'], 'largest_drop_kind': big['kind']}
    return ev, summary


def print_summary(runs, id_col):
    cols = [id_col, 'duration_s', 'pipeline_frames', 'pipeline_dropped', 'pipeline_pct', 'camera_dropped', 'camera_pct',
            'offline_s', 'offline_spans_s']
    print(runs[[c for c in cols if c in runs]].to_string(index=False, float_format=lambda v: f'{v:.2f}'))
    for kind in ('pipeline', 'camera'):
        tot, d = runs[f'{kind}_frames'].sum(), runs[f'{kind}_dropped'].sum()
        print(f'{kind}: {d} of {tot} frames dropped ({100 * d / max(tot, 1):.2f}%)')


def make_figure(runs, events, id_col, name, title):
    """One row per run: its span (grey), each drop event at its time (marker area ~ frames dropped, red pipeline,
    blue camera), offline spans hatched, and the run's total % dropped on the right."""
    ids = list(runs[id_col])

    def panel(ax, _):
        for y, rid in enumerate(ids):
            r = runs.iloc[y]
            ax.plot([0, r['duration_s']], [y, y], color='#d0d0d0', lw=3, solid_capstyle='butt', zorder=1)
            for span in (r['offline_spans_s'] if isinstance(r.get('offline_spans_s'), str) else '').split():
                a, b = map(float, span.split('-'))
                ax.add_patch(plt.Rectangle((a, y - 0.3), b - a, 0.6, facecolor='none', edgecolor='#888888',
                                           hatch='///', lw=0.8, zorder=2))
            ev = events[events[id_col] == rid]
            for kind, g in ev.groupby('kind'):
                ax.scatter(g['t_s'], np.full(len(g), y), s=12 + 10 * g['n_dropped'], color=KIND_COLOR[kind],
                           alpha=0.6, linewidths=0, zorder=3)
            ax.text(1.01, y, f"{r['pipeline_pct']:.2f}% | {r['camera_pct']:.2f}%", transform=ax.get_yaxis_transform(),
                    va='center', fontsize=plt.rcParams['ytick.labelsize'])
        ax.set_yticks(range(len(ids)), ids)
        ax.set_ylim(len(ids) - 0.5, -0.5)
        ax.set_xlim(0, None)
        ax.set_xlabel('Time in run (s)')
        ax.set_title(title, loc='left')
        ax.text(1.01, 1.0, 'dropped:\npipeline | camera', transform=ax.transAxes, va='bottom', fontweight='bold',
                fontsize=plt.rcParams['ytick.labelsize'])
        handles = [ax.scatter([], [], s=40, color=c, label=f'{k} drop') for k, c in KIND_COLOR.items()]
        handles.append(plt.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#888888', hatch='///', label='pipeline offline'))
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0), ncol=3, borderaxespad=0.3)
        ax.grid(axis='x', color='#ececec')
        for kind in ('pipeline', 'camera'):
            describe(ax, f'{kind} drops (% of frames, per run)', runs[f'{kind}_pct'], unit='run', n_runs=len(runs),
                     n_frames=runs[f'{kind}_frames'].sum(), n_dropped=runs[f'{kind}_dropped'].sum(),
                     n_events=runs[f'{kind}_events'].sum(),
                     pct_all_frames=100 * runs[f'{kind}_dropped'].sum() / max(runs[f'{kind}_frames'].sum(), 1))
        if 'offline_s' in runs:
            describe(ax, 'pipeline offline (s, per run)', runs['offline_s'], unit='run', n_runs=len(runs),
                     runs_with_offline=int((runs['offline_s'] > 0).sum()), total_offline_s=runs['offline_s'].sum())

    return compose([['drops']], {'drops': panel}, None, (16, 2.5 + 0.42 * len(ids)), name, labels={}, stats_bars=False)
