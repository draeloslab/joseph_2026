"""Figure 2 supplement: end-to-end latency of the Freebie pipeline in the earlier sessions, next to the War Machine runs
(fig 2's data, data/war_machine/), plus where frames were dropped in the earlier sessions."""
from pathlib import Path

import numpy as np
import pandas as pd

import frame_drops
import war_machine_figures as wmf
from utils import COLORS, DATA, FRAME_BUDGET_MS, boxes, compose, describe, load_master

ACTORS = [('frame_acq', 'Frame acquisition'), ('inference', 'Inference'),
          ('sender', 'Sender'), ('e2e', 'End-to-end')]
WM_COLOR = '#1baf7a'

LAYOUT = [['hist', 'stages']]   # panel B (actors, the per-actor example trace) hidden for now: A and C side by side
SIZE = (18, 6.5)


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    ex = cfg['examples']
    skip = ex['latency_skip_first_frames']
    sessions = load_master(cfg)['sessions']

    pooled = [s['e2e_latency_s'][skip:] * 1000 for s in sessions.values() if s['e2e_latency_s'] is not None]
    np.savez(DATA / 'e2e_pooled.npz', e2e_ms=np.concatenate(pooled), n_sessions=len(pooled))

    # per session: mean processing time of each actor, and of the true Freebie frame -> UART latency
    rows = [{'session': sid, 'n_frames': len(s['actors']['e2e']), **{k: s['actors'][k][skip:].mean() for k, _ in ACTORS}}
            for sid, s in sessions.items() if s['actors'] is not None and len(s['actors']['e2e']) >= 100]
    pd.DataFrame(rows).to_csv(DATA / 'latency_stage_summary.csv', index=False)

    sid, n = ex['latency_session'], ex['latency_window_frames'] - 1
    np.savez(DATA / 'latency_example_actors.npz', session=sid,
             **{k: v[:n] for k, v in sessions[sid]['actors'].items()})

    analyze_drops(cfg, [r['session'] for r in rows])
    wmf.analyze_fig2(cfg)      # the War Machine runs shown next to the earlier sessions


def _session_log(cfg, sid):
    """The prediction folder of an earlier session; its camera is the one the master's per-actor logs came from
    (the true_e2e / TIS log whose length matches them: cam0 / 'right' on the sessions checked)."""
    folder = Path(cfg['war_machine']['predictions']).expanduser() / sid.split('-')[0] / sid
    n_e2e = len(load_master(cfg)['sessions'][sid]['actors']['e2e'])
    n_acq = len(load_master(cfg)['sessions'][sid]['actors']['frame_acq'])
    cams = [c for c in ('cam0', 'cam2') if (folder / f'true_e2e_{c}_frame_nums.npy').exists()]
    tis = [c for c in ('right', 'top') if (folder / f'TISstarts_{c}.npy').exists()]
    if not cams or not tis:
        return None
    cam = min(cams, key=lambda c: abs(len(np.load(folder / f'true_e2e_{c}.npy', mmap_mode='r')) - 1 - n_e2e))
    tc = min(tis, key=lambda c: abs(len(np.load(folder / f'TISstarts_{c}.npy', mmap_mode='r')) - n_acq))
    return frame_drops.session_log(folder, cam, tc)


def analyze_drops(cfg, sids):
    """data/latency_dropped_frames_{sessions,events}.csv: dropped frames of every session with per-actor logs, over
    the whole session after the warm-up frames (frame_drops.py)."""
    skip = cfg['examples']['latency_skip_first_frames']
    rows, events = [], []
    for sid in sids:
        log = _session_log(cfg, sid)
        if log is None:
            print(f'{sid}: no frame-number logs in the prediction folder, dropped frames skipped')
            continue
        t = log['pipeline'][0]
        ev, summary = frame_drops.drops([log], t[skip], t[-1])
        rows.append({'session': sid, **summary})
        events.append(ev.assign(session=sid))
    runs = pd.DataFrame(rows)
    runs.to_csv(DATA / 'latency_dropped_frames_sessions.csv', index=False)
    pd.concat([e for e in events if len(e)] or events[:1], ignore_index=True).to_csv(DATA / 'latency_dropped_frames_events.csv', index=False)
    frame_drops.print_summary(runs, 'session')


def load():
    pooled = np.load(DATA / 'e2e_pooled.npz')
    wm_runs = pd.read_csv(wmf.OUT / 'latency_runs.csv')
    wm_frames = np.load(wmf.OUT / 'latency_e2e_frames.npz')
    return {'e2e': pooled['e2e_ms'], 'n_sessions': int(pooled['n_sessions']),
            'wm_runs': wm_runs, 'wm_e2e': np.concatenate([wm_frames[r] for r in wm_runs['run_id']]),
            'stages': pd.read_csv(DATA / 'latency_stage_summary.csv'),
            'actors': dict(np.load(DATA / 'latency_example_actors.npz'))}


# ---------------------------------------------------------------- panels
def actor_panel(ax, a, title):
    for key, label in ACTORS:
        ax.plot(a[f'{key}_t'], a[key], color=COLORS[key], lw=1.2, alpha=0.85,
                label=f'{label} (mean {a[key].mean():.1f} ms)')
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--')
    ax.set_ylim(0, 100)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Processing time (ms)')
    ax.set_title(title)
    ax.legend(loc='upper right')


def panel_hist(ax, d):
    bins = np.arange(0, 80.5, 0.5)
    ax.hist(d['e2e'], bins=bins, color=COLORS['e2e'], alpha=0.6,
            label=f"Earlier sessions (n={d['n_sessions']}, {len(d['e2e']):,} frames)")
    ax.hist(d['wm_e2e'], bins=bins, color=WM_COLOR, alpha=0.6,
            label=f"War Machine runs (n={len(d['wm_runs'])}, {len(d['wm_e2e']):,} frames)")
    ax.axvline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--', label=f'30 FPS budget ({FRAME_BUDGET_MS:.1f} ms)')
    describe(ax, 'Earlier sessions', d['e2e'], unit='frame', n_sessions=d['n_sessions'],
             pct_over_budget=100 * np.mean(d['e2e'] > FRAME_BUDGET_MS), budget_ms=FRAME_BUDGET_MS)
    describe(ax, 'War Machine runs', d['wm_e2e'], unit='frame', n_runs=len(d['wm_runs']),
             n_sessions=d['wm_runs']['run_id'].str[:10].nunique(), pct_over_budget_30fps=100 * np.mean(d['wm_e2e'] > FRAME_BUDGET_MS))
    ax.axvline(np.median(d['e2e']), color='#111111', ls=':', label=f"Median (earlier sessions) {np.median(d['e2e']):.1f} ms")
    ax.axvline(np.median(d['wm_e2e']), color=WM_COLOR, ls=':', label=f"Median (War Machine) {np.median(d['wm_e2e']):.1f} ms")
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames')
    ax.set_title('End-to-end latency distribution')
    ax.legend()


def panel_actors(ax, d):
    actor_panel(ax, d['actors'], f"Per-actor processing time ({d['actors']['session']})")


def panel_stages(ax, d):
    # per actor: mean over the earlier sessions (one point each); War Machine: end-to-end mean of each run
    groups = [(label, d['stages'][key]) for key, label in ACTORS] + [('War Machine\nend-to-end', d['wm_runs']['e2e'])]
    colors = [COLORS[k] for k, _ in ACTORS] + [WM_COLOR]
    boxes(ax, groups, colors, ylabel='Processing time (ms)', describe_groups=False,
          title=f"Per actor: session means (n={len(d['stages'])})\nvs War Machine: run means (n={len(d['wm_runs'])})")
    for (label, v), unit in zip(groups, ['session'] * len(ACTORS) + ['run']):
        frames = d['stages']['n_frames'].sum() if unit == 'session' else d['wm_runs']['frames'].sum()
        describe(ax, label, v, unit=unit, n_frames=frames)
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--')
    ax.set_ylim(0, FRAME_BUDGET_MS * 1.15)
    # label the line directly (a legend with one entry floated mid-panel)
    ax.text(0.99, FRAME_BUDGET_MS, f'30 FPS budget ({FRAME_BUDGET_MS:.1f} ms)', transform=ax.get_yaxis_transform(),
            ha='right', va='bottom', color=COLORS['budget'], fontweight='bold')


# panel_actors (the per-actor example trace, panel B) isn't in PANELS/LAYOUT right now -- see LAYOUT above
# -- but stays defined and reachable if it comes back (relabel then: it was B, stages was C).
PANELS = {'hist': panel_hist, 'stages': panel_stages}
LABELS = {'hist': 'A', 'stages': 'B'}


def make_figure(cfg):
    fig = compose(LAYOUT, PANELS, load(), SIZE, 'fig2_supplement_latency', labels=LABELS,
                  stats_bars=cfg['stats_bars']['2_supplement'])
    runs = pd.read_csv(DATA / 'latency_dropped_frames_sessions.csv')
    ev = pd.read_csv(DATA / 'latency_dropped_frames_events.csv')
    tot = lambda k: 100 * runs[f'{k}_dropped'].sum() / runs[f'{k}_frames'].sum()
    frame_drops.make_figure(runs, ev, 'session', 'fig2_supplement_dropped_frames',
                            f'Earlier sessions dropped frames (all: pipeline {tot("pipeline"):.2f}%, camera {tot("camera"):.2f}%)')
    return fig
