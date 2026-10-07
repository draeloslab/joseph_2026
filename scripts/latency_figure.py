"""Figure 2 supplement: end-to-end latency of the Freebie pipeline in the earlier sessions, next to the War Machine runs
(fig 3's data, data/war_machine/)."""
import numpy as np
import pandas as pd

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

    wmf.analyze_fig2(cfg)      # the War Machine runs shown next to the earlier sessions


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
    # each group as % of its own frames: Monkey R has ~5x the frames and would flatten Monkey N; the legend only names the colours
    bins = np.arange(10, 45.25, 0.25)
    pct = lambda x: np.full(len(x), 100 / len(x))
    ax.hist(d['e2e'], bins=bins, weights=pct(d['e2e']), color=COLORS['e2e'], alpha=0.75, label='Monkey N')
    ax.hist(d['wm_e2e'], bins=bins, weights=pct(d['wm_e2e']), color=WM_COLOR, alpha=0.6, label='Monkey R')
    describe(ax, 'Monkey N', d['e2e'], unit='frame', n_sessions=d['n_sessions'], median_ms=np.median(d['e2e']),
             pct_over_budget=100 * np.mean(d['e2e'] > FRAME_BUDGET_MS), budget_ms=FRAME_BUDGET_MS)
    describe(ax, 'Monkey R', d['wm_e2e'], unit='frame', n_runs=len(d['wm_runs']), median_ms=np.median(d['wm_e2e']),
             n_sessions=d['wm_runs']['run_id'].str[:10].nunique(), pct_over_budget_30fps=100 * np.mean(d['wm_e2e'] > FRAME_BUDGET_MS))
    ax.set_xlim(10, 45)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames (%)')
    ax.legend(loc='upper right')


def panel_actors(ax, d):
    actor_panel(ax, d['actors'], f"Per-actor processing time ({d['actors']['session']})")


def panel_stages(ax, d):
    # per actor: mean over the earlier sessions (one point each); War Machine: end-to-end mean of each run
    groups = [(label, d['stages'][key]) for key, label in ACTORS] + [('Monkey R\nend-to-end', d['wm_runs']['e2e'])]
    colors = [COLORS[k] for k, _ in ACTORS] + [WM_COLOR]
    boxes(ax, groups, colors, ylabel='Processing time (ms)', describe_groups=False)
    for (label, v), unit in zip(groups, ['session'] * len(ACTORS) + ['run']):
        frames = d['stages']['n_frames'].sum() if unit == 'session' else d['wm_runs']['frames'].sum()
        describe(ax, label, v, unit=unit, n_frames=frames)
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--')
    ax.set_ylim(0, FRAME_BUDGET_MS * 1.15)


# panel_actors (the per-actor example trace, panel B) isn't in PANELS/LAYOUT right now -- see LAYOUT above
# -- but stays defined and reachable if it comes back (relabel then: it was B, stages was C).
PANELS = {'hist': panel_hist, 'stages': panel_stages}
LABELS = {'hist': 'A', 'stages': 'B'}


def make_figure(cfg):
    fig = compose(LAYOUT, PANELS, load(), SIZE, 'figS1_latency_old_pipeline', labels=LABELS,
                  stats_bars=cfg['stats_bars']['S1'])
    return fig
