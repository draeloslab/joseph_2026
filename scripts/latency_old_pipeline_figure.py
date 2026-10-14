"""Figure S2: end-to-end latency of the Freebie pipeline in the Monkey N sessions, next to the Monkey R runs (fig 3's
data, data/monkey_r/)."""
import numpy as np
import pandas as pd

import monkey_r_figures as mr
from utils import COLORS, DATA, FRAME_BUDGET_MS, boxes, compose, describe

ACTORS = [('frame_acq', 'Frame acquisition'), ('inference', 'Inference'),
          ('sender', 'Sender'), ('e2e', 'End-to-end')]
MONKEY_R_COLOR = '#1baf7a'

LAYOUT = [['hist', 'stages']]
SIZE = (18, 6.5)


def load():
    pooled = np.load(DATA / 'e2e_pooled.npz')
    r_runs = pd.read_csv(mr.OUT / 'latency_runs.csv')
    r_frames = np.load(mr.OUT / 'latency_e2e_frames.npz')
    return {'e2e': pooled['e2e_ms'], 'n_sessions': int(pooled['n_sessions']),
            'r_runs': r_runs, 'r_e2e': np.concatenate([r_frames[r] for r in r_runs['run_id']]),
            'stages': pd.read_csv(DATA / 'latency_stage_summary.csv')}


# ---------------------------------------------------------------- panels
def panel_hist(ax, d):
    # each monkey as % of its own frames (Monkey R has ~5x the frames)
    bins = np.arange(10, 45.25, 0.25)
    pct = lambda x: np.full(len(x), 100 / len(x))
    ax.hist(d['e2e'], bins=bins, weights=pct(d['e2e']), color=COLORS['e2e'], alpha=0.75, label='Monkey N')
    ax.hist(d['r_e2e'], bins=bins, weights=pct(d['r_e2e']), color=MONKEY_R_COLOR, alpha=0.6, label='Monkey R')
    describe(ax, 'Monkey N', d['e2e'], unit='frame', n_sessions=d['n_sessions'], median_ms=np.median(d['e2e']),
             pct_over_budget=100 * np.mean(d['e2e'] > FRAME_BUDGET_MS), budget_ms=FRAME_BUDGET_MS)
    describe(ax, 'Monkey R', d['r_e2e'], unit='frame', n_runs=len(d['r_runs']), median_ms=np.median(d['r_e2e']),
             n_sessions=d['r_runs']['run_id'].str[:10].nunique(), pct_over_budget_30fps=100 * np.mean(d['r_e2e'] > FRAME_BUDGET_MS))
    ax.set_xlim(10, 45)
    ax.set_xticks([10, 20, 30, 40])
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames (%)')
    ax.legend(loc='upper right')


def panel_stages(ax, d):
    # per actor: mean of each Monkey N session; Monkey R: end-to-end mean of each run
    groups = [(label, d['stages'][key]) for key, label in ACTORS] + [('Monkey R\nend-to-end', d['r_runs']['e2e'])]
    colors = [COLORS[k] for k, _ in ACTORS] + [MONKEY_R_COLOR]
    boxes(ax, groups, colors, ylabel='Processing time (ms)', describe_groups=False)
    for (label, v), unit in zip(groups, ['session'] * len(ACTORS) + ['run']):
        frames = d['stages']['n_frames'].sum() if unit == 'session' else d['r_runs']['frames'].sum()
        describe(ax, label, v, unit=unit, n_frames=frames)
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--')
    ax.set_ylim(0, FRAME_BUDGET_MS * 1.15)


PANELS = {'hist': panel_hist, 'stages': panel_stages}
LABELS = {'hist': 'A', 'stages': 'B'}


def make_figure(cfg):
    fig = compose(LAYOUT, PANELS, load(), SIZE, 'figS2_latency_old_pipeline', labels=LABELS,
                  stats_bars=cfg['stats_bars']['S2'])
    return fig
