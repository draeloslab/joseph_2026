"""Figure 2: end-to-end latency of the Freebie pipeline, plus the bonus real-time (Sept 2026) figure."""
import numpy as np
import pandas as pd

from utils import COLORS, DATA, FRAME_BUDGET_MS, boxes, compose, load_master, significance

ACTORS = [('frame_acq', 'Frame acquisition'), ('inference', 'Inference'),
          ('sender', 'Sender'), ('e2e', 'End-to-end')]

LAYOUT = [['hist', 'stages']]   # panel B (actors, the per-actor example trace) hidden for now: A and C side by side
SIZE = (18, 6.5)
RT_LAYOUT = [['hist', 'actors', 'old_vs_new']]
RT_SIZE = (21, 5.5)


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

    analyze_rt(cfg)      # fig2's hist/stages panels now also show the real-time tuned session


def analyze_rt(cfg):
    sid = cfg['examples']['rt_best_session']
    np.savez(DATA / 'rt_best_actors.npz', session=sid, **load_master(cfg)['rt_runs'][sid])


def load(rt=False):
    pooled = np.load(DATA / 'e2e_pooled.npz')
    d = {'e2e': pooled['e2e_ms'], 'n_sessions': int(pooled['n_sessions']), 'rt': dict(np.load(DATA / 'rt_best_actors.npz'))}
    if rt:
        return d
    return d | {'stages': pd.read_csv(DATA / 'latency_stage_summary.csv'),
                'actors': dict(np.load(DATA / 'latency_example_actors.npz'))}


# ---------------------------------------------------------------- panels
def e2e_hist(ax, e2e_ms, title, color):
    ax.hist(e2e_ms, bins=np.arange(0, 80.5, 0.5), color=color, alpha=0.9)
    ax.axvline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--', label=f'30 FPS budget ({FRAME_BUDGET_MS:.1f} ms)')
    ax.axvline(np.median(e2e_ms), color='#111111', ls=':', label=f'Median {np.median(e2e_ms):.1f} ms')
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames')
    ax.set_title(title)
    ax.legend()


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
    ax.hist(d['e2e'], bins=np.arange(0, 80.5, 0.5), color=COLORS['e2e'], alpha=0.6,
            label=f"All sessions (n={d['n_sessions']}, {len(d['e2e']):,} frames)")
    ax.hist(d['rt']['e2e'], bins=np.arange(0, 80.5, 0.5), color=COLORS['new_rt'], alpha=0.6,
            label=f"Real-time tuned ({d['rt']['session']}, {len(d['rt']['e2e']):,} frames)")
    ax.axvline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--', label=f'30 FPS budget ({FRAME_BUDGET_MS:.1f} ms)')
    ax.axvline(np.median(d['e2e']), color='#111111', ls=':', label=f"Median (all sessions) {np.median(d['e2e']):.1f} ms")
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames')
    ax.set_title('End-to-end latency distribution')
    ax.legend()


def panel_actors(ax, d):
    actor_panel(ax, d['actors'], f"Per-actor processing time ({d['actors']['session']})")


def panel_stages(ax, d):
    # per actor: mean over sessions (one point each). Real-time tuned: only one session exists, so its box
    # is every individual frame's e2e latency instead (all frames shown as dots), not a mean.
    groups = [(label, d['stages'][key]) for key, label in ACTORS] + [('Real-time tuned', d['rt']['e2e'])]
    colors = [COLORS[k] for k, _ in ACTORS] + [COLORS['new_rt']]
    boxes(ax, groups, colors, ylabel='Processing time (ms)',
          title=f"Per actor: session means (n={len(d['stages'])})\nvs real-time tuned: per frame ({d['rt']['session']})")
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--', label='30 FPS budget')
    ax.set_ylim(0, max(FRAME_BUDGET_MS * 1.15, np.percentile(d['rt']['e2e'], 99.5)))
    ax.legend(loc='center left')


def panel_rt_hist(ax, d):
    e2e_hist(ax, d['rt']['e2e'], f"Real-time tuned ({d['rt']['session']})", COLORS['new_rt'])


def panel_rt_actors(ax, d):
    actor_panel(ax, d['rt'], 'Per-actor processing time')


def panel_old_vs_new(ax, d):
    data = [d['e2e'], d['rt']['e2e']]
    bp = ax.boxplot(data, tick_labels=['Previous sessions\n(pooled)', 'Real-time tuned'], widths=0.55,
                    patch_artist=True, showfliers=False, medianprops={'color': '#111111', 'lw': 2})
    for patch, c in zip(bp['boxes'], (COLORS['old'], COLORS['new_rt'])):
        patch.set(facecolor=c, alpha=0.5, edgecolor='#444444')
    for i, x in enumerate(data):
        ax.text(i + 1, np.percentile(x, 75) + 1, f'mean {x.mean():.1f}', ha='center', fontweight='bold')
    ax.axhline(FRAME_BUDGET_MS, color=COLORS['budget'], ls='--')
    ax.set_ylim(0, None)
    ax.set_ylabel('End-to-end latency (ms)')
    ax.set_title('Old vs new pipeline')
    significance(ax, [('Previous sessions', data[0]), ('Real-time tuned', data[1])])


# panel_actors (the per-actor example trace, panel B) isn't in PANELS/LAYOUT right now -- see LAYOUT above
# -- but stays defined and reachable if it comes back; LABELS keeps A/C so it's clear B was skipped on purpose.
PANELS = {'hist': panel_hist, 'stages': panel_stages}
LABELS = {'hist': 'A', 'stages': 'C'}
RT_PANELS = {'hist': panel_rt_hist, 'actors': panel_rt_actors, 'old_vs_new': panel_old_vs_new}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'fig2_latency', labels=LABELS, stats_bars=cfg['stats_bars'][2])


def make_rt_figure(cfg):
    return compose(RT_LAYOUT, RT_PANELS, load(rt=True), RT_SIZE, 'fig2_bonus_realtime',
                   stats_bars=cfg['stats_bars']['rt_bonus'])
