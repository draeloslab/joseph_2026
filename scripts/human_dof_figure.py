"""
Figure S3: human 2-4 DOF hand control with 3D hand tracking (four cameras, triangulated), one run per DOF count.
A: the task screen (assets/figS3_task_screen.png), B: bit rate per DOF count, C: end-to-end vision latency.
Data: data/figS3_human/.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import COLORS, DATA, ROOT, STYLE, boxes, compose, describe

DOF_COLORS = {1: '#2a78d6', 2: '#1baf7a', 3: '#eb6834', 4: '#b455e0'}
OUT = DATA / 'figS3_human'


def load():
    return {'runs': pd.read_csv(OUT / 'runs.csv'), 'trials': pd.read_csv(OUT / 'trials.csv'),
            'e2e': {r: v.min(axis=1) for r, v in np.load(OUT / 'e2e_cam_frames.npz').items()},
            'image': plt.imread(ROOT / 'assets' / 'figS3_task_screen.png')}


def _dofs(d):
    return sorted(d['runs']['n_dof'])


def panel_dof_latency(ax, d):
    """End-to-end vision latency as in fig 3 (send time - camera_start), measured from the newest of the four cameras' frames
    in each step (the oldest is up to ~10 ms earlier: the cameras free-run), every step inside a trial, % per DOF count."""
    bins = np.arange(10, 20.01, 0.25)
    budget = 1000 / 60
    handles = []
    for n, (run, v) in zip(_dofs(d), [(r, d['e2e'][r]) for r in d['runs'].sort_values('n_dof')['run_id']]):
        ax.hist(v, bins=bins, weights=np.full(len(v), 100 / len(v)), color=DOF_COLORS[n], alpha=0.6, label=f'{n} DOF')
        describe(ax, f'{n} DOF', v, 'frame', n_runs=1, median_ms=np.median(v), pct_over_budget=100 * np.mean(v > budget), budget_ms=budget)
    ax.axvline(budget, ymax=0.6, color=COLORS['budget'], ls='--')      # stops below the legend
    ax.plot([], [], color=COLORS['budget'], ls='--', label='60 fps interval')
    ax.set_xlim(10, 20)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.3)       # headroom: the legend sits above the bars
    ax.set_xticks([10, 15, 20])
    ax.set_xlabel('End-to-end latency (ms)')
    ax.set_ylabel('Frames (%)')
    ax.legend(loc='upper right')


def panel_dof_bitrate(ax, d):
    """Bit rate of every successful trial, one box per DOF count."""
    groups = [(f'{n} DOF', d['trials'].loc[d['trials']['n_dof'] == n, 'bitrate_bits_per_s']) for n in _dofs(d)]
    boxes(ax, groups, [DOF_COLORS[n] for n in _dofs(d)], ylabel='Bit rate (bits/s)', unit='trial')
    plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_ylim(0, None)


def panel_task_image(ax, d):
    """The task screen: virtual hand and the target spheres, cropped square."""
    ax.imshow(d['image'], aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def make_figure(cfg):
    """Fig S3, one row: A the task screen, B bit rate per DOF count, C end-to-end latency histogram."""
    layout = [['img', 'bit', 'lat']]
    panels = {'img': panel_task_image, 'bit': panel_dof_bitrate, 'lat': panel_dof_latency}

    def place(fig, axd):
        b, l = axd['bit'].get_position(), axd['lat'].get_position()
        W, H = fig.get_size_inches()
        h = b.height
        side = h * H / W                                          # square image, as tall as the bit-rate axes
        x0 = 0.03                                                 # room for the letter: the image is black, so A sits left of it
        axd['img'].set_position([x0, b.y1 - h, side, h])
        fig.text(0.0, b.y1, 'A', ha='left', va='top', fontweight='bold',
                 fontsize=STYLE['panel_label_size'] * plt.rcParams['font.size'] / STYLE['font_size'])
        bx = x0 + side + 0.075
        axd['bit'].set_position([bx, b.y0, b.width, h])
        lx = bx + b.width + 0.085
        axd['lat'].set_position([lx, l.y0, 0.995 - lx, l.height])

    return compose(layout, panels, load(), (18, 5.6), 'figS3_human_hand_control', labels={'bit': 'B', 'lat': 'C'},
                   stats_bars=False, post_layout=place)
