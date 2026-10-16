"""
Figure S5: Monkey R without the manipulandum. A: one camera frame (assets/figS5_frame.png, already cropped) with the live
Freebie keypoint; B: bit rate per trial of bend-sensor hand control (manipulandum) and Freebie hand control (without it).
Drawn from data/monkey_r/figS5_*.

data/monkey_r/figS5_frame.json: the live keypoint (before the Kalman filter) in the 960 x 720 px camera frame, and the crop
[x0, x1, y0, y1] of that frame that the asset is.
"""
import json
from functools import partial

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd

import utils
import monkey_r_figures as mr
from utils import COLORS, ROOT, STYLE, compose

OUT = mr.OUT
# panel B boxes: group -> label for the stats csv (the tick labels are TICKS)
GROUPS = {'bend': 'Hand control, bend sensor', 'freebie': 'Hand control, Freebie'}
TICKS = ['Bend\nsensor', 'Freebie']
BRACKETS = [('Hand control', (1, 2))]                      # group labels under the boxes
GROUP_LINE_EM = 3.4                                        # the line over each group label: this many font sizes below the axis
LAYOUT = [['frame', 'bit']]
# drawn at fig 2's scale (20 design inches = 180 mm), so the frame prints at fig 2's image size (52.5 mm square)
MM_PER_IN = 180 / 20
SIZE = (10.9, 7.5)
WIDTH_MM = SIZE[0] * MM_PER_IN
LEFT_IN = 0.55                 # room for the letter A left of the frame
FRAME_IN = 52.5 / MM_PER_IN    # width of the frame; its height (the crop's aspect) is the height of A and B
GAP_IN = 1.5                   # between the frame and the bit-rate axes: the y label and ticks


# ---------------------------------------------------------------- figure
def load(cfg):
    return {'runs': pd.read_csv(OUT / 'figS5_run_metrics.csv'), 'trials': pd.read_csv(OUT / 'figS5_trial_metrics.csv'),
            'frame': json.loads((OUT / 'figS5_frame.json').read_text(encoding='utf-8')), 'img': plt.imread(ROOT / 'assets' / 'figS5_frame.png')}


def panel_frame(ax, d):
    """The camera frame with the live Freebie keypoint (the DLC output for that frame, before the Kalman filter)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    img, f = d['img'], d['frame']
    x0, x1, y0, y1 = f['crop']                         # the asset is this crop of the 960 px frame, at its own resolution
    scale = img.shape[1] / (x1 - x0)
    x, y = (np.array(f['keypoint_xy']) - [x0, y0]) * scale
    ax.imshow(img, aspect='equal', cmap='gray' if img.ndim == 2 else None)
    ax.scatter([x], [y], s=utils.marker_area(1), color=COLORS['freebie'], edgecolors='white',
               linewidths=plt.rcParams['lines.linewidth'] * 0.6, zorder=3)


def panel_metric(ax, d, metric):
    # one bend run and two Freebie runs, too few for run means: every trial is drawn and tested (Mann-Whitney U on trials)
    groups = [(label, d['runs'][d['runs']['group'] == k], COLORS[k]) for k, label in GROUPS.items()]
    utils.metric_panel(ax, groups, metric, trials=d['trials'])
    ax.set_xticks(range(1, len(TICKS) + 1), TICKS, rotation=0, ha='center')
    # group label under the two-line tick labels, with a line across its boxes
    fs = plt.rcParams['font.size']
    below = lambda em: ax.get_xaxis_transform() + ScaledTranslation(0, -em * fs / 72, ax.figure.dpi_scale_trans)
    for text, (a, b) in BRACKETS:
        ax.plot([a - 0.4, b + 0.4], [0, 0], transform=below(GROUP_LINE_EM), color='#111111', lw=plt.rcParams['axes.linewidth'],
                clip_on=False, solid_capstyle='butt')
        ax.text((a + b) / 2, 0, text, transform=below(GROUP_LINE_EM + 0.25), ha='center', va='top', clip_on=False)


PANELS = {'frame': panel_frame, 'bit': partial(panel_metric, metric='bitrate_bits_per_s')}


def spread(fig, axd):
    """A (the crop's aspect) and B (the bit-rate boxes at the usual box width): the same height, the same top."""
    W, H = fig.get_size_inches()
    top = axd['bit'].get_position().y1
    x0, x1, y0, y1 = d_crop[0]
    x_a = LEFT_IN / W
    x_b = x_a + (FRAME_IN + GAP_IN) / W
    h = FRAME_IN * (y1 - y0) / (x1 - x0) / H
    axd['frame'].set_position([x_a, top - h, FRAME_IN / W, h])
    axd['bit'].set_position([x_b, top - h, len(GROUPS) * utils.BOX_IN / W, h])
    fig.text(0.0, top, 'A', ha='left', va='top', fontweight='bold',
             fontsize=STYLE['panel_label_size'] * plt.rcParams['font.size'] / STYLE['font_size'])


d_crop = [None]               # the crop, for spread (set by make_figure)


def make_figure(cfg):
    data = load(cfg)
    d_crop[0] = data['frame']['crop']
    return compose(LAYOUT, PANELS, data, SIZE, 'figS5_no_manipulandum', labels={'bit': 'B'},
                   stats_bars=cfg['stats_bars']['S5'], post_layout=spread, width_mm=WIDTH_MM)
