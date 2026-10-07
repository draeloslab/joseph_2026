"""Figure 5: Freebie driving FES (real-time tracking session + stimulation session)."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from utils import COLORS, DATA, DOF_NAMES, compose, marker_area

# 21-keypoint hand: wrist, then 4 (thumb 5 incl. wrist) points per finger in DOF_NAMES order
FINGERS = dict(zip(DOF_NAMES, [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]))
BONES = [(f, a, b) for f, idx in FINGERS.items() for a, b in zip(idx, idx[1:])] + \
        [('palm', a, b) for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17))]
CROP_PAD_PX = 60
CROP_ASPECT = {'rt': 1.0, 'stim': 1.48}   # width / height of the cropped frames

LAYOUT = [['rt_0', 'rt_1', 'stim_0', 'stim_1'],
          ['rt_trace', 'rt_trace', 'stim_trace', 'stim_trace']]
SIZE = (16, 9)      # same scale as the other figures (16-18 in wide), so the shared fonts / line widths read the same
WIDTH_RATIOS = [1, 1, 1.4, 1.4]
HEIGHT_RATIOS = [1, 0.8]


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    pass   # data/fig5_fes.npz is exported by the FES tracking/stim notebooks (improv/experiments/FES/analysis)


def load():
    return dict(np.load(DATA / 'fig5_fes.npz'))


# ---------------------------------------------------------------- panels
TRACKED = {'rt': 8, 'stim': 9}     # keypoint followed in the trace below each image: index fingertip (A), middle MCP knuckle (B)


def draw_skeleton(ax, xy, tracked=None):
    """Skeleton at a light weight (alpha) so it does not hide the hand; the keypoint plotted below is ringed."""
    lw = plt.rcParams['axes.linewidth']
    for f, a, b in BONES:
        if np.isfinite(xy[[a, b]]).all():
            ax.plot(*xy[[a, b]].T, color=COLORS['fingers'][f], lw=0.9 * lw, alpha=0.8, solid_capstyle='round', zorder=2)
    for f, idx in FINGERS.items():
        m = np.isfinite(xy[idx, 0])
        ax.scatter(*xy[idx][m].T, s=marker_area(1.2), color=COLORS['fingers'][f], alpha=0.85, edgecolors='white',
                   linewidths=0.3 * lw, zorder=3)
    if tracked is not None and np.isfinite(xy[tracked]).all():     # white halo + black ring: visible on dark and light frames
        for ec, w, size in (('white', 1.6 * lw, 3.4), ('#111111', 0.8 * lw, 3.4)):
            ax.scatter(*xy[tracked], s=marker_area(size), facecolors='none', edgecolors=ec, linewidths=w, zorder=4)


def panel_image(ax, d, kind, i):
    """frame cropped around the hand (keypoints + pad) to the panel's aspect ratio, skeleton on top"""
    img, xy = d[f'{kind}_imgs'][i], d[f'{kind}_xy'][i]
    ax.imshow(img)
    ax.set_axis_off()
    draw_skeleton(ax, xy, TRACKED[kind])
    ok = np.isfinite(xy[:, 0])
    (x0, y0), (x1, y1) = xy[ok].min(0) - CROP_PAD_PX, xy[ok].max(0) + CROP_PAD_PX
    aspect = CROP_ASPECT[kind]
    hw = max((x1 - x0) / 2, (y1 - y0) / 2 * aspect)
    cx, cy, hh = (x0 + x1) / 2, (y0 + y1) / 2, hw / aspect
    ax.set_xlim(cx - hw, cx + hw)
    ax.set_ylim(cy + hh, cy - hh)
    ax.set_title(f"{d['rt_img_t'][i]:.1f} s" if kind == 'rt' else f"Camera {d['stim_cams'][i]}")
    ax.keep_title = True                          # frame time / camera name: kept by compose


def panel_rt_trace(ax, d):
    for k, name in enumerate(('x', 'y')):
        ax.plot(d['rt_t'], d['rt_indextip'][:, k], color=COLORS[f'fes_{name}'], marker='.', label=f"{ {'x': 'Horizontal', 'y': 'Vertical'}[name]} (px)")
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Index fingertip (px)')
    ax.set_title('Real-time tracking (camera 3)', pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1, handlelength=1.5, markerscale=2)


def panel_stim_trace(ax, d):
    for a, b in d['stim_windows']:
        ax.axvspan(a, b, color=COLORS['stim_window'], alpha=0.8, lw=0)     # stimulation periods (conditions listed in the legend)
    ax.plot(d['stim_t'], d['stim_x'] - np.nanmedian(d['stim_x']), color=COLORS['fingers']['middle'])   # = the middle knuckle on the images
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Middle knuckle x (mm)')


PANELS = {'rt_0': partial(panel_image, kind='rt', i=0), 'rt_1': partial(panel_image, kind='rt', i=1),
          'stim_0': partial(panel_image, kind='stim', i=0), 'stim_1': partial(panel_image, kind='stim', i=1),
          'rt_trace': panel_rt_trace, 'stim_trace': panel_stim_trace}
LABELS = {'rt_0': 'A', 'stim_0': 'B'}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'fig6_fes', labels=LABELS, stats_bars=cfg['stats_bars'][6],
                   height_ratios=HEIGHT_RATIOS, width_ratios=WIDTH_RATIOS)
