"""Figure 6: Freebie driving FES (real-time tracking session + stimulation session). The four frames are
assets/fig6_{rt,stim}_{0,1}.png: face-masked and cut to the window the panel shows (the hand's keypoints + a pad, at the
panel's aspect); data/fig6_fes.npz holds the keypoints in those crops' pixels and the traces.

data/fig6_detection.csv (written as figures/fig6_fes_descriptives.csv): panel A's live run, % of processing steps in which
each camera, and at least one / two cameras at once, had the index fingertip -- counting every stored hand, only the
monkey's hands, or only the reaching hand (stored hands labelled by eye from the videos)."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from utils import COLORS, DATA, DOF_NAMES, FIGS, ROOT, STYLE, compose, marker_area

# 21-keypoint hand: wrist, then 4 (thumb 5 incl. wrist) points per finger in DOF_NAMES order
FINGERS = dict(zip(DOF_NAMES, [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]]))
BONES = [(f, a, b) for f, idx in FINGERS.items() for a, b in zip(idx, idx[1:])] + \
        [('palm', a, b) for a, b in ((0, 5), (5, 9), (9, 13), (13, 17), (0, 17))]
CROP_PAD_PX = 60
CROP_ASPECT = {'rt': 1.0, 'stim': 1.48}   # width / height of the cropped frames

LAYOUT = [['rt_0', 'rt_1', 'stim_0', 'stim_1'],
          ['rt_trace', 'rt_trace', 'stim_trace', 'stim_trace']]
SIZE = (16, 9)
WIDTH_RATIOS = [1, 1, 1.4, 1.4]
HEIGHT_RATIOS = [1, 0.8]
LINE_MM = 68.0     # width of each line plot's axes at print size, about that of the image pair above it
TRACE_LW = 5


def load():
    d = dict(np.load(DATA / 'fig6_fes.npz'))
    for kind in ('rt', 'stim'):
        d[f'{kind}_imgs'] = [np.array(Image.open(ROOT / 'assets' / f'fig6_{kind}_{i}.png').convert('RGB')) for i in range(2)]
    return d


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
    """the frame (already cut around the hand) with the skeleton on top; limits: keypoints + pad at the panel's aspect ratio"""
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
        ax.plot(d['rt_t'], d['rt_indextip'][:, k], color=COLORS[f'fes_{name}'], lw=TRACE_LW, label=f"{ {'x': 'Horizontal', 'y': 'Vertical'}[name]} (px)")
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Index fingertip (px)')
    ax.set_title('Real-time tracking (camera 3)', pad=34)
    ax.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.1, handlelength=1.5, markerscale=2)


def panel_stim_trace(ax, d):
    for a, b in d['stim_windows']:
        ax.axvspan(a, b, color=COLORS['stim_window'], alpha=0.8, lw=0)     # stimulation periods
    ax.plot(d['stim_t'], d['stim_x'] - np.nanmedian(d['stim_x']), color=COLORS['fingers']['middle'], lw=TRACE_LW)   # = the middle knuckle on the images
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Middle knuckle x (mm)')


PANELS = {'rt_0': partial(panel_image, kind='rt', i=0), 'rt_1': partial(panel_image, kind='rt', i=1),
          'stim_0': partial(panel_image, kind='stim', i=0), 'stim_1': partial(panel_image, kind='stim', i=1),
          'rt_trace': panel_rt_trace, 'stim_trace': panel_stim_trace}
LABELS = {'rt_0': 'A', 'stim_0': 'B'}


def make_figure(cfg):
    def align(fig, axd):
        # both line plots LINE_MM wide at print size, left edges under the images
        w = LINE_MM / STYLE['figure_width_mm']
        for kind in ('rt', 'stim'):
            a, t = axd[f'{kind}_0'].get_position(), axd[f'{kind}_trace'].get_position()
            axd[f'{kind}_trace'].set_position([a.x0, t.y0, w, t.height])

    fig = compose(LAYOUT, PANELS, load(), SIZE, 'fig6_fes', labels=LABELS, stats_bars=cfg['stats_bars'][6],
                  height_ratios=HEIGHT_RATIOS, width_ratios=WIDTH_RATIOS, post_layout=align)
    det = pd.read_csv(DATA / 'fig6_detection.csv')
    det[['panel', *det.columns.drop('panel')]].to_csv(FIGS / 'fig6_fes_descriptives.csv', index=False)
    return fig
