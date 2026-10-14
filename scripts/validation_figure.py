"""Figure 2: Freebie keypoints vs manual labels on held-out frames (never in the training split of the model that
predicted them). The example frames are assets/fig2_{top,side}.png, already cropped; their labels / predictions
(data/fig2_keypoint_labels.npz) are in the full camera frame's pixels, <view>_crop_x0 the crop's first column."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from utils import COLORS, DATA, ROOT, compose, significance, describe, marker_area

TOP_KPS = ['DIP', 'PIP', 'MCP', 'WRIST']
KEYPOINTS = TOP_KPS + ['MRS']
THRESHOLD_PX = 20      # size of the smallest joint
LABEL_GRID = [(0.03, 0.64), (0.03, 0.42), (0.03, 0.20), (0.52, 0.20)]   # DIP PIP MCP WRIST label boxes (axes fraction)

LAYOUT = [['top', 'side', 'box']]
SIZE = (20, 7)


def load():
    return {'errors': dict(np.load(DATA / 'fig2_keypoint_errors.npz')),
            'labels': dict(np.load(DATA / 'fig2_keypoint_labels.npz')),
            **{v: np.array(Image.open(ROOT / 'assets' / f'fig2_{v}.png').convert('RGB')) for v in ('top', 'side')}}


# ---------------------------------------------------------------- panels
def panel_frame(ax, d, view):
    kps = TOP_KPS if view == 'top' else ['MRS']
    colors = COLORS['keypoints'][:4] if view == 'top' else COLORS['keypoints'][4:]
    img = d[view]
    man, pred, lik = (d['labels'][f'{view}_{k}'] for k in ('manual_xy', 'pred_xy', 'pred_lik'))
    x0 = int(d['labels'][f'{view}_crop_x0'])
    ax.imshow(img, extent=(x0 - 0.5, x0 + img.shape[1] - 0.5, img.shape[0] - 0.5, -0.5))   # in full-frame pixels
    ax.set_xlim(x0, x0 + img.shape[1] - 1)          # first to last pixel centre: the frame as cropped
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    if view == 'top':
        ax.plot(man[:, 0], man[:, 1], color='white', lw=3.5, alpha=0.85, zorder=2)
    for k, (kp, c) in enumerate(zip(kps, colors)):
        (x, y), (px, py) = man[k], pred[k]
        ax.scatter(x, y, s=200, color=c, edgecolors='white', linewidths=2, zorder=5, label=kp)
        ax.scatter(px, py, s=140, marker='x', color=c, linewidths=3, zorder=6)
        # labels in a fixed 2 x 2 grid over the empty lower part of the image, an arrow to each keypoint
        gx, gy = LABEL_GRID[k] if view == 'top' else (0.03, 0.30)
        # arrow from the box's middle, drawn under the boxes so one keypoint's arrow never crosses another's label
        arrow = ax.annotate('', xy=(x, y), xytext=(gx + 0.2, gy - 0.08), textcoords='axes fraction', zorder=6,
                            annotation_clip=False, arrowprops=dict(arrowstyle='->', color=c, shrinkB=4,
                                                                   lw=plt.rcParams['lines.linewidth']))
        ann = ax.text(gx, gy, f'{kp}  c={lik[k, 0]:.2f}\nLabel {x:.0f}, {y:.0f}\nFreebie {px:.0f}, {py:.0f}',
                      transform=ax.transAxes, fontsize=plt.rcParams['font.size'], fontweight='bold', color=c,
                      ha='left', va='top', zorder=8,
                      bbox=dict(boxstyle='round,pad=0.45', fc='#111111', alpha=0.85, ec=c, lw=plt.rcParams['axes.linewidth']))
        for a in (arrow, ann):
            a.set_in_layout(False)                # labels sit on the image; don't shrink the image to fit them
    ax.set_title('Top view' if view == 'top' else 'Side view')
    ax.keep_title = True                          # camera name: kept by compose


def panel_box(ax, d):
    data = [d['errors'][kp] for kp in KEYPOINTS]
    bp = ax.boxplot(data, tick_labels=KEYPOINTS, widths=0.55, patch_artist=True, showfliers=False,
                    medianprops={'color': '#111111'})
    rng = np.random.default_rng(0)
    for i, (patch, x, c) in enumerate(zip(bp['boxes'], data, COLORS['keypoints'])):
        patch.set(facecolor=c, alpha=0.6, edgecolor='#444444')
        ax.scatter(rng.normal(i + 1, 0.06, len(x)), x, s=marker_area(0.9), color=c, alpha=0.4, zorder=2, linewidths=0)
    ax.scatter(np.arange(1, len(data) + 1), [np.mean(x) for x in data], marker='D', s=marker_area(1.8),
               color='#111111', zorder=4, label='Mean')
    ax.axhline(THRESHOLD_PX, color='gray', ls='--', label=f'Smallest joint ({THRESHOLD_PX} px)')
    ax.set_ylim(0, max(THRESHOLD_PX * 1.4, np.percentile(np.concatenate(data), 99.5)))
    ax.set_ylabel('Euclidean distance error (px)')
    ax.n_boxes = len(data)                        # compose() sets the width from this
    ax.legend(loc='upper right')
    for kp, x in zip(KEYPOINTS, data):
        describe(ax, kp, x, unit='frame')
    significance(ax, list(zip(KEYPOINTS, data)), tested_on='frames')


PANELS = {'top': partial(panel_frame, view='top'), 'side': partial(panel_frame, view='side'),
          'box': panel_box}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'fig2_validation',
                   stats_bars=cfg['stats_bars'][2])
