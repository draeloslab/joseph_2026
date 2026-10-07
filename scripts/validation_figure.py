"""Figure 1: Freebie tracking validation (labelled frames, error histogram, per-keypoint error)."""
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

from utils import COLORS, DATA, compose, load_master, significance, describe, marker_area

TOP_KPS = ['DIP', 'PIP', 'MCP', 'WRIST']
KEYPOINTS = TOP_KPS + ['MRS']
LIKELIHOOD_MIN = 0.99
THRESHOLD_PX = 20      # size of the smallest joint
LABEL_GRID = [(0.03, 0.42), (0.52, 0.42), (0.03, 0.20), (0.52, 0.20)]   # DIP PIP MCP WRIST label boxes (axes fraction)

LAYOUT = [['top', 'top', 'side'],
          ['hist', 'hist', 'box']]
SIZE = (18, 11)


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    master = load_master(cfg)
    views = master['freebie_validation']

    # euclidean error (px) of every labelled frame, Freebie prediction vs manual label
    errors = {kp: [] for kp in KEYPOINTS}
    for view, kps in (('top', TOP_KPS), ('side', ['MRS'])):
        d = views[view]
        for i in range(d['n_videos']):
            pred, man, lik = (np.asarray(d[k][i]).reshape(len(d['frame_indices'][i]), len(kps), -1)
                              for k in ('pred_xy', 'manual_xy', 'pred_lik'))
            for k, kp in enumerate(kps):
                keep = lik[:, k, 0] >= LIKELIHOOD_MIN
                errors[kp].append(np.linalg.norm(pred[keep, k] - man[keep, k], axis=1))
    np.savez(DATA / 'keypoint_errors.npz', **{kp: np.concatenate(v) for kp, v in errors.items()})

    # example frames (png bytes) with their manual labels / predictions
    frames = master['keypoint_frames']
    (DATA / 'keypoint_frames').mkdir(exist_ok=True)
    for view in ('top', 'side'):
        (DATA / 'keypoint_frames' / f'{view}.png').write_bytes(frames[view])
    np.savez(DATA / 'keypoint_frames' / 'labels.npz', **frames['labels'])


def load():
    return {'errors': dict(np.load(DATA / 'keypoint_errors.npz')),
            'labels': dict(np.load(DATA / 'keypoint_frames' / 'labels.npz')),
            'top': np.array(Image.open(DATA / 'keypoint_frames' / 'top.png').convert('RGB')),
            'side': np.array(Image.open(DATA / 'keypoint_frames' / 'side.png').convert('RGB'))}


# ---------------------------------------------------------------- panels
def panel_frame(ax, d, view):
    kps = TOP_KPS if view == 'top' else ['MRS']
    colors = COLORS['keypoints'][:4] if view == 'top' else COLORS['keypoints'][4:]
    img = d[view]
    man, pred, lik = (d['labels'][f'{view}_{k}'] for k in ('manual_xy', 'pred_xy', 'pred_lik'))
    ax.imshow(img)
    ax.set_axis_off()
    if view == 'top':
        ax.plot(man[:, 0], man[:, 1], color='white', lw=3.5, alpha=0.85, zorder=2)
    h, w = img.shape[:2]
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
        ann = ax.text(gx, gy, f'{kp}  p={lik[k, 0]:.2f}\nLabel {x:.0f}, {y:.0f}\nFreebie {px:.0f}, {py:.0f}',
                      transform=ax.transAxes, fontsize=plt.rcParams['font.size'], fontweight='bold', color=c,
                      ha='left', va='top', zorder=8,
                      bbox=dict(boxstyle='round,pad=0.45', fc='#111111', alpha=0.85, ec=c, lw=plt.rcParams['axes.linewidth']))
        for a in (arrow, ann):
            a.set_in_layout(False)                # labels sit on the image; don't shrink the image to fit them
    ax.set_title('Top view' if view == 'top' else 'Side view')
    ax.keep_title = True                          # camera name: kept by compose


def panel_hist(ax, d):
    for kp, c in zip(KEYPOINTS, COLORS['keypoints']):
        e = d['errors'][kp]
        _, bins, _ = ax.hist(e, bins=15, alpha=0.6, color=c, label=f'{kp} (mean {e.mean():.2f} px)')
        x = np.linspace(bins[0], bins[-1], 500)
        ax.plot(x, gaussian_kde(e)(x) * len(e) * (bins[1] - bins[0]), color=c)
    ax.axvline(THRESHOLD_PX, color='gray', ls='--', label=f'Smallest joint ({THRESHOLD_PX} px)')
    ax.set_xlabel('Euclidean distance error (px)')
    ax.set_ylabel('Count')
    ax.set_title('Tracking error distribution')
    ax.legend()


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
    ax.axhline(THRESHOLD_PX, color='gray', ls='--')
    ax.set_ylim(0, max(THRESHOLD_PX * 1.4, np.percentile(np.concatenate(data), 99.5)))
    ax.set_ylabel('Euclidean distance error (px)')
    ax.set_title('Error per keypoint')
    ax.n_boxes = len(data)                        # compose() sets the width: same box width in every figure
    ax.legend()
    for kp, x in zip(KEYPOINTS, data):
        describe(ax, kp, x, unit='frame')
    significance(ax, list(zip(KEYPOINTS, data)), tested_on='frames')


PANELS = {'top': partial(panel_frame, view='top'), 'side': partial(panel_frame, view='side'),
          'hist': panel_hist, 'box': panel_box}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'validation_all_frames',
                   stats_bars=cfg['stats_bars']['validation_all'])
