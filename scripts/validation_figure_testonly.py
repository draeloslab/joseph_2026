"""Figure 1 (held-out only): same as validation_figure.py, restricted to DLC test-set frames.

master['freebie_validation'] mixes frames the DLC models were trained on with frames they never
saw (Buildnpy.py, upstream, just kept every frame with both a manual label and a prediction).
master['freebie_validation_heldout'] / master['keypoint_frames_heldout'] (added by
freebie/script/add_heldout_validation.py) are filtered to frames absent from the DLC train split
of whichever shuffle produced that video's predictions -- including the example frame drawn in
the image panel, which needed swapping (the original hardcoded frame turned out to be a train
frame; see that script's docstring)."""
from functools import partial

import numpy as np
from PIL import Image
from scipy.stats import gaussian_kde

from utils import COLORS, DATA, compose, load_master, significance, describe

TOP_KPS = ['DIP', 'PIP', 'MCP', 'WRIST']
KEYPOINTS = TOP_KPS + ['MRS']
LIKELIHOOD_MIN = 0.99
THRESHOLD_PX = 20      # size of the smallest joint
TOP_OFFSETS = {'DIP': (60, -40), 'PIP': (-60, -50), 'MCP': (-60, 40), 'WRIST': (-40, 160)}   # px; kept inside CROP_X

LAYOUT = [['top', 'side', 'box']]
SIZE = (20, 7)
CROP_X = {'top': (360, 1440), 'side': (760, 1840)}   # 1080 px wide x full 1080 px height = square, keypoints + labels inside
LABEL_SIZE = 12

OUT_DIR = DATA / 'keypoint_frames_heldout'


# ---------------------------------------------------------------- analysis
def analyze(cfg):
    master = load_master(cfg)
    views = master['freebie_validation_heldout']

    # euclidean error (px) of every held-out labelled frame, Freebie prediction vs manual label
    errors = {kp: [] for kp in KEYPOINTS}
    for view, kps in (('top', TOP_KPS), ('side', ['MRS'])):
        d = views[view]
        for i in range(d['n_videos']):
            pred, man, lik = (np.asarray(d[k][i]).reshape(len(d['frame_indices'][i]), len(kps), -1)
                              for k in ('pred_xy', 'manual_xy', 'pred_lik'))
            for k, kp in enumerate(kps):
                keep = lik[:, k, 0] >= LIKELIHOOD_MIN
                errors[kp].append(np.linalg.norm(pred[keep, k] - man[keep, k], axis=1))
    np.savez(DATA / 'keypoint_errors_heldout.npz', **{kp: np.concatenate(v) for kp, v in errors.items()})

    # example frames (png bytes) with their manual labels / predictions
    frames = master['keypoint_frames_heldout']
    OUT_DIR.mkdir(exist_ok=True)
    for view in ('top', 'side'):
        (OUT_DIR / f'{view}.png').write_bytes(frames[view])
    np.savez(OUT_DIR / 'labels.npz', **frames['labels'])


def load():
    return {'errors': dict(np.load(DATA / 'keypoint_errors_heldout.npz')),
            'labels': dict(np.load(OUT_DIR / 'labels.npz')),
            'top': np.array(Image.open(OUT_DIR / 'top.png').convert('RGB')),
            'side': np.array(Image.open(OUT_DIR / 'side.png').convert('RGB'))}


# ---------------------------------------------------------------- panels
def panel_frame(ax, d, view):
    kps = TOP_KPS if view == 'top' else ['MRS']
    colors = COLORS['keypoints'][:4] if view == 'top' else COLORS['keypoints'][4:]
    img = d[view]
    man, pred, lik = (d['labels'][f'{view}_{k}'] for k in ('manual_xy', 'pred_xy', 'pred_lik'))
    ax.imshow(img)
    ax.set_xlim(*CROP_X[view])
    ax.set_ylim(img.shape[0], 0)
    ax.set_axis_off()
    if view == 'top':
        ax.plot(man[:, 0], man[:, 1], color='white', lw=3.5, alpha=0.85, zorder=2)
    h, w = img.shape[:2]
    for k, (kp, c) in enumerate(zip(kps, colors)):
        (x, y), (px, py) = man[k], pred[k]
        ax.scatter(x, y, s=200, color=c, edgecolors='white', linewidths=2, zorder=5, label=kp)
        ax.scatter(px, py, s=140, marker='x', color=c, linewidths=3, zorder=6)
        dx, dy = TOP_OFFSETS[kp] if view == 'top' else (25 if x < 0.78 * w else -25, -38 if y > 0.1 * h else 38)
        ax.annotate(f'{kp}\nLabel: ({x:.0f}, {y:.0f})\nFreebie: ({px:.0f}, {py:.0f})  p={lik[k, 0]:.2f}',
                    xy=(x, y), xytext=((x + px) / 2 + dx, (y + py) / 2 + dy), fontsize=LABEL_SIZE, fontweight='bold',
                    color=c, ha='left' if dx > 0 else 'right', va='center', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.35', fc='black', alpha=0.65, ec=c, lw=0.8),
                    arrowprops=dict(arrowstyle='->', color=c, lw=1.4))
    ax.set_title('Top view' if view == 'top' else 'Side view')


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
                    medianprops={'color': '#111111', 'lw': 2})
    rng = np.random.default_rng(0)
    for i, (patch, x, c) in enumerate(zip(bp['boxes'], data, COLORS['keypoints'])):
        patch.set(facecolor=c, alpha=0.6, edgecolor='#444444')
        ax.scatter(rng.normal(i + 1, 0.06, len(x)), x, s=6, color=c, alpha=0.35, zorder=2, linewidths=0)
    ax.scatter(np.arange(1, len(data) + 1), [np.mean(x) for x in data], marker='D', s=40,
               color='#111111', zorder=4, label='Mean')
    ax.axhline(THRESHOLD_PX, color='gray', ls='--', label=f'Smallest joint ({THRESHOLD_PX} px)')
    ax.set_ylim(0, max(THRESHOLD_PX * 1.15, np.percentile(np.concatenate(data), 99.5)))
    ax.set_ylabel('Euclidean distance error (px)')
    ax.set_title('Error per keypoint (held-out)')
    ax.n_boxes = len(data)                        # compose() sets the width: same box width as figs 3/4
    ax.legend(loc='upper right')
    for kp, x in zip(KEYPOINTS, data):
        describe(ax, kp, x, unit='frame')
    significance(ax, list(zip(KEYPOINTS, data)), tested_on='frames')


PANELS = {'top': partial(panel_frame, view='top'), 'side': partial(panel_frame, view='side'),
          'box': panel_box}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'fig1_validation_heldout',
                   stats_bars=cfg['stats_bars'].get('1_heldout', cfg['stats_bars'][1]))
