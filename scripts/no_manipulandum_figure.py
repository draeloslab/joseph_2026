"""
Figure S5: War Machine without the manipulandum. A: one camera frame (assets/figS5_frame.png, already cropped) with the
live Freebie keypoint; B: an example of Freebie hand control without the manipulandum and of bend-sensor hand control with
it, each with the raw EMG of the 8 decoder channels; C: bit rate per trial of hand control and of the online KFs trained on
each sensor. Drawn from data/war_machine/figS5_* only.

data/war_machine/figS5_frame.json: the frame's time in the Freebie example and the cursor position then, the live
keypoint (before the Kalman filter) in the 960 px frame the model saw, and the crop [x0, x1, y0, y1] of that frame that
the asset is.
"""
import json
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
import war_machine_figures as wmf
from utils import COLORS, ROOT, STYLE, compose

OUT = wmf.OUT
# panel C boxes: group -> label for the stats csv. Hand control (bend sensor in the manipulandum; Freebie without it),
# then online KFs trained on each (the tick labels name what the KF was trained on, `TICKS`)
GROUPS = {'bend': 'Hand control, bend sensor', 'freebie': 'Hand control, Freebie',
          'kf_bend': 'KF, bend sensor', 'kf_freebie': 'KF, Freebie'}
TICKS = ['Bend\nsensor', 'Freebie', 'Bend\nsensor', 'Freebie']
BRACKETS = [('Hand control', (1, 2)), ('KF', (3, 4))]      # group labels under the boxes
GROUP_LINE_Y = -0.21                                       # (axes fraction) the line over each group label
EXAMPLES = {'freebie': ('example', 'Freebie, no manipulandum'), 'bend': ('example_bend', 'Bend sensor, manipulandum')}
UV_PER_COUNT = 0.25            # the raw EMG is stored as int16 ADC counts, 0.25 µV per count
# A frame | B one example trace of each condition with the raw EMG of the 8 decoder channels stacked under it | C bit rate per
# trial: hand control (bend sensor / Freebie) and the online KFs trained on each. A, B and C are the same height.
LAYOUT = [['frame', 'trace_freebie', 'bit'], ['frame', 'emg_freebie', 'bit'],
          ['frame', 'trace_bend', 'bit'], ['frame', 'emg_bend', 'bit']]
SIZE = (24, 9)
HEIGHT_RATIOS = None
LEFT_IN = 1.0                  # room for the letter A left of the frame
FRAME_IN = 6.4                 # width of the frame; its height (the crop's aspect) is the height of A, B and C
B_IN = 5.2                     # width of B's traces
C_GAP_IN = 1.8                 # between panels: the y label and ticks of the panel on the right
PAIR_GAP_IN = 1.2              # between B's two trace + EMG pairs: the upper EMG's ticks and label, the lower trace's title
EMG_SHARE = 0.6                # of a pair's height, the raw EMG (the rest is the behaviour trace); they touch
EMG_SPACING = 1.2              # channel spacing in units of the 99.5th percentile of |EMG| in the largest example


# ---------------------------------------------------------------- figure
def _nice(v, up=False):
    """v rounded down (up) to 1, 2 or 5 x 10^k."""
    k = 10 ** np.floor(np.log10(v))
    return float(k * (min(f for f in (1, 2, 5, 10) if f * k >= v) if up else max(f for f in (1, 2, 5) if f * k <= v)))


def load(cfg):
    trace = {k: wmf._load_example(f'figS5_example_{k}') for k in EXAMPLES}
    emg = {k: dict(np.load(OUT / f'figS5_emg_{k}.npz')) for k in EXAMPLES}
    for e in emg.values():
        e['x'] = e['x'] * UV_PER_COUNT                  # counts -> microvolts
    tmax = max(t['stop_time'].iloc[-1] - t['start_time'].iloc[0] for _, t in trace.values())
    spacing = EMG_SPACING * max(np.percentile(np.abs(e['x'] - np.median(e['x'], axis=0)), 99.5) for e in emg.values())
    return {'runs': pd.read_csv(OUT / 'figS5_run_metrics.csv'), 'trials': pd.read_csv(OUT / 'figS5_trial_metrics.csv'),
            'trace': trace, 'emg': emg, 'tmax': tmax, 'emg_spacing': spacing, 'emg_bar': _nice(spacing, up=True),
            'frame': json.loads((OUT / 'figS5_frame.json').read_text()), 'img': plt.imread(ROOT / 'assets' / 'figS5_frame.png')}


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


def panel_trace(ax, d, key):
    """One example: the target windows and the signal that drove the cursor. The Freebie one marks panel A's frame. It touches
    the raw EMG below it, so it has no x axis of its own."""
    ex, tr = d['trace'][key]
    color = COLORS[key]
    wmf._boxes_and_trace(ax, tr, [(ex['cursor_t'], ex['cursor'], None, color, 2.5, '-')], 'Flexion\n(%)', legend=False)      # calibrated 0 (extended) to 1 (flexed)
    ax.set_xlim(0, d['tmax'])
    ax.set_yticks([0, 0.5, 1], ['0', '50', '100'])
    ax.set_xlabel('')
    ax.tick_params(bottom=False, labelbottom=False)
    ax.spines['bottom'].set_visible(False)
    ax.set_title(EXAMPLES[key][1], loc='left', pad=4)
    if key == 'freebie':
        f = d['frame']
        ax.axvline(f['time_s'], color=COLORS['budget'], ls=':', lw=1, zorder=2)
        ax.plot(f['time_s'], f['position'], 'o', ms=7, mfc='white', mec=color, mew=2, zorder=4, label='Frame in A')
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), borderaxespad=0, handlelength=0.8)


def panel_emg(ax, d, key):
    """The raw EMG of the 8 decoder channels, stacked (channel order from the top), same time axis as the trace above and one
    spacing / scale for both examples."""
    e, sp = d['emg'][key], d['emg_spacing']
    x = e['x'] - np.median(e['x'], axis=0)
    for k in range(x.shape[1]):
        ax.plot(e['t'], x[:, k] + (x.shape[1] - 1 - k) * sp, color=COLORS[key], lw=0.3, rasterized=True)
    ax.set_xlim(0, d['tmax'])
    ax.set_ylim(-0.6 * sp, (e['x'].shape[1] - 0.4) * sp)
    ax.set_yticks([])
    ax.set_ylabel('EMG')
    bar = d['emg_bar']                  # a vertical scale bar just right of the axes, the same length on both examples
    x = d['tmax'] * 1.015
    ax.plot([x, x], [-0.5 * sp, -0.5 * sp + bar], color='#111111', lw=plt.rcParams['axes.linewidth'] * 1.5, clip_on=False, solid_capstyle='butt')
    ax.text(x + 0.015 * d['tmax'], -0.5 * sp + bar / 2, f'{bar:g} µV', ha='left', va='center', clip_on=False)
    ax.set_xlabel('Time (s)' if key == 'bend' else '')       # one label for both pairs (the same x axis)
    if key == 'freebie':
        ax.axvline(d['frame']['time_s'], color=COLORS['budget'], ls=':', lw=1, zorder=2)


def panel_metric(ax, d, metric):
    # bit rate: the library's own value for each trial (not a running average). Hand control is one bend run and two Freebie
    # runs, each KF group one run, so there are no run means to test: every trial is plotted and tested (Mann-Whitney U on
    # trials, Holm-corrected over the six pairs; stats csv tested_on = trials)
    groups = [(label, d['runs'][d['runs']['group'] == k], COLORS[k]) for k, label in GROUPS.items()]
    utils.metric_panel(ax, groups, metric, trials=d['trials'])
    ax.set_xticks(range(1, len(TICKS) + 1), TICKS, rotation=0, ha='center')
    # group labels under the tick labels, each with a line across its boxes (y in axes fraction, below the two-line ticks)
    for text, (a, b) in BRACKETS:
        ax.annotate('', xy=(b + 0.4, GROUP_LINE_Y), xytext=(a - 0.4, GROUP_LINE_Y), xycoords=('data', 'axes fraction'),
                    annotation_clip=False, arrowprops=dict(arrowstyle='-', lw=plt.rcParams['axes.linewidth'], shrinkA=0, shrinkB=0))
        ax.text((a + b) / 2, GROUP_LINE_Y - 0.015, text, transform=ax.get_xaxis_transform(), ha='center', va='top', clip_on=False)


PANELS = {'frame': panel_frame, **{f'trace_{k}': partial(panel_trace, key=k) for k in EXAMPLES},
          **{f'emg_{k}': partial(panel_emg, key=k) for k in EXAMPLES}, 'bit': partial(panel_metric, metric='bitrate_bits_per_s')}


def spread(fig, axd):
    """Left to right in inches: A (the crop's aspect), B (two pairs, each a behaviour trace touching its raw EMG, one x axis),
    C (the bit-rate boxes at the usual box width). All three have A's height and the same top."""
    pos = lambda k: axd[k].get_position()
    W, H = fig.get_size_inches()
    top = pos('trace_freebie').y1
    crop = d_crop[0]
    aspect = (crop[1] - crop[0]) / (crop[3] - crop[2])
    x_a = LEFT_IN / W
    x_b = x_a + (FRAME_IN + C_GAP_IN) / W
    x_c = x_b + (B_IN + C_GAP_IN + 1.2) / W
    h = FRAME_IN / aspect / H
    axd['frame'].set_position([x_a, top - h, FRAME_IN / W, h])
    axd['bit'].set_position([x_c, top - h, len(GROUPS) * utils.BOX_IN / W, h])
    pair = (h - PAIR_GAP_IN / H) / 2
    for n, key in enumerate(EXAMPLES):
        y1 = top - n * (pair + PAIR_GAP_IN / H)
        axd[f'trace_{key}'].set_position([x_b, y1 - (1 - EMG_SHARE) * pair, B_IN / W, (1 - EMG_SHARE) * pair])
        axd[f'emg_{key}'].set_position([x_b, y1 - pair, B_IN / W, EMG_SHARE * pair])
    fig.text(0.0, top, 'A', ha='left', va='top', fontweight='bold',
             fontsize=STYLE['panel_label_size'] * plt.rcParams['font.size'] / STYLE['font_size'])


d_crop = [None]               # the crop, for spread (set by make_figure)


def make_figure(cfg):
    data = load(cfg)
    d_crop[0] = data['frame']['crop']
    return compose(LAYOUT, PANELS, data, SIZE, 'figS5_no_manipulandum',
                   labels={'trace_freebie': 'B', 'bit': 'C'},
                   stats_bars=cfg['stats_bars']['S5'], height_ratios=HEIGHT_RATIOS, post_layout=spread)
