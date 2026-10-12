"""
Figure S5: War Machine without the manipulandum (2026-10-09, config `war_machine_s5`). A: one camera frame with the
live Freebie keypoint, next to example Freebie hand-control trials (the frame's moment marked); B: online metrics of
bend hand control, Freebie hand control without the manipulandum, and the KFs trained on each.

analyze reads its own master (<master>/war_machine_s5.pkl, built here from the server rdbs via war_machine.build) and
the improv logs (~/predictions/20261009), writes data/war_machine/s5_*; make_figure draws from those only.

The frame: the value BRAND logged in `vision_data` is the exact float improv's sender sent (index tip y px / resize),
and improv logs every sent value with the camera frame it came from (sender_sent_angles / sender_sent_frame_nums), so
matching the value gives the frame without any clock alignment. Camera frame k is video frame k (the saver writes
every frame the camera delivers; checked below). The frame itself is never opened here: analyze prints the ffmpeg
command that writes it, and config frame_image points at the result.
"""
import json
import subprocess
import tempfile
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils
import war_machine as wm
import war_machine_figures as wmf
from utils import COLORS, ONLINE_METRICS, ROOT, STYLE, compose

KEY = 'war_machine_s5'
OUT = wmf.OUT
# panel C boxes: key (config groups) -> label for the stats csv. Hand control (bend sensor in the manipulandum; Freebie without it),
# then online KFs trained on each (the tick labels name what the KF was trained on, `TICKS`)
GROUPS = {'bend': 'Hand control, bend sensor', 'freebie': 'Hand control, Freebie',
          'kf_bend': 'KF, bend sensor', 'kf_freebie': 'KF, Freebie'}
TICKS = ['Bend\nsensor', 'Freebie', 'Bend\nsensor', 'Freebie']
BRACKETS = [('Hand control', (1, 2)), ('KF', (3, 4))]      # group labels under the boxes
GROUP_LINE_Y = -0.21                                       # (axes fraction) the line over each group label
CAM = 2                        # improv camera_num of the rig camera (TIS front_left)
NATIVE_W = 960                 # px, the frames the model saw = the saved video (camera_config resolution, prescaled)
HOLD_END_S = 0.05              # auto frame: this long before the success of the most-flexed target's trial (in the target)
m = list(ONLINE_METRICS)
OFFLINE = {'nmse_valid': 'Normalized MSE', 'correlation_valid': 'Correlation (r)'}
EXAMPLES = {'freebie': ('example', 'Freebie, no manipulandum'), 'bend': ('example_bend', 'Bend sensor, manipulandum')}
UV_PER_COUNT = 0.25            # the rdb holds int16 ADC counts; brand-monkeyrig rdb2nwb converts them with 1e-6 * 0.25 V per count
LAYOUT_CHECK = False           # draw neither the frame nor the raw EMG (so the layout can be looked at without them)
# A frame | B one example trace of each condition with the raw EMG of the 8 decoder channels stacked under it | C bit rate per
# trial: hand control (bend sensor / Freebie) and the online KFs trained on each. A, B and C are the same height. The other
# online metrics are computed by analyze but not drawn.
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


# ---------------------------------------------------------------- analysis
def _session(cfg, r):
    """The improv session (live, with sender logs) whose frames cover most of the run's task window."""
    pred = Path(cfg['war_machine']['predictions']).expanduser() / r['day'].replace('-', '')
    lo, hi = r['window_unix']
    best, cover = None, 0
    for s in sorted(pred.iterdir()):
        f = s / f'proc_timestamps_cam{CAM}.npy'
        if not (f.exists() and (s / f'sender_sent_frame_nums_cam{CAM}.npy').exists()):
            continue
        t = np.load(f)
        c = min(hi, t[-1]) - max(lo, t[0])
        if c > cover:
            best, cover = s, c
    return best


def frame_for(cfg, r, t):
    """The camera frame behind the Freebie sample BRAND used at time t (BRAND clock): the last `vision_data` sample at
    or before t, matched by value to improv's sent packets. Returns that sample's time and the frame's details."""
    vt, vx = r['streams']['vision_raw']
    i = np.searchsorted(vt, t, side='right') - 1
    s = _session(cfg, r)
    L = lambda n: np.load(s / f'{n}.npy')
    sent, sent_frame = L(f'sender_sent_angles_cam{CAM}'), L(f'sender_sent_frame_nums_cam{CAM}')
    same = np.flatnonzero(sent.astype('float32') == np.float32(vx[i]))      # BRAND keeps the value as float32
    if len(same) != 1:
        raise ValueError(f'{r["run_id"]}: {len(same)} sent packets match the Freebie value {vx[i]} at t = {vt[i]}')
    j = int(same[0])
    frame = int(sent_frame[j])
    if len(L(f'saverstarts_cam_{CAM}')) != len(L('TISstarts_front_left')):
        print(f'  [S5 frame] {s.name}: the saver wrote a different number of frames than the camera delivered; '
              'video frame numbers may be off')
    k = int(np.searchsorted(L(f'proc_frame_nums_cam{CAM}'), frame))
    raw, smooth = L(f'raw_predictions_cam{CAM}')[k, 0], L(f'predictions_cam{CAM}')[k, 0]
    videos = sorted((Path.home() / 'camera_video' / r['day'] / f'{s.name[-4:]}00').glob('*.mp4'))
    return vt[i], {'session': s.name, 'frame': frame, 'video': str(videos[0]) if videos else None,
                   'keypoint_raw_xy': raw[:2].tolist(), 'keypoint_likelihood': float(raw[2]),
                   'keypoint_kalman_xy': smooth[:2].tolist(), 'sent_value': float(sent[j])}


def offline_kf(cfg, s5):
    """data/war_machine/s5_offline_kf.csv: a KF per hand-control run (no run has both sensors now, the manipulandum was
    off), trained on that run's neural data and the `cursor` stream = whichever sensor drove the cursor (bend, or Freebie)."""
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in wm.run_ids(cfg):
            r = wm.run(cfg, rid, neural=True)
            if s5['groups'][rid] not in ('bend', 'freebie'):
                continue
            print('training offline KF:', rid, flush=True)
            mt = wmf._train(cfg, r, 'cursor', tmp)
            rows.append({'run_id': rid, 'group': s5['groups'][rid], 'n_trials': len(wm.in_mode(r)),
                         **{k: mt[k] for k in (*OFFLINE, 'mse_valid')}})
    pd.DataFrame(rows).to_csv(OUT / 's5_offline_kf.csv', index=False)


def emg_bins(r, lo, hi, step=0.032):
    """32 ms MAV of each decoder channel on a grid over [lo, hi] (NaN where the neural stream has a gap): the feature the
    online KF reads. The master holds 4 ms bins of |x|, so a 32 ms MAV is the mean of the bins inside it."""
    nt, nx = r['neural']
    c = np.r_[np.zeros((1, nx.shape[1])), np.cumsum(nx, 0)]
    g = np.arange(lo, hi, step)
    i1, i0 = np.searchsorted(nt, g + step, 'right'), np.searchsorted(nt, g, 'right')
    n = i1 - i0
    mav = (c[i1] - c[i0]) / np.maximum(n, 1)[:, None]
    mav[n < 6] = np.nan                                # a 32 ms bin holds 8 4 ms bins; fewer means a stream gap
    return g, mav


def raw_emg(sub, run_id, t0, t1, out):
    """The raw 2 kHz samples of the decoder channels between two BRAND times, read from the run's rdb (a copy of it is loaded
    by war_machine_rdb; the server file is only read)."""
    wmc = sub['war_machine']
    day, run = run_id.split('_')
    cmd = [str(Path(wmc['rdb_python']).expanduser()), str(wm.HERE / 'war_machine_rdb.py'), '--raw',
           str(Path(wmc['server']).expanduser() / day / run / f'{run}.rdb'), str(out), str(t0), str(t1),
           *map(str, wmc['decoder_channels'])]
    for attempt in range(6):        # the redis env's interpreter / redis-server occasionally dies at startup; retry
        if subprocess.run(cmd, capture_output=True).returncode == 0:
            return np.load(out)
        print(f'    raw EMG extraction failed, retrying ({attempt + 1})', flush=True)
    raise RuntimeError(f'{run_id}: raw EMG extraction kept failing')


def save_examples(sub, s5):
    """Per example (Freebie without the manipulandum, bend with it): its trials, the signal that drove the cursor and the raw
    EMG of the decoder channels over the same window (time from the first trial's start)."""
    for key, (cfg_key, _) in EXAMPLES.items():
        wmf._save_example(sub, cfg_key, ['cursor'], f's5_example_{key}')
        ex, trials = wmf._load_example(f's5_example_{key}')
        t0, t1 = trials['start_time'].iloc[0], trials['stop_time'].iloc[-1]
        z = raw_emg(sub, s5[cfg_key]['run'], t0, t1, OUT / f's5_emg_{key}.npz')
        np.savez(OUT / f's5_emg_{key}.npz', t=z['t'] - t0, x=z['x'], run=s5[cfg_key]['run'])


def emg_scale(sub, s5):
    """data/war_machine/s5_emg_scale.csv: how much bigger the EMG is without the manipulandum. Per hand-control run, the
    mean over the decoder channels of the 32 ms MAV over the task window: its median, 5th-95th percentile range (dynamic
    range) and the per-channel medians, then the Freebie runs (pooled) over the bend run."""
    vals, chan = {}, {}
    for rid in wm.run_ids(sub):
        if s5['groups'][rid] not in ('bend', 'freebie'):
            continue
        r = wm.run(sub, rid, neural=True)
        tr = wm.in_mode(r)
        _, mav = emg_bins(r, tr['start_time'].iloc[0], tr['stop_time'].iloc[-1])
        ok = ~np.isnan(mav).any(axis=1)
        vals[rid], chan[rid] = np.nanmean(mav[ok], axis=1), mav[ok]
    rows = []
    for name, ids in {**{rid: [rid] for rid in vals}, 'freebie_pooled': [r for r in vals if s5['groups'][r] == 'freebie']}.items():
        v, c = np.concatenate([vals[i] for i in ids]), np.concatenate([chan[i] for i in ids])
        p5, p50, p95 = np.percentile(v, [5, 50, 95])
        rows.append({'run_id': name, 'group': s5['groups'].get(name, 'freebie'), 'n_bins': len(v), 'mean': v.mean(), 'median': p50,
                     'p5': p5, 'p95': p95, 'range_p95_p5': p95 - p5,
                     **{f'ch{k}_median': np.median(c[:, k]) for k in range(c.shape[1])}})
    df = pd.DataFrame(rows)
    bend = df[df['group'] == 'bend'].iloc[0]
    for col in ('mean', 'median', 'p95', 'range_p95_p5', *[c for c in df if c.endswith('_median') and c.startswith('ch')]):
        df[f'x_{col}'] = df[col] / bend[col]
    df.to_csv(OUT / 's5_emg_scale.csv', index=False)
    print(df[['run_id', 'group', 'n_bins', 'median', 'p5', 'p95', 'range_p95_p5', 'x_mean', 'x_median', 'x_p95', 'x_range_p95_p5']].round(2).to_string(index=False))
    ch = [c for c in df if c.startswith('x_ch')]
    print('per-channel median ratio, Freebie pooled / bend:', df.loc[df['run_id'] == 'freebie_pooled', ch].round(1).to_numpy().ravel().tolist())


def offline_kf_folds(cfg, s5):
    """data/war_machine/s5_offline_kf_folds.csv + _summary.csv: the offline KF on `kf_folds` chronological blocks per run
    (a window of half the run each, the library's 80/20 chronological split inside it, so the validation segments are
    consecutive, non-overlapping tenths of the run), for a spread of validation metrics and a 95% CI (t, folds - 1)."""
    from scipy import stats
    K, rows = s5['kf_folds'], []
    with tempfile.TemporaryDirectory() as tmp:
        for rid in wm.run_ids(cfg):
            if s5['groups'][rid] not in ('bend', 'freebie'):
                continue
            r = wm.run(cfg, rid, neural=True)
            lo, hi = wmf._window(r)
            for k, end in enumerate(np.linspace(0.5, 1, K)):
                w = (lo + (end - 0.5) * (hi - lo), lo + end * (hi - lo))
                mt = wmf._train(cfg, r, 'cursor', tmp, window=w)
                rows.append({'run_id': rid, 'group': s5['groups'][rid], 'fold': k, **{c: mt[c] for c in OFFLINE}})
                print('  offline KF fold', rid, k, flush=True)
    folds = pd.DataFrame(rows)
    folds.to_csv(OUT / 's5_offline_kf_folds.csv', index=False)
    out = []
    for name, g in [*folds.groupby('run_id'), *[(f'{k} (pooled folds)', folds[folds['group'] == k]) for k in ('bend', 'freebie')]]:
        row = {'unit': name, 'n_folds': len(g)}
        for c in OFFLINE:
            v = g[c].to_numpy()
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            row.update({f'{c}_mean': v.mean(), f'{c}_sd': v.std(ddof=1), f'{c}_median': np.median(v),
                        f'{c}_ci95_lo': v.mean() - half, f'{c}_ci95_hi': v.mean() + half})
        out.append(row)
    summary = pd.DataFrame(out)
    summary.to_csv(OUT / 's5_offline_kf_summary.csv', index=False)
    print(summary.round(3).T.to_string(header=False))


def analyze(cfg):
    sub = wm.sub_config(cfg, KEY)
    s5 = cfg[KEY]
    wm.build(sub)                   # resumes: only runs not yet in <master>/war_machine/ are read from the server
    wmf.export_online_metrics(sub, prefix='s5_')
    for name in ('s5_run_metrics', 's5_trial_metrics'):       # what each run is, from the session notes (config groups)
        df = pd.read_csv(OUT / f'{name}.csv')
        df['group'] = df['run_id'].map(s5['groups'])
        df.to_csv(OUT / f'{name}.csv', index=False)
    save_examples(sub, s5)
    emg_scale(sub, s5)

    ex, trials = wmf._load_example('s5_example_freebie')
    t0 = trials['start_time'].iloc[0]
    if s5['frame_time_s'] is None:
        inner = trials.iloc[1:-1]                      # not at the trace's edges
        row = inner[inner['success']].sort_values('target').iloc[-1]
        t = row['stop_time'] - HOLD_END_S
    else:
        t = t0 + s5['frame_time_s']
    r = wm.run(sub, s5['example']['run'])
    tb, info = frame_for(sub, r, t)
    info.update(run=r['run_id'], time_s=float(tb - t0), position=float(np.interp(tb, ex['cursor_t'], ex['cursor'])))
    (OUT / 's5_frame.json').write_text(json.dumps(info, indent=2))
    offline_kf(sub, s5)
    offline_kf_folds(sub, s5)
    out = ROOT / 'assets' / 'figS5_frame.png'
    print(f"fig S5 frame: {info['session']} camera frame {info['frame']} ({info['time_s']:.2f} s into the trace). Write it with\n"
          f"  ffmpeg -i \"{info['video']}\" -vf \"select=eq(n\\,{info['frame']})\" -vsync 0 -frames:v 1 {out}\n"
          f"then set config war_machine_s5.frame_image: assets/figS5_frame.png")


# ---------------------------------------------------------------- figure
def _nice(v, up=False):
    """v rounded down (up) to 1, 2 or 5 x 10^k."""
    k = 10 ** np.floor(np.log10(v))
    return float(k * (min(f for f in (1, 2, 5, 10) if f * k >= v) if up else max(f for f in (1, 2, 5) if f * k <= v)))


def load(cfg):
    s5 = cfg[KEY]
    trace = {k: wmf._load_example(f's5_example_{k}') for k in EXAMPLES}
    emg = {k: dict(np.load(OUT / f's5_emg_{k}.npz')) for k in EXAMPLES}
    for e in emg.values():
        e['x'] = e['x'] * UV_PER_COUNT                  # counts -> microvolts
    tmax = max(t['stop_time'].iloc[-1] - t['start_time'].iloc[0] for _, t in trace.values())
    spacing = EMG_SPACING * max(np.percentile(np.abs(e['x'] - np.median(e['x'], axis=0)), 99.5) for e in emg.values())
    img = plt.imread(ROOT / s5['frame_image']) if s5['frame_image'] and not LAYOUT_CHECK else None
    return {'runs': pd.read_csv(OUT / 's5_run_metrics.csv'), 'trials': pd.read_csv(OUT / 's5_trial_metrics.csv'),
            'trace': trace, 'emg': emg, 'tmax': tmax, 'emg_spacing': spacing, 'emg_bar': _nice(spacing, up=True), 'frame': json.loads((OUT / 's5_frame.json').read_text()), 'img': img,
            'crop': s5['crop'], 'offline': pd.read_csv(OUT / 's5_offline_kf.csv'), 'show_trials': cfg['war_machine']['all_trials']}


def panel_frame(ax, d):
    """The camera frame with the live Freebie keypoint (the DLC output for that frame, before the Kalman filter)."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    if d['img'] is None:
        ax.text(0.5, 0.5, 'Frame not extracted yet\n(see analyze)', ha='center', va='center', transform=ax.transAxes)
        return
    img = d['img']
    scale = img.shape[1] / NATIVE_W
    x, y = (np.array(d['frame']['keypoint_raw_xy']) * scale)
    x0, x1, y0, y1 = [int(round(v * scale)) for v in d['crop']] if d['crop'] else (0, img.shape[1], 0, img.shape[0])
    ax.imshow(img[y0:y1, x0:x1], aspect='equal', cmap='gray' if img.ndim == 2 else None)
    ax.scatter([x - x0], [y - y0], s=utils.marker_area(1), color=COLORS['freebie'], edgecolors='white',
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
    if not LAYOUT_CHECK:
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


def panel_offline(ax, d, metric):
    off = d['offline']
    groups = [(GROUPS[k], off.loc[off['group'] == k, metric]) for k in ('bend', 'freebie')]
    utils.boxes(ax, groups, [COLORS['bend'], COLORS['freebie']], ylabel=OFFLINE[metric], title='Offline KF, index',
                unit='run')


PANELS = {'frame': panel_frame, **{f'trace_{k}': partial(panel_trace, key=k) for k in EXAMPLES},
          **{f'emg_{k}': partial(panel_emg, key=k) for k in EXAMPLES}, 'bit': partial(panel_metric, metric='bitrate_bits_per_s')}


def spread(fig, axd):
    """Left to right in inches: A (the crop's aspect), B (two pairs, each a behaviour trace touching its raw EMG, one x axis),
    C (the bit-rate boxes at the usual box width). All three have A's height and the same top."""
    pos = lambda k: axd[k].get_position()
    W, H = fig.get_size_inches()
    top = pos('trace_freebie').y1
    crop = d_crop[0]
    aspect = (crop[1] - crop[0]) / (crop[3] - crop[2]) if crop else 960 / 720
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
    d_crop[0] = data['crop']
    return compose(LAYOUT, PANELS, data, SIZE, 'figS5_no_manipulandum',
                   labels={'trace_freebie': 'B', 'bit': 'C'},
                   stats_bars=cfg['stats_bars']['S5'], height_ratios=HEIGHT_RATIOS, post_layout=spread)
