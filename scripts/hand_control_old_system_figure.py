"""Figure S3: Freebie tracking vs bend sensors during real-time hand control, Monkey N (earlier system)."""
from functools import partial

import numpy as np
import pandas as pd

from utils import COLORS, DATA, ONLINE_METRICS, boxes, compose, metric_panel, split_groups, target_trace

SIGNALS = {1: [('index_angle', 'Freebie index angle'), ('pip_y', 'Freebie PIP y')], 3: [('mrs_x', 'Freebie MRS x')]}

# grid in box units: every boxplot panel spans one column per box (3 here), so boxes come out the same width
m = list(ONLINE_METRICS)
LAYOUT = [['trace_1'] * 4 + ['corr'] * 3 + ['.'] * 2,
          ['trace_3'] * 4 + ['corr'] * 3 + ['.'] * 2,
          [k for k in m[:3] for _ in range(3)],
          [k for k in m[3:] for _ in range(3)]]
SIZE = (20, 17)
HEIGHT_RATIOS = [0.6, 0.6, 1, 1]
TRACE_T_MAX = 6        # s; panel A only shows the first 6 s of the example trials


def load():
    runs = pd.read_csv(DATA / 'run_metrics.csv')
    hand = runs[runs['mode'] == 'hand_control']
    return {'trace': dict(np.load(DATA / 'figS3_example_trace.npz')),
            'ex_trials': pd.read_csv(DATA / 'figS3_example_trials.csv'),
            'corr': pd.read_csv(DATA / 'freebie_bend_corr.csv'),
            'groups': split_groups(hand), 'trials': pd.read_csv(DATA / 'run_trial_metrics.csv')}


# ---------------------------------------------------------------- panels
def panel_trace(ax, d, dof):
    tr = d['trace']
    lines = [(tr['t'], tr[f'bend_{dof}'], 'Bend sensor', COLORS['bend'], 2.5)]
    lines += [(tr[f'{key}_t'], tr[key], label, COLORS[key], 2) for key, label in SIGNALS[dof]]
    target_trace(ax, d['ex_trials'], f'target_{dof}', lines, f"{'Index' if dof == 1 else 'MRS'}\nposition (a.u.)")
    ax.set_xlim(0, TRACE_T_MAX)
    if dof == 1:
        ax.set_title('Hand control example', pad=34)   # clear the legend row above the axes


def panel_corr(ax, d):
    c = d['corr'].set_index('session_id').abs()
    boxes(ax, [('Index angle\nvs index', c['corr_index_angle']), ('PIP y\nvs index', c['corr_pip_y']),
               ('MRS x\nvs MRS', c['corr_mrs_x'])],
          [COLORS['index_angle'], COLORS['pip_y'], COLORS['mrs_x']],
          ylabel='|Pearson r| (per session)', title='Freebie vs bend sensor', test='paired', unit='session')
    ax.set_ylim(bottom=0)


def panel_metric(ax, d, metric):
    # every trial drawn; descriptives and tests on one mean per session
    metric_panel(ax, d['groups'], metric, stats_trials=d['trials'], by_session=True)


PANELS = {'trace_1': partial(panel_trace, dof=1), 'trace_3': partial(panel_trace, dof=3), 'corr': panel_corr,
          **{m: partial(panel_metric, metric=m) for m in ONLINE_METRICS}}
LABELS = {'trace_1': 'A', 'corr': 'B', 'success_rate': 'C'}


def make_figure(cfg):
    return compose(LAYOUT, PANELS, load(), SIZE, 'figS3_hand_control_old_system', labels=LABELS,
                   stats_bars=cfg['stats_bars']['S3'], height_ratios=HEIGHT_RATIOS)
