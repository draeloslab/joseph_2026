"""
Make the paper's figures and their statistics.

    python scripts/generate_figs.py                  # draw every figure from data/ and assets/ -> figures/
    python scripts/generate_figs.py --figs 3 S2      # just some of them
    python scripts/generate_figs.py --from-raw       # first recompute data/ from raw/ -> data_from_raw/, draw -> figures_from_raw/
"""
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--figs', nargs='+', help='which figures (default: all)')
parser.add_argument('--from-raw', action='store_true',
                    help='recompute every number from raw/ (needs the raw package and neuraldecoding; see README)')
args = parser.parse_args()
if args.from_raw:
    os.environ['FREEBIE_FROM_RAW'] = '1'        # utils reads this when it is first imported

import matplotlib
matplotlib.use('Agg')

import utils
import fes_figure, hand_control_old_system_figure, human_dof_figure, kalman_filter_old_system_figure, \
    latency_old_pipeline_figure, no_manipulandum_figure, validation_figure
import monkey_r_figures

FIGURES = {
    '2': validation_figure.make_figure,                  # Freebie keypoints vs manual labels, held-out frames
    '3': monkey_r_figures.make_fig3,                     # Monkey R vision-pipeline latency
    '4': monkey_r_figures.make_fig4,                     # Monkey R hand control
    '5': monkey_r_figures.make_fig5,                     # Monkey R Kalman filter
    '6': fes_figure.make_figure,                         # FES
    'S1': human_dof_figure.make_figure,                  # human 2-4 DOF hand control
    'S2': latency_old_pipeline_figure.make_figure,       # Monkey N pipeline latency, next to Monkey R
    'S3': hand_control_old_system_figure.make_figure,    # Monkey N hand control
    'S4': kalman_filter_old_system_figure.make_figure,   # Monkey N Kalman filter
    'S5': no_manipulandum_figure.make_figure,            # Monkey R without the manipulandum
}
figs = args.figs or list(FIGURES)
if unknown := [k for k in figs if k not in FIGURES]:
    parser.error(f'unknown figure(s) {unknown}; choose from {list(FIGURES)}')

cfg = utils.load_config()
utils.apply_style(cfg)
if args.from_raw:
    import analysis
    done = analysis.run(cfg, figs)
for key in figs:
    FIGURES[key](cfg)
utils.write_all_stats()       # <figures>/all_stats.csv: every figure's descriptives + tests, from the csvs on disk
if args.from_raw:
    from analysis.compare import compare
    compare(done).to_csv(utils.FIGS / 'rederivation_check.csv', index=False)
