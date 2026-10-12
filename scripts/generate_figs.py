"""
Make the paper figures and their statistics from data/ and assets/.

    python scripts/generate_figs.py                 # every figure enabled in config.yaml
    python scripts/generate_figs.py --figs 3 S1     # just some of them
"""
import argparse

import matplotlib
matplotlib.use('Agg')

import utils
import fes_figure, hand_control_old_system_figure, human_dof_figure, kalman_filter_old_system_figure, \
    latency_old_pipeline_figure, no_manipulandum_figure, validation_figure
import war_machine_figures as wmf

FIGURES = {
    '2': validation_figure.make_figure,                  # validation, held-out frames
    '3': wmf.make_fig3,                                  # War Machine vision latency
    '4': wmf.make_fig4,                                  # War Machine hand control
    '5': wmf.make_fig5,                                  # War Machine Kalman filter
    '6': fes_figure.make_figure,                         # FES
    'S1': latency_old_pipeline_figure.make_figure,       # old pipeline latency (next to War Machine)
    'S2': hand_control_old_system_figure.make_figure,    # real-time hand control, old system
    'S3': human_dof_figure.make_figure,                  # human 2-4 DOF hand control
    'S4': kalman_filter_old_system_figure.make_figure,   # Kalman filter, old system
    'S5': no_manipulandum_figure.make_figure,            # War Machine without the manipulandum
}

parser = argparse.ArgumentParser()
parser.add_argument('--figs', nargs='+', choices=list(FIGURES), help='override config: which figures')
args = parser.parse_args()

cfg = utils.load_config()
utils.apply_style(cfg)
for key in args.figs or [str(k) for k, on in cfg['figures'].items() if on]:
    FIGURES[key](cfg)
utils.write_all_stats()       # figures/all_stats.csv: every figure's descriptives + tests, from the csvs on disk
