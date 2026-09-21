"""
Make the paper figures.

    python scripts/generate_figs.py                 # figures enabled in config.yaml, from data/
    python scripts/generate_figs.py --figs 2 rt_bonus
    python scripts/generate_figs.py --analysis      # rerun the analysis from data/master.pkl first
    python scripts/generate_figs.py --analysis --retrain   # ... and retrain the offline KFs (else cached)
"""
import argparse

import matplotlib
matplotlib.use('Agg')

import utils
import validation_figure, latency_figure, hand_control_figure, kf_decode_figure

FIGURES = {
    '1': (validation_figure.analyze, validation_figure.make_figure),
    '2': (latency_figure.analyze, latency_figure.make_figure),
    'rt_bonus': (latency_figure.analyze_rt, latency_figure.make_rt_figure),
    '3': (hand_control_figure.analyze, hand_control_figure.make_figure),
    '4': (kf_decode_figure.analyze, kf_decode_figure.make_figure),
}

parser = argparse.ArgumentParser()
parser.add_argument('--figs', nargs='+', help='override config: which figures (1 2 3 4 rt_bonus)')
parser.add_argument('--analysis', action='store_true', help='override config: rerun the analysis from the master file first')
parser.add_argument('--retrain', action='store_true', help='retrain the cached offline KFs instead of reusing them')
args = parser.parse_args()

cfg = utils.load_config()
utils.apply_style(cfg)
figs = args.figs or [str(k) for k, on in cfg['figures'].items() if on]
run_analysis = args.analysis or cfg['run_analysis']
cfg['retrain'] = args.retrain

for key in figs:
    analyze, make_figure = FIGURES[key]
    if run_analysis:
        analyze(cfg)
    make_figure(cfg)
