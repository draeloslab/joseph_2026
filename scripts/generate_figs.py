"""
Make the paper figures.

    python scripts/generate_figs.py                 # figures enabled in config.yaml, from data/
    python scripts/generate_figs.py --figs 2 2_supplement
    python scripts/generate_figs.py --analysis      # rerun the analysis from the master folder first
    python scripts/war_machine.py                   # (re)build the War Machine master from the server first (figs 2-4)
    python scripts/generate_figs.py --analysis --retrain   # ... and retrain the offline KFs (else cached)
"""
import argparse

import matplotlib
matplotlib.use('Agg')

import utils
import validation_figure, validation_figure_testonly, latency_figure, fig3_supplement_figure, fig4_supplement_figure, fes_figure, human_figure
import war_machine_figures as wmf

FIGURES = {
    '1': (validation_figure.analyze, validation_figure.make_figure),
    '1_heldout': (validation_figure_testonly.analyze, validation_figure_testonly.make_figure),
    '2': (wmf.analyze_fig2, wmf.make_fig2),                                    # War Machine (config war_machine)
    '2_supplement': (latency_figure.analyze, latency_figure.make_figure),     # pooled earlier sessions + War Machine
    '3': (wmf.analyze_fig3, wmf.make_fig3),
    '3_supplement': (fig3_supplement_figure.analyze, fig3_supplement_figure.make_figure),
    '4': (wmf.analyze_fig4, wmf.make_fig4),
    '4_supplement': (fig4_supplement_figure.analyze, fig4_supplement_figure.make_figure),
    '5': (fes_figure.analyze, fes_figure.make_figure),
    '6': (human_figure.analyze, human_figure.make_figure),
}

parser = argparse.ArgumentParser()
parser.add_argument('--figs', nargs='+', help='override config: which figures (1 1_heldout 2 2_supplement 3 3_supplement 4 4_supplement 5 6); 2-4 are War Machine')
parser.add_argument('--analysis', action='store_true', help='override config: rerun the analysis from the master file first')
parser.add_argument('--retrain', action='store_true', help='retrain the cached offline KFs instead of reusing them')
parser.add_argument('--dofs', nargs='+', type=int, help='fig 6: which DOF tasks to draw, e.g. --dofs 1 2 (default: config human.plot_dofs)')
args = parser.parse_args()

cfg = utils.load_config()
utils.apply_style(cfg)
figs = args.figs or [str(k) for k, on in cfg['figures'].items() if on]
run_analysis = args.analysis or cfg['run_analysis']
cfg['retrain'] = args.retrain
if args.dofs:
    cfg['human']['plot_dofs'] = args.dofs

for key in figs:
    analyze, make_figure = FIGURES[key]
    if run_analysis:
        analyze(cfg)
    make_figure(cfg)
utils.write_all_stats()       # figures/all_stats.csv: every figure's descriptives + tests, from the csvs on disk
