"""
From-scratch analysis: raw/ -> data_from_raw/, in the layout of data/. Run with `python scripts/generate_figs.py --from-raw`.

One analysis per figure. They recompute every number in data/ except the files in STATIC, which come from upstream steps
that the raw package does not cover (frames and their labels, the FES panel traces) and are copied from data/.
"""
import shutil

import utils
from . import fes, human, monkey_r, old_system, validation

ANALYSES = {'2': validation.analyze, '3': monkey_r.analyze_fig3, '4': monkey_r.analyze_fig4, '5': monkey_r.analyze_fig5,
            '6': fes.analyze, 'S1': human.analyze, 'S2': old_system.analyze_s2, 'S3': old_system.analyze_s3,
            'S4': old_system.analyze_s4, 'S5': monkey_r.analyze_s5}
NEEDS = {'S2': ['3']}        # fig S2 draws fig 3's latency next to its own
STATIC = ['fig2_keypoint_labels.npz', 'fig6_fes.npz', 'monkey_r/figS5_frame.json']
_EX = ['fig4_example', 'fig5_example_freebie', 'fig5_example_bend']
PRODUCES = {            # the files of data_from_raw/ each analysis writes (compared with data/ afterwards)
    '2': ['fig2_keypoint_errors.npz'],
    '3': ['monkey_r/latency_runs.csv', 'monkey_r/latency_e2e_frames.npz', 'monkey_r/latency_drops.csv'],
    '4': ['monkey_r/run_metrics.csv', 'monkey_r/trial_metrics.csv', 'monkey_r/sensor_corr.csv',
          'monkey_r/fig4_example.npz', 'monkey_r/fig4_example_trials.csv'],
    '5': ['monkey_r/run_metrics.csv', 'monkey_r/trial_metrics.csv', 'monkey_r/offline_kf.csv',
          *[f'monkey_r/{e}{x}' for e in _EX[1:] for x in ('.npz', '_trials.csv')]],
    '6': ['fig6_detection.csv'],
    'S1': ['figS1_human/runs.csv', 'figS1_human/trials.csv', 'figS1_human/e2e_cam_frames.npz'],
    'S2': ['e2e_pooled.npz', 'latency_stage_summary.csv'],
    'S3': ['run_metrics.csv', 'run_trial_metrics.csv', 'freebie_bend_corr.csv', 'figS3_example_trace.npz', 'figS3_example_trials.csv'],
    'S4': ['run_metrics.csv', 'run_trial_metrics.csv', 'offline_kf.csv',
           *[f'figS4_example_{k}{x}' for k in ('freebie', 'bend') for x in ('_trace.npz', '_trials.csv')]],
    'S5': ['monkey_r/figS5_run_metrics.csv', 'monkey_r/figS5_trial_metrics.csv']}


def run(cfg, figs):
    if not utils.RAW.exists():
        raise SystemExit(f'{utils.RAW} not found: unzip raw_data.zip into the repository folder first')
    utils.use_neuraldecoding()
    utils.DATA.mkdir(exist_ok=True)
    for f in STATIC:
        (utils.DATA / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(utils.ROOT / 'data' / f, utils.DATA / f)
    done = []
    for key in figs:
        for k in [*NEEDS.get(key, []), key]:
            if k not in done:
                print(f'--- analysis for figure {k}', flush=True)
                ANALYSES[k](cfg)
                done.append(k)
    return done
