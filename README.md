# joseph_2026

Code and data for the paper's figures 2–6 and supplementary figures S1–S5. Every figure and every statistic in it is
drawn from `data/` (the numbers behind each panel) and `assets/` (the camera frames and the task screen shown in the
figures).

```
config.yaml                       which figures to make, style + palette, significance bars
scripts/generate_figs.py          entry point
scripts/utils.py                  style, figure composition, shared panels, significance tests
scripts/validation_figure.py      fig 2   keypoint validation on held-out frames
scripts/war_machine_figures.py    figs 3, 4, 5   War Machine: vision latency, hand control, Kalman filter
scripts/fes_figure.py             fig 6   FES
scripts/latency_old_pipeline_figure.py       fig S1  latency of the earlier pipeline, next to War Machine
scripts/hand_control_old_system_figure.py    fig S2  real-time hand control, earlier system
scripts/human_dof_figure.py                  fig S3  human 2-4 DOF hand control
scripts/kalman_filter_old_system_figure.py   fig S4  Kalman filter, earlier system
scripts/no_manipulandum_figure.py            fig S5  War Machine without the manipulandum
data/                             the numbers each figure is drawn from (see below)
assets/                           the images drawn in figs 2, 6, S3 and S5
figures/                          output: <fig>.svg / .png, <fig>_stats.csv, <fig>_descriptives.csv, all_stats.csv
```

## Make the figures
Needs Python 3 with numpy, pandas, scipy, matplotlib >= 3.9, pyyaml and pillow.
```
python scripts/generate_figs.py               # every figure enabled in config.yaml
python scripts/generate_figs.py --figs 3 S1   # just some of them
```
Each figure is saved to `figures/` as SVG and PNG at 180 mm wide (double column), with
`<fig>_descriptives.csv` (n, mean, sd, median, ... of every group drawn) and `<fig>_stats.csv` (every significance
test: two-sided Mann-Whitney U for independent groups, Wilcoxon signed-rank for paired comparisons, Holm-corrected
within each panel). `figures/all_stats.csv` collects both for every figure.

## Data
| Figure | Files |
|---|---|
| 2 | `keypoint_errors_heldout.npz` (error per held-out frame and keypoint, px), `keypoint_labels_heldout.npz` (labels / predictions on the two example frames); `assets/fig2_top.png`, `assets/fig2_side.png` |
| 3 | `war_machine/latency_runs.csv`, `latency_e2e_frames.npz` (end-to-end latency of every frame, per run), `latency_drops.csv` |
| 4 | `war_machine/run_metrics.csv`, `trial_metrics.csv`, `sensor_corr.csv`, `fig4_example.npz` + `_trials.csv` |
| 5 | `war_machine/run_metrics.csv`, `trial_metrics.csv`, `offline_kf.csv`, `fig5_example_{freebie,bend}.npz` + `_trials.csv` |
| 6 | `fig6_fes.npz` (keypoints, traces, stimulation windows), `fig6_detection.csv` (panel A: % of steps with the index fingertip detected, per camera and across cameras); `assets/fig6_{rt,stim}_{0,1}.png` |
| S1 | `e2e_pooled.npz`, `latency_stage_summary.csv`, and fig 3's `war_machine/latency_*` |
| S2 | `run_metrics.csv`, `run_trial_metrics.csv`, `freebie_bend_corr.csv`, `figS2_example_trace.npz` + `_trials.csv` |
| S3 | `figS3_human/runs.csv`, `trials.csv`, `e2e_cam_frames.npz`; `assets/figS3_task_screen.png` |
| S4 | `run_metrics.csv`, `run_trial_metrics.csv`, `offline_kf.csv`, `figS4_example_{freebie,bend}_trace.npz` + `_trials.csv` |
| S5 | `war_machine/figS5_run_metrics.csv`, `figS5_trial_metrics.csv`, `figS5_example_{freebie,bend}.npz` + `_trials.csv`, `figS5_emg_{freebie,bend}.npz`, `figS5_frame.json`; `assets/figS5_frame.png` |

All paths are under `data/` unless they start with `assets/`. Example traces (`*_example*.npz`) hold the time series
drawn, `*_trials.csv` their trials (start / stop time, target, target radius, success). Run and trial tables hold one row
per run / per trial with the online metrics: success rate, bit rate (bits/s), targets per second, time to target (s),
orbiting time (s) and path efficiency. Success rate and targets per second have no per-trial value, so the trial
tables carry a running rate up to each trial (`*_running`); bit rate and the time / path metrics are per trial.

The images in `assets/` are the frames as they appear in the figures (cropped, faces masked in fig 6) before the
keypoints, skeletons and labels are drawn on them. Fig 6 keypoints are in the cropped images' pixels; fig 2 labels
are in the full camera frame's pixels (the crop's first column is `<view>_crop_x0`), fig S5's keypoint in the 960 px
camera frame (the crop is in `figS5_frame.json`).

## Changing a figure
Each figure module has one function per panel and a `LAYOUT` (matplotlib `subplot_mosaic`) at the top; move or
reorder panels by editing `LAYOUT`. Fonts, line widths, the colour palette and whether each figure draws
significance bars are in `config.yaml`. Every figure is drawn in its own design inches and saved at
`style.figure_width_mm`: fonts, lines and markers in `style` are the printed sizes.
