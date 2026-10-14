# Freebie: figure code, data and analysis

Code and data that draw every figure in the paper (figures 2–6 and supplementary figures S1–S5), compute every statistic
reported in them, and, optionally, recompute every number in them from the raw inputs. There are two packages:

| Package | Contents | Needed for |
|---|---|---|
| `figure_data.zip` | `data/` (the numbers behind each panel), `assets/` (the images shown in figures 2, 6, S1, S5) | drawing the figures and statistics |
| `raw_data.zip` | `raw/` (per-trial, per-frame and per-run inputs, and trimmed NWB files) | recomputing `data/` from scratch (`--from-raw`) |

## Quick start: draw the figures and statistics
```
git clone https://github.com/draeloslab/joseph_2026 && cd joseph_2026
python -m venv .venv && source .venv/bin/activate        # or: conda create -n freebie python=3.11 && conda activate freebie
pip install -r requirements.txt
# download figure_data.zip into this folder, then:
unzip figure_data.zip                                     # creates data/ and assets/
python scripts/generate_figs.py                           # every figure + its statistics -> figures/
```
Requires Python ≥ 3.10 
`python scripts/generate_figs.py --figs 3 S2` draws just some figures.

## Recompute everything from the raw data
`data/` is derived from `raw/` by the code in `scripts/analysis/`: online metrics (bit rate, time to target, ...), pipeline
latency and dropped frames, Freebie-vs-bend correlations, offline Kalman filters. `--from-raw` reruns all of it, then draws
every figure from the recomputed numbers.
```
# 1. the lab's neuraldecoding package
git clone <neuraldecoding repository URL> ../neuraldecoding
pip install -r ../neuraldecoding/requirements.txt && pip install -r requirements-raw.txt
pip install -e ../neuraldecoding                          # or set NEURALDECODING_PATH to the clone instead
# 2. the packages: figure_data.zip and raw_data.zip, both unzipped into this folder (creates data/, assets/, raw/)
python scripts/generate_figs.py --from-raw                # about 20 minutes on one CPU core
```
- `data_from_raw/` has the layout of `data/`; `figures_from_raw/` the layout of `figures/`.
- At the end the run compares every recomputed file with the shipped one. `figures_from_raw/rederivation_check.csv` lists
  each as `identical`, `within tolerance` (relative difference ≤ 1e-6, floating-point
  noise) or `DIFFERENT`.
- `--figs` works here too: `python scripts/generate_figs.py --from-raw --figs 4 5`.
- neuraldecoding is used for the online metrics (`neuraldecoding.utils.online_metrics`), for reading the NWB files, and
  for training the Kalman filters. Its `requirements.txt` lists PyWavelets and it imports networkx without listing it,
  both are in `requirements-raw.txt`.

### What the analysis does not recompute
Three files of `data/` come from steps that start from video frames, which are not part of either package. They are
copied unchanged into `data_from_raw/`:
- `fig2_keypoint_labels.npz`: Freebie's prediction and the manual label on the two example frames of figure 2;
- `fig6_fes.npz`: the keypoints and traces of the four example frames and the two line plots of figure 6;
- `monkey_r/figS5_frame.json`: the live keypoint on the frame shown in figure S5A, and its position in the example trial.

### What `raw/` holds
| Path | Content | Feeds |
|---|---|---|
| `fig2_heldout_predictions.npz` | per held-out frame, Freebie's prediction, the manual label and the prediction's likelihood, per video | 2 |
| `monkey_r/runs.csv`, `trials/`, `runs/` | per run: trials, bend sensor, Freebie and cursor signals, per-frame pipeline stage times, the 8 decoder channels (hand-control runs, 4 ms bins of mean absolute EMG), camera and processor frame logs | 3, 4, 5, S5 |
| `monkey_n/runs.csv`, `nwb/` | per run of the earlier system: trimmed NWB (trials, finger angles, decoder output, EMG) | S3, S4 |
| `monkey_n/sessions.csv`, `sessions/` | per session: Freebie signals, clock offset, run window, bend packets inside it, per-frame latency | S2, S3, S4 |
| `human/` | per run: trials, cursor, vision packets; the pipeline's frame logs | S1 |
| `fes_live_run.npz`, `fes_detection_labels.csv` | per processing step of the live run, which cameras had the index fingertip, and what each stored detection is | 6 |

The file formats are documented at the top of each module in `scripts/analysis/`.

## Output
For each figure `figures/` holds:
- `<fig>.svg` and `<fig>.png`: the figure at its print width (180 mm, double column; fig S5 98 mm), with text and lines
  at the print sizes in `config.yaml` `style`.
- `<fig>_descriptives.csv`: n, mean, SD, SEM, median, quartiles and range of every group drawn. Online-metric panels
  also give the number of sessions, runs and trials, and run- and trial-level summaries.
- `<fig>_stats.csv`: every significance test. Two-sided Mann-Whitney U for independent groups and Wilcoxon signed-rank
  for paired comparisons, Holm-corrected within each panel. Each row records what was tested (trials, run means or
  session means).

`figures/all_stats.csv` collects the descriptives and tests of every figure in one table.

Text is set in Arial when it is installed, otherwise DejaVu Sans.

## Figures
| Figure | Content | Drawing script | Analysis (`--from-raw`) |
|---|---|---|---|
| 2 | Freebie keypoints vs manual labels on held-out frames | `validation_figure.py` | `analysis/validation.py` |
| 3 | Monkey R: vision-pipeline latency | `monkey_r_figures.py` | `analysis/monkey_r.py` |
| 4 | Monkey R: hand control, bend sensor vs Freebie | `monkey_r_figures.py` | `analysis/monkey_r.py` |
| 5 | Monkey R: Kalman filter decoders trained on each sensor; offline KFs | `monkey_r_figures.py` | `analysis/monkey_r.py` |
| 6 | Freebie driving FES | `fes_figure.py` | `analysis/fes.py` |
| S1 | Human 2–4 DOF hand control with 3D hand tracking | `human_dof_figure.py` | `analysis/human.py` |
| S2 | Monkey N: pipeline latency, next to Monkey R | `latency_old_pipeline_figure.py` | `analysis/old_system.py` |
| S3 | Monkey N: hand control, bend sensors vs Freebie | `hand_control_old_system_figure.py` | `analysis/old_system.py` |
| S4 | Monkey N: Kalman filter decoders trained on each sensor; offline KFs | `kalman_filter_old_system_figure.py` | `analysis/old_system.py` |
| S5 | Monkey R: hand control without the manipulandum | `no_manipulandum_figure.py` | `analysis/monkey_r.py` |

`scripts/generate_figs.py` is the entry point and `scripts/utils.py` holds the shared style, layout, box plots and
statistics. `config.yaml` sets the print style and colours, which figures draw significance bars (the tests are written
to the stats files either way) and the settings of the analysis (`analysis:`).

## Data
| Figure | Files (`data/` unless noted) |
|---|---|
| 2 | `fig2_keypoint_errors.npz` (error of every held-out frame per keypoint, px), `fig2_keypoint_labels.npz` (labels and predictions on the example frames); `assets/fig2_top.png`, `assets/fig2_side.png` |
| 3 | `monkey_r/latency_runs.csv`, `monkey_r/latency_e2e_frames.npz` (end-to-end latency of every frame, per run), `monkey_r/latency_drops.csv` |
| 4 | `monkey_r/run_metrics.csv`, `monkey_r/trial_metrics.csv`, `monkey_r/sensor_corr.csv`, `monkey_r/fig4_example.npz` + `_trials.csv` |
| 5 | `monkey_r/run_metrics.csv`, `monkey_r/trial_metrics.csv`, `monkey_r/offline_kf.csv`, `monkey_r/fig5_example_{freebie,bend}.npz` + `_trials.csv` |
| 6 | `fig6_fes.npz` (keypoints, traces, stimulation windows), `fig6_detection.csv` (% of steps with the index fingertip detected, per camera and across cameras); `assets/fig6_{rt,stim}_{0,1}.png` |
| S1 | `figS1_human/runs.csv`, `figS1_human/trials.csv`, `figS1_human/e2e_cam_frames.npz`; `assets/figS1_task_screen.png` |
| S2 | `e2e_pooled.npz`, `latency_stage_summary.csv`, and fig 3's `monkey_r/latency_*` |
| S3 | `run_metrics.csv`, `run_trial_metrics.csv`, `freebie_bend_corr.csv`, `figS3_example_trace.npz` + `_trials.csv` |
| S4 | `run_metrics.csv`, `run_trial_metrics.csv`, `offline_kf.csv`, `figS4_example_{freebie,bend}_trace.npz` + `_trials.csv` |
| S5 | `monkey_r/figS5_run_metrics.csv`, `monkey_r/figS5_trial_metrics.csv`, `monkey_r/figS5_frame.json`; `assets/figS5_frame.png` |

- **Run and trial tables** hold one row per run or per trial, with the online metrics: success rate, bit rate (bits/s),
  targets per second, time to target (s), orbiting time (s) and path efficiency. Success rate and targets per second
  have no per-trial value, so the trial tables give the running rate up to each trial (`*_running`).
- **Example traces** (`*example*.npz`) hold the time series drawn; their `*_trials.csv` hold those trials' start and
  stop times, targets, target radius and success.
- **Images in `assets/`** are the frames as they appear in the figures (cropped; faces masked in fig 6) before any
  keypoints, skeletons or labels are drawn. Fig 6 keypoints are in the cropped images' pixels. Fig 2 labels are in the
  full camera frame's pixels, with the crop's first column stored as `<view>_crop_x0`. Fig S5's keypoint is in the
  960 × 720 px camera frame, with the crop in `figS5_frame.json`.

## Changing a figure
Each figure module has one function per panel and a `LAYOUT` (a matplotlib `subplot_mosaic`) at the top. Fonts, line
widths and colours are in `config.yaml` `style`, given as printed sizes.
