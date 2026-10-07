# joseph_2026

Code for the paper's figures 2–6 and supplements S1–S4. Everything is rebuilt from the
**master folder `data/master/`**: `master.pkl` plus one small NWB per xPC run. You do not need the lab
server, the raw Freebie folders or any other lab repo. The only outside dependency is a clone of the lab's
[neuraldecoding](https://github.com/chesteklab/neuraldecoding) repo (set `neuraldecoding_path` in `config.yaml`),
which reads the NWBs and provides the metrics and the Kalman filter.

```
config.yaml                which figures, style + palette, significance bars, example sessions
scripts/generate_figs.py   entry point
scripts/utils.py           style, figure composition, shared panels, significance tests, master loader
scripts/metrics.py         online metrics of a run, from its NWB
scripts/freebie.py         Freebie signals vs the bend sensors: correlations and the Freebie KFs
scripts/kf.py              offline Kalman filter (neuraldecoding pipeline)
notebooks/walkthrough.ipynb   guided tour of the master data, every figure and every analysis step
scripts/validation_figure.py | latency_figure.py | fig3_supplement_figure.py | fig4_supplement_figure.py | human_figure.py
scripts/war_machine.py     War Machine master: build from the server (BRAND .rdb + vision logs) and load
scripts/war_machine_rdb.py one run's .rdb -> npz (run by war_machine.py in the redis env, config war_machine.rdb_python)
scripts/war_machine_figures.py   figs 3, 4 and 5 (War Machine)
notebooks/war_machine_sensors/   bend sensor vs Freebie check on every War Machine run (was fig 3_sensors)
notebooks/human_vision/    human hand-tracking latency figure (was fig 6_latency) + vision diagnostics
data/master/               master.pkl + nwb/<run_id>.nwb; war_machine.pkl + war_machine/<run_id>{.pkl,_neural.npz}
data/master/sessions_master.csv   the master sessions list (one row per session)
data/cache/freebie_kf.csv  trained offline-KF results, reused until the master changes
data/*.csv, *.npz          derived numbers each figure is drawn from (War Machine: data/war_machine/)
figures/                   output (svg + png) and <fig>_stats.csv with every significance test
figures/all_stats.csv      every figure's descriptives (n sessions / runs / trials, mean, sd, median, ...) + every test
                           (rebuilt at the end of each generate_figs.py run from the <fig>_descriptives / _stats csvs)
```

## Make the figures
Needs numpy, pandas, scipy, matplotlib >= 3.9, pyyaml, pillow, plus the neuraldecoding environment.
```
python scripts/generate_figs.py                     # draw the figures enabled in config.yaml from data/
python scripts/generate_figs.py --figs 3 S1   # just some of them
python scripts/generate_figs.py --analysis          # rerun the analysis from the master first (KFs from cache)
python scripts/generate_figs.py --analysis --retrain  # ... and retrain the offline KFs
```

## What the master holds
Built by `freebie/script/build_master.py` from the audit (`session_audit.py`); every check and decision lives
there, this repo only reads the result.
- `master.pkl`:
  - `sessions[session_id]`: fpos bend packets, the Freebie signals the audit chose (per signal the stable-model
    rerun when usable, else live; image coordinates negated so flexion is positive) and which source that was,
    the fpos -> xPC clock offset (timestamps only: fpos arrival stamp minus its eTime), the xPC run window on the Freebie clock
    (for the correlations), whether the audit trained Freebie KFs (`freebie_kf`), true end-to-end latency,
    per-actor latency logs, the audit's notes.
  - `runs[run_id]`: mode (hand control / closed loop), active DOFs, Freebie / decoder flags, group,
    `excluded` + reason, path to its NWB.
  - `freebie_validation`, `keypoint_frames` (Fig 1), `rt_runs` (no longer used).
- `nwb/<run_id>.nwb`: trimmed NWB per run. It keeps only what neuraldecoding reads: the trials table (raw columns
  plus target / target_radius / hold_time / success / blank / decoded / in_mode in the form the library's generic
  metric functions expect), `finger_angles`, `decode` (closed-loop runs) and `EMG` (runs behind a Freebie KF, first
  16 of the 96 channels; the rest carry no signal).

## War Machine (figs 3, 4, 5)
Figures 3-5 come from War Machine on the BRAND rig (1 DOF, index). The earlier-system figures are S1, S2 and S4.
```
python scripts/war_machine.py                       # 1. server -> data/master/war_machine.pkl (resumes; --rebuild re-extracts)
python scripts/generate_figs.py --figs 3 4 5 --analysis   # 2. master -> data/war_machine/ -> figures/
```
Step 1 needs the CNPL share mounted and the redis env (`war_machine.rdb_python`); step 2 only the master. Which runs
are in, and why the others are not, is `config.yaml` `war_machine.runs`; the rest of that block holds the decoder
training runs, the example runs and four switches (`all_trials`, `fig3_merge_fps`, `exclude_closed_loop`,
`offline_kf_min_trials`).

- Figs 4/5 online metrics: the box and dots show every trial (`all_trials: true`) or one mean per run (`false`);
  the significance tests are always on the run means, and every tick label gives n runs and n trials.
- Fig 3: `fig2_merge_fps: true` pools every run into one histogram with only the 60 fps budget
  (`figures/fig3_latency_war_machine_merged`).
- Dropped frames vs frames over the latency budget (exact counts per fps): `notebooks/war_machine_latency/frame_numbers.py`.
  `war_machine.predictions` at analysis time.

Things the data needs, all handled in `war_machine.py` / `war_machine_rdb.py`:
- The bend sensor is the rdb `manipulandum` stream and Freebie `vision_dofs`. The server NWB's
  `acquisition/continuous` is the control mux's `behavior` output (whichever sensor drove the cursor), so it is never
  used -- correlating it with Freebie on Freebie-driven runs compares Freebie with itself (r ~ 0.999).
- What drove each trial comes from the mux's own log (`control_state`), not the notes (09-30 Run-006, 010, 012 differ).
- Neural: the online decoder's 8 channels (`kf_rhodes.yaml`), raw `continuousNeural` order (the server NWB reorders by
  electrode id), timed from the packet header clock (the packet's own `timestamps` field wraps), pre-binned to 4 ms
  mean |x| so kf.py's binning is fast (same 32 ms MAV).
- The camera ran at 30 fps on 09-30 and 60 fps from 10-02; fig 2 draws a budget line for each.

## Excluded runs
Runs with `excluded = True` are dropped from the online-metrics figures. Currently `20251216_Run-002`: it is
flagged closed loop but its decoder output is constant (-1) for the whole run and the bend sensor is what hits the
targets. Its Freebie-vs-bend correlations and offline KFs are still used.

## Print size
Every figure is drawn in its own design inches and saved at `style.figure_width_mm` (180 mm, double column): fonts,
lines and markers in `config.yaml` `style` are the printed sizes (`utils.print_rc`). Subplot titles are removed (except
panels flagged `ax.keep_title`), automatic axes get at most 4 ticks, and frame-budget lines are black.

## Changing a figure
Each figure module has one function per panel and a `LAYOUT` (matplotlib `subplot_mosaic`) at the top; move or
reorder panels by editing `LAYOUT`. Fonts, line widths, the colour palette and whether each figure draws
significance bars are in `config.yaml`.

`data/master/` is a few hundred MB; host it outside git (LFS, Zenodo, ...) if you publish the repo.

## Figure naming
Fig 2 validation (held-out) | 3 War Machine latency | 4 hand control | 5 Kalman filter | 6 FES | S1 old-pipeline latency |
S2 real-time hand control, old system (`fig3_supplement_figure.py`) | S3 human 2-4 DOF hand control | S4 Kalman filter, old system
(`fig4_supplement_figure.py`). Output files are `fig<N>_<name>` / `figS<N>_<name>` in `figures/`.
Fig S1 (`latency_figure.py`) is the earlier pooled-session latency figure next to the War Machine runs.
Fig S3 (`human_figure.make_figure`, analysis `human_tuning.analyze_dof`, config `human_dof`) is the 3D human hand tracking figure on the
BRAND triangulation set-up (2026-10-07, one run each of 2, 3 and 4 DOF): per-trial stage latency by DOF count, bit rate, the task screen
(`assets/`). Key `human_1006` is the 2026-10-06 tuning figure (config `human_bare`); `human_0930` the earlier 09-30 version (config `human`).
Its latency figure and the vision diagnostics are in `notebooks/human_vision/` (`python notebooks/human_vision/human_latency.py`,
`vision_diagnostics.py`); the sensor check in `notebooks/war_machine_sensors/sensor_check.py`. They write next to themselves.
