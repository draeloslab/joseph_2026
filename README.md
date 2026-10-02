# joseph_2026

Code for the paper's Figures 1–4 (plus a bonus real-time latency figure). Everything is rebuilt from the
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
data/master/               master.pkl + nwb/<run_id>.nwb
data/master/sessions_master.csv   the master sessions list (one row per session)
data/cache/freebie_kf.csv  trained offline-KF results, reused until the master changes
data/*.csv, *.npz          derived numbers each figure is drawn from
figures/                   output (svg + png) and <fig>_stats.csv with every significance test
```

## Make the figures
Needs numpy, pandas, scipy, matplotlib >= 3.9, pyyaml, pillow, plus the neuraldecoding environment.
```
python scripts/generate_figs.py                     # draw the figures enabled in config.yaml from data/
python scripts/generate_figs.py --figs 2 rt_bonus   # just some of them
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
  - `freebie_validation`, `keypoint_frames` (Fig 1), `rt_runs` (bonus figure).
- `nwb/<run_id>.nwb`: trimmed NWB per run. It keeps only what neuraldecoding reads: the trials table (raw columns
  plus target / target_radius / hold_time / success / blank / decoded / in_mode in the form the library's generic
  metric functions expect), `finger_angles`, `decode` (closed-loop runs) and `EMG` (runs behind a Freebie KF, first
  16 of the 96 channels; the rest carry no signal).

## Excluded runs
Runs with `excluded = True` are dropped from the online-metrics figures. Currently `20251216_Run-002`: it is
flagged closed loop but its decoder output is constant (-1) for the whole run and the bend sensor is what hits the
targets. Its Freebie-vs-bend correlations and offline KFs are still used.

## Changing a figure
Each figure module has one function per panel and a `LAYOUT` (matplotlib `subplot_mosaic`) at the top; move or
reorder panels by editing `LAYOUT`. Fonts, line widths, the colour palette and whether each figure draws
significance bars are in `config.yaml`.

`data/master/` is a few hundred MB; host it outside git (LFS, Zenodo, ...) if you publish the repo.

## Figure naming
Fig 3 and Fig 4 are the War Machine figures, built by `notebooks/war_machine_2026-09-30/build_fig3.py` / `build_fig4.py`.
The earlier 3 and 4 are now `fig3_supplement_figure.py` / `fig4_supplement_figure.py` (config keys `3_supplement`, `4_supplement`).
Fig 6 (`human_figure.py`, keys `6` and `6_latency`) is the 3D human hand tracking figure; its data paths are under `human:` in `config.yaml`.
