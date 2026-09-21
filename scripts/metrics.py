"""
Online metrics of one xPC run, straight from its trimmed NWB: neuraldecoding's bitrate / success-rate /
targets-per-second functions, plus time to target, orbiting time and path efficiency (not in the library).
The run's mode and the trial columns they need (target, in_mode, ...) were fixed when the master was built.
"""
import numpy as np
import pandas as pd

from utils import DATA, load_master, open_nwb


def first_entry_time(pos, target, radius, dt):
    """Seconds from trial start until the cursor is inside the target box on every DOF."""
    hits = np.flatnonzero(np.all(np.abs(pos - target) <= radius, axis=1))
    return hits[0] * dt if len(hits) else np.nan


def path_efficiency(pos):
    path = np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1))
    return np.linalg.norm(pos[-1] - pos[0]) / path if path > 0 else np.nan


def run_metrics(nwb, closed, dofs):
    """Per-run means over the trials that belong to the run's mode (hand control: finger_angles, closed loop: decode)."""
    from neuraldecoding.utils.online_metrics import (bitrate, bitrate_zstruct, get_timeseries_from_trial, success_rate,
                                                     success_rate_zstruct, targets_per_second_zstruct)
    tr = nwb.trials.to_dataframe()
    feature = int(np.median(tr['DecodeFeature']))          # closed loop: decoder output columns 0-4 TCFR, 5-9 SBP
    path, cols = ('processing.behavior.decode', [d + 5 * feature for d in dofs]) if closed else \
                 ('processing.behavior.finger_angles', list(dofs))
    dt = float(np.median(np.diff(nwb.processing['behavior']['finger_angles'].timestamps[:1000])))
    if closed:
        bits = bitrate_zstruct(nwb, exclude_failed_trials=True, exclude_intarget_trials=True,
                               pos_indices=list(dofs), feature=feature)
        rate, _ = success_rate_zstruct(nwb)
    else:
        bits = bitrate(nwb, 'processing.behavior.finger_angles', trial_start_label='start_time', trial_stop_label='stop_time',
                       target_label='target', target_radius_label='target_radius', exclude_failed_trials=True,
                       exclude_intarget_trials=True, pos_indices=list(dofs))
        rate, _ = success_rate(nwb, success_field='success', exclude_trials=np.flatnonzero(~tr['in_mode'].to_numpy()))

    ttt, orbit, eff = [], [], []
    for i in np.flatnonzero(tr['in_mode'] & tr['success']):          # only successful in-mode trials enter the means
        pos = get_timeseries_from_trial(nwb, i, path)[:, cols]
        t_in = first_entry_time(pos, tr['target'][i], tr['target_radius'][i], dt)
        ttt.append(t_in)
        orbit.append(max(len(pos) * dt - tr['hold_time'][i] - t_in, 0.0))     # stop_time is the last sample
        eff.append(path_efficiency(pos))
    return {'n_trials': int(tr['in_mode'].sum()), 'success_rate': rate, 'bitrate_bits_per_s': np.nanmean(np.asarray(bits, float)),
            'targets_per_s': targets_per_second_zstruct(nwb),
            'time_to_target_s': pd.Series(ttt).mean(), 'orbiting_time_s': pd.Series(orbit).mean(),
            'path_efficiency': pd.Series(eff).mean()}


def export_run_metrics(cfg):
    """data/run_metrics.csv: one row per run. Runs the audit excluded (their online metrics do not measure the
    labelled mode) are skipped."""
    rows = []
    for run_id, r in load_master(cfg)['runs'].items():
        if r['excluded']:
            print(f"excluding {run_id} from online metrics: {r['exclude_reason']}")
            continue
        ds = open_nwb(cfg, run_id)
        rows.append({'run_id': run_id, 'sessions': ','.join(r['sessions']), 'group': r['group'], 'freebie': r['freebie'],
                     'mode': r['mode'], 'n_dof': len(r['dofs']), **run_metrics(ds.dataset, r['mode'] == 'closed_loop', r['dofs'])})
        ds.close()
    pd.DataFrame(rows).to_csv(DATA / 'run_metrics.csv', index=False)
