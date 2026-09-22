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
    """
    Per-run means over the trials that belong to the run's mode (hand control: finger_angles, closed loop:
    decode), plus a per-trial dataframe the means above are computed from:
      - bitrate_bits_per_s: the library's own per-trial throughput (bitrate/bitrate_zstruct already
        return one value per trial, NaN on excluded/failed/in-target trials) -- a real per-trial number,
        not an average.
      - success_rate, targets_per_s: neither has a per-trial value in the library (success is a single
        0/1 per trial, targets/s is only defined over a whole run's total time), so these are a running
        cumulative rate instead -- successes-so-far / trials-so-far, and targets-so-far / elapsed-time-so-far
        -- which converges to the run's aggregate value and reads far better as a per-trial curve than raw
        0/1 success flags would.
      - time_to_target_s / orbiting_time_s / path_efficiency: real per-trial values (successful trials only).
    """
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
    bits = np.asarray(bits, float)                          # one entry per NWB trial (NaN where excluded)
    durations = (tr['stop_time'] - tr['start_time']).to_numpy()

    trial_rows, n_success, n_seen, target_time = [], 0, 0, 0.0
    for i in np.flatnonzero(tr['in_mode'].to_numpy()):               # one row per in-mode trial, in order
        success = bool(tr['success'][i])
        n_seen += 1
        n_success += success
        target_time += durations[i]
        row = {'trial': int(i), 'success': success, 'bitrate_bits_per_s': bits[i],
               'success_rate_running': n_success / n_seen, 'targets_per_s_running': n_seen / target_time,
               'time_to_target_s': np.nan, 'orbiting_time_s': np.nan, 'path_efficiency': np.nan}
        if success:                                                  # ttt/orbit/path only defined for hits
            pos = get_timeseries_from_trial(nwb, i, path)[:, cols]
            t_in = first_entry_time(pos, tr['target'][i], tr['target_radius'][i], dt)
            row['time_to_target_s'] = t_in
            row['orbiting_time_s'] = max(len(pos) * dt - tr['hold_time'][i] - t_in, 0.0)   # stop_time is the last sample
            row['path_efficiency'] = path_efficiency(pos)
        trial_rows.append(row)
    trials = pd.DataFrame(trial_rows, columns=['trial', 'success', 'bitrate_bits_per_s', 'success_rate_running',
                                               'targets_per_s_running', 'time_to_target_s', 'orbiting_time_s',
                                               'path_efficiency'])

    agg = {'n_trials': int(tr['in_mode'].sum()), 'success_rate': rate, 'bitrate_bits_per_s': np.nanmean(bits),
           'targets_per_s': targets_per_second_zstruct(nwb), 'time_to_target_s': trials['time_to_target_s'].mean(),
           'orbiting_time_s': trials['orbiting_time_s'].mean(), 'path_efficiency': trials['path_efficiency'].mean()}
    return agg, trials


def export_run_metrics(cfg):
    """
    data/run_metrics.csv: one row per run (session/run-level means). data/run_trial_metrics.csv: one row per
    in-mode trial of every run (success flag, and time-to-target/orbiting/path-efficiency when it succeeded),
    for panels that plot every trial instead of one mean per run. Runs the audit excluded (their online
    metrics do not measure the labelled mode) are skipped from both.
    """
    rows, trial_rows = [], []
    for run_id, r in load_master(cfg)['runs'].items():
        if r['excluded']:
            print(f"excluding {run_id} from online metrics: {r['exclude_reason']}")
            continue
        ds = open_nwb(cfg, run_id)
        agg, trials = run_metrics(ds.dataset, r['mode'] == 'closed_loop', r['dofs'])
        rows.append({'run_id': run_id, 'sessions': ','.join(r['sessions']), 'group': r['group'], 'freebie': r['freebie'],
                     'mode': r['mode'], 'n_dof': len(r['dofs']), **agg})
        trial_rows.append(trials.assign(run_id=run_id, freebie=r['freebie'], mode=r['mode'], n_dof=len(r['dofs'])))
        ds.close()
    pd.DataFrame(rows).to_csv(DATA / 'run_metrics.csv', index=False)
    pd.concat(trial_rows, ignore_index=True).to_csv(DATA / 'run_trial_metrics.csv', index=False)
