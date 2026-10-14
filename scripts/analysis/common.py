"""Helpers shared by the from-raw analyses."""

import utils


def open_nwb_path(path):
    """An NWB file as neuraldecoding reads it (Dataset of type 'nwb'); close with ds.close()."""
    utils.use_neuraldecoding()
    from omegaconf import OmegaConf
    from neuraldecoding.dataset import Dataset
    ds = Dataset(OmegaConf.create({'dataset_type': 'nwb', 'autoload': False, 'save_path': None,
                                   'dataset_parameters': {'nwb_file': str(path)}}))
    ds.load_data()
    return ds


def trials_frame(nwb, dofs):
    """The run's trials with the target on each active DOF as target_<dof>."""
    tr = nwb.trials.to_dataframe()
    for j, d in enumerate(dofs):
        tr[f'target_{d}'] = [t[j] for t in tr['target']]
    return tr


def trial_window(trials, first_trial, n_trials, keep):
    """n_trials consecutive trials from first_trial; None = a third of the way through the trials selected by `keep`."""
    if first_trial is None:
        ok = trials.index[keep]
        first_trial = int(trials.index.get_loc(ok[len(ok) // 3]))
    return trials.iloc[first_trial:first_trial + n_trials]


def shortest_run_of_successes(trials, n):
    """The n consecutive successful trials that took the least time in total."""
    ok = trials['success'].to_numpy()
    starts = [i for i in range(len(trials) - n + 1) if ok[i:i + n].all()]
    first = min(starts, key=lambda i: trials['stop_time'].iloc[i + n - 1] - trials['start_time'].iloc[i])
    return trials.iloc[first:first + n]


def csv_trials(path):
    """A trials table written by the package, floats read back exactly."""
    import pandas as pd
    return pd.read_csv(path, float_precision='round_trip')


def stream(z, name):
    return z[f'{name}_t'], z[f'{name}_x']
