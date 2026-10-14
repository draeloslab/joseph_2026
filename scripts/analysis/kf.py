"""
Offline Kalman filter, EMG -> finger position, entirely through neuraldecoding's preprocessing blocks and
KalmanFilter. Values come from the lab's configs: `mav` EMG feature, KalmanFilter params, a chronological
80/20 split of the bins, and the 32 ms bins the online decoders used.
"""
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from utils import DOF_NAMES, use_neuraldecoding

EMG = 'processing.ecephys.EMG'
FINGER_ANGLES = 'processing.behavior.finger_angles'
GOOD_CHANS_EMG = [0, 2, 4, 6, 8, 10, 12, 14]
KF_BIN_MS = 32


def kf_config(dofs):
    return OmegaConf.create({
        'preprocessing': {
            'order': ['dict', 'xfilt', 'yfilt', 'feat', 'split', 'trainer'],
            'content': {
                'dict': {'type': 'Dataset2DictBlock', 'params': {
                    'neural_nwb_loc': EMG, 'behavior_nwb_loc': FINGER_ANGLES,
                    'skip_first_n_trials': 1, 'data_keys': ['neural', 'behavior'],
                    'nwb_trial_start_times_loc': 'trials.start_time',
                    'nwb_trial_end_times_loc': 'trials.stop_time',
                    'nwb_targets_loc': 'trials.TargetPos'}},
                'xfilt': {'type': 'IndexSelectorBlock', 'params': {
                    'location': ['neural'], 'indices': GOOD_CHANS_EMG}},
                'yfilt': {'type': 'IndexSelectorBlock', 'params': {
                    'location': ['behavior'], 'indices': [int(d) for d in dofs]}},
                'feat': {'type': 'FeatureExtractionBlock', 'params': {
                    'location_data': ['neural', 'behavior'],
                    'location_ts': ['neural_ts', 'behavior_ts'],
                    'feature_extractor_config': {
                        'channels': len(GOOD_CHANS_EMG), 'feature_type': [['mav'], ['mean', 'vel']],
                        'feature_params': [{}, {}], 'bin_size_ms': KF_BIN_MS}}},
                'split': {'type': 'DataSplitBlock', 'params': {
                    'location': ['neural', 'behavior'], 'split_ratio': 0.8,
                    'split_seed': 42, 'shuffle': False,
                    'data_keys': ['neural_train', 'neural_test', 'behavior_train', 'behavior_test']}},
                'trainer': {'type': 'Dict2TrainerBlock', 'params': {
                    'location': ['neural_train', 'neural_test', 'behavior_train', 'behavior_test']}},
            },
        },
        'trainer': {
            'model': {'type': 'KalmanFilter', 'params': {
                'append_ones_y': True, 'zero_position_uncertainty': True, 'start_y': None}},
            'evaluation': {'metrics': ['mse', 'correlation']},
            'training': {'print_results': False},
        },
    })


def train_kf(ds, dofs, behavior=None, behavior_ts_ms=None):
    """
    Validation metrics per output of a KF on the run's EMG. With `behavior` (n, 5) and its timestamps in xPC ms
    (e.g. Freebie frames at their own frame times) the library's `dict` step output is swapped for it; the library
    bins each stream by its own timestamps, so the EMG is not resampled. nmse = mse / variance of the validation
    target, comparable across signals with different spread.
    """
    use_neuraldecoding()
    from neuraldecoding.preprocessing import Preprocessing
    from neuraldecoding.trainer.LinearTrainer import LinearTrainer

    cfg = kf_config(dofs)
    pre = Preprocessing(cfg.preprocessing)
    data, interpipe = pre.preprocess_step(ds, 'dict', {'save_keys_ram': [], 'is_train': True})
    if behavior is not None:
        data['behavior'], interpipe['behavior_ts'] = np.asarray(behavior, float), np.asarray(behavior_ts_ms, float)
    for step in cfg.preprocessing.order[1:]:
        data, interpipe = pre.preprocess_step(data, step, interpipe)
    trainer = LinearTrainer(pre, cfg.trainer, dataset=None)
    trainer.train_X, trainer.train_Y = data['X_train'], data['Y_train']
    trainer.valid_X, trainer.valid_Y = data['X_val'], data['Y_val']
    trainer.train_model()
    out = pd.DataFrame({f'{m}_{split}': np.asarray(trainer.logger[m][split][-1]).ravel()
                        for m in trainer.metrics for split in ('train', 'valid')},
                       index=[f'{DOF_NAMES[d]}_pos' for d in dofs] + [f'{DOF_NAMES[d]}_vel' for d in dofs])
    out['nmse_valid'] = out['mse_valid'] / np.var(np.asarray(trainer.valid_Y), axis=0)
    return out
