"""Figure 2, from raw/fig2_heldout_predictions.npz: Freebie's prediction, the manual label and the prediction's likelihood
of every held-out labelled frame, per video ({view}_{video}_{pred_xy, manual_xy, pred_lik}, frames x keypoints x value)."""
import numpy as np

from utils import DATA, RAW

TOP_KPS = ['DIP', 'PIP', 'MCP', 'WRIST']


def analyze(cfg):
    """data/fig2_keypoint_errors.npz: euclidean error (px) of every scored frame per keypoint, prediction vs label.
    Predictions below the likelihood threshold are not scored."""
    z, min_lik = np.load(RAW / 'fig2_heldout_predictions.npz'), cfg['analysis']['keypoint_likelihood_min']
    errors = {kp: [] for kp in TOP_KPS + ['MRS']}
    for view, kps in (('top', TOP_KPS), ('side', ['MRS'])):
        i = 0
        while f'{view}_{i:02d}_pred_xy' in z:
            pred, man, lik = (z[f'{view}_{i:02d}_{k}'] for k in ('pred_xy', 'manual_xy', 'pred_lik'))
            for k, kp in enumerate(kps):
                keep = lik[:, k, 0] >= min_lik
                errors[kp].append(np.linalg.norm(pred[keep, k] - man[keep, k], axis=1))
            i += 1
    np.savez(DATA / 'fig2_keypoint_errors.npz', **{kp: np.concatenate(v) for kp, v in errors.items()})
