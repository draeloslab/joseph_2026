"""Figure 6 panel A, from raw/fes_live_run.npz and raw/fes_detection_labels.csv: the live free-behaviour run, 7 camera slots
wired (5 delivered frames), MediaPipe with up to two hands per camera, 2D only. Per processing step: when it pulled its
frames (timestamps), whether each camera had the index fingertip of each MediaPipe hand block (index_tip_detected: steps x
cameras x hands), and each camera's frame starts and latencies. The labels say what each stored detection is (reaching
hand, the monkey's resting hand, glove, background or face, ...), labelled by eye from the videos."""
import numpy as np
import pandas as pd

from utils import DATA, RAW


def analyze(cfg):
    """data/fig6_detection.csv: how often each camera on its own, and at least one / two cameras at the same step, had the
    index fingertip, counting every stored detection, only the monkey's hands, or only the reaching hand. A step is used
    when each camera's newest frame (the last one that had arrived when the step pulled) lies inside the camera's video."""
    z = np.load(RAW / 'fes_live_run.npz')
    ts, det, names = z['timestamps'], z['index_tip_detected'], [str(n) for n in z['cameras']]
    slots, hands = [int(s) for s in z['slots']], [str(h) for h in z['hands']]
    frame = [np.searchsorted(z[f'starts_{n}'] + z[f'latencies_{n}'], ts) - 1 for n in names]
    keep = np.all([(f >= 0) & (f < cfg['analysis']['fes']['video_frames']) for f in frame], 0)
    lab = pd.read_csv(RAW / 'fes_detection_labels.csv')
    kinds = {'all detections': None, 'monkey': {'reaching hand', 'other hand (resting)'}, 'reaching hand': {'reaching hand'}}
    rows = []
    for kind, ok in kinds.items():
        seen = np.zeros((len(ts), len(names)), bool)
        for j, slot in enumerate(slots):
            if ok is None:
                seen[:, j] = det[:, j].any(1)
                continue
            for r in lab[(lab.camera == slot) & lab.label.isin(ok)].itertuples():
                b = hands.index(r.mediapipe_hand)
                seen[r.first_step:r.last_step + 1, j] |= det[r.first_step:r.last_step + 1, j, b]
        seen = seen[keep]
        rows += [dict(group=f'camera {s}, {kind}', mean=100 * seen[:, j].mean()) for j, s in enumerate(slots)]
        rows += [dict(group=f'at least {n} cameras, {kind}', mean=100 * (seen.sum(1) >= n).mean()) for n in (1, 2)]
    pd.DataFrame(rows).assign(panel='A', measure='index fingertip detected', unit='% of steps',
                              n_frames=int(keep.sum())).to_csv(DATA / 'fig6_detection.csv', index=False)
