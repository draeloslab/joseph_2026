"""
One War Machine BRAND run's Redis dump (Run-XXX.rdb) -> npz, every signal from the stream its own node wrote:

    manip_*      `manipulandum`      (manipulandum_python)  bend sensor, calibrated 0-1
    manip_raw_*  `manipulandum_raw`  (manipulandum_python)  bend sensor ADC counts
    vision_*     `vision_dofs`       (vision_dofs)          Freebie, calibrated 0-1 (column 0 = position)
    vision_raw_* `vision_data`       (vision)               Freebie raw: PIP keypoint y pixel (improv 1dof_dlc_udp.yaml)
    behavior_*   `behavior`          (control_mux)          copy of whichever sensor the mux selected
    cursor_*     `cursor`            (control_mux)          what the task scored: behavior, or the decoder
    decoded_*    `decoded_output`    (generic_decoder)      online KF output
    neural_*     `continuousNeural`  (gemini_adapter)       raw samples, the decoder's channels only
plus the trials (trial_info / trial_success), the mux's own log (control_state), decoder loads
(decoder_config_updates) and the task's wall-clock span (Redis entry ids are Unix ms).

The server-built NWB's `acquisition/continuous` is `behavior`, not the bend sensor (rdb2nwb config in
brand-monkeyrig/graphs/monkey/rhodes_1d_universal.yaml), so nothing reads the NWB.
`continuousNeural` packets: 2 samples x 96 channels, channel-major. The packet's own `timestamps` field is a separate,
wrapping clock, so samples are timed from the header ts (the BRAND clock all streams share), 0.5 ms apart.

Needs redis-server on PATH plus redis and msgpack (config war_machine.rdb_python). Called by war_machine.py.

    python war_machine_rdb.py <run.rdb> <out.npz> <channel> [<channel> ...]
    python war_machine_rdb.py --raw <run.rdb> <out.npz> <t0> <t1> <channel> [<channel> ...]
        the raw 2 kHz `continuousNeural` samples of those channels between two BRAND times (s), for example traces
"""
import sys
sys.path.remove(sys.path[0])   # scripts/ holds utils.py etc., which shadow modules numpy/pandas import
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import msgpack
import numpy as np
import redis

SAMPLE_DT = 0.0005             # 2 kHz
DOF_STREAMS = {'manip': 'manipulandum', 'manip_raw': 'manipulandum_raw', 'vision': 'vision_dofs',
               'behavior': 'behavior', 'cursor': 'cursor', 'decoded': 'decoded_output'}


def load_rdb(rdb):
    """A throwaway redis-server on a copy of the dump (the server copy stays untouched), on a random port."""
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(rdb, tmp / 'dump.rdb')
    port = int(np.random.randint(20000, 40000))
    proc = subprocess.Popen(['redis-server', '--port', str(port), '--dir', str(tmp), '--dbfilename', 'dump.rdb',
                             '--save', '', '--appendonly', 'no'], stdout=subprocess.DEVNULL)
    c = redis.Redis(port=port, socket_timeout=120)
    for _ in range(400):
        try:
            c.dbsize()
            return proc, c, tmp
        except (redis.exceptions.ConnectionError, redis.exceptions.BusyLoadingError):
            time.sleep(0.5)
    raise RuntimeError('redis-server did not come up')


def read_stream(c, key, count=2000):
    out, last = [], '-'
    while True:
        chunk = c.xrange(key, min=last if last == '-' else '(' + last, count=count)
        if not chunk:
            return out
        out += chunk
        last = chunk[-1][0].decode()


def unpack(v):
    return msgpack.unpackb(v, raw=False)


def ts(d):
    return unpack(d[b'_hdr'])['ts'] / 1e9


def dof_stream(c, key):
    t, x = [], []
    for _, d in read_stream(c, key):
        a = unpack(d[b'data'])['index']
        t.append(ts(d))
        x.append(np.frombuffer(a[b'data'], dtype=a[b'type'])[0])
    return np.array(t), np.array(x, dtype=np.float64)


def extract(rdb, out, channels):
    proc, c, tmp = load_rdb(rdb)
    try:
        kw = {'channels': channels}
        for name, key in DOF_STREAMS.items():
            if c.exists(key):
                kw[f'{name}_t'], kw[f'{name}_x'] = dof_stream(c, key)
        if c.exists('vision_data'):
            rows = read_stream(c, 'vision_data')
            kw['vision_raw_t'] = np.array([ts(d) for _, d in rows])
            kw['vision_raw_x'] = np.array([unpack(d[b'data'])['angles_vec'][0] for _, d in rows], dtype=np.float64)
        info = read_stream(c, 'trial_info')
        succ = read_stream(c, 'trial_success')
        trial_info = [unpack(d[b'data']) for _, d in info]
        kw.update(start_t=[ts(d) for _, d in info], result_t=[ts(d) for _, d in succ],
                  success=[bool(unpack(d[b'data'])['success']) for _, d in succ],
                  target=[b['targets']['index'] for b in trial_info], target_radius=[b['target_radius'] for b in trial_info],
                  hold_time=[b['hold_time'] for b in trial_info], screen_on=[b.get('screen_on', 1) for b in trial_info],
                  window_unix=[int(info[0][0].decode().split('-')[0]) / 1000, int(succ[-1][0].decode().split('-')[0]) / 1000])
        if c.exists('control_state'):
            st = [(ts(d), unpack(d[b'data'])) for _, d in read_stream(c, 'control_state')]
            kw.update(mux_t=[a for a, _ in st], mux_behavior=[b['behavior_source'] for _, b in st],
                      mux_cursor=[b['cursor_source'] for _, b in st])
        if c.exists('decoder_config_updates'):
            dc = [(ts(d), unpack(d[b'data'])) for _, d in read_stream(c, 'decoder_config_updates')]
            kw.update(decoder_t=[a for a, _ in dc], decoder_model=[Path(str(b.get('model_path', ''))).name for _, b in dc])
        nt, nx = [], []
        for _, d in read_stream(c, 'continuousNeural'):
            a = unpack(d[b'data'])
            n = len(a['timestamps'])
            nt.append(ts(d) - SAMPLE_DT * np.arange(n - 1, -1, -1))
            nx.append(np.asarray(a['samples'], dtype=np.float32).reshape(96, n).T[:, channels])
        kw['neural_t'], kw['neural_x'] = np.concatenate(nt), np.concatenate(nx)
        np.savez(out, **kw)
    finally:
        proc.terminate(); proc.wait(); shutil.rmtree(tmp)


def extract_raw(rdb, out, t0, t1, channels):
    proc, c, tmp = load_rdb(rdb)
    try:
        nt, nx = [], []
        for _, d in read_stream(c, 'continuousNeural'):
            if not t0 <= ts(d) <= t1 + 0.01:
                continue
            a = unpack(d[b'data'])
            n = len(a['timestamps'])
            nt.append(ts(d) - SAMPLE_DT * np.arange(n - 1, -1, -1))
            nx.append(np.asarray(a['samples'], dtype=np.float32).reshape(96, n).T[:, channels])
        t, x = np.concatenate(nt), np.concatenate(nx)
        keep = (t >= t0) & (t <= t1)
        np.savez(out, t=t[keep], x=x[keep], channels=channels)
    finally:
        proc.terminate(); proc.wait(); shutil.rmtree(tmp)


if __name__ == '__main__':
    if sys.argv[1] == '--raw':
        extract_raw(sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), [int(ch) for ch in sys.argv[6:]])
    else:
        extract(sys.argv[1], sys.argv[2], [int(ch) for ch in sys.argv[3:]])
