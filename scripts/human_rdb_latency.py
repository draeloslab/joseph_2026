"""
Latencies from one human-hand-tracking BRAND Redis dump (.rdb). The dumps hold no latency stream, so they are
derived from the node timestamps every stream entry carries (`_hdr.ts`, one shared clock across nodes, ns):
  - 'hand3d -> control_mux' (outputs with no hand3d sample in the last 100 ms are dropouts and excluded): age of the newest hand3d_dofs sample at the moment control_mux publishes `behavior`
    (pose-to-cursor latency, includes up to one 10 ms hand3d period of sampling delay).
  - '<stream> publish delay': redis arrival (entry id, ms) minus node timestamp, minus its own minimum. The two
    clocks differ by an unknown constant, so this is the delay above the best case, not an absolute latency.
Needs redis-server on PATH plus redis and msgpack (the improvPytorch conda env has them).

    python human_rdb_latency.py <run.rdb> <out.csv>      # columns: actor, latency_ms
"""
import sys
import json
sys.path.remove(sys.path[0])   # scripts/ holds utils.py etc., which shadow modules numpy/pandas import
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd
import redis

DROPOUT_S = 0.1
PUBLISH_STREAMS = ['hand3d_dofs', 'behavior', 'cursor', 'targetData']


def read_stream(c, key, count=20000):
    out, last = [], '-'
    while True:
        chunk = c.xrange(key, min=last if last == '-' else '(' + last, count=count)
        if not chunk:
            return out
        out += chunk
        last = chunk[-1][0].decode()


def stamps(c, key):
    """(node ts in s, redis arrival in s) per entry that carries a header."""
    ts, arr = [], []
    for eid, e in read_stream(c, key):
        if b'_hdr' in e:
            ts.append(msgpack.unpackb(e[b'_hdr'], raw=False)['ts'] / 1e9)
            arr.append(int(eid.decode().split('-')[0]) / 1e3)
    return np.array(ts), np.array(arr)


def main(rdb, out):
    port = int(np.random.randint(20000, 40000))
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(rdb, tmp / 'dump.rdb')
    assert (tmp / 'dump.rdb').stat().st_size == Path(rdb).stat().st_size, 'truncated copy from the share'
    proc = subprocess.Popen(['redis-server', '--port', str(port), '--dir', str(tmp), '--dbfilename', 'dump.rdb',
                             '--save', '', '--appendonly', 'no'], stdout=subprocess.DEVNULL)
    rows, trials = [], []
    try:
        c = redis.Redis(port=port)
        for _ in range(200):
            try:
                c.dbsize()
                break
            except (redis.exceptions.ConnectionError, redis.exceptions.BusyLoadingError):
                time.sleep(0.5)
        for _, e in read_stream(c, 'trial_info'):      # every trial's full target set (the NWB export keeps only some DOFs)
            hdr, info = msgpack.unpackb(e[b'_hdr'], raw=False), msgpack.unpackb(e[b'data'], raw=False)
            trials.append({'start': hdr['ts'] / 1e9, 'targets': {k: float(v) for k, v in info['targets'].items()}})
        if c.exists('hand3d_dofs') and c.exists('behavior'):
            t_in, _ = stamps(c, 'hand3d_dofs')
            t_out, _ = stamps(c, 'behavior')
            i = np.searchsorted(t_in, t_out, side='right') - 1          # newest input at or before each output
            age = np.where(i >= 0, t_out - t_in[np.maximum(i, 0)], np.inf)
            ok = age <= DROPOUT_S      # older inputs are hand3d dropouts (its stream has gaps, worst in Run-015), not latency
            print(f'hand3d -> control_mux: {(~ok).sum()} of {len(ok)} outputs had no input in the last {DROPOUT_S * 1e3:.0f} ms, dropped', file=sys.stderr)
            rows.append(pd.DataFrame({'actor': 'hand3d -> control_mux', 'latency_ms': (t_out[ok] - t_in[i[ok]]) * 1e3}))
        for key in PUBLISH_STREAMS:
            if c.exists(key):
                ts, arr = stamps(c, key)
                if len(ts):
                    d = (arr - ts) * 1e3
                    rows.append(pd.DataFrame({'actor': f'{key} publish delay', 'latency_ms': d - d.min()}))
    finally:
        proc.terminate()
        proc.wait()
        shutil.rmtree(tmp)
    pd.concat(rows, ignore_index=True).to_csv(out, index=False)
    Path(str(out) + '.trials.json').write_text(json.dumps(trials))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
