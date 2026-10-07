"""
One human 3D-hand-tracking BRAND run (2026-10-06 on, graph rhodes_2d_vision_bare) .rdb -> npz:

    vision_*   `vision_data`  (vision node)   joint angles from improv (deg), the joints in JOINTS, + improv frame_index
    dofs_*     `vision_dofs`  (vision_dofs)   normalised + smoothed DOFs (position column)
    cursor_*   `cursor`       (control_mux)   what the task scored, one column per task DOF
    trials     trial_info / trial_success (targets per task DOF, radius, hold, screen_on, success)
Times are BRAND header clock (s); `*_rid` are the Redis entry ids (Unix ms, the BRAND machine's wall clock), and
`window_unix` the task span. Needs redis-server + redis + msgpack (config human.rdb_python). Copies the .rdb to a temp
dir first, so the server copy is never touched.

    python human_rdb.py <run.rdb> <out.npz>
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

JOINTS = ['right_INDEX_MCP', 'right_RING_MCP', 'right_MIDDLE_MCP', 'right_PINKY_MCP']


def load_rdb(rdb):
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


def read_stream(c, key, count=5000):
    out, last = [], '-'
    while True:
        chunk = c.xrange(key, min=last if last == '-' else '(' + last, count=count)
        if not chunk:
            return out
        out += chunk
        last = chunk[-1][0].decode()


unpack = lambda v: msgpack.unpackb(v, raw=False)
ts = lambda d: unpack(d[b'_hdr'])['ts'] / 1e9
rid = lambda i: int(i.decode().split('-')[0]) / 1000


def arr(field):
    return np.frombuffer(field[b'data'], dtype=field[b'type'])


def dof_stream(c, key, kw, name):
    rows = read_stream(c, key)
    if not rows:
        return
    names = list(unpack(rows[0][1][b'data']))
    kw[f'{name}_t'] = np.array([ts(d) for _, d in rows])
    kw[f'{name}_rid'] = np.array([rid(i) for i, _ in rows])
    kw[f'{name}_names'] = np.array(names)
    kw[f'{name}_x'] = np.array([[arr(unpack(d[b'data'])[n])[0] for n in names] for _, d in rows], dtype=np.float64)


def extract(rdb, out):
    proc, c, tmp = load_rdb(rdb)
    try:
        kw = {}
        rows = read_stream(c, 'vision_data')
        kw['vision_t'] = np.array([ts(d) for _, d in rows])
        kw['vision_rid'] = np.array([rid(i) for i, _ in rows])
        data = [unpack(d[b'data']) for _, d in rows]
        kw['vision_frame'] = np.array([v.get('frame_index', -1) for v in data])
        kw['vision_joints'] = np.array(JOINTS)
        kw['vision_x'] = np.array([[v.get('angles', {}).get(j, np.nan) if v.get('angles', {}).get(j) is not None else np.nan
                                    for j in JOINTS] for v in data], dtype=np.float64)
        dof_stream(c, 'vision_dofs', kw, 'dofs')
        dof_stream(c, 'cursor', kw, 'cursor')
        info, succ = read_stream(c, 'trial_info'), read_stream(c, 'trial_success')
        ti = [unpack(d[b'data']) for _, d in info]
        dof_names = list(ti[0]['targets'])
        kw.update(start_t=[ts(d) for _, d in info], result_t=[ts(d) for _, d in succ],
                  success=[bool(unpack(d[b'data'])['success']) for _, d in succ],
                  target_dofs=np.array(dof_names), targets=np.array([[b['targets'].get(n, np.nan) for n in dof_names] for b in ti]),
                  target_radius=[b['target_radius'] for b in ti], hold_time=[b['hold_time'] for b in ti],
                  screen_on=[b.get('screen_on', 1) for b in ti],
                  window_unix=[rid(info[0][0]), rid(succ[-1][0])])
        np.savez(out, **kw)
    finally:
        proc.terminate(); proc.wait(); shutil.rmtree(tmp)


if __name__ == '__main__':
    extract(sys.argv[1], sys.argv[2])
