"""
Freebie signals against the bend sensors. The master already holds the Freebie signal the audit chose for
each session (stable-model rerun when usable, else live; image coordinates negated so flexion is positive), the
fpos bend packets on the same clock, the fpos eTime -> xPC clock offset (timestamps only, no fitting), the xPC run window on the Freebie clock, and
whether the audit trained Freebie KFs.
"""
import numpy as np
import pandas as pd

import kf
from utils import DATA, load_master, open_nwb

BEND_DOF = {'index_angle': 1, 'pip_y': 1, 'mrs_x': 3}      # the bend-sensor DOF each Freebie signal is compared with
CACHE = DATA / 'cache' / 'freebie_kf.csv'


def unit_range(x, lo=1, hi=99):
    """Robust 0-1 scaling so pixel / degree signals share the bend sensors' 0-1 units."""
    p_lo, p_hi = np.nanpercentile(x, [lo, hi])
    return (x - p_lo) / (p_hi - p_lo)


def xpc_signals(s, t_x, pct=(1, 99)):
    """The session's Freebie signals on xPC time inside the run, 0-1 scaled (pct = the percentiles mapped to 0 and 1):
    {name: (t, v)}."""
    out = {}
    for name, (t, v) in s['signals'].items():
        tx = t - s['xpc_offset_s']
        ok = np.isfinite(v) & (tx >= t_x[0]) & (tx <= t_x[-1])
        out[name] = (tx[ok], unit_range(v[ok], *pct))
    return out


def correlations(master):
    """Pearson r of each Freebie signal with the fpos bend value at the Freebie frame times, over the frames
    inside the xPC run, per session with a bend sensor on the xPC (bend sensor on the xPC) and a known run window."""
    rows = []
    for sid, s in master['sessions'].items():
        if s['freebie'] or s['run_window_s'] is None:
            continue
        tr, (lo, hi), row = s['fpos_t'], s['run_window_s'], {'session_id': sid}
        for name, (t, v) in s['signals'].items():
            ok = np.isfinite(v) & (t >= max(tr[0], lo)) & (t <= min(tr[-1], hi))
            bend = s['fpos_bend'][:, s['fpos_dofs'].index(BEND_DOF[name])]
            row[f'corr_{name}'] = np.corrcoef(v[ok], np.interp(t[ok], tr, bend))[0, 1]
        rows.append(row)
    return pd.DataFrame(rows).reindex(columns=['session_id', 'corr_index_angle', 'corr_pip_y', 'corr_mrs_x'])


def freebie_kfs(cfg, sid):
    """KFs on the run's EMG with the behavior replaced by Freebie signals at their own frame times, next to a bend
    baseline on the same frames. The MRS signal is interpolated onto the index-signal frames."""
    master = load_master(cfg)
    s, run = master['sessions'][sid], master['runs'][master['sessions'][sid]['run_id']]
    ds = open_nwb(cfg, s['run_id'])
    fa = ds.dataset.processing['behavior']['finger_angles']
    t_x, bend = fa.timestamps[:], fa.data[:]
    dofs = run['dofs'][:1] if len(run['dofs']) == 2 and 'mrs_x' not in s['signals'] else run['dofs']
    need = ['index_angle', 'pip_y'] + (['mrs_x'] if len(dofs) == 2 else [])
    tg = s['signals']['pip_y'][0]                                  # index-signal frame times
    vals = {n: np.interp(tg, *s['signals'][n], left=np.nan, right=np.nan) for n in need if n in s['signals']}
    tx = tg - s['xpc_offset_s']                                    # xPC seconds
    ok = np.all([np.isfinite(v) for v in vals.values()], axis=0) & (tx >= t_x[0]) & (tx <= t_x[-1])
    span, tx = tx[ok][-1] - tx[ok][0], tx[ok]

    def behavior(cols):                                            # (n, 5) like finger_angles, DOF columns filled
        b = np.full((len(tx), 5), np.nan)
        for col, dof in zip(cols, dofs):
            b[:, dof] = col
        return b

    conditions = [('bend (same window)', '', behavior([np.interp(tx, t_x, bend[:, d]) for d in dofs]))]
    for lead in ('index_angle', 'pip_y'):
        if lead in vals:
            names = [lead] + (['mrs_x'] if len(dofs) == 2 else [])
            conditions.append(('+'.join(names), ','.join(f"{n}:{s['signal_source'][n]}" for n in names),
                               behavior([unit_range(vals[n][ok]) for n in names])))
    rows = []
    for label, sources, b in conditions:
        for output, m in kf.train_kf(ds, dofs, behavior=b, behavior_ts_ms=tx * 1000).iterrows():
            rows.append({'session_id': sid, 'condition': label, 'output': output, 'n_dof': len(dofs), 'window_s': span,
                         'sources': sources, 'mse_valid': m['mse_valid'], 'nmse_valid': m['nmse_valid'],
                         'correlation_valid': m['correlation_valid']})
    ds.close()
    return rows


def offline_kf_rows(cfg):
    """All Freebie-KF rows for the sessions the audit trained KFs for. Trained KFs are cached in data/cache/ and
    reused while the master is unchanged; cfg['retrain'] forces retraining."""
    master = load_master(cfg)
    built = master['meta']['built']
    if CACHE.exists() and not cfg.get('retrain'):
        cached = pd.read_csv(CACHE, float_precision='round_trip')
        if (cached['master_built'] == built).all():
            return cached
    rows = []
    for sid, s in master['sessions'].items():
        if s['freebie_kf']:
            print(f'training Freebie KFs: {sid}', flush=True)
            rows += freebie_kfs(cfg, sid)
    CACHE.parent.mkdir(exist_ok=True)
    out = pd.DataFrame(rows).assign(master_built=built)
    out.to_csv(CACHE, index=False)
    return out
