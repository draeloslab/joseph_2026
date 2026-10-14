"""Compare data_from_raw/ (recomputed) with data/ (shipped): every file the analysis wrote."""
import numpy as np
import pandas as pd

import utils
from . import PRODUCES


def _same(a, b, rtol):
    """(equal, largest relative difference) of two arrays."""
    if a.shape != b.shape:
        return False, np.inf
    if a.dtype.kind in 'USOb' or b.dtype.kind in 'USOb':        # text / booleans; empty cells compare equal
        return bool((a.astype(str) == b.astype(str)).all()), 0.0
    a, b = a.astype(float), b.astype(float)
    if not np.array_equal(np.isnan(a), np.isnan(b)):
        return False, np.inf
    m = ~np.isnan(a)
    d = np.max(np.abs(a[m] - b[m]) / np.maximum(np.abs(b[m]), 1e-300)) if m.any() else 0.0
    return bool(np.allclose(a[m], b[m], rtol=rtol, atol=0)), float(d)


def compare(done, rtol=1e-6):
    """Prints a summary and returns a table: one row per file the analyses `done` wrote, identical / within rtol /
    DIFFERENT / missing."""
    rows = []
    for rel in sorted({f for k in done for f in PRODUCES[k]}):
        ref, new = utils.ROOT / 'data' / rel, utils.DATA / rel
        row = {'file': rel, 'status': 'missing', 'max_rel_diff': np.nan}
        if new.exists():
            if ref.suffix == '.csv':
                a, b = pd.read_csv(ref, float_precision='round_trip'), pd.read_csv(new, float_precision='round_trip')
                same_cols = list(a.columns) == list(b.columns) and len(a) == len(b)
                parts = [_same(a[c].to_numpy(), b[c].to_numpy(), rtol) for c in a.columns] if same_cols else [(False, np.inf)]
            elif ref.suffix == '.npz':
                za, zb = np.load(ref), np.load(new)
                same_keys = sorted(za.files) == sorted(zb.files)
                parts = [_same(za[k], zb[k], rtol) for k in za.files] if same_keys else [(False, np.inf)]
            else:
                parts = [(ref.read_bytes() == new.read_bytes(), 0.0)]
            diff = max(d for _, d in parts)
            row.update(status='identical' if diff == 0 and all(ok for ok, _ in parts) else
                       'within tolerance' if all(ok for ok, _ in parts) else 'DIFFERENT', max_rel_diff=diff)
        rows.append(row)
    out = pd.DataFrame(rows)
    print('\nrecomputed data vs data/:', out['status'].value_counts().to_dict())
    for r in out[~out['status'].isin(['identical'])].itertuples():
        print(f'  {r.status:17s} {r.file}  (max relative difference {r.max_rel_diff:.2e})')
    return out
