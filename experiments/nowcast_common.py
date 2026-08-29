"""Shared neighbour-gauge donor features (see hgb_nowcast.py for the design
and the shuffled-control validation). Same-day + lag-1 observed flow at each
catchment's k nearest usable gauges, scaled by the donor's train-window q95.
"""
import numpy as np
import pandas as pd

from common import ROOT, read_attr, TRAIN_END, TEST_START

K, MIN_COV = 3, 0.5

_flows = pd.read_parquet(ROOT / "cache" / "daily_discharge_spec.parquet")
_ftr = _flows.loc[:TRAIN_END]
USABLE = _flows.columns[(_ftr.notna().mean() >= MIN_COV)
                        & (_flows.loc[TEST_START:].notna().mean() >= MIN_COV)]
D0 = (_flows[USABLE] / _ftr[USABLE].quantile(0.95)).astype("float32")
D1 = D0.shift(1)

_topo = read_attr("topographic").set_index("gauge_id")
_ex, _ny = _topo.gauge_easting.astype(float), _topo.gauge_northing.astype(float)


def nearest_donors(gids, pool=None, k=K):
    """k nearest donors per catchment from `pool` (default: all usable),
    never itself. Returns {gid: (donor_list, dist_km_list)}."""
    pool = USABLE if pool is None else pd.Index(pool)
    dx = _ex.loc[gids].values[:, None] - _ex.loc[pool].values[None, :]
    dy = _ny.loc[gids].values[:, None] - _ny.loc[pool].values[None, :]
    Dk = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
    self_col = {g: j for j, g in enumerate(pool)}
    out = {}
    for i, g in enumerate(gids):
        row = Dk[i].copy()
        if g in self_col:
            row[self_col[g]] = np.inf
        j = np.argsort(row)[:k]
        out[g] = (list(pool[j]), row[j].tolist())
    return out


def donor_features(index, gid, assignment, k=K):
    """2k donor columns aligned to (date index, gid array)."""
    cols = {f"nb{r}_{lag}": np.full(len(index), np.nan, dtype="float32")
            for r in range(k) for lag in ("d0", "d1")}
    for g in np.unique(gid):
        donors, _ = assignment[g]
        rows = np.flatnonzero(gid == g)
        dts = index[rows]
        for r, d in enumerate(donors):
            cols[f"nb{r}_d0"][rows] = D0[d].reindex(dts).values
            cols[f"nb{r}_d1"][rows] = D1[d].reindex(dts).values
    return pd.DataFrame(cols, index=index)
