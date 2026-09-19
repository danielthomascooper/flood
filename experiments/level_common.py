"""Shared pieces for the Phase 8 level-forecast scripts (8b point, 8b-2 ladder)."""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors

ROOT = Path(__file__).resolve().parents[1]
EA = ROOT / "cache/ea/daily_wd"
ENS = ROOT / "cache/nwp/gefs_catchment_leads_ens.parquet"
MEMMAX = ROOT / "cache/nwp/gefs_catchment_leads_memmax.parquet"
FC_START = "2000-01-01"
OKQ = ["Good", "Estimated"]
MIN_TRAIN, MIN_TEST = 3650, 2920


def level_series(gid):
    d = pd.read_parquet(EA / f"{gid}.parquet")
    h = d.level_max.where(d.level_max_q.isin(OKQ))
    return h[~h.index.duplicated()]


def stations():
    keep, stats = [], {}
    good = set(good_catchments())
    for f in sorted(EA.glob("*.parquet")):
        gid = int(f.stem)
        if gid not in good:
            continue
        h = level_series(gid)
        tr = h[(h.index >= "1970-10-01") & (h.index <= TRAIN_END)].dropna()
        te = h[(h.index >= TEST_START) & (h.index <= "2022-09-30")].dropna()
        if len(tr) >= MIN_TRAIN and len(te) >= MIN_TEST and tr.std() > 0:
            keep.append(gid)
            stats[gid] = dict(mu=float(tr.mean()), sd=float(tr.std()),
                              q90=float(tr.quantile(.90)), q95=float(tr.quantile(.95)),
                              q99=float(tr.quantile(.99)))
    return keep, pd.DataFrame(stats).T


def build(L, gids, st):
    assign = nearest_donors(np.array(good_catchments()))
    frames, ids = [], []
    t0 = time.time()
    for gid in gids:
        X = features(gid)
        y = X.pop("y")
        X["y_now"] = y
        for lag in range(1, 7):
            X[f"y_lag{lag}"] = y.shift(lag)
        X["y_mean30"] = y.rolling(30, min_periods=15).mean()
        for r, d in enumerate(assign[gid][0]):
            X[f"nb{r}_d0"] = D0[d].reindex(X.index).values
            X[f"nb{r}_d1"] = D1[d].reindex(X.index).values
        h = ((level_series(gid) - st.loc[gid, "mu"]) / st.loc[gid, "sd"]).reindex(X.index)
        X["h_now"] = h
        for lag in range(1, 7):
            X[f"h_lag{lag}"] = h.shift(lag)
        X["h_mean30"] = h.rolling(30, min_periods=15).mean()
        X["h_rise1"] = h - h.shift(1)
        for k in range(1, L + 1):
            X[f"p_next{k}"] = X["p_0"].shift(-k)
        X["target"] = h.shift(-L)
        X["y_target"] = y.shift(-L)               # observed flow on the target day (flowmap fit)
        frames.append(X.astype("float32"))
        ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames); GID = np.concatenate(ids)
    print(f"  {len(gids)} stations, {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values], names=["gid", "date"])
    ens = pd.read_parquet(ENS).reindex(mi).set_axis(DATA.index)
    mm = pd.read_parquet(MEMMAX).reindex(mi).set_axis(DATA.index)
    for k in range(1, L + 1):
        DATA[f"fc_next{k}"] = ens[f"p_fc{k}"].values
        DATA[f"mx_next{k}"] = mm[f"p_fc{k}"].values
    return DATA, GID


