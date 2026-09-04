"""Ensemble-of-flows vs flow-of-ensemble-mean — Phase 7 upgrade 3 stage 2.

Upgrade 3 showed the 5-member ensemble-MEAN rain lifts point skill but
smooths extremes (AMAX/peak-day bias worsen vs c00). The hydrological
question: does the flood signal survive in member space? Same
train-on-obs tree as hgb_forecast_gefs.py; at test time it is driven
separately with each member's catchment-mean rain path (5 flow
forecasts per row) and with the member-MAX rain path.

MODE (argv[1]): L1 | L2 | L3 (fit + predict, one per process) | score.
Fit outputs: results/forecast_members_L{L}.parquet with pred_c00..pred_p04,
pred_rainmax (+ gid, obs, covered). Score compares, against the
ensemble-mean-rain run: mean / max / q-implied combos of member flows,
and the rainmax-driven flow.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from evaluate import evaluate, per_catchment, report

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]
FC = ROOT / "cache/nwp/gefs_catchment_leads_members.parquet"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)


def build():
    gauges = good_catchments()
    assign = nearest_donors(np.array(gauges))
    print("building forecast frames...", flush=True)
    t0 = time.time()
    frames, ids = [], []
    for gid in gauges:
        X = features(gid)
        y = X.pop("y")
        X["y_now"] = y
        for lag in range(1, 7):
            X[f"y_lag{lag}"] = y.shift(lag)
        X["y_mean30"] = y.rolling(30, min_periods=15).mean()
        X["y_mean90"] = y.rolling(90, min_periods=45).mean()
        for r, d in enumerate(assign[gid][0]):
            X[f"nb{r}_d0"] = D0[d].reindex(X.index).values
            X[f"nb{r}_d1"] = D1[d].reindex(X.index).values
        for k in (1, 2, 3):
            X[f"p_next{k}"] = X["p_0"].shift(-k)
            X[f"target{k}"] = y.shift(-k)
        frames.append(X.astype("float32"))
        ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames)
    GID = np.concatenate(ids)
    print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)
    fc = pd.read_parquet(FC)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values],
                                   names=["gid", "date"])
    FCA = fc.reindex(mi).set_axis(DATA.index)
    del fc
    return DATA, GID, FCA


def run(L):
    dst = OUT / f"forecast_members_L{L}.parquet"
    if dst.exists():
        print(f"L{L}: exists, skipping"); return
    DATA, GID, FCA = build()
    own = ["y_now"] + [f"y_lag{l}" for l in range(1, 7)] + ["y_mean30", "y_mean90"]
    don = [f"nb{r}_{l}" for r in range(3) for l in ("d0", "d1")]
    drop = own + don + [f"p_next{k}" for k in (1, 2, 3)] \
        + [f"target{k}" for k in (1, 2, 3)]
    weather = [c for c in DATA.columns if c not in drop]
    rain = [f"p_next{k}" for k in range(1, L + 1)]
    cols = weather + own + don + rain

    tgt = DATA[f"target{L}"]
    ok = tgt.notna().values & DATA["y_now"].notna().values
    is_tr = np.asarray(DATA.index <= TRAIN_END) & ok
    is_te = np.asarray(DATA.index >= TEST_START) & ok
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE).fit(DATA.loc[is_tr, cols],
                                                  tgt[is_tr].values)
    print(f"L{L}: fitted in {time.time()-t0:.0f}s", flush=True)

    idx = pd.DatetimeIndex(DATA.index[is_te], name="date") + pd.Timedelta(days=L)
    out = pd.DataFrame({"gid": GID[is_te], "obs": tgt[is_te].values,
                        "covered": FCA.loc[is_te, "p_fc1_c00"].notna().values},
                       index=idx)
    Xte = DATA.loc[is_te, cols].copy()
    for mem in MEMBERS:
        for k in range(1, L + 1):
            Xte[f"p_next{k}"] = FCA.loc[is_te, f"p_fc{k}_{mem}"].values
        out[f"pred_{mem}"] = np.clip(m.predict(Xte), 0, None).astype("float32")
        print(f"L{L}: predicted {mem}", flush=True)
    for k in range(1, L + 1):                     # member-max rain path
        Xte[f"p_next{k}"] = FCA.loc[
            is_te, [f"p_fc{k}_{mem}" for mem in MEMBERS]].max(axis=1).values
    out["pred_rainmax"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    out.to_parquet(dst)
    print(f"L{L}: wrote {dst.name}", flush=True)


if MODE != "score":
    run(int(MODE[1]))
    sys.exit(0)

# ---- score: expand combos into plain parquets for analysis_forecast_skill
for L in (1, 2, 3):
    src = OUT / f"forecast_members_L{L}.parquet"
    if not src.exists():
        print(f"L{L}: no members parquet yet"); continue
    df = pd.read_parquet(src)
    P = df[[f"pred_{mem}" for mem in MEMBERS]].values
    combos = {"flowmean": P.mean(1), "flowmax": P.max(1),
              "rainmax": df.pred_rainmax.values}
    for name, pred in combos.items():
        pd.DataFrame({"gid": df.gid.values, "obs": df.obs.values,
                      "pred": pred.astype("float32"),
                      "covered": df.covered.values}, index=df.index
                     ).to_parquet(OUT / f"forecast_{name}_L{L}.parquet")
    print(f"L{L}: wrote {', '.join(f'forecast_{n}_L{L}.parquet' for n in combos)}")
