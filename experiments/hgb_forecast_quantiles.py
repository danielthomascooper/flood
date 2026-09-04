"""Scenario-matched forecast ladder — Phase 7c follow-up (tree, lead 1).

Arc's 7c found the rain-channel quantile ladder keeps ceiling sharpness
under real forecast rain and so under-covers annual peaks (LSTM q99:
87.7% with perfect rain -> 73.0% with ensemble-mean rain). The queued
no-retraining fix: drive the UPPER quantiles with a pessimistic rain
scenario instead of the mean. This tests it in tree land: one lead-1
ladder (q50/q90/q95/q99) trained on observed next-day rain, each
quantile predicted three ways — observed rain (ceiling), ensemble-mean
rain, and member-MAX rain in the same p_next1 channel.

MODE (argv[1]): q50 | q90 | q95 | q99 (fit + 3 predictions, one per
process) | score. Fit outputs results/forecast_q{XX}_L1.parquet with
pred_perfect / pred_ens / pred_memmax. Score: pooled coverage, AMAX-day
coverage (one event per gid x water-year with >=350 obs days), and the
median envelope/obs ratio on AMAX days, per rain path; covered rows only.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from evaluate import per_catchment

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
MEMBERS = ["c00", "p01", "p02", "p03", "p04"]
FC = ROOT / "cache/nwp/gefs_catchment_leads_members.parquet"
ENS = ROOT / "cache/nwp/gefs_catchment_leads_ens.parquet"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
ALPHAS = {"q50": 0.50, "q90": 0.90, "q95": 0.95, "q99": 0.99}

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
        X["p_next1"] = X["p_0"].shift(-1)
        X["target1"] = y.shift(-1)
        frames.append(X.astype("float32"))
        ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames)
    GID = np.concatenate(ids)
    print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values],
                                   names=["gid", "date"])
    ens = pd.read_parquet(ENS)[["p_fc1"]].reindex(mi).set_axis(DATA.index)
    mem = pd.read_parquet(FC)[[f"p_fc1_{m}" for m in MEMBERS]] \
        .reindex(mi).set_axis(DATA.index)
    return DATA, GID, ens, mem


def run(name):
    dst = OUT / f"forecast_{name}_L1.parquet"
    if dst.exists():
        print(f"{name}: exists, skipping"); return
    a = ALPHAS[name]
    DATA, GID, ens, mem = build()
    own = ["y_now"] + [f"y_lag{l}" for l in range(1, 7)] + ["y_mean30", "y_mean90"]
    don = [f"nb{r}_{l}" for r in range(3) for l in ("d0", "d1")]
    drop = own + don + ["p_next1", "target1"]
    weather = [c for c in DATA.columns if c not in drop]
    cols = weather + own + don + ["p_next1"]

    tgt = DATA["target1"]
    ok = tgt.notna().values & DATA["y_now"].notna().values
    is_tr = np.asarray(DATA.index <= TRAIN_END) & ok
    is_te = np.asarray(DATA.index >= TEST_START) & ok
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=a) \
        .fit(DATA.loc[is_tr, cols], tgt[is_tr].values)
    print(f"{name}: fitted in {time.time()-t0:.0f}s", flush=True)

    idx = pd.DatetimeIndex(DATA.index[is_te], name="date") + pd.Timedelta(days=1)
    out = pd.DataFrame({"gid": GID[is_te], "obs": tgt[is_te].values,
                        "covered": ens.loc[is_te, "p_fc1"].notna().values},
                       index=idx)
    Xte = DATA.loc[is_te, cols]
    out["pred_perfect"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    Xte = Xte.copy()
    Xte["p_next1"] = ens.loc[is_te, "p_fc1"].values
    out["pred_ens"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    Xte["p_next1"] = mem.loc[is_te].max(axis=1).values
    out["pred_memmax"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    out.to_parquet(dst)
    print(f"{name}: wrote {dst.name}", flush=True)


if MODE != "score":
    run(MODE)
    sys.exit(0)

# ---- score ---------------------------------------------------------------
Q = {n: pd.read_parquet(OUT / f"forecast_{n}_L1.parquet") for n in ALPHAS}
base = Q["q99"]
cov = base.covered.values
# never .loc on the date index (non-unique, 416 rows/date): go positional
sub = base[cov][["gid", "obs"]].reset_index()
sub["wy"] = sub.date.dt.year + (sub.date.dt.month >= 10)
nobs = sub.groupby(["gid", "wy"]).obs.transform("size")
amax_pos = sub[nobs >= 350].groupby(["gid", "wy"]).obs.idxmax().values
oa = sub.obs.values[amax_pos]

rows = []
for path in ("perfect", "ens", "memmax"):
    for n, a in ALPHAS.items():
        p = Q[n][f"pred_{path}"].values[cov]
        pa = p[amax_pos]
        rows.append({"rain": path, "q": n, "nominal": a,
                     "pooled": float((sub.obs.values <= p).mean()),
                     "amax_days": float((oa <= pa).mean()),
                     "amax_median_ratio": float(np.median(pa / oa)),
                     "median_width_q99_q50":
                         float(np.median(Q["q99"][f"pred_{path}"].values[cov]
                                         - Q["q50"][f"pred_{path}"].values[cov]))
                         if n == "q99" else np.nan})
df = pd.DataFrame(rows)
print(f"{len(amax_pos):,} AMAX events (covered rows only)")
print(df.to_string(index=False))
df.to_csv(OUT / "forecast_quantile_scenario_L1.csv", index=False)

q50 = Q["q50"]
for path in ("perfect", "ens"):
    nse = per_catchment(pd.DataFrame(
        {"gid": q50.gid.values[cov], "obs": q50.obs.values[cov],
         "pred": q50[f"pred_{path}"].values[cov]}, index=q50.index[cov])).nse
    print(f"q50-as-point ({path} rain): median NSE {nse.median():+.3f}")
print("wrote forecast_quantile_scenario_L1.csv")
