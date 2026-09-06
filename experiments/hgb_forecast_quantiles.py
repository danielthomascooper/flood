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
NWP = ROOT / "cache/nwp"
ENS = NWP / "gefs_catchment_leads_ens.parquet"
# rain sources that can drive the trained ladder (p_fc1 column each);
# 'perfect' is observed rain, 'ens'/'memmax' the GEFS 5-member mean / max
SOURCES = {"ens": ENS, "memmax": NWP / "gefs_catchment_leads_memmax.parquet",
           "tigge": NWP / "tigge_catchment_leads_mean.parquet",
           "tiggeq90": NWP / "tigge_catchment_leads_q90.parquet",
           "tiggeq95": NWP / "tigge_catchment_leads_q95.parquet",
           "tiggeq98": NWP / "tigge_catchment_leads_q98.parquet",
           "tiggemax": NWP / "tigge_catchment_leads_max.parquet"}
MODELS = OUT / "models"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
import joblib
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
    rain = {k: pd.read_parquet(v)[["p_fc1"]].reindex(mi).set_axis(DATA.index)["p_fc1"]
            for k, v in SOURCES.items() if v.exists()}
    return DATA, GID, rain


def run(name):
    dst = OUT / f"forecast_{name}_L1.parquet"
    a = ALPHAS[name]
    DATA, GID, rain = build()
    own = ["y_now"] + [f"y_lag{l}" for l in range(1, 7)] + ["y_mean30", "y_mean90"]
    don = [f"nb{r}_{l}" for r in range(3) for l in ("d0", "d1")]
    drop = own + don + ["p_next1", "target1"]
    weather = [c for c in DATA.columns if c not in drop]
    cols = weather + own + don + ["p_next1"]

    tgt = DATA["target1"]
    ok = tgt.notna().values & DATA["y_now"].notna().values
    is_tr = np.asarray(DATA.index <= TRAIN_END) & ok
    is_te = np.asarray(DATA.index >= TEST_START) & ok
    MODELS.mkdir(exist_ok=True)
    mpath = MODELS / f"hgb_fq_{name}_L1.joblib"
    if mpath.exists():
        m = joblib.load(mpath); print(f"{name}: loaded {mpath.name}", flush=True)
    else:
        t0 = time.time()
        m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=a) \
            .fit(DATA.loc[is_tr, cols], tgt[is_tr].values)
        joblib.dump(m, mpath)
        print(f"{name}: fitted in {time.time()-t0:.0f}s -> {mpath.name}", flush=True)

    idx = pd.DatetimeIndex(DATA.index[is_te], name="date") + pd.Timedelta(days=1)
    if dst.exists():
        out = pd.read_parquet(dst)
        assert len(out) == is_te.sum()
    else:
        out = pd.DataFrame({"gid": GID[is_te], "obs": tgt[is_te].values,
                            "covered": rain["ens"].loc[is_te].notna().values}, index=idx)
    Xte = DATA.loc[is_te, cols]
    if "pred_perfect" not in out:
        out["pred_perfect"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    Xte = Xte.copy()
    for src, col in rain.items():
        if f"pred_{src}" in out:
            continue
        Xte["p_next1"] = col.loc[is_te].values
        out[f"pred_{src}"] = np.clip(m.predict(Xte), 0, None).astype("float32")
        out[f"covered_{src}"] = col.loc[is_te].notna().values
        print(f"{name}: predicted {src}", flush=True)
    out.to_parquet(dst)
    print(f"{name}: wrote {dst.name} ({[c for c in out if c.startswith('pred_')]})", flush=True)


if MODE != "score":
    run(MODE)
    sys.exit(0)

# ---- score: `score [subset]` — subset = ens (GEFS-covered rows, default) or
# tigge (rows covered by BOTH archives, so every rain path is comparable)
SUBSET = sys.argv[2] if len(sys.argv) > 2 else "ens"
Q = {n: pd.read_parquet(OUT / f"forecast_{n}_L1.parquet") for n in ALPHAS}
base = Q["q99"]
cov = base.covered.values.copy()
if SUBSET != "ens":
    cov &= base[f"covered_{SUBSET}"].values
paths = [c[5:] for c in base.columns if c.startswith("pred_")
         and (f"covered_{c[5:]}" not in base or base[f"covered_{c[5:]}"].values[cov].mean() > 0.99)]
# AMAX days are defined on the FULL observed test record (gid x water-year
# with >=350 obs days) and then restricted to covered rows, so a rain
# archive with patchy month coverage is scored on the true annual peaks
# that fall inside it. Never .loc on the date index (non-unique): positional.
full = base[["gid", "obs"]].reset_index()
full["wy"] = full.date.dt.year + (full.date.dt.month >= 10)
nobs = full.groupby(["gid", "wy"]).obs.transform("size")
amax_full = full[nobs >= 350].groupby(["gid", "wy"]).obs.idxmax().values
is_amax = np.zeros(len(base), bool); is_amax[amax_full] = True
sub = base[cov][["gid", "obs"]].reset_index()
amax_pos = np.flatnonzero(is_amax[cov])
oa = sub.obs.values[amax_pos]

rows = []
for path in paths:
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
print(f"subset {SUBSET}: {cov.sum():,} rows, {len(amax_pos):,} AMAX events")
print(df.round(3).to_string(index=False))
csv = OUT / f"forecast_quantile_scenario_L1{'' if SUBSET == 'ens' else '_' + SUBSET}.csv"
df.to_csv(csv, index=False)

q50 = Q["q50"]
for path in paths:
    nse = per_catchment(pd.DataFrame(
        {"gid": q50.gid.values[cov], "obs": q50.obs.values[cov],
         "pred": q50[f"pred_{path}"].values[cov]}, index=q50.index[cov])).nse
    print(f"q50-as-point ({path} rain): median NSE {nse.median():+.3f}")
print(f"wrote {csv.name}")
