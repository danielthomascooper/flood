"""Real forecast rain — Phase 7 (tree ladder driven by GEFS).

The training years predate our GEFS archive, so the model that uses future
rain is trained on *observed* future rain (the ar_perfect setup, the
standard train-on-obs / drive-with-forecast approach) and then, at test
time, predicted twice: once with observed future rain (the ceiling) and
once with GEFS c00 lead-1..L rain substituted into the same columns. The
gap between the two is the price of a real 2010s-era NWP forecast; where
the GEFS archive has a hole (2020-01..09) the substituted value is NaN and
the trees route it like any missing feature.

For lead L the model sees the full forecast rain path p_next1..p_nextL
(rain on t+1..t+L), unlike Phase 6's ar_perfect which had only t+1.

MODE (argv[1]): L1 | L2 | L3 (fit, one per process - OOM-safe, resumable)
                | score (cards + skill vs persistence from parquets).
Outputs in results/: forecast_ar_perfect2_L{L}.parquet,
forecast_ar_gefs_L{L}.parquet, forecast_persist2_L{L}.parquet,
forecast_gefs_cards.csv, forecast_gefs_per_catchment.csv.
"""
import gc, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from evaluate import evaluate, per_catchment, amax_bias, report

OUT = Path(__file__).resolve().parent / "results"
FC = Path(__file__).resolve().parents[1] / "cache/nwp/gefs_catchment_leads.parquet"
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
            X[f"p_next{k}"] = X["p_0"].shift(-k)     # observed rain on t+k
            X[f"target{k}"] = y.shift(-k)
        frames.append(X.astype("float32"))
        ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames)
    GID = np.concatenate(ids)
    print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)
    fc = pd.read_parquet(FC)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values],
                                   names=["gid", "date"])
    FCA = fc.reindex(mi).set_axis(DATA.index)        # aligned forecast rain
    del fc
    return DATA, GID, FCA


def run(L):
    if (OUT / f"forecast_ar_gefs_L{L}.parquet").exists():
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
    gte, yte = GID[is_te], tgt[is_te].values
    base = dict(gid=gte, obs=yte)
    pd.DataFrame({**base, "pred": DATA.loc[is_te, "y_now"].values.astype("float32")},
                 index=idx).to_parquet(OUT / f"forecast_persist2_L{L}.parquet")

    Xte = DATA.loc[is_te, cols]
    pd.DataFrame({**base, "pred": np.clip(m.predict(Xte), 0, None).astype("float32")},
                 index=idx).to_parquet(OUT / f"forecast_ar_perfect2_L{L}.parquet")
    Xte = Xte.copy()
    for k in range(1, L + 1):                        # drive with GEFS rain
        Xte[f"p_next{k}"] = FCA.loc[is_te, f"p_fc{k}"].values
    pd.DataFrame({**base, "pred": np.clip(m.predict(Xte), 0, None).astype("float32"),
                  "covered": FCA.loc[is_te, "p_fc1"].notna().values},
                 index=idx).to_parquet(OUT / f"forecast_ar_gefs_L{L}.parquet")
    print(f"L{L}: wrote perfect2 / gefs / persist2 parquets", flush=True)


if MODE != "score":
    run(int(MODE[1]))
    sys.exit(0)

# ---- score ---------------------------------------------------------------
rows, pcs, lines = [], {}, []
for L in (1, 2, 3):
    P = {k: pd.read_parquet(OUT / f"forecast_{k}_L{L}.parquet")
         for k in ("persist2", "ar_perfect2", "ar_gefs")}
    cov = P["ar_gefs"].covered.values
    for k, v in P.items():
        rows.append(evaluate(v[["gid", "obs", "pred"]], f"{k}_L{L}")[0])
        pcs[f"{k}_L{L}"] = per_catchment(v).nse
        pcs[f"{k}_L{L}_cov"] = per_catchment(v[cov]).nse
    for k in ("ar_perfect2", "ar_gefs"):
        d = pcs[f"{k}_L{L}"] - pcs[f"persist2_L{L}"]
        dc = pcs[f"{k}_L{L}_cov"] - pcs[f"persist2_L{L}_cov"]
        lines.append(
            f"  {k:12s} L{L}: dNSE vs persistence {d.median():+.3f} "
            f"(GEFS-covered rows only {dc.median():+.3f}), beats it in "
            f"{(d > 0).mean():.0%}; AMAX bias {amax_bias(P[k]).median():+.1f}% "
            f"(persistence {amax_bias(P['persist2']).median():+.1f}%)")
print()
df = report(rows)
df.to_csv(OUT / "forecast_gefs_cards.csv")
print("\n=== skill vs persistence (paired per catchment, median) ===")
print("\n".join(lines))
pd.DataFrame(pcs).to_csv(OUT / "forecast_gefs_per_catchment.csv")
print("\nwrote forecast_gefs_cards.csv, forecast_gefs_per_catchment.csv")
