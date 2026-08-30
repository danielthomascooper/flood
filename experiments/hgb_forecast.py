"""From simulation to forecasting — Phase 6 F1 (tree).

Every model so far estimates the flow on day t from information complete
only at the end of day t. This asks what survives when the target moves
into the future. Issue day = t; target = flow at t+L. Inputs are everything
known at the end of day t: the standard forcings and rolling windows up to
t (rain on day t included - it is observed by then), and optionally the
target's own past flow and the neighbours' flow at t.

Variants at lead L = 1 day (each one fit on the identical split):

  persistence   no model: predict flow(t+1) = flow(t)      (the bar to beat)
  weather       standard features at t, target t+1        (no own flow)
  ar            + own flow at t, t-1, ..., t-6 and 30/90-day means
  ar_donor      + 3 nearest donors' flow at t and t-1
  ar_perfect    ar_donor + the actual rain on t+1 ("perfect rainfall
                forecast": the ceiling an NWP-driven forecaster could reach)

plus ar_donor at L = 2 and L = 3 to see how skill decays with lead.

All targets/feature alignments are built on each catchment's full daily
index BEFORE dropping missing days, so shifts never cross gaps. Reports the
standard card, skill vs persistence, and paired per-catchment NSE.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from evaluate import evaluate, per_catchment, amax_bias, report

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

gauges = good_catchments()
assign = nearest_donors(np.array(gauges))

print("building per-catchment forecast frames...", flush=True)
t0 = time.time()
frames, ids = [], []
for gid in gauges:
    X = features(gid)                           # full daily index, has 'y'
    y = X.pop("y")
    X["y_now"] = y                              # flow at t (persistence)
    for lag in range(1, 7):
        X[f"y_lag{lag}"] = y.shift(lag)
    X["y_mean30"] = y.rolling(30, min_periods=15).mean()
    X["y_mean90"] = y.rolling(90, min_periods=45).mean()
    for r, d in enumerate(assign[gid][0]):
        X[f"nb{r}_d0"] = D0[d].reindex(X.index).values
        X[f"nb{r}_d1"] = D1[d].reindex(X.index).values
    X["p_next"] = X["p_0"].shift(-1)            # rain on t+1 (perfect forecast)
    for L in (1, 2, 3):
        X[f"target{L}"] = y.shift(-L)
    frames.append(X.astype("float32"))
    ids.append(np.full(len(X), gid, dtype=np.int32))
DATA = pd.concat(frames); GID = np.concatenate(ids)
del frames
print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)

OWN = ["y_now"] + [f"y_lag{l}" for l in range(1, 7)] + ["y_mean30", "y_mean90"]
DON = [f"nb{r}_{l}" for r in range(3) for l in ("d0", "d1")]
EXTRA = OWN + DON + ["p_next"] + [f"target{L}" for L in (1, 2, 3)]
WEATHER = [c for c in DATA.columns if c not in EXTRA]

VARIANTS = [
    ("weather",    1, WEATHER),
    ("ar",         1, WEATHER + OWN),
    ("ar_donor",   1, WEATHER + OWN + DON),
    ("ar_perfect", 1, WEATHER + OWN + DON + ["p_next"]),
    ("ar_donor",   2, WEATHER + OWN + DON),
    ("ar_donor",   3, WEATHER + OWN + DON),
]

dates = DATA.index
is_tr = np.asarray(dates <= TRAIN_END)
is_te = np.asarray(dates >= TEST_START)


def split(L, cols):
    tgt = DATA[f"target{L}"]
    ok = tgt.notna().values & DATA["y_now"].notna().values
    tr, te = is_tr & ok, is_te & ok
    return (DATA.loc[tr, cols], tgt[tr].values, DATA.loc[te, cols],
            tgt[te].values, GID[te], DATA.loc[te, "y_now"].values, dates[te])



# ---- run mode: one variant per process (OOM-safe), resumable -------------
import gc
MODE = sys.argv[2] if len(sys.argv) > 2 else "all"


def run_variant(name, L, cols):
    key = f"{name}_L{L}"
    out_p = SCRATCH / f"forecast_{key}.parquet"
    if out_p.exists():
        print(f"  {key}: exists, skipping", flush=True)
        return
    Xtr, ytr, Xte, yte, gte, ynow, dte = split(L, cols)
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE).fit(Xtr, ytr)
    idx = pd.DatetimeIndex(dte, name="date") + pd.Timedelta(days=L)
    res = pd.DataFrame({"gid": gte, "obs": yte,
                        "pred": np.clip(m.predict(Xte), 0, None).astype("float32")},
                       index=idx)
    res.to_parquet(out_p)
    pp = SCRATCH / f"forecast_persistence_L{L}.parquet"
    if not pp.exists():
        pd.DataFrame({"gid": gte, "obs": yte, "pred": ynow.astype("float32")},
                     index=idx).to_parquet(pp)
    print(f"  {key}: fitted in {time.time()-t0:.0f}s", flush=True)
    del Xtr, Xte, m, res; gc.collect()


if MODE != "score":
    todo = VARIANTS if MODE == "all" else [v for v in VARIANTS if f"{v[0]}_L{v[1]}" == MODE]
    for name, L, cols in todo:
        run_variant(name, L, cols)
    if MODE != "all":
        sys.exit(0)

# ---- score mode: everything from parquets ---------------------------------
del DATA; gc.collect()
order = ["persistence_L1", "weather_L1", "ar_L1", "ar_donor_L1", "ar_perfect_L1",
         "persistence_L2", "ar_donor_L2", "persistence_L3", "ar_donor_L3"]
preds = {k: pd.read_parquet(SCRATCH / f"forecast_{k}.parquet") for k in order}
preds["ar_donor_L1"].to_parquet(OUT / "forecast_L1_test_predictions.parquet")
rows = [evaluate(preds[k], k)[0] for k in order]
print()
df = report(rows)
df.to_csv(OUT / "forecast_cards.csv")

print("\n=== skill vs persistence (paired per catchment) ===")
pcs = {k: per_catchment(v).nse for k, v in preds.items()}
for L in (1, 2, 3):
    for k in [x for x in order if x.endswith(f"L{L}") and not x.startswith("persist")]:
        d = pcs[k] - pcs[f"persistence_L{L}"]
        print(f"  {k:16s} median dNSE vs persistence {d.median():+.3f}, "
              f"beats it in {(d>0).mean():.0%} of catchments; "
              f"AMAX bias {amax_bias(preds[k]).median():+.1f}% "
              f"(persistence {amax_bias(preds[f'persistence_L{L}']).median():+.1f}%)")
pd.DataFrame(pcs).to_csv(OUT / "forecast_per_catchment.csv")
print(f"\nwrote forecast_cards.csv, forecast_per_catchment.csv, forecast_L1_test_predictions.parquet")
