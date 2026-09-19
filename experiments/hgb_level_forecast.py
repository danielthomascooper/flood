"""Phase 8b - LEVEL as the target (tree, mixed regime, leads 1-3).

The research models predict daily-MEAN specific discharge. A practical
product needs the day's PEAK LEVEL. For the CAMELS-GB catchments with an
EA level station (cache/ea/daily/<nrfa>.parquet) this trains the
mixed-regime forecast tree on next-day(s) DAILY-MAX level, standardised
per station on the training years (stage datums are arbitrary), and
compares three routes on identical rows:

  direct   tree -> z-level, features = forecast set + own level lags
  flowmap  the existing mixed-regime FLOW forecast -> per-station monotone
           (isotonic) flow_mean -> level_max map fitted on training years;
           the rating-curve route, clipped at the training range
  persist  level(t+L) = level(t)

Metrics follow the aim: exceedance skill at each station's training-era
q90/q95/q99 level (hit rate, false-alarm ratio, CSI), error on the
station's annual-max-level days in metres, MAE in metres; NSE of z-level
only as a bridge to the research numbers. GEFS-covered test rows only.

MODE: L1|L2|L3 (fit + predict, one process per lead) | score
Outputs: results/level_fc_L{L}.parquet, results/level_fc_cards.csv,
results/level_fc_per_station.csv
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from level_common import FC_START  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAIN_END, TEST_START

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
import os
SMOKE = int(os.environ.get("LEVEL_SMOKE", "0"))   # N -> first N stations, *_smoke outputs
BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)


from level_common import level_series, stations, build  # noqa: E402


def run(L):
    dst = OUT / f"level_fc{'_smoke' if SMOKE else ''}_L{L}.parquet"
    if dst.exists():
        print(f"L{L}: exists, skipping"); return
    gids, st = stations()
    if SMOKE:
        gids = gids[:SMOKE]; st = st.loc[gids]
    DATA, GID = build(L, gids, st)
    aux = [c for c in DATA.columns if c.startswith("fc_next")] + ["target", "y_target"]
    cols = [c for c in DATA.columns if c not in aux]
    ok = DATA["target"].notna().values & DATA["h_now"].notna().values & DATA["y_now"].notna().values
    dates = DATA.index
    has_fc = DATA["fc_next1"].notna().values
    is_tr = np.asarray(dates <= TRAIN_END) & ok & (np.asarray(dates < FC_START) | has_fc)
    is_te = np.asarray(dates >= TEST_START) & ok

    def rain(mask, forecast_everywhere):
        X = DATA.loc[mask, cols].copy()
        sub = DATA.loc[mask]
        for k in range(1, L + 1):
            fc = sub[f"fc_next{k}"].values
            X[f"p_next{k}"] = fc if forecast_everywhere else np.where(np.isnan(fc), sub[f"p_next{k}"].values, fc)
        return X

    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE).fit(rain(is_tr, False), DATA.loc[is_tr, "target"].values)
    print(f"L{L}: {is_tr.sum():,} train rows, fitted in {time.time()-t0:.0f}s", flush=True)
    idx = pd.DatetimeIndex(dates[is_te], name="date") + pd.Timedelta(days=L)
    out = pd.DataFrame({"gid": GID[is_te], "obs": DATA.loc[is_te, "target"].values,
                        "direct": m.predict(rain(is_te, True)).astype("float32"),
                        "persist": DATA.loc[is_te, "h_now"].values,
                        "covered": has_fc[is_te]}, index=idx)

    # flowmap route: existing mixed-regime flow forecast -> isotonic flow->level per station
    fl = pd.read_parquet(OUT / f"forecast_fctrain_mixeds_L{L}.parquet")
    fl = fl.set_index([fl.gid.values, fl.index]).pred
    key = pd.MultiIndex.from_arrays([out.gid.values, out.index])
    qhat = fl.reindex(key).to_numpy("float32")
    fm = np.full(len(out), np.nan, "float32")
    trm = np.asarray(dates <= TRAIN_END) & DATA["target"].notna().values & DATA["y_target"].notna().values
    for gid in gids:
        a = trm & (GID == gid)
        iso = IsotonicRegression(out_of_bounds="clip").fit(DATA.loc[a, "y_target"].values, DATA.loc[a, "target"].values)
        b = out.gid.values == gid
        v = ~np.isnan(qhat[b])
        tmp = np.full(b.sum(), np.nan, "float32"); tmp[v] = iso.predict(qhat[b][v]); fm[b] = tmp
    out["flowmap"] = fm
    out.to_parquet(dst)
    st.to_csv(OUT / f"level_fc{'_smoke' if SMOKE else ''}_station_stats.csv")
    print(f"L{L}: wrote {dst.name} ({len(out):,} test rows, {len(gids)} stations)", flush=True)


if MODE != "score":
    run(int(MODE[1])); sys.exit(0)

# ---- score ---------------------------------------------------------------
TAG = "_smoke" if SMOKE else ""
st = pd.read_csv(OUT / f"level_fc{TAG}_station_stats.csv", index_col=0)
cards, per = [], []
for L in (1, 2, 3):
    p = OUT / f"level_fc{TAG}_L{L}.parquet"
    if not p.exists():
        continue
    d = pd.read_parquet(p)
    d = d[d.covered & d.flowmap.notna()].reset_index()
    d["wy"] = d.date.dt.year + (d.date.dt.month >= 10)
    sd = d.gid.map(st.sd).values
    amax = d.loc[d.groupby(["gid", "wy"]).obs.idxmax()]
    for route in ("direct", "flowmap", "persist"):
        g = d.groupby("gid")
        nse = g.apply(lambda x: 1 - ((x.obs - x[route]) ** 2).sum() / ((x.obs - x.obs.mean()) ** 2).sum(),
                      include_groups=False)
        row = dict(lead=L, route=route, n_stations=d.gid.nunique(), median_NSE_z=nse.median(),
                   MAE_m=float(np.median(g.apply(lambda x: (x.obs - x[route]).abs().mean(), include_groups=False)
                                         * st.sd.reindex(nse.index))),
                   amax_day_err_m=float(np.median((amax[route] - amax.obs) * amax.gid.map(st.sd))),
                   amax_day_err_sd=float(np.median(amax[route] - amax.obs)))
        for q in ("q90", "q95", "q99"):
            thr = ((d.gid.map(st[q]) - d.gid.map(st.mu)) / d.gid.map(st.sd)).values
            o, f = d.obs.values > thr, d[route].values > thr
            hits, miss, fa = (o & f).sum(), (o & ~f).sum(), (~o & f).sum()
            row[f"{q}_hit"] = hits / max(hits + miss, 1); row[f"{q}_FAR"] = fa / max(hits + fa, 1)
            row[f"{q}_CSI"] = hits / max(hits + miss + fa, 1)
        cards.append(row)
        per.append(nse.rename(f"{route}_L{L}"))
df = pd.DataFrame(cards)
print(df.round(3).to_string(index=False))
df.to_csv(OUT / f"level_fc{TAG}_cards.csv", index=False)
pd.concat(per, axis=1).to_csv(OUT / f"level_fc{TAG}_per_station.csv")
