"""Phase 8c - operational-rain substitution (tree, level target, lead 1).

The level models are trained and scored on HadUK-Grid rain, which is
published months in arrears. At issue time the observed-rain window must
come from the live EA gauge network. This builds a catchment daily rain
series from EA gauges (gauges inside the CAMELS boundary, else the 3
nearest within 30 km; mean of Good/Estimated/Unchecked values), checks its
alignment and agreement with HadUK on the test years, then drives the
8b point model (refit on HadUK, as deployed) and the saved 8b-2 ladder
models with EA rain in the observed-rain features for every test row
where EA rain exists. Cost of going live = HadUK-driven skill minus
EA-driven skill on the same rows.

MODE: build (catchment rain parquet + agreement stats) | run | score
Outputs: cache/ea/catchment_rain_daily.parquet, results/oprain_rain_agreement.csv,
results/level_oprain_L1.parquet, results/level_oprain_cards.csv
"""
import sys, time
from pathlib import Path

import geopandas as gpd
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from level_common import stations, level_series, ENS, MEMMAX, FC_START

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
MODELS = OUT / "models"
RAIN = ROOT / "cache/ea/rain_daily"
CATCH_RAIN = ROOT / "cache/ea/catchment_rain_daily.parquet"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
OKQ = ["Good", "Estimated", "Unchecked"]
ALPHAS = {"q50": .50, "q75": .75, "q90": .90, "q95": .95, "q99": .99}
BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)


def build_rain():
    st = pd.read_parquet(ROOT / "cache/ea/rain_stations.parquet").dropna(subset=["lat", "lon"])
    st = st[[(RAIN / f"{s}.parquet").exists() for s in st.station]]
    pts = gpd.GeoDataFrame(st, geometry=[Point(x, y) for x, y in zip(st.lon, st.lat)], crs=4326).to_crs(27700)
    shp = gpd.read_file(ROOT / "data/Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp")
    idcol = next(c for c in shp.columns if "id" in c.lower())
    shp[idcol] = pd.to_numeric(shp[idcol], errors="coerce")
    shp = shp.set_index(idcol).to_crs(27700)
    gids, _ = stations()
    series = {}
    def load(s):
        if s not in series:
            d = pd.read_parquet(RAIN / f"{s}.parquet")
            series[s] = d.value.where(d.quality.astype(str).isin(OKQ))
        return series[s]
    out, meta = {}, []
    for gid in gids:
        poly = shp.geometry[gid]
        inside = pts[pts.within(poly)]
        if len(inside) >= 2:
            chosen = inside.station.tolist(); how = f"inside:{len(inside)}"
        else:
            dist = pts.geometry.distance(poly.centroid)
            near = pts.assign(dist=dist).sort_values("dist")
            near = near[near.dist <= 30_000].head(3)
            chosen = sorted(set(inside.station.tolist() + near.station.tolist())); how = f"nearest:{len(chosen)}"
        if not chosen:
            meta.append(dict(gid=gid, gauges=0, how="none")); continue
        m = pd.concat([load(s) for s in chosen], axis=1).mean(axis=1, skipna=True)
        out[gid] = m; meta.append(dict(gid=gid, gauges=len(chosen), how=how))
    R = pd.DataFrame(out).sort_index().astype("float32"); R.index.name = "date"
    R.to_parquet(CATCH_RAIN)
    meta = pd.DataFrame(meta).set_index("gid")
    # alignment + agreement vs HadUK on the test years, shift in {-1, 0, +1} days
    rows = []
    for gid in R.columns:
        hk = features(gid).p_0.loc[TEST_START:"2022-09-30"]
        ea = R[gid].reindex(hk.index)
        best = None
        for s in (-1, 0, 1):
            r = hk.corr(R[gid].shift(s).reindex(hk.index))
            if best is None or r > best[1]:
                best = (s, r)
        rows.append(dict(gid=gid, gauges=meta.loc[gid, "gauges"], how=meta.loc[gid, "how"],
                         coverage=float(ea.notna().mean()), best_shift=best[0], r_best=best[1],
                         r0=hk.corr(ea), bias_pct=float((ea.mean() / hk.mean() - 1) * 100),
                         wet_q95_ratio=float(ea.quantile(.95) / hk.quantile(.95))))
    A = pd.DataFrame(rows).set_index("gid")
    A.to_csv(OUT / "oprain_rain_agreement.csv")
    print(f"{len(R.columns)} catchments with EA gauge rain; gauges/catchment median {int(A.gauges.median())}")
    print("best shift counts:", A.best_shift.value_counts().to_dict())
    print(f"median r (shift 0) {A.r0.median():.3f}, r (best) {A.r_best.median():.3f}, bias {A.bias_pct.median():+.1f}%, "
          f"q95 ratio {A.wet_q95_ratio.median():.2f}, coverage {A.coverage.median():.2f}")


def build_data(gids, st, shift):
    R = pd.read_parquet(CATCH_RAIN)
    assign = nearest_donors(np.array(good_catchments()))
    frames, ids = [], []
    for gid in gids:
        ea = R[gid].shift(shift) if gid in R.columns else None
        if ea is not None:
            ea = ea[ea.index >= TEST_START]          # deployed model: trained on HadUK, fed EA rain at issue time
        X = features(gid, rain=ea)
        y = X.pop("y"); X["y_now"] = y
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
        X["p_next1"] = X["p_0"].shift(-1)
        X["target"] = h.shift(-1)
        X["ea_ok"] = (ea.reindex(X.index).notna() if ea is not None else pd.Series(False, index=X.index)).values
        frames.append(X.astype("float32")); ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames); GID = np.concatenate(ids)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values], names=["gid", "date"])
    ens = pd.read_parquet(ENS).reindex(mi).set_axis(DATA.index)
    mm = pd.read_parquet(MEMMAX).reindex(mi).set_axis(DATA.index)
    DATA["fc_next1"] = ens["p_fc1"].values; DATA["mx_next1"] = mm["p_fc1"].values
    return DATA, GID


def run():
    A = pd.read_csv(OUT / "oprain_rain_agreement.csv", index_col=0)
    shift = int(A.best_shift.mode().iloc[0])
    print(f"using EA rain shift {shift:+d} day(s) (modal best alignment)", flush=True)
    gids, st = stations()
    DATA, GID = build_data(gids, st, shift)
    aux = ["fc_next1", "mx_next1", "target", "ea_ok"]
    cols = [c for c in DATA.columns if c not in aux]
    ok = DATA["target"].notna().values & DATA["h_now"].notna().values & DATA["y_now"].notna().values
    dates = DATA.index; has_fc = DATA["fc_next1"].notna().values
    is_tr = np.asarray(dates <= TRAIN_END) & ok & (np.asarray(dates < FC_START) | has_fc)
    is_te = np.asarray(dates >= TEST_START) & ok
    # training rows are HadUK by construction (override starts at TEST_START)
    Xtr = DATA.loc[is_tr, cols].copy()
    fc = DATA.loc[is_tr, "fc_next1"].values
    Xtr["p_next1"] = np.where(np.isnan(fc), DATA.loc[is_tr, "p_next1"].values, fc)
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE).fit(Xtr, DATA.loc[is_tr, "target"].values)
    print(f"point model refit in {time.time()-t0:.0f}s", flush=True)
    Xte = DATA.loc[is_te, cols].copy(); Xte["p_next1"] = DATA.loc[is_te, "fc_next1"].values
    idx = pd.DatetimeIndex(dates[is_te], name="date") + pd.Timedelta(days=1)
    out = pd.DataFrame({"gid": GID[is_te], "obs": DATA.loc[is_te, "target"].values,
                        "covered": has_fc[is_te], "ea_ok": DATA.loc[is_te, "ea_ok"].values.astype(bool),
                        "direct_ea": m.predict(Xte).astype("float32")}, index=idx)
    for q in ALPHAS:
        mq = joblib.load(MODELS / f"hgb_level_{q}_L1.joblib")
        out[f"{q}_ea"] = mq.predict(Xte).astype("float32")
        print(f"ladder {q} predicted with EA rain", flush=True)
    out.to_parquet(OUT / "level_oprain_L1.parquet")
    print(f"wrote level_oprain_L1.parquet ({len(out):,} rows, EA rain on {out.ea_ok.mean():.1%})", flush=True)


if MODE == "build":
    build_rain(); sys.exit(0)
if MODE == "run":
    run(); sys.exit(0)

# ---- score: HadUK-driven vs EA-driven on identical rows -------------------
st = pd.read_csv(OUT / "level_fc_station_stats.csv", index_col=0)
ea = pd.read_parquet(OUT / "level_oprain_L1.parquet")
hk_pt = pd.read_parquet(OUT / "level_fc_L1.parquet")
hkQ = {q: pd.read_parquet(OUT / f"level_lq_{q}_L1.parquet") for q in ALPHAS}
assert len(ea) == len(hk_pt) and (ea.gid.values == hk_pt.gid.values).all()
sel = ea.covered.values & ea.ea_ok.values
obs = ea.obs.values[sel]; gid = ea.gid.values[sel]
alphas = np.array(list(ALPHAS.values()))


def p_exceed(M, thr):
    lo = M[:, 0] - 2 * (M[:, 1] - M[:, 0]); hi = M[:, -1] + 2 * (M[:, -1] - M[:, -2])
    xs = np.column_stack([lo, M, hi]); ys = np.concatenate([[0.0], alphas, [1.0]])
    return 1.0 - np.array([np.interp(t, x, ys) for t, x in zip(thr, xs)])


def nse_by_station(pred):
    df = pd.DataFrame({"gid": gid, "o": obs, "p": pred})
    return df.groupby("gid").apply(lambda x: 1 - ((x.o - x.p) ** 2).sum() / ((x.o - x.o.mean()) ** 2).sum(), include_groups=False)


rows = []
for name, pt, lad in (("HadUK", hk_pt.direct.values[sel], np.sort(np.stack([hkQ[q].pred_ens.values[sel] for q in ALPHAS], 1), 1)),
                      ("EA gauges", ea.direct_ea.values[sel], np.sort(np.stack([ea[f"{q}_ea"].values[sel] for q in ALPHAS], 1), 1))):
    row = dict(rain=name, n_rows=int(sel.sum()), n_stations=int(pd.Series(gid).nunique()),
               point_NSE_z=nse_by_station(pt).median(),
               point_MAE_m=float(np.median(pd.DataFrame({"gid": gid, "e": np.abs(pt - obs) * pd.Series(gid).map(st.sd).values}).groupby("gid").e.mean())))
    for thr_name in ("q90", "q95", "q99"):
        thr = ((pd.Series(gid).map(st[thr_name]) - pd.Series(gid).map(st.mu)) / pd.Series(gid).map(st.sd)).values
        o = obs > thr; prob = p_exceed(lad, thr)
        cuts = np.linspace(0.05, 0.95, 19)
        row[f"{thr_name}_bestCSI"] = max((((o & (prob >= c)).sum()) / max((o | (prob >= c)).sum(), 1)) for c in cuts)
        hr = 0.0
        for c in cuts:
            f = prob >= c; hits = (o & f).sum(); fa = (~o & f).sum()
            if hits + fa and fa / (hits + fa) <= 0.30:
                hr = hits / max(o.sum(), 1); break
        row[f"{thr_name}_hit@FAR30"] = hr
        row[f"{thr_name}_brier"] = float(np.mean((prob - o) ** 2))
    rows.append(row)
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
df.to_csv(OUT / "level_oprain_cards.csv", index=False)
