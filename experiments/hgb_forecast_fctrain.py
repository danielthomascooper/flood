"""Forecast-rain TRAINING — Phase 7g (tree). The Taccari question in tree land.

Every forecast run so far trained on observed future rain and swapped a
forecast in at test time, so the model treats the rain channel as exact.
The GEFS reforecast now covers 2000-01 -> TRAIN_END (2010-09-30), the
last ~11 of the 40 training water-years, so training CAN see forecast
rain and member spread. Variants (all driven with GEFS ens-mean rain at
test; spread variants also get s_fc at test):

  obs_2000      train 2000-2010, observed next-day rain   (data-size control)
  fc_2000       train 2000-2010, GEFS ens-mean rain       (forecast-trained)
  fcs_2000      fc_2000 + s_fc1..L spread features        (spread channel)
  mixed         train 1970-2010: obs rain <2000, GEFS rain >=2000 (tree
                analogue of pre-train + fine-tune)
  mixeds        mixed + spread (NaN before 2000: trees route it)
Reference: forecast_ar_gefs_ens_L{L} (train 1970-2010 obs rain, ens at test).

MODE: L1|L2|L3 (point, all variants, one process per lead)
      Q<variant>_<q>  (lead-1 quantile fit, e.g. Qfcs_2000_q99; saves model
                       to results/models/, predicts ens + memmax rain)
      score | scoreq
Outputs: results/forecast_fctrain_<variant>_L{L}.parquet (point),
         results/forecast_fq_<variant>_<q>_L1.parquet (ladder).
"""
import gc, sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import features, good_catchments, TRAIN_END, TEST_START
from nowcast_common import D0, D1, nearest_donors
from evaluate import per_catchment

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "results"
MODELS = OUT / "models"
ENS = ROOT / "cache/nwp/gefs_catchment_leads_ens.parquet"
MEMMAX = ROOT / "cache/nwp/gefs_catchment_leads_memmax.parquet"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
FC_START = "2000-01-01"
VARIANTS = ["obs_2000", "fc_2000", "fcs_2000", "mixed", "mixeds"]
ALPHAS = {"q50": 0.50, "q90": 0.90, "q95": 0.95, "q99": 0.99}
BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)


def build(L):
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
        for k in range(1, L + 1):
            X[f"p_next{k}"] = X["p_0"].shift(-k)
        X["target"] = y.shift(-L)
        frames.append(X.astype("float32"))
        ids.append(np.full(len(X), gid, dtype=np.int32))
    DATA = pd.concat(frames)
    GID = np.concatenate(ids)
    print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)
    mi = pd.MultiIndex.from_arrays([GID, DATA.index.values], names=["gid", "date"])
    ens = pd.read_parquet(ENS).reindex(mi).set_axis(DATA.index)
    for k in range(1, L + 1):                      # forecast rain + spread, NaN outside archive
        DATA[f"fc_next{k}"] = ens[f"p_fc{k}"].values
        DATA[f"s_next{k}"] = ens[f"s_fc{k}"].values
    mm = pd.read_parquet(MEMMAX).reindex(mi).set_axis(DATA.index)
    for k in range(1, L + 1):
        DATA[f"mx_next{k}"] = mm[f"p_fc{k}"].values
    del ens, mm
    return DATA, GID


def columns(DATA, L, spread):
    own = ["y_now"] + [f"y_lag{l}" for l in range(1, 7)] + ["y_mean30", "y_mean90"]
    don = [f"nb{r}_{l}" for r in range(3) for l in ("d0", "d1")]
    aux = [c for c in DATA.columns if c.startswith(("fc_next", "s_next", "mx_next", "p_next"))] + ["target"]
    weather = [c for c in DATA.columns if c not in own + don + aux]
    cols = weather + own + don + [f"p_next{k}" for k in range(1, L + 1)]
    if spread:
        cols += [f"s_next{k}" for k in range(1, L + 1)]
    return cols


def train_frame(DATA, variant, L):
    """Rows + rain-channel values used to train `variant`. Returns (mask, X)."""
    ok = DATA["target"].notna().values & DATA["y_now"].notna().values
    dates = DATA.index
    pre = np.asarray(dates <= TRAIN_END) & ok
    since2000 = np.asarray(dates >= FC_START)
    has_fc = DATA["fc_next1"].notna().values
    if variant.endswith("_2000"):
        mask = pre & since2000 & has_fc
    else:                                          # mixed
        mask = pre & (~since2000 | has_fc)
    X = DATA.loc[mask, columns(DATA, L, variant.endswith("s") and variant != "obs_2000")].copy()
    if variant != "obs_2000":                      # forecast rain where the archive has it
        sub = DATA.loc[mask]
        for k in range(1, L + 1):
            fc = sub[f"fc_next{k}"].values
            X[f"p_next{k}"] = np.where(np.isnan(fc), sub[f"p_next{k}"].values, fc)
    return mask, X


def test_frame(DATA, L, spread, rain="fc"):
    ok = DATA["target"].notna().values & DATA["y_now"].notna().values
    is_te = np.asarray(DATA.index >= TEST_START) & ok
    X = DATA.loc[is_te, columns(DATA, L, spread)].copy()
    for k in range(1, L + 1):
        X[f"p_next{k}"] = DATA.loc[is_te, f"{rain}_next{k}"].values
    return is_te, X


def run_point(L):
    DATA, GID = build(L)
    for v in VARIANTS:
        dst = OUT / f"forecast_fctrain_{v}_L{L}.parquet"
        if dst.exists():
            print(f"{v} L{L}: exists, skipping"); continue
        spread = v in ("fcs_2000", "mixeds")
        mask, Xtr = train_frame(DATA, v, L)
        t0 = time.time()
        m = HistGradientBoostingRegressor(**BASE).fit(Xtr, DATA.loc[mask, "target"].values)
        print(f"{v} L{L}: {mask.sum():,} train rows, fitted in {time.time()-t0:.0f}s", flush=True)
        is_te, Xte = test_frame(DATA, L, spread)
        idx = pd.DatetimeIndex(DATA.index[is_te], name="date") + pd.Timedelta(days=L)
        pd.DataFrame({"gid": GID[is_te], "obs": DATA.loc[is_te, "target"].values,
                      "pred": np.clip(m.predict(Xte), 0, None).astype("float32"),
                      "covered": DATA.loc[is_te, "fc_next1"].notna().values},
                     index=idx).to_parquet(dst)
        del m, Xtr, Xte; gc.collect()
        print(f"{v} L{L}: wrote {dst.name}", flush=True)


def run_quantile(variant, q):
    dst = OUT / f"forecast_fq_{variant}_{q}_L1.parquet"
    if dst.exists():
        print(f"{variant} {q}: exists, skipping"); return
    DATA, GID = build(1)
    spread = variant in ("fcs_2000", "mixeds")
    mask, Xtr = train_frame(DATA, variant, 1)
    MODELS.mkdir(exist_ok=True)
    mpath = MODELS / f"hgb_fq_{variant}_{q}_L1.joblib"
    if mpath.exists():
        m = joblib.load(mpath)
    else:
        t0 = time.time()
        m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=ALPHAS[q]) \
            .fit(Xtr, DATA.loc[mask, "target"].values)
        joblib.dump(m, mpath)
        print(f"{variant} {q}: {mask.sum():,} train rows, fitted in {time.time()-t0:.0f}s", flush=True)
    is_te, Xte = test_frame(DATA, 1, spread, rain="fc")
    idx = pd.DatetimeIndex(DATA.index[is_te], name="date") + pd.Timedelta(days=1)
    out = pd.DataFrame({"gid": GID[is_te], "obs": DATA.loc[is_te, "target"].values,
                        "covered": DATA.loc[is_te, "fc_next1"].notna().values}, index=idx)
    out["pred_ens"] = np.clip(m.predict(Xte), 0, None).astype("float32")
    _, Xmx = test_frame(DATA, 1, spread, rain="mx")
    out["pred_memmax"] = np.clip(m.predict(Xmx), 0, None).astype("float32")
    out.to_parquet(dst)
    print(f"{variant} {q}: wrote {dst.name}", flush=True)


def amax_positions(base, cov):
    full = base[["gid", "obs"]].reset_index()
    full["wy"] = full.date.dt.year + (full.date.dt.month >= 10)
    nobs = full.groupby(["gid", "wy"]).obs.transform("size")
    am = full[nobs >= 350].groupby(["gid", "wy"]).obs.idxmax().values
    is_am = np.zeros(len(base), bool); is_am[am] = True
    return np.flatnonzero(is_am[cov])


if MODE.startswith("L"):
    run_point(int(MODE[1])); sys.exit(0)
if MODE.startswith("Q"):
    v, q = MODE[1:].rsplit("_", 1); run_quantile(v, q); sys.exit(0)

if MODE == "score":
    rows = []
    for L in (1, 2, 3):
        ref = pd.read_parquet(OUT / f"forecast_ar_gefs_ens_L{L}.parquet")
        cov = ref.covered.values
        am = amax_positions(ref, cov)
        base_nse = per_catchment(ref[cov]).nse
        for v in ["obs_full"] + VARIANTS:
            p = OUT / (f"forecast_ar_gefs_ens_L{L}.parquet" if v == "obs_full"
                       else f"forecast_fctrain_{v}_L{L}.parquet")
            if not p.exists():
                continue
            d = pd.read_parquet(p)
            nse = per_catchment(d[cov]).nse
            pk = d.pred.values[cov][am] / d.obs.values[cov][am] - 1
            rows.append({"lead": L, "variant": v, "median_NSE": nse.median(),
                         "dNSE_vs_obs_full": (nse - base_nse).median(),
                         "better_frac": (nse > base_nse).mean(),
                         "peak_day_bias_pct": np.median(pk) * 100})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    df.to_csv(OUT / "forecast_fctrain_cards.csv", index=False)

if MODE == "scoreq":
    rows = []
    for v in VARIANTS:
        Q = {}
        for q in ALPHAS:
            p = OUT / f"forecast_fq_{v}_{q}_L1.parquet"
            if p.exists():
                Q[q] = pd.read_parquet(p)
        if len(Q) < 4:
            continue
        base = Q["q99"]; cov = base.covered.values
        am = amax_positions(base, cov); obs = base.obs.values[cov]; oa = obs[am]
        for path in ("ens", "memmax"):
            for q, a in ALPHAS.items():
                p = Q[q][f"pred_{path}"].values[cov]
                rows.append({"variant": v, "rain": path, "q": q, "pooled": (obs <= p).mean(),
                             "amax_days": (oa <= p[am]).mean(),
                             "width_q99_q50": np.median(Q["q99"][f"pred_{path}"].values[cov]
                                                        - Q["q50"][f"pred_{path}"].values[cov])
                             if q == "q99" else np.nan})
    df = pd.DataFrame(rows)
    print(df.round(3).to_string(index=False))
    df.to_csv(OUT / "forecast_fctrain_ladder_L1.csv", index=False)
