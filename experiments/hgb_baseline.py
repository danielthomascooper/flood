"""Does a gradient-boosted tree hold up on long-horizon streamflow prediction?

One global HistGradientBoostingRegressor over many CAMELS-GB catchments, with
hand-engineered memory features (rolling rainfall windows) so the tree gets a
fair shot at the state-dependence an LSTM would learn.

Split is TEMPORAL: train on the early record, test on the later record. That is
the setting the question is about -- does it stay good years later?
"""
import time, sys
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path("/home/habrt/source/flood")
DAILY = ROOT/"data/Catchment_Timeseries/hydro-meteorological/daily"
ATTR  = ROOT/"data/Catchment_Attributes"
CACHE = ROOT/"cache"

TRAIN_END = "2010-09-30"      # 40 water years to train
TEST_START = "2010-10-01"     # 12 water years held out, strictly later

def read_attr(name):
    p = ATTR/f"camels_gb_v2_{name}_attributes.csv"
    try: return pd.read_csv(p, na_values=["NaN"])
    except pd.errors.ParserError:
        n = len(pd.read_csv(p, nrows=0).columns)
        return pd.read_csv(p, engine="python", na_values=["NaN"],
                           on_bad_lines=lambda f: f[:n-1]+[",".join(f[n-1:])])

# ---- static attributes. NOTHING derived from discharge: no q_mean, runoff_ratio,
# ---- baseflow_index, Q5/Q95 etc, or the model would be reading its own target.
top, cli, soil, hgeo, lc = (read_attr(x) for x in
    ["topographic","climatic","soil","hydrogeology","landcover"])
STATIC = (top[["gauge_id","area","dpsbar","elev_mean","elev_50","gauge_elev"]]
    .merge(cli[["gauge_id","p_mean","pet_mean","aridity","p_seasonality","frac_snow",
                "high_prec_freq","low_prec_freq"]], on="gauge_id")
    .merge(soil[["gauge_id","sand_perc","clay_perc","silt_perc","organic_perc",
                 "porosity_cosby","conductivity_cosby","soil_depth_pelletier","tawc"]], on="gauge_id")
    .merge(hgeo[["gauge_id","inter_high_perc","frac_high_perc","no_gw_perc"]], on="gauge_id")
    .merge(lc[["gauge_id","urban_perc_2015","dwood_perc_2015","grass_perc_2015",
               "crop_perc_2015","bares_perc_2015"]], on="gauge_id"))
STATIC["area"] = np.log10(STATIC["area"])
STATIC = STATIC.set_index("gauge_id")

hm = read_attr("hydrometry")
GOOD = hm.loc[hm.daily_flow_perc_complete >= 95, "gauge_id"].tolist()
print(f"catchments with >=95% complete daily flow: {len(GOOD)}")

def features(gid):
    f = sorted(DAILY.glob(f"*_{gid}_*.csv"))[0]
    d = pd.read_csv(f, parse_dates=["date"], na_values=["NaN"],
                    usecols=["date","precipitation_haduk","pet_hydrope",
                             "temperature_haduk","discharge_spec"]).set_index("date")
    p, e, t = d.precipitation_haduk, d.pet_hydrope, d.temperature_haduk
    X = {}
    for lag in range(1, 8):
        X[f"p_lag{lag}"] = p.shift(lag)
    X["p_0"] = p
    for w in (3, 7, 14, 30, 90, 180, 365):
        X[f"p_sum{w}"] = p.rolling(w, min_periods=max(1, w//2)).sum()
    for w in (7, 30, 90):
        X[f"pet_mean{w}"] = e.rolling(w, min_periods=max(1, w//2)).mean()
    X["t_0"] = t
    X["t_mean30"] = t.rolling(30, min_periods=15).mean()
    doy = d.index.dayofyear.values
    X["doy_sin"] = np.sin(2*np.pi*doy/365.25)
    X["doy_cos"] = np.cos(2*np.pi*doy/365.25)
    X = pd.DataFrame(X, index=d.index)
    for c, v in STATIC.loc[gid].items():
        X[c] = v
    X["y"] = d.discharge_spec
    return X.astype("float32")

t0 = time.time()
frames, ids = [], []
for i, gid in enumerate(GOOD):
    X = features(gid)
    X = X.dropna(subset=["y"])
    frames.append(X); ids.append(np.full(len(X), gid, dtype=np.int32))
    if (i+1) % 100 == 0: print(f"  {i+1}/{len(GOOD)} ({time.time()-t0:.0f}s)")
DATA = pd.concat(frames); GID = np.concatenate(ids)
del frames
print(f"built {DATA.shape[0]:,} rows x {DATA.shape[1]-1} features in {time.time()-t0:.0f}s")

dates = DATA.index
tr = dates <= TRAIN_END
te = dates >= TEST_START
FEATS = [c for c in DATA.columns if c != "y"]
Xtr, ytr = DATA.loc[tr, FEATS], DATA.loc[tr, "y"]
Xte, yte = DATA.loc[te, FEATS], DATA.loc[te, "y"]
gid_tr, gid_te = GID[np.asarray(tr)], GID[np.asarray(te)]
print(f"train {len(ytr):,} rows ({dates[tr].min().date()} to {dates[tr].max().date()})")
print(f"test  {len(yte):,} rows ({dates[te].min().date()} to {dates[te].max().date()})")

t0 = time.time()
model = HistGradientBoostingRegressor(
    max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
    min_samples_leaf=100, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.1, random_state=0)
model.fit(Xtr, ytr)
print(f"fitted in {time.time()-t0:.0f}s, {model.n_iter_} iterations")

pred = model.predict(Xte).astype("float32")
np.save("/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/pred.npy", pred)
res = pd.DataFrame({"gid": gid_te, "obs": yte.values, "pred": pred}, index=dates[te])
res.to_parquet("/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/hgb_test.parquet")

# training-period maximum per catchment -- the tree's structural ceiling
trmax = pd.Series(ytr.values, index=gid_tr).groupby(level=0).max()
trmax.to_frame("train_max").to_parquet("/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/trainmax.parquet")
print("saved predictions")
