"""Shared data pipeline for CAMELS-GB v2 model experiments.

One place for the feature construction so every experiment trains on
identical inputs. Uses the HadUK-Grid / Hydro-PE forcing columns throughout:
the CEH-GEAR / CHESS columns end 2019-12-31 in the daily files and are NaN
afterwards (see README, quirk 5).
"""
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DAILY = ROOT / "data/Catchment_Timeseries/hydro-meteorological/daily"
ATTR = ROOT / "data/Catchment_Attributes"

TRAIN_END = "2010-09-30"   # 40 water years to train
TEST_START = "2010-10-01"  # 12 water years held out, strictly later


def read_attr(name):
    """Read an attribute table, repairing the two ragged hydrometry rows."""
    p = ATTR / f"camels_gb_v2_{name}_attributes.csv"
    try:
        return pd.read_csv(p, na_values=["NaN"])
    except pd.errors.ParserError:
        n = len(pd.read_csv(p, nrows=0).columns)
        return pd.read_csv(p, engine="python", na_values=["NaN"],
                           on_bad_lines=lambda f: f[:n - 1] + [",".join(f[n - 1:])])


def static_table():
    """23 static attributes. NOTHING derived from discharge: no q_mean,
    runoff_ratio, baseflow_index, Q5/Q95 etc — those are computed from the
    target and including them is leakage."""
    top, cli, soil, hgeo, lc = (read_attr(x) for x in
        ["topographic", "climatic", "soil", "hydrogeology", "landcover"])
    static = (top[["gauge_id", "area", "dpsbar", "elev_mean", "elev_50", "gauge_elev"]]
        .merge(cli[["gauge_id", "p_mean", "pet_mean", "aridity", "p_seasonality",
                    "frac_snow", "high_prec_freq", "low_prec_freq"]], on="gauge_id")
        .merge(soil[["gauge_id", "sand_perc", "clay_perc", "silt_perc", "organic_perc",
                     "porosity_cosby", "conductivity_cosby", "soil_depth_pelletier",
                     "tawc"]], on="gauge_id")
        .merge(hgeo[["gauge_id", "inter_high_perc", "frac_high_perc", "no_gw_perc"]],
               on="gauge_id")
        .merge(lc[["gauge_id", "urban_perc_2015", "dwood_perc_2015", "grass_perc_2015",
                   "crop_perc_2015", "bares_perc_2015"]], on="gauge_id"))
    static["area"] = np.log10(static["area"])
    return static.set_index("gauge_id")


STATIC = static_table()


def good_catchments(min_complete=95):
    hm = read_attr("hydrometry")
    return hm.loc[hm.daily_flow_perc_complete >= min_complete, "gauge_id"].tolist()


def features(gid):
    """Dynamic features for one catchment. The lags and rolling windows are a
    hand-built stand-in for catchment storage (see hgb_ablation.py for what
    they are worth: 0.47 median NSE)."""
    f = sorted(DAILY.glob(f"*_{gid}_*.csv"))[0]
    d = pd.read_csv(f, parse_dates=["date"], na_values=["NaN"],
                    usecols=["date", "precipitation_haduk", "pet_hydrope",
                             "temperature_haduk", "discharge_spec"]).set_index("date")
    p, e, t = d.precipitation_haduk, d.pet_hydrope, d.temperature_haduk
    X = {}
    for lag in range(1, 8):
        X[f"p_lag{lag}"] = p.shift(lag)
    X["p_0"] = p
    for w in (3, 7, 14, 30, 90, 180, 365):
        X[f"p_sum{w}"] = p.rolling(w, min_periods=max(1, w // 2)).sum()
    for w in (7, 30, 90):
        X[f"pet_mean{w}"] = e.rolling(w, min_periods=max(1, w // 2)).mean()
    X["t_0"] = t
    X["t_mean30"] = t.rolling(30, min_periods=15).mean()
    doy = d.index.dayofyear.values
    X["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    X["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    X = pd.DataFrame(X, index=d.index)
    for c, v in STATIC.loc[gid].items():
        X[c] = v
    X["y"] = d.discharge_spec
    return X.astype("float32")


def build_dataset(gauges=None):
    """Stack all catchments. Returns (DATA, GID): DATA has feature columns
    plus 'y', indexed by date; GID is a parallel int array of gauge ids."""
    gauges = gauges if gauges is not None else good_catchments()
    frames, ids = [], []
    for gid in gauges:
        X = features(gid).dropna(subset=["y"])
        frames.append(X)
        ids.append(np.full(len(X), gid, dtype=np.int32))
    return pd.concat(frames), np.concatenate(ids)


def temporal_split(DATA, GID):
    """Train on the early record, test on the later record, all catchments."""
    dates = DATA.index
    tr = np.asarray(dates <= TRAIN_END)
    te = np.asarray(dates >= TEST_START)
    feats = [c for c in DATA.columns if c != "y"]
    return (DATA.loc[tr, feats], DATA.loc[tr, "y"], GID[tr],
            DATA.loc[te, feats], DATA.loc[te, "y"], GID[te])
