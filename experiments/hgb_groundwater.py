"""Does an observed groundwater level fix the chalk catchments?

The ungauged split (hgb_spatial.py) failed worst on small chalk catchments,
and even the gauged tree cannot see the aquifer state that drives them: the
365-day precipitation window is a poor proxy for a Chalk water table that
integrates several years of recharge. CAMELS-GB v2 ships 55 observation
wells. This experiment feeds the nearest usable well to each catchment as
two extra features and asks whether it helps where it should (chalk) and
nowhere else.

Features added to the standard set from common.py, both lagged one day so
the model only sees a level that was already observed:

  gw_z       per-well z-score of the level, standardised with the
             TRAIN-window mean/std only (no test leakage)
  gw_z_d90   90-day change in gw_z (is the aquifer filling or draining)

Matching: nearest well with >=50% coverage in both train and test windows,
within 40 km; catchments without one get NaN, which HistGradientBoosting
handles natively, so one model covers all 416 catchments.

Three fits on the identical temporal split:

  raw       no-GW control -- reuses targets_raw.parquet from hgb_targets.py
            when present (identical model, split, seed), else refits
  gw        real well assignment
  shuffled  well assignments permuted among the matched catchments: any gain
            that survives shuffling is the model exploiting a well as a
            regional wetness index, not the catchment's own aquifer

Success = chalk improves, non-chalk flat, shuffled flat. Reported as paired
per-catchment NSE deltas vs raw on: all / chalk / chalk with a matched well /
non-chalk / the five named spatial-split failures.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STATIC, build_dataset, temporal_split, read_attr, TRAIN_END
from evaluate import evaluate, per_catchment, report

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT
CACHE = ROOT / "cache"

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
MAX_KM, MIN_COV, LAG = 40.0, 0.5, 1
FAILURES = ["Lea Brook at Beck Bridge", "Colne at Denham", "Heacham at Heacham",
            "Tilling Bourne at Shalford", "Granta at Stapleford"]

# ---- wells -----------------------------------------------------------------

levels = pd.read_parquet(CACHE / "gw_levels_daily.parquet")       # date x well
wells = read_attr("groundwaterwell").set_index("gw_well_id")
train_lv = levels.loc[:TRAIN_END]
test_lv = levels.loc[pd.Timestamp(TRAIN_END) + pd.Timedelta(days=1):]
cov_tr, cov_te = train_lv.notna().mean(), test_lv.notna().mean()
usable = levels.columns[(cov_tr >= MIN_COV) & (cov_te >= MIN_COV)]
Z = (levels[usable] - train_lv[usable].mean()) / train_lv[usable].std()
Z_D90 = Z - Z.shift(90)
Z, Z_D90 = Z.shift(LAG), Z_D90.shift(LAG)
print(f"{len(usable)} usable wells of {levels.shape[1]} "
      f"(train & test coverage >= {MIN_COV})", flush=True)

# ---- catchment -> well matching --------------------------------------------

topo = read_attr("topographic").set_index("gauge_id")
gx, gy = topo.gauge_easting.astype(float), topo.gauge_northing.astype(float)
wx = wells.loc[usable, "gw_well_easting"].astype(float).values
wy = wells.loc[usable, "gw_well_northing"].astype(float).values


def match_wells(gids):
    D = np.sqrt((gx.loc[gids].values[:, None] - wx[None, :]) ** 2
                + (gy.loc[gids].values[:, None] - wy[None, :]) ** 2) / 1000.0
    j = D.argmin(axis=1)
    d = D[np.arange(len(gids)), j]
    m = pd.DataFrame({"well_id": usable[j], "dist_km": d}, index=pd.Index(gids, name="gid"))
    m.loc[m.dist_km > MAX_KM, "well_id"] = None
    m["aquifer"] = wells.aquifer.reindex(m.well_id).values
    return m


def gw_features(index, gid, assignment):
    """Two GW columns aligned to (date index, gid array) for a well assignment."""
    z = np.full(len(index), np.nan, dtype="float32")
    dz = np.full(len(index), np.nan, dtype="float32")
    for g, w in assignment.well_id.dropna().items():
        rows = np.flatnonzero(gid == g)
        z[rows] = Z[w].reindex(index[rows]).values
        dz[rows] = Z_D90[w].reindex(index[rows]).values
    return pd.DataFrame({"gw_z": z, "gw_z_d90": dz}, index=index)


# ---- data ------------------------------------------------------------------

print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")
gids = np.unique(GID)
print(f"  {len(ytr):,} train / {len(yte):,} test rows, {len(gids)} catchments "
      f"in {time.time()-t0:.0f}s", flush=True)

real = match_wells(gids)
matched = real.well_id.notna()
rng = np.random.default_rng(0)
shuffled = real.copy()
shuffled.loc[matched, ["well_id", "dist_km", "aquifer"]] = \
    real.loc[matched, ["well_id", "dist_km", "aquifer"]].sample(frac=1, random_state=0).values
print(f"  {matched.sum()} catchments matched to a usable well within {MAX_KM:.0f} km "
      f"(median {real.dist_km[matched].median():.1f} km)", flush=True)

# ---- subsets ---------------------------------------------------------------

chalk = STATIC.frac_high_perc.reindex(gids) >= 50
fail_ids = topo.index[topo.gauge_name.str.lower().isin([n.lower() for n in FAILURES])]
SUBSETS = {
    "all":            pd.Series(True, index=gids),
    "chalk":          chalk,
    "chalk_matched":  chalk & matched,
    "non_chalk":      ~chalk,
    "failures":       pd.Series(np.isin(gids, fail_ids), index=gids),
}
print("  subsets: " + ", ".join(f"{k}={v.sum()}" for k, v in SUBSETS.items()), flush=True)

# ---- fits ------------------------------------------------------------------

def fit(name, assignment):
    t0 = time.time()
    Xa = pd.concat([Xtr, gw_features(Xtr.index, gtr, assignment)], axis=1)
    Xb = pd.concat([Xte, gw_features(Xte.index, gte, assignment)], axis=1)
    m = HistGradientBoostingRegressor(**BASE).fit(Xa, ytr)
    res = pd.DataFrame({"gid": gte, "obs": yte.values,
                        "pred": np.clip(m.predict(Xb), 0, None).astype("float32")},
                       index=Xte.index)
    res.to_parquet(SCRATCH / f"gw_{name}.parquet")
    print(f"  {name}: fitted in {time.time()-t0:.0f}s "
          f"({m.n_iter_} iterations)", flush=True)
    return res


reuse = SCRATCH / "targets_raw.parquet"
if reuse.exists():
    preds = {"raw": pd.read_parquet(reuse)}
    print("  raw: reused from hgb_targets run", flush=True)
else:
    preds = {"raw": fit("raw", real.assign(well_id=None))}
preds["gw"] = fit("gw", real)
preds["shuffled"] = fit("shuffled", shuffled)

# ---- evaluation ------------------------------------------------------------

pcs = {k: per_catchment(v) for k, v in preds.items()}
paired = pd.DataFrame({f"nse_{k}": pc.nse for k, pc in pcs.items()})
paired["d_gw"] = paired.nse_gw - paired.nse_raw
paired["d_shuffled"] = paired.nse_shuffled - paired.nse_raw
paired = paired.join(real[["well_id", "dist_km", "aquifer"]]).join(
    STATIC[["area", "frac_high_perc"]]).join(topo.gauge_name)
for k, v in SUBSETS.items():
    paired[k] = v.reindex(paired.index).values
paired.to_csv(OUT / "groundwater_paired.csv")

rows, cards = [], []
for sub, mask in SUBSETS.items():
    ids = mask[mask].index
    p = paired.loc[ids]
    rows.append({
        "subset": sub, "n": len(p),
        "raw_median_NSE": p.nse_raw.median(),
        "gw_median_NSE": p.nse_gw.median(),
        "shuffled_median_NSE": p.nse_shuffled.median(),
        "d_gw_median": p.d_gw.median(),
        "d_gw_mean": p.d_gw.mean(),
        "gw_frac_improved": (p.d_gw > 0).mean(),
        "d_shuffled_median": p.d_shuffled.median(),
        "shuffled_frac_improved": (p.d_shuffled > 0).mean(),
    })
    for k, res in preds.items():
        row, _ = evaluate(res[res.gid.isin(ids)], f"{sub}/{k}", train_max=train_max)
        cards.append(row)

summary = pd.DataFrame(rows).set_index("subset")
print("\n=== paired per-catchment NSE deltas vs raw ===")
with pd.option_context("display.width", 200, "display.max_columns", 30,
                       "display.float_format", lambda v: f"{v:+.3f}"):
    print(summary.to_string())
summary.to_csv(OUT / "groundwater_summary.csv")

print("\n=== standard card by subset ===")
report(cards).to_csv(OUT / "groundwater_cards.csv")

print("\n=== the five spatial-split failures ===")
cols = ["gauge_name", "well_id", "dist_km", "nse_raw", "nse_gw", "nse_shuffled", "d_gw"]
with pd.option_context("display.width", 200, "display.float_format", lambda v: f"{v:+.3f}"):
    print(paired.loc[SUBSETS["failures"][SUBSETS["failures"]].index, cols].to_string())
print(f"\nwrote {OUT/'groundwater_summary.csv'}, groundwater_cards.csv, groundwater_paired.csv")
