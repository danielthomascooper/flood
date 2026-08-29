"""Neighbour-gauge nowcasting — the Gate 2 build.

Gate 2 closed on "statistic + information": calibrated q99 envelopes from
two model classes miss the same ~15% of AMAX days, so on those days the
daily forcings do not flag the event. But the flood wave itself is observed
in real time — by the neighbouring gauges. This feeds each catchment the
same-day and 1-day-lagged observed flow at its k=3 nearest usable gauges
(each scaled by that donor's own train-window q95, so the feature reads
"fraction of donor's high flow"), which is operationally legitimate
nowcasting: every donor value is a measurement available at prediction time.

Two settings, one dataset build:

  temporal   the standard gauged split. Control 1 = the persisted raw tree
             (results/tree_test_predictions.parquet). Control 2 = a
             shuffled-donor fit (donor sets permuted among catchments):
             any gain that survives shuffling is regional wetness, not the
             local flood wave.
  spatial    the 5 rotated area-stratified folds from hgb_spatial_folds.py,
             now with donor features. For an ungauged site the neighbours
             are exactly what exists (the site itself has no gauge), so
             this is the operationally honest ungauged model. Paired
             against results/spatial_folds_per_catchment.csv.

Success criteria: AMAX bias and top-1% NSE move in the temporal setting
where the shuffled control does not; the ungauged chalk penalty (+0.097)
shrinks in the spatial setting.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, STATIC, build_dataset, temporal_split, read_attr, \
    good_catchments, TRAIN_END, TEST_START
from evaluate import evaluate, per_catchment, amax_bias, report

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
K, MIN_COV = 3, 0.5

# ---- donor pool ------------------------------------------------------------

flows = pd.read_parquet(ROOT / "cache" / "daily_discharge_spec.parquet")
ftr = flows.loc[:TRAIN_END]
usable = flows.columns[(ftr.notna().mean() >= MIN_COV)
                       & (flows.loc[TEST_START:].notna().mean() >= MIN_COV)]
q95 = ftr[usable].quantile(0.95)
D0 = (flows[usable] / q95).astype("float32")       # same-day donor index
D1 = D0.shift(1)                                   # yesterday's
print(f"{len(usable)} usable donor gauges of {flows.shape[1]}", flush=True)

topo = read_attr("topographic").set_index("gauge_id")
ex, ny = topo.gauge_easting.astype(float), topo.gauge_northing.astype(float)


def nearest_donors(gids):
    """k nearest usable donor gauges per catchment (never itself)."""
    dx = ex.loc[gids].values[:, None] - ex.loc[usable].values[None, :]
    dy = ny.loc[gids].values[:, None] - ny.loc[usable].values[None, :]
    Dk = np.sqrt(dx ** 2 + dy ** 2) / 1000.0
    self_col = {g: j for j, g in enumerate(usable)}
    out = {}
    for i, g in enumerate(gids):
        row = Dk[i].copy()
        if g in self_col:
            row[self_col[g]] = np.inf
        j = np.argsort(row)[:K]
        out[g] = (list(usable[j]), row[j].tolist())
    return out


def donor_features(index, gid, assignment):
    """2K donor columns aligned to (date index, gid array)."""
    cols = {f"nb{r}_{lag}": np.full(len(index), np.nan, dtype="float32")
            for r in range(K) for lag in ("d0", "d1")}
    for g in np.unique(gid):
        donors, _ = assignment[g]
        rows = np.flatnonzero(gid == g)
        dts = index[rows]
        for r, d in enumerate(donors):
            cols[f"nb{r}_d0"][rows] = D0[d].reindex(dts).values
            cols[f"nb{r}_d1"][rows] = D1[d].reindex(dts).values
    return pd.DataFrame(cols, index=index)


# ---- data ------------------------------------------------------------------

print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")
gids = np.unique(GID)
print(f"  {len(ytr):,} train / {len(yte):,} test rows in {time.time()-t0:.0f}s", flush=True)

real = nearest_donors(gids)
dists = np.array([d for g in gids for d in real[g][1]])
print(f"  donor distances: median {np.median(dists):.1f} km, "
      f"p90 {np.quantile(dists, .9):.1f} km", flush=True)
rng = np.random.default_rng(0)
perm = rng.permutation(gids)
shuffled = {g: real[p] for g, p in zip(gids, perm)}


def fit(name, Xa, ya, Xb, gb, yb):
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE).fit(Xa, ya)
    res = pd.DataFrame({"gid": gb, "obs": yb,
                        "pred": np.clip(m.predict(Xb), 0, None).astype("float32")},
                       index=Xb.index)
    print(f"  {name}: fitted in {time.time()-t0:.0f}s", flush=True)
    return res

# ---- temporal setting ------------------------------------------------------

print("temporal setting...", flush=True)
preds = {"raw": pd.read_parquet(OUT / "tree_test_predictions.parquet")}
for name, assign in [("nowcast", real), ("shuffled", shuffled)]:
    Xa = pd.concat([Xtr, donor_features(Xtr.index, gtr, assign)], axis=1)
    Xb = pd.concat([Xte, donor_features(Xte.index, gte, assign)], axis=1)
    preds[name] = fit(name, Xa, ytr.values, Xb, gte, yte.values)
    del Xa, Xb
preds["nowcast"].to_parquet(OUT / "nowcast_test_predictions.parquet")

rows = [evaluate(r, k, train_max=train_max)[0] for k, r in preds.items()]
print("\n=== temporal split ===")
report(rows).to_csv(OUT / "nowcast_cards.csv")

pcs = {k: per_catchment(v) for k, v in preds.items()}
ambs = {k: amax_bias(v).groupby("gid").median() for k, v in preds.items()}
paired = pd.DataFrame({
    "nse_raw": pcs["raw"].nse, "nse_now": pcs["nowcast"].nse,
    "nse_shuf": pcs["shuffled"].nse,
    "top1_raw": pcs["raw"].top1_nse, "top1_now": pcs["nowcast"].top1_nse,
    "amax_raw": ambs["raw"], "amax_now": ambs["nowcast"],
    "amax_shuf": ambs["shuffled"],
}).join(STATIC.frac_high_perc).join(topo.gauge_name)
paired.to_csv(OUT / "nowcast_paired.csv")
print("\npaired medians: dNSE now "
      f"{(paired.nse_now-paired.nse_raw).median():+.3f} / shuf "
      f"{(paired.nse_shuf-paired.nse_raw).median():+.3f}; "
      f"dTop1NSE now {(paired.top1_now-paired.top1_raw).median():+.3f}; "
      f"AMAX bias raw {paired.amax_raw.median():+.1f}% -> now "
      f"{paired.amax_now.median():+.1f}% (shuf {paired.amax_shuf.median():+.1f}%)",
      flush=True)

# ---- spatial (ungauged) setting --------------------------------------------

print("\nspatial setting (5 rotated folds, donor features)...", flush=True)
by_area = sorted(good_catchments(), key=lambda g: STATIC.loc[g, "area"])
dates = DATA.index
feats = [c for c in DATA.columns if c != "y"]
is_tr = np.asarray(dates <= TRAIN_END)
is_te = np.asarray(dates >= TEST_START)
NB_ALL = donor_features(DATA.index, GID, real)
pcs_sp = []
for off in range(5):
    held = set(by_area[off::5])
    in_held = np.isin(GID, list(held))
    tr = is_tr & ~in_held
    te = is_te & in_held
    Xa = pd.concat([DATA.loc[tr, feats], NB_ALL.loc[tr]], axis=1)
    Xb = pd.concat([DATA.loc[te, feats], NB_ALL.loc[te]], axis=1)
    res = fit(f"fold{off}", Xa, DATA.loc[tr, "y"].values, Xb, GID[te],
              DATA.loc[te, "y"].values)
    del Xa, Xb
    res.to_parquet(SCRATCH / f"nowcast_fold{off}.parquet")
    pcs_sp.append(per_catchment(res))
ung = pd.concat(pcs_sp)

base_sp = pd.read_csv(OUT / "spatial_folds_per_catchment.csv", index_col=0)
sp = pd.DataFrame({"nse_gauged": base_sp.nse_gauged,
                   "nse_ung_raw": base_sp.nse_ungauged,
                   "nse_ung_now": ung.nse,
                   "chalk": base_sp.chalk, "gauge_name": base_sp.gauge_name})
sp["pen_raw"] = sp.nse_gauged - sp.nse_ung_raw
sp["pen_now"] = sp.nse_gauged - sp.nse_ung_now
sp.to_csv(OUT / "nowcast_spatial_per_catchment.csv")

print("\n=== ungauged penalty, raw vs nowcast (median) ===")
for name, m in [("all", sp.index == sp.index), ("chalk", sp.chalk.astype(bool)),
                ("non_chalk", ~sp.chalk.astype(bool))]:
    t = sp[m]
    print(f"  {name:10s} n={len(t):3d}  raw {t.pen_raw.median():+.3f} -> "
          f"nowcast {t.pen_now.median():+.3f}   ungauged failures "
          f"{(t.nse_ung_raw<0).sum()} -> {(t.nse_ung_now<0).sum()}")
print("\nworst 8 raw-ungauged catchments, with nowcast:")
cols = ["gauge_name", "nse_ung_raw", "nse_ung_now", "chalk"]
print(sp.nsmallest(8, "nse_ung_raw")[cols].round(3).to_string())
print(f"\nwrote nowcast_cards.csv, nowcast_paired.csv, "
      f"nowcast_spatial_per_catchment.csv, nowcast_test_predictions.parquet")
