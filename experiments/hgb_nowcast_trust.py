"""Donor-trust features for ungauged nowcasting — Phase 3 C3.

hgb_nowcast_similar.py halved the ungauged-chalk penalty by filtering
donors on geology, but Law Brook and Mimram stayed worse than having no
donors: a hard filter cannot teach the model *how much* to believe a donor.
This keeps the similarity-filtered donor assignment and adds, per donor,
three gauge-free dissimilarity features — |frac_high_perc gap|, |log10-area
gap|, distance km — so the tree can learn to discount a donor's flow where
the dissimilarity is large, per catchment and per split of the trees.

Same 5 rotated ungauged folds; paired against raw / nearest / similar from
results/nowcast_similar_per_catchment.csv.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STATIC, build_dataset, good_catchments, read_attr, \
    TRAIN_END, TEST_START
from nowcast_common import USABLE, K, nearest_donors, donor_features
from evaluate import per_catchment

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
SIM_TOL = 20.0

hg = read_attr("hydrogeology").set_index("gauge_id")
fhp = pd.to_numeric(hg.frac_high_perc, errors="coerce")
topo = read_attr("topographic").set_index("gauge_id")
larea = np.log10(topo.area.astype(float))


def similar_donors(gids):
    """Same rule as hgb_nowcast_similar.py: donors within SIM_TOL of the
    target's frac_high_perc, nearest K among them; plain nearest fallback."""
    out = {}
    fallback = nearest_donors(gids)
    donors_fhp = fhp.reindex(USABLE)
    for g in gids:
        tgt = fhp.get(g, np.nan)
        if np.isnan(tgt):
            out[g] = fallback[g]
            continue
        pool = USABLE[(donors_fhp - tgt).abs().values <= SIM_TOL]
        pool = pool[pool != g]
        out[g] = fallback[g] if len(pool) < K else nearest_donors([g], pool=pool)[g]
    return out


def trust_features(index, gid, assignment):
    """3 gauge-free dissimilarity columns per donor, constant per catchment."""
    cols = {f"nb{r}_{n}": np.full(len(index), np.nan, dtype="float32")
            for r in range(K) for n in ("dfhp", "darea", "dist")}
    for g in np.unique(gid):
        donors, dists = assignment[g]
        rows = np.flatnonzero(gid == g)
        for r, (d, km) in enumerate(zip(donors, dists)):
            cols[f"nb{r}_dfhp"][rows] = abs(fhp.get(g, np.nan) - fhp.get(d, np.nan))
            cols[f"nb{r}_darea"][rows] = abs(larea.get(g, np.nan) - larea.get(d, np.nan))
            cols[f"nb{r}_dist"][rows] = km
    return pd.DataFrame(cols, index=index)


print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
gids = np.unique(GID)
print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)

assign = similar_donors(gids)
NB = pd.concat([donor_features(DATA.index, GID, assign),
                trust_features(DATA.index, GID, assign)], axis=1)
print(f"  {NB.shape[1]} donor+trust columns", flush=True)

by_area = sorted(good_catchments(), key=lambda g: STATIC.loc[g, "area"])
dates = DATA.index
feats = [c for c in DATA.columns if c != "y"]
is_tr = np.asarray(dates <= TRAIN_END)
is_te = np.asarray(dates >= TEST_START)

pcs = []
for off in range(5):
    held = set(by_area[off::5])
    in_held = np.isin(GID, list(held))
    tr, te = is_tr & ~in_held, is_te & in_held
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE)
    m.fit(pd.concat([DATA.loc[tr, feats], NB.loc[tr]], axis=1), DATA.loc[tr, "y"])
    res = pd.DataFrame({"gid": GID[te], "obs": DATA.loc[te, "y"].values,
                        "pred": np.clip(
                            m.predict(pd.concat([DATA.loc[te, feats], NB.loc[te]],
                                                axis=1)), 0, None).astype("float32")},
                       index=DATA.index[te])
    res.to_parquet(SCRATCH / f"nowcast_trust_fold{off}.parquet")
    pcs.append(per_catchment(res))
    print(f"  fold {off}: fitted in {time.time()-t0:.0f}s", flush=True)

ung = pd.concat(pcs)
prev = pd.read_csv(OUT / "nowcast_similar_per_catchment.csv", index_col=0)
sp = prev.assign(nse_ung_trust=ung.nse)
sp["pen_trust"] = sp.nse_gauged - sp.nse_ung_trust
sp.to_csv(OUT / "nowcast_trust_per_catchment.csv")

print("\n=== ungauged penalty: raw | nearest | similar | +trust (median) ===")
for name, m in [("all", sp.index == sp.index), ("chalk", sp.chalk.astype(bool)),
                ("non_chalk", ~sp.chalk.astype(bool))]:
    t = sp[m]
    print(f"  {name:10s} n={len(t):3d}  {t.pen_raw.median():+.3f} | "
          f"{t.pen_now.median():+.3f} | {t.pen_sim.median():+.3f} | "
          f"{t.pen_trust.median():+.3f}   failures {(t.nse_ung_raw<0).sum()} | "
          f"{(t.nse_ung_now<0).sum()} | {(t.nse_ung_sim<0).sum()} | "
          f"{(t.nse_ung_trust<0).sum()}")
print("\nworst 8 raw-ungauged, all four:")
cols = ["gauge_name", "nse_ung_raw", "nse_ung_now", "nse_ung_sim", "nse_ung_trust"]
print(sp.nsmallest(8, "nse_ung_raw")[cols].round(3).to_string())
print(f"\nwrote {OUT/'nowcast_trust_per_catchment.csv'}")
