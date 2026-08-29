"""Similarity-restricted donors for ungauged chalk — Phase 3 C2.

hgb_nowcast.py rescued the median ungauged site but left (or worsened) the
groundwater-dominated ones: a chalk stream's nearest gauges are often flashy
surface-water catchments whose flood wave actively misleads. Hypothesis: for
an ungauged site, donors should be hydrogeologically similar first and near
second. Similarity uses only attributes computable without a gauge
(frac_high_perc from hydrogeology maps), so the setting stays honestly
ungauged; the donor's own observed flow is fair game (donors are gauged).

Rule: donor pool = usable gauges with |frac_high_perc − target's| <= 20
(fall back to the plain nearest-k when fewer than k qualify); take the k
nearest within the pool. Rerun the 5 rotated ungauged folds; paired against
both the raw ungauged tree and the nearest-donor nowcast.
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


def similar_donors(gids):
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
        if len(pool) < K:
            out[g] = fallback[g]
        else:
            out[g] = nearest_donors([g], pool=pool)[g]
    return out


print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
gids = np.unique(GID)
print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)

assign = similar_donors(gids)
plain = nearest_donors(gids)
changed = sum(assign[g][0] != plain[g][0] for g in gids)
d_sim = np.array([d for g in gids for d in assign[g][1]])
print(f"  donor sets changed for {changed}/{len(gids)} catchments; "
      f"similar-donor distances: median {np.median(d_sim):.1f} km, "
      f"p90 {np.quantile(d_sim, .9):.1f} km", flush=True)

NB = donor_features(DATA.index, GID, assign)
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
    res.to_parquet(SCRATCH / f"nowcast_sim_fold{off}.parquet")
    pcs.append(per_catchment(res))
    print(f"  fold {off}: fitted in {time.time()-t0:.0f}s", flush=True)

ung = pd.concat(pcs)
prev = pd.read_csv(OUT / "nowcast_spatial_per_catchment.csv", index_col=0)
sp = prev.assign(nse_ung_sim=ung.nse)
sp["pen_sim"] = sp.nse_gauged - sp.nse_ung_sim
sp.to_csv(OUT / "nowcast_similar_per_catchment.csv")

print("\n=== ungauged penalty: raw vs nearest-donor vs similar-donor (median) ===")
for name, m in [("all", sp.index == sp.index), ("chalk", sp.chalk.astype(bool)),
                ("non_chalk", ~sp.chalk.astype(bool))]:
    t = sp[m]
    print(f"  {name:10s} n={len(t):3d}  raw {t.pen_raw.median():+.3f} | "
          f"nearest {t.pen_now.median():+.3f} | similar {t.pen_sim.median():+.3f}"
          f"   failures {(t.nse_ung_raw<0).sum()} | {(t.nse_ung_now<0).sum()} | "
          f"{(t.nse_ung_sim<0).sum()}")
print("\nworst 8 raw-ungauged, all three:")
cols = ["gauge_name", "nse_ung_raw", "nse_ung_now", "nse_ung_sim"]
print(sp.nsmallest(8, "nse_ung_raw")[cols].round(3).to_string())
print(f"\nwrote {OUT/'nowcast_similar_per_catchment.csv'}")
