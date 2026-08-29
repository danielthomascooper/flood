"""Ungauged penalty as a distribution — Phase 2 C4.

The committed 0.026-NSE ungauged penalty came from one deterministic fold
(83 catchments, offset 2 of an area-ordered 1-in-5). The audit's point: the
same card shows failures quintupling, and a single fold cannot say which
catchments are genuinely hard ungauged. Rotating the offset 0-4 gives every
one of the 416 catchments exactly one held-out score, paired against its
gauged score from the temporal split (identical test window).

Five raw-target fits (~6 min each). Writes per-catchment paired penalties
and a summary; prediction parquets to the scratch dir.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STATIC, build_dataset, good_catchments, TRAIN_END, TEST_START
from evaluate import per_catchment

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

GOOD = good_catchments()
by_area = sorted(GOOD, key=lambda g: STATIC.loc[g, "area"])

print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
dates = DATA.index
feats = [c for c in DATA.columns if c != "y"]
is_train_win = np.asarray(dates <= TRAIN_END)
is_test_win = np.asarray(dates >= TEST_START)
print(f"  {len(DATA):,} rows in {time.time()-t0:.0f}s", flush=True)

pcs = []
for off in range(5):
    held = set(by_area[off::5])
    in_held = np.isin(GID, list(held))
    tr = is_train_win & ~in_held
    te = is_test_win & in_held
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE)
    m.fit(DATA.loc[tr, feats], DATA.loc[tr, "y"])
    res = pd.DataFrame({"gid": GID[te], "obs": DATA.loc[te, "y"].values,
                        "pred": np.clip(m.predict(DATA.loc[te, feats]), 0, None)
                                  .astype("float32")},
                       index=DATA.index[te])
    res.to_parquet(SCRATCH / f"spatial_fold{off}.parquet")
    pc = per_catchment(res)
    pc["fold"] = off
    pcs.append(pc)
    print(f"  fold {off}: {len(held)} held out, fitted in {time.time()-t0:.0f}s "
          f"(median ungauged NSE {pc.nse.median():+.3f})", flush=True)

ung = pd.concat(pcs)                      # one row per catchment (folds disjoint)
base = pd.read_csv(OUT / "groundwater_paired.csv", index_col=0)
tab = pd.DataFrame({"nse_gauged": base.nse_raw, "nse_ungauged": ung.nse,
                    "top1_nse_ungauged": ung.top1_nse, "fold": ung.fold,
                    "chalk": base.chalk, "gauge_name": base.gauge_name})
tab["penalty"] = tab.nse_gauged - tab.nse_ungauged
tab["area"] = STATIC.area.reindex(tab.index)          # log10 km2
tab.to_csv(OUT / "spatial_folds_per_catchment.csv")

print("\n=== ungauged penalty, all 416 catchments (5 rotated folds) ===")
d = tab.penalty
print(f"median {d.median():+.3f}   mean {d.mean():+.3f}   "
      f"q10 {d.quantile(.1):+.3f}   q90 {d.quantile(.9):+.3f}   "
      f"worst {d.max():+.3f}")
print(f"failures (ungauged NSE<0): {(tab.nse_ungauged<0).sum()} "
      f"vs gauged {(tab.nse_gauged<0).sum()}")
print(f"per-fold median penalty: "
      + ", ".join(f"{g.penalty.median():+.3f}" for _, g in tab.groupby("fold")))
print("\nby subset (median penalty):")
small = tab.area <= tab.area.quantile(0.25)
for name, m in [("chalk", tab.chalk.astype(bool)), ("non_chalk", ~tab.chalk.astype(bool)),
                ("small (area Q1)", small), ("small & chalk", small & tab.chalk.astype(bool))]:
    t = tab[m]
    print(f"  {name:16s} n={len(t):3d}  penalty {t.penalty.median():+.3f}  "
          f"ungauged failures {(t.nse_ungauged<0).sum()}")
print("\nworst 10 ungauged catchments:")
cols = ["gauge_name", "nse_gauged", "nse_ungauged", "penalty", "chalk"]
print(tab.nsmallest(10, "nse_ungauged")[cols].round(3).to_string())
print(f"\nwrote {OUT/'spatial_folds_per_catchment.csv'}")
