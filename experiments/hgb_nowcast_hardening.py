"""Nowcast hardening: lead time and nestedness — Phase 5 C1.

Two adversarial questions about the donor-gauge result, answered with two
fits on the identical temporal split:

  lagged    donors at lag-1 and lag-2 ONLY (no same-day flow). The original
            uses same-day donor flow, which is legitimate nowcasting but has
            zero lead time; this measures what one day of lead costs, which
            any operational framing must state.
  dropnear  donors at ranks 2-4 by distance (the nearest gauge excluded).
            The nearest donor is the one most likely to sit on the same
            river immediately up/downstream — where its "observed flow"
            partially IS the target's flow. If the gains collapse without
            the nearest donor, part of the result is measurement rather
            than prediction; if they survive, the network-information claim
            stands.

Compared against the committed raw tree and full nowcast predictions.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset, temporal_split
from nowcast_common import D0, D1, nearest_donors
from evaluate import evaluate, per_catchment, amax_bias, report

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
D2 = D0.shift(2)

print("building dataset...", flush=True)
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")
gids = np.unique(GID)

near3 = nearest_donors(gids)                 # ranks 0-2
near4 = nearest_donors(gids, k=4)            # ranks 0-3

VARIANTS = {
    "lagged":   {g: (near3[g][0], [D1, D2]) for g in gids},
    "dropnear": {g: (near4[g][0][1:], [D0, D1]) for g in gids},
}


def build_feats(index, gid, assignment):
    k = 3
    cols = {f"nb{r}_l{i}": np.full(len(index), np.nan, dtype="float32")
            for r in range(k) for i in range(2)}
    for g in np.unique(gid):
        donors, lags = assignment[g]
        rows = np.flatnonzero(gid == g)
        dts = index[rows]
        for r, d in enumerate(donors[:k]):
            for i, L in enumerate(lags):
                cols[f"nb{r}_l{i}"][rows] = L[d].reindex(dts).values
    return pd.DataFrame(cols, index=index)


preds = {"raw": pd.read_parquet(OUT / "tree_test_predictions.parquet"),
         "nowcast": pd.read_parquet(OUT / "nowcast_test_predictions.parquet")}
for name, assign in VARIANTS.items():
    t0 = time.time()
    Xa = pd.concat([Xtr, build_feats(Xtr.index, gtr, assign)], axis=1)
    Xb = pd.concat([Xte, build_feats(Xte.index, gte, assign)], axis=1)
    m = HistGradientBoostingRegressor(**BASE).fit(Xa, ytr)
    res = pd.DataFrame({"gid": gte, "obs": yte.values,
                        "pred": np.clip(m.predict(Xb), 0, None).astype("float32")},
                       index=Xb.index)
    res.to_parquet(SCRATCH / f"nowcast_{name}.parquet")
    preds[name] = res
    del Xa, Xb
    print(f"  {name}: fitted in {time.time()-t0:.0f}s", flush=True)

rows = [evaluate(r, k, train_max=train_max)[0] for k, r in preds.items()]
print()
report(rows).to_csv(OUT / "nowcast_hardening_cards.csv")

pcs = {k: per_catchment(v).nse for k, v in preds.items()}
tab = pd.DataFrame(pcs)
print("\npaired medians vs raw: "
      + ", ".join(f"{k} {(tab[k]-tab.raw).median():+.3f}"
                  for k in ["nowcast", "lagged", "dropnear"]))
print("share of full nowcast NSE gain retained: "
      + ", ".join(f"{k} {((tab[k]-tab.raw).median()/(tab.nowcast-tab.raw).median())*100:.0f}%"
                  for k in ["lagged", "dropnear"]))
amx = {k: amax_bias(v).median() for k, v in preds.items()}
print("AMAX bias: " + ", ".join(f"{k} {v:+.1f}%" for k, v in amx.items()))
tab.to_csv(OUT / "nowcast_hardening_per_catchment.csv")
print(f"\nwrote nowcast_hardening_cards.csv, nowcast_hardening_per_catchment.csv")
