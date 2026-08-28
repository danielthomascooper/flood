"""Does the tree's flood failure come from the target, the leaves, or the data?

Five variants of the identical model on the identical temporal split, changing
one thing each:

  raw        control -- the committed baseline configuration
  fine_leaf  min_samples_leaf 100 -> 20: tests the "it's just smoothing" (i.e.
             tuning-knob) hypothesis. Extreme days are rare; a 100-sample leaf
             must pool them with ordinary days, dragging predictions down.
  log1p      train on log1p(y), back-transform with expm1: MSE in log space is
             relative error, so the model stops spending all capacity on big
             absolute residuals; leaf averages become geometric means.
  norm       train on y / p_mean(catchment): a runoff-index target, so wet and
             dry catchments share structure and the ceiling is per-climate.
  q99        quantile loss, alpha=0.99: targets the tail directly. NSE is
             meaningless for this one by design -- read coverage (should be
             ~0.99), top-1% bias and AMAX bias instead.

If none of these moves top-1% performance much, the binding constraint is
rare-event scarcity (Martel et al. 2025), not model structure.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset, temporal_split
from evaluate import evaluate, report

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")
p_mean_tr, p_mean_te = Xtr["p_mean"].values, Xte["p_mean"].values
print(f"  {len(ytr):,} train / {len(yte):,} test rows in {time.time()-t0:.0f}s", flush=True)

VARIANTS = {
    "raw":       dict(params={}, fwd=lambda y, pm: y,             inv=lambda p, pm: p),
    "fine_leaf": dict(params={"min_samples_leaf": 20},
                      fwd=lambda y, pm: y,                        inv=lambda p, pm: p),
    "log1p":     dict(params={}, fwd=lambda y, pm: np.log1p(y),   inv=lambda p, pm: np.expm1(p)),
    "norm":      dict(params={}, fwd=lambda y, pm: y / pm,        inv=lambda p, pm: p * pm),
    "q99":       dict(params={"loss": "quantile", "quantile": 0.99},
                      fwd=lambda y, pm: y,                        inv=lambda p, pm: p),
}

rows = []
for name, v in VARIANTS.items():
    t0 = time.time()
    model = HistGradientBoostingRegressor(**{**BASE, **v["params"]})
    model.fit(Xtr, v["fwd"](ytr.values, p_mean_tr))
    pred = v["inv"](model.predict(Xte), p_mean_te)
    res = pd.DataFrame({"gid": gte, "obs": yte.values,
                        "pred": np.clip(pred, 0, None).astype("float32")},
                       index=Xte.index)
    res.to_parquet(SCRATCH / f"targets_{name}.parquet")
    row, _ = evaluate(res, name, train_max=train_max)
    row["fit_s"] = round(time.time() - t0)
    rows.append(row)
    print(f"  {name}: fitted+evaluated in {row['fit_s']}s "
          f"(median NSE {row['median_NSE']:+.3f}, top1 {row['top1_NSE']:+.3f})", flush=True)

df = report(rows)
df.to_csv(OUT / "target_transforms.csv")
print(f"\nwrote {OUT/'target_transforms.csv'}")
