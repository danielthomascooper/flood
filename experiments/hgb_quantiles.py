"""Full quantile sweep: turn the calibrated alpha=0.99 envelope into a
predictive distribution.

Fits alpha = 0.05 / 0.25 / 0.50 / 0.75 / 0.95 on the identical temporal split
(0.99 is reused from hgb_targets.py if its predictions are present). Reports:

  calibration   empirical coverage per alpha vs nominal
  sharpness     pinball loss per alpha; median central-interval widths
  intervals     coverage of the 50% [q25,q75] and 90% [q05,q95] bands
  q50 as point  the median-regression model scored on the standard card

Independently fitted quantiles can cross; rows are sorted across alphas
before interval statistics and the crossing rate is reported.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset, temporal_split
from evaluate import evaluate, per_catchment, report

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
ALPHAS = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]

print("building dataset...", flush=True)
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")

Q = pd.DataFrame({"gid": gte, "obs": yte.values}, index=Xte.index)
for a in ALPHAS:
    col = f"q{int(a*100):02d}"
    reuse = SCRATCH / "targets_q99.parquet"
    if a == 0.99 and reuse.exists():
        Q[col] = pd.read_parquet(reuse)["pred"].values
        print(f"  {col}: reused from hgb_targets run", flush=True)
        continue
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=a)
    m.fit(Xtr, ytr)
    Q[col] = np.clip(m.predict(Xte), 0, None).astype("float32")
    print(f"  {col}: fitted in {time.time()-t0:.0f}s", flush=True)

qcols = [f"q{int(a*100):02d}" for a in ALPHAS]
raw = Q[qcols].to_numpy()
crossing = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
Q[qcols] = np.sort(raw, axis=1)          # enforce monotone quantiles
Q.to_parquet(OUT / "quantile_predictions.parquet")   # persisted: audit flagged scratch dirs as perishable

obs = Q["obs"].to_numpy()
rows = []
for a, col in zip(ALPHAS, qcols):
    p = Q[col].to_numpy()
    pinball = float(np.mean(np.where(obs >= p, (obs - p) * a, (p - obs) * (1 - a))))
    rows.append({"alpha": a, "coverage": float((obs <= p).mean()),
                 "pinball": pinball, "median_pred": float(np.median(p))})
cal = pd.DataFrame(rows)
print("\n=== calibration (nominal vs empirical coverage) ===")
print(cal.round(4).to_string(index=False))
print(f"quantile-crossing rows before sort-fix: {crossing*100:.2f}%")

for lo, hi, nom in [("q25", "q75", 50), ("q05", "q95", 90)]:
    inside = float(((obs >= Q[lo]) & (obs <= Q[hi])).mean())
    width = float((Q[hi] - Q[lo]).median())
    print(f"{nom}% central interval: coverage {inside*100:.1f}%  "
          f"median width {width:.2f} mm/day")

res50 = Q[["gid", "obs"]].assign(pred=Q["q50"])
row, _ = evaluate(res50, "q50_as_point_forecast", train_max=train_max)
report([row])

cal.to_csv(OUT / "quantile_sweep.csv", index=False)
print(f"\nwrote {OUT/'quantile_sweep.csv'}")
