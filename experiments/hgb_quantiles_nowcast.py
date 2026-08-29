"""Quantile ladder + nowcast donors — closing the Gate 2 residual? (Phase 3 C1)

Gate 2 left one number unexplained: calibrated q99 envelopes miss ~15% of
AMAX days for both model classes, because the daily forcings don't flag
those events. hgb_nowcast.py showed the missing signal lives in the
neighbouring gauges. This refits the full 6-alpha ladder WITH the donor
features and reports the same conditional calibration as
analysis_quantile_calibration.py — the question is a single column: does
AMAX-day q99 coverage climb from 0.829 towards 0.99, and how much sharper
does the ladder get?
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset, temporal_split
from nowcast_common import nearest_donors, donor_features
from evaluate import evaluate, report

OUT = Path(__file__).resolve().parent / "results"

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
ALPHAS = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QCOLS = [f"q{int(a*100):02d}" for a in ALPHAS]

print("building dataset...", flush=True)
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
train_max = pd.Series(ytr.values, index=gtr).groupby(level=0).max().rename("train_max")
assign = nearest_donors(np.unique(GID))
Xtr = pd.concat([Xtr, donor_features(Xtr.index, gtr, assign)], axis=1)
Xte = pd.concat([Xte, donor_features(Xte.index, gte, assign)], axis=1)

Q = pd.DataFrame({"gid": gte, "obs": yte.values}, index=Xte.index)
for a, col in zip(ALPHAS, QCOLS):
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=a)
    m.fit(Xtr, ytr)
    Q[col] = np.clip(m.predict(Xte), 0, None).astype("float32")
    print(f"  {col}: fitted in {time.time()-t0:.0f}s", flush=True)

raw = Q[QCOLS].to_numpy()
crossing = float((np.diff(raw, axis=1) < 0).any(axis=1).mean())
Q[QCOLS] = np.sort(raw, axis=1)
Q.to_parquet(OUT / "quantile_nowcast_predictions.parquet")
print(f"crossing rows before sort: {crossing*100:.2f}%", flush=True)

# conditional calibration, same construction as analysis_quantile_calibration
q2 = Q.reset_index()
q2["wy"] = q2.date.dt.year + (q2.date.dt.month >= 10).astype(int)
counts = q2.groupby(["gid", "wy"]).obs.count()
full_years = counts[counts >= 350].index
sub = q2[pd.MultiIndex.from_frame(q2[["gid", "wy"]]).isin(full_years)]
amax = sub.loc[sub.groupby(["gid", "wy"]).obs.idxmax()]
thr = q2.groupby("gid").obs.transform(lambda o: o.quantile(0.99))
top1 = q2[q2.obs >= thr]

old = pd.read_csv(OUT / "quantile_calibration_conditional.csv")
rows = []
for a, c in zip(ALPHAS, QCOLS):
    rows.append({"alpha": a,
                 "pooled": (q2.obs <= q2[c]).mean(),
                 "amax_days": (amax.obs <= amax[c]).mean(),
                 "amax_days_no_donors": float(old.loc[old.alpha == a, "amax_days"].iloc[0]),
                 "top1_days": (top1.obs <= top1[c]).mean(),
                 "amax_median_ratio": (amax[c] / amax.obs).median()})
cond = pd.DataFrame(rows)
print("\n=== conditional calibration WITH donors (vs without) ===")
print(cond.round(3).to_string(index=False))
cond.to_csv(OUT / "quantile_nowcast_calibration.csv", index=False)

for lo, hi, nom in [("q25", "q75", 50), ("q05", "q95", 90)]:
    inside = float(((q2.obs >= q2[lo]) & (q2.obs <= q2[hi])).mean())
    width = float((q2[hi] - q2[lo]).median())
    print(f"{nom}% interval: coverage {inside*100:.1f}%  median width {width:.2f} mm/day")

row, _ = evaluate(Q[["gid", "obs"]].assign(pred=Q.q50), "q50_nowcast_as_point",
                  train_max=train_max)
report([row])
print(f"\nwrote quantile_nowcast_predictions.parquet, quantile_nowcast_calibration.csv")
