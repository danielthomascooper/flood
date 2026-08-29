"""Conditional calibration of the quantile ladder — Phase 2 C2.

The sweep reported pooled coverage over 1.8M test rows, which ordinary days
dominate; pooled 0.983 can coexist with badly miscalibrated catchments or
flood days exactly where the envelope would be used. This persists the
sweep's predictions into results/ (they previously lived only in a session
scratch dir — the audit flagged that as perishable evidence) and computes
coverage per catchment, on AMAX days only, and on top-1% obs days.

Usage: analysis_quantile_calibration.py <dir containing quantile_predictions.parquet>
(no refit; if the parquet is gone, rerun hgb_quantiles.py first).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "results"
SRC = Path(sys.argv[1]) / "quantile_predictions.parquet" if len(sys.argv) > 1 \
    else OUT / "quantile_predictions.parquet"

ALPHAS = [0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
QCOLS = [f"q{int(a*100):02d}" for a in ALPHAS]

Q = pd.read_parquet(SRC)
dest = OUT / "quantile_predictions.parquet"
if SRC != dest:
    Q.to_parquet(dest)
    print(f"persisted {SRC} -> {dest} ({len(Q):,} rows)")

q2 = Q.reset_index()
q2["wy"] = q2.date.dt.year + (q2.date.dt.month >= 10).astype(int)

# subsets: all days / AMAX days (one per catchment-water-year with >=350 obs
# days) / top-1% obs days per catchment
counts = q2.groupby(["gid", "wy"]).obs.count()
full_years = counts[counts >= 350].index
sub = q2[pd.MultiIndex.from_frame(q2[["gid", "wy"]]).isin(full_years)]
amax = sub.loc[sub.groupby(["gid", "wy"]).obs.idxmax()]
thr = q2.groupby("gid").obs.transform(lambda o: o.quantile(0.99))
top1 = q2[q2.obs >= thr]

print(f"\n{len(amax):,} AMAX events, {len(top1):,} top-1% days")
print("\n=== coverage by subset (nominal vs empirical) ===")
rows = []
for a, c in zip(ALPHAS, QCOLS):
    rows.append({"alpha": a,
                 "pooled": (q2.obs <= q2[c]).mean(),
                 "amax_days": (amax.obs <= amax[c]).mean(),
                 "top1_days": (top1.obs <= top1[c]).mean(),
                 "amax_median_ratio": (amax[c] / amax.obs).median()})
cond = pd.DataFrame(rows)
print(cond.round(3).to_string(index=False))

# per-catchment coverage spread
print("\n=== per-catchment coverage (416 catchments) ===")
rows = []
for a, c in zip(ALPHAS, QCOLS):
    cov = q2.groupby("gid").apply(lambda g: (g.obs <= g[c]).mean(),
                                  include_groups=False)
    rows.append({"alpha": a, "median": cov.median(),
                 "q10": cov.quantile(0.1), "q90": cov.quantile(0.9),
                 "frac_within_2pp": ((cov - a).abs() <= 0.02).mean()})
pc = pd.DataFrame(rows)
print(pc.round(3).to_string(index=False))

cond.merge(pc, on="alpha", suffixes=("", "_pc")) \
    .to_csv(OUT / "quantile_calibration_conditional.csv", index=False)
print(f"\nwrote {OUT/'quantile_calibration_conditional.csv'}")
