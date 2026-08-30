"""Post-processing fixes demanded by the adversarial conclusions review — Phase 5 C2.

  (a) daily-null capture: how many of the 145 pilot both-missed AMAX events
      does the DAILY tree ladder catch when its q99 is inflated to match the
      hourly model's overall AMAX-day coverage? The hourly pilot's 72-77%
      has no baseline until this number exists.
  (b) oracle coverage: AMAX days are selected on the realisation, so even a
      perfectly calibrated q99 under-covers them. Simulate observations from
      the fitted conditional ladders (GPD-consistent tail beyond q99, with
      sensitivity) and report the AMAX-day q99 coverage a calibrated model
      of this sharpness WOULD achieve - the correct benchmark, not 0.99.
  (c) event-clustered bootstrap CIs on AMAX-day coverage (annual maxima
      cluster on shared storm dates; rows are not independent).
  (d) nowcast gain vs donor distance - the transferability curve.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nowcast_common import nearest_donors

OUT = Path(__file__).resolve().parent / "results"
ALPHAS = np.array([0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
QCOLS = [f"q{int(a*100):02d}" for a in ALPHAS]
rng = np.random.default_rng(0)

Q = pd.read_parquet(OUT / "quantile_nowcast_predictions.parquet").reset_index()
Q["wy"] = Q.date.dt.year + (Q.date.dt.month >= 10).astype(int)
counts = Q.groupby(["gid", "wy"]).obs.count()
full = counts[counts >= 350].index
sub = Q[pd.MultiIndex.from_frame(Q[["gid", "wy"]]).isin(full)]
am = sub.loc[sub.groupby(["gid", "wy"]).obs.idxmax()].copy()
print(f"{len(am)} AMAX events (donor tree ladder)")

# ---- (a) daily-null capture of the 145 pilot both-missed events ------------
m = pd.read_csv(OUT / "missed_amax_events.csv", parse_dates=["date"])
pilot = set(int(x) for x in (OUT / "hourly_pilot_catchments.txt").read_text().split())
mp = m[m.gid.isin(pilot)]
ev = am.merge(mp[["gid", "date"]], on=["gid", "date"])
print(f"(a) {len(ev)} of the pilot both-missed events matched in the donor-ladder frame")
base_cov = (am.obs <= am.q99).mean()
target_cov = 0.904          # hourly model's all-events AMAX coverage (verified)
lo, hi = 1.0, 3.0
for _ in range(40):
    c = (lo + hi) / 2
    cov = (am.obs <= am.q99 * c).mean()
    lo, hi = (lo, c) if cov > target_cov else (c, hi)
cap = (ev.obs <= ev.q99 * c).mean()
print(f"    inflating daily q99 by x{c:.3f} lifts overall AMAX coverage "
      f"{base_cov:.3f} -> {target_cov:.3f}; it then captures {cap:.1%} of the "
      f"both-missed events (hourly model: 72-77%)")

# ---- (b) oracle coverage under perfect calibration -------------------------
qm = sub[QCOLS].to_numpy()
gidwy = pd.MultiIndex.from_frame(sub[["gid", "wy"]])
codes = gidwy.factorize()[0]
xi = {"gpd_from_ladder": None, "light(x1.2)": 1.2, "heavy(x2)": 2.0}
print("(b) oracle AMAX-day q99 coverage under PERFECT calibration "
      "(20 sims each):")
for name, mult in xi.items():
    covs = []
    for s in range(20):
        U = rng.uniform(size=len(sub))
        # piecewise-linear inverse CDF across the ladder, vectorised per row
        j = np.searchsorted(ALPHAS, np.clip(U, 0.05, 0.99), side="right") - 1
        j = np.clip(j, 0, 4)
        a0, a1 = ALPHAS[j], ALPHAS[j + 1]
        q0 = qm[np.arange(len(sub)), j]
        q1 = qm[np.arange(len(sub)), j + 1]
        w = (np.clip(U, 0.05, 0.99) - a0) / (a1 - a0)
        obs_sim = q0 + w * (q1 - q0)
        tail = U > 0.99
        if mult is None:
            with np.errstate(divide="ignore", invalid="ignore"):
                shape = np.log(np.maximum(qm[:, 5] / np.maximum(qm[:, 4], 1e-6), 1.0)) / np.log(5)
            obs_sim[tail] = qm[tail, 5] * ((0.01 / (1 - U[tail])) ** shape[tail])
        else:
            obs_sim[tail] = qm[tail, 5] * (1 + (mult - 1) * (U[tail] - 0.99) / 0.01)
        df = pd.DataFrame({"c": codes, "sim": obs_sim, "U": U})
        idx = df.groupby("c").sim.idxmax()
        covs.append((df.loc[idx, "U"] <= 0.99).mean())
    print(f"    tail={name:16s} oracle coverage {np.mean(covs):.3f} "
          f"(+/- {np.std(covs):.3f})  [observed: 0.896]")

# ---- (c) event-clustered bootstrap CI on observed AMAX coverage ------------
am["hit"] = am.obs <= am.q99
by_date = am.groupby(am.date.dt.floor("D")).hit.agg(["sum", "count"])
vals = []
for _ in range(2000):
    pick = rng.integers(0, len(by_date), len(by_date))
    s = by_date.iloc[pick]
    vals.append(s["sum"].sum() / s["count"].sum())
print(f"(c) observed AMAX-day q99 coverage 0.896; date-clustered 95% CI "
      f"[{np.quantile(vals, .025):.3f}, {np.quantile(vals, .975):.3f}] "
      f"({len(by_date)} distinct event dates for {len(am)} events)")

# ---- (d) nowcast gain vs donor distance ------------------------------------
paired = pd.read_csv(OUT / "nowcast_paired.csv", index_col=0)
assign = nearest_donors(np.array(paired.index))
dist = pd.Series({g: np.mean(assign[g][1]) for g in paired.index})
gain = paired.nse_now - paired.nse_raw
qd = pd.qcut(dist, 4)
print("(d) paired NSE gain by mean-donor-distance quartile:")
print(gain.groupby(qd, observed=True).median().round(3).to_string())
print(f"    Spearman(gain, distance) = {gain.corr(dist, method='spearman'):+.3f}")

pd.DataFrame({"gain": gain, "mean_donor_km": dist}).to_csv(
    OUT / "nowcast_gain_vs_distance.csv")
print("\nwrote nowcast_gain_vs_distance.csv")
