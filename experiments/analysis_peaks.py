"""Zero-fit peak diagnostics — Phase 2 C3.

Four analyses from files already on disk, no model fitting. Together with
C2's flood-day calibration they answer Q2: is the peak failure a statistic
problem (right information, wrong functional), an information problem (the
forcings don't contain the peak), or an observation problem (the "truth" is
itself extrapolated / clipped)?

  (a) AMAX event anatomy   timing vs amplitude vs 5-day volume per event
  (b) rating noise floor   AMAX bias stratified by rating-curve uncertainty
  (c) 4-member ensemble    mean + upper-member cards from committed runs
  (d) bankfull skill       POD/FAR/CSI for the threshold decision

Inputs: results/lstm_*/ prediction parquets, the raw-tree predictions
(scratch targets_raw.parquet, pass its dir as argv[1]), cache/, hydrometry
and topographic attributes.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import read_attr
from evaluate import evaluate, amax_bias, report

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

MODELS = {
    "tree": OUT / "tree_test_predictions.parquet",
    "lstm_e16": OUT / "lstm_e16" / "lstm_test_predictions.parquet",
}
ENSEMBLE = {
    "e8s0": OUT / "lstm_test_predictions.parquet",
    "e16": OUT / "lstm_e16" / "lstm_test_predictions.parquet",
    "s1": OUT / "lstm_seed1" / "lstm_test_predictions.parquet",
    "tail1": OUT / "lstm_tail1" / "lstm_test_predictions.parquet",
}

res = {k: pd.read_parquet(p) for k, p in MODELS.items()}

topo = read_attr("topographic").set_index("gauge_id")
hyd = read_attr("hydrometry")
hyd["gauge_id"] = pd.to_numeric(hyd.gauge_id, errors="coerce")
hyd = hyd.set_index("gauge_id")
area = topo.area
to_mmday = 86.4 / area          # m3/s -> mm/day specific discharge

# ---- (a) AMAX event anatomy ------------------------------------------------

def event_anatomy(r):
    rows = []
    for gid, g in r.groupby("gid"):
        g = g.sort_index()
        wy = g.index.year + (g.index.month >= 10).astype(int)
        for _, e in g.groupby(wy):
            if e.obs.count() < 350:
                continue
            d = e.obs.idxmax()
            w = e.loc[d - pd.Timedelta(days=3): d + pd.Timedelta(days=3)]
            v = e.loc[d - pd.Timedelta(days=2): d + pd.Timedelta(days=2)]
            own = e.pred.idxmax()
            rows.append({
                "gid": gid, "obs_peak": e.obs.max(),
                "same_day": e.pred[d] / e.obs.max(),
                "matched_peak": w.pred.max() / e.obs.max(),
                "volume5": v.pred.sum() / v.obs.sum() if v.obs.sum() > 0 else np.nan,
                "timing_hit": abs((own - d).days) <= 3,
            })
    return pd.DataFrame(rows)

print("=== (a) AMAX event anatomy (medians over catchment-year events) ===")
print(f"{'model':10s} {'n_events':>8s} {'same-day':>9s} {'peak±3d':>9s} "
      f"{'5d-volume':>10s} {'own-max within 3d':>18s}")
anat = {}
for k, r in res.items():
    a = event_anatomy(r)
    anat[k] = a
    print(f"{k:10s} {len(a):8d} {a.same_day.median():9.3f} "
          f"{a.matched_peak.median():9.3f} {a.volume5.median():10.3f} "
          f"{a.timing_hit.mean():18.2%}")
pd.concat(anat, names=["model"]).to_csv(OUT / "amax_event_anatomy.csv")

# ---- (b) rating-curve noise floor ------------------------------------------

print("\n=== (b) AMAX bias vs rating-curve uncertainty ===")
ab = {k: amax_bias(r).groupby("gid").median() for k, r in res.items()}
strat = pd.DataFrame(ab)
strat["q99_halfwidth"] = (hyd.q99_uncert_upper - hyd.q99_uncert_lower).reindex(strat.index) / 2
strat["extrap_dur"] = hyd.daily_flow_extrap_dur.reindex(strat.index)
obs_amax = {gid: g.obs.max() for gid, g in res["tree"].groupby("gid")}
mgf = (hyd.max_gauging_flow * to_mmday).reindex(strat.index)
strat["peak_beyond_gauged"] = pd.Series(obs_amax) > mgf
strat.to_csv(OUT / "amax_bias_vs_rating.csv")

for col, label in [("q99_halfwidth", "q99 uncertainty half-width quartile"),
                   ("extrap_dur", "flow extrapolation-duration quartile")]:
    ok = strat.dropna(subset=[col])
    qt = pd.qcut(ok[col].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
    print(f"\nby {label} (median AMAX bias %):")
    print(ok.groupby(qt, observed=True)[["tree", "lstm_e16"]].median().round(1).to_string())
n_bg = strat.peak_beyond_gauged.sum()
print(f"\ncatchments whose test AMAX exceeds the largest ever-gauged flow: "
      f"{n_bg}/{strat.peak_beyond_gauged.notna().sum()}")
print(strat.groupby("peak_beyond_gauged")[["tree", "lstm_e16"]].median().round(1).to_string())
hw = strat.q99_halfwidth.median()
print(f"\nmedian rating half-width at q99: ±{hw:.1f}% "
      f"(vs tree AMAX bias {strat.tree.median():+.1f}%, "
      f"lstm {strat.lstm_e16.median():+.1f}%)")

# ---- (c) free 4-member LSTM ensemble ---------------------------------------

print("\n=== (c) 4-member LSTM ensemble ===")
members = {k: pd.read_parquet(p) for k, p in ENSEMBLE.items()}
first = next(iter(members.values()))
P = np.stack([m.pred.values for m in members.values()])
rows = []
for name, pred in [("ens_mean", P.mean(0)), ("ens_upper", P.max(0))]:
    e = first[["gid", "obs"]].assign(pred=pred)
    rows.append(evaluate(e, name)[0])
rows.append(evaluate(res["lstm_e16"], "lstm_e16 (best single)")[0])
rows.append(evaluate(res["tree"], "tree")[0])
report(rows).to_csv(OUT / "lstm_ensemble_cards.csv")

# ---- (d) bankfull-exceedance skill -----------------------------------------

print("\n=== (d) bankfull-exceedance skill (POD/FAR/CSI, pooled days) ===")
bf = (hyd.bankfull_flow * to_mmday).dropna()
rows = []
for k, r in {**res, "ens_upper": first[["gid", "obs"]].assign(pred=P.max(0))}.items():
    rr = r[r.gid.isin(bf.index)]
    th = rr.gid.map(bf).values
    o, p = rr.obs.values >= th, rr.pred.values >= th
    hits, miss, fa = (o & p).sum(), (o & ~p).sum(), (~o & p).sum()
    rows.append({"model": k, "n_gauges": rr.gid.nunique(),
                 "obs_days": int(o.sum()),
                 "POD": hits / (hits + miss), "FAR": fa / max(hits + fa, 1),
                 "CSI": hits / max(hits + miss + fa, 1)})
bk = pd.DataFrame(rows).set_index("model")
print(bk.round(3).to_string())
bk.to_csv(OUT / "bankfull_skill.csv")

print(f"\nwrote amax_event_anatomy.csv, amax_bias_vs_rating.csv, "
      f"lstm_ensemble_cards.csv, bankfull_skill.csv in {OUT}")
