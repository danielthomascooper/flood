"""The missing ~10% — Phase 4 C1.

Both daily model classes, given nowcast donors, leave the same ~10% of AMAX
days outside a nominal-99% envelope (tree ladder 0.896, LSTM ladder 0.892).
This names those events and asks what they look like: which catchments,
which season, how big the same-day rain was, how badly daily averaging
clips them. Output feeds two things: the hourly-pilot catchment list
(hourly_pilot_catchments.txt) and the artifact's account of the residual.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DAILY, STATIC, read_attr

OUT = Path(__file__).resolve().parent / "results"

def amax_table(path):
    Q = pd.read_parquet(path)
    q2 = Q.reset_index()
    q2["wy"] = q2.date.dt.year + (q2.date.dt.month >= 10).astype(int)
    counts = q2.groupby(["gid", "wy"]).obs.count()
    full = counts[counts >= 350].index
    sub = q2[pd.MultiIndex.from_frame(q2[["gid", "wy"]]).isin(full)]
    am = sub.loc[sub.groupby(["gid", "wy"]).obs.idxmax()]
    return am.set_index(["gid", "wy"])

tree = amax_table(OUT / "quantile_nowcast_predictions.parquet")
lstm = amax_table(OUT / "lstm_qnow" / "lstm_test_predictions.parquet")

tab = pd.DataFrame({
    "date": tree.date, "obs": tree.obs,
    "miss_tree": tree.obs > tree.q99,
    "miss_lstm": (lstm.obs > lstm.q99).reindex(tree.index),
    "tree_q99_ratio": tree.q99 / tree.obs,
})
tab["miss_both"] = tab.miss_tree & tab.miss_lstm.fillna(False)
n = len(tab)
print(f"{n} AMAX events: tree misses {tab.miss_tree.sum()} "
      f"({tab.miss_tree.mean():.1%}), lstm {tab.miss_lstm.sum()} "
      f"({tab.miss_lstm.mean():.1%}), BOTH {tab.miss_both.sum()} "
      f"({tab.miss_both.mean():.1%})")
ov = tab.miss_both.sum() / max(tab.miss_tree.sum(), 1)
print(f"overlap: {ov:.0%} of tree misses are also lstm misses "
      f"(same events, not same rate by chance)")

# characterise the both-missed events
m = tab[tab.miss_both].reset_index()
m["month"] = m.date.dt.month
print("\nby season (both-missed vs all AMAX):")
season = lambda mo: "DJF" if mo in (12, 1, 2) else "MAM" if mo in (3, 4, 5) \
    else "JJA" if mo in (6, 7, 8) else "SON"
all_s = tab.reset_index().date.dt.month.map(season).value_counts(normalize=True)
mis_s = m.month.map(season).value_counts(normalize=True)
print(pd.DataFrame({"all_amax": all_s, "missed": mis_s}).round(2).to_string())

# same-day rain percentile (within the catchment's own test-window rain)
print("\nsame-day rain percentile for both-missed events (per catchment):")
pcts = []
for gid, g in m.groupby("gid"):
    f = sorted(DAILY.glob(f"*_{gid}_*.csv"))[0]
    p = pd.read_csv(f, parse_dates=["date"], na_values=["NaN"],
                    usecols=["date", "precipitation_haduk"]).set_index("date")
    p = p.loc["2010-10-01":, "precipitation_haduk"].dropna()
    for dt in g.date:
        pcts.append((p <= p.get(dt, np.nan)).mean() if dt in p.index else np.nan)
pcts = pd.Series(pcts)
print(f"  median {pcts.median():.3f}, <=q90 rain: {(pcts <= 0.90).mean():.0%}, "
      f"<=q95 rain: {(pcts <= 0.95).mean():.0%} "
      f"(an event with <=q90 same-day rain is invisible to a daily rain model)")

m["area_log10"] = STATIC.area.reindex(m.gid).values
small = m.area_log10 <= STATIC.area.quantile(0.25)
print(f"\nsmall-catchment (area Q1) share of missed events: {small.mean():.0%} "
      f"(base rate 25%)")

per_cat = m.groupby("gid").size().rename("missed_events")
m.to_csv(OUT / "missed_amax_events.csv", index=False)

# hourly-pilot list: most-missed catchments with good hourly flow records
hyd = read_attr("hydrometry")
hyd["gauge_id"] = pd.to_numeric(hyd.gauge_id, errors="coerce")
hcov = hyd.set_index("gauge_id").hourly_flow_perc_complete
pool = per_cat.to_frame().join(hcov)
pool = pool[pd.to_numeric(pool.hourly_flow_perc_complete, errors="coerce") >= 90]
picks = pool.sort_values("missed_events", ascending=False).head(48).index.tolist()
# + 12 zero-miss controls, area-matched-ish (every 35th of the rest by area)
rest = [g for g in STATIC.reindex(tab.reset_index().gid.unique()).sort_values("area").index
        if g not in picks and g not in per_cat.index and hcov.get(g, 0) >= 90]
picks += rest[:: max(1, len(rest) // 12)][:12]
Path(OUT / "hourly_pilot_catchments.txt").write_text("\n".join(map(str, picks)) + "\n")
print(f"\nhourly pilot list: {len(picks)} catchments "
      f"({len(picks)-12} most-missed + 12 zero-miss controls) -> "
      f"results/hourly_pilot_catchments.txt")
print(f"wrote {OUT/'missed_amax_events.csv'}")
