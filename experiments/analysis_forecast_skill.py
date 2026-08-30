"""Score any forecast parquet(s) the way a forecast must be scored — Phase 6 C2.

Usage: analysis_forecast_skill.py LEAD path.parquet [path.parquet ...]

Each parquet: gid / obs / pred indexed by the TARGET date (train_lstm.py
--lead and hgb_forecast.py both write this). Persistence is rebuilt from
cache/daily_discharge_spec.parquet as obs shifted by LEAD days, on exactly
the rows each model predicted. Reports the standard card, paired skill vs
persistence, and the two numbers evaluate.py's AMAX bias cannot give a
forecast: bias on the observed peak day, and whether the model's own annual
maximum lands within +/-1 day of the observed one.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT
from evaluate import evaluate, per_catchment, report

LEAD = int(sys.argv[1])
paths = [Path(p) for p in sys.argv[2:]]
flows = pd.read_parquet(ROOT / "cache" / "daily_discharge_spec.parquet")


def persistence_for(res):
    """pred(t) = obs(t - LEAD), aligned to the model's own rows."""
    out = res[["gid", "obs"]].copy()
    src = flows.shift(LEAD)                       # value LEAD days earlier
    out["pred"] = [src.at[d, g] if (d in src.index and g in src.columns) else np.nan
                   for d, g in zip(res.index, res.gid)]
    out["pred"] = out["pred"].astype("float32")
    return out.dropna(subset=["pred"])


def peak_day(res):
    r = res.reset_index()
    r["wy"] = r.date.dt.year + (r.date.dt.month >= 10).astype(int)
    c = r.groupby(["gid", "wy"]).obs.count()
    full = c[c >= 350].index
    sub = r[pd.MultiIndex.from_frame(r[["gid", "wy"]]).isin(full)]
    am = sub.loc[sub.groupby(["gid", "wy"]).obs.idxmax()]
    bias = ((am.pred / am.obs) - 1).median() * 100
    own = sub.loc[sub.groupby(["gid", "wy"]).pred.idxmax()].set_index(["gid", "wy"]).date
    hit = (abs((own - am.set_index(["gid", "wy"]).date).dt.days) <= 1).mean()
    return bias, hit


rows, extra = [], []
for p in paths:
    res = pd.read_parquet(p)
    name = p.parent.name if p.name.startswith("lstm") else p.stem
    pers = persistence_for(res)
    res = res.loc[pers.index] if len(pers) < len(res) else res
    for label, r in [(f"persistence_L{LEAD}", pers), (name, res)]:
        if any(x["model"] == label for x in rows):
            continue
        rows.append(evaluate(r, label)[0])
        b, h = peak_day(r)
        extra.append({"model": label, "peak_day_bias_pct": round(b, 1),
                      "own_max_within_1d": round(h, 3)})
    d = per_catchment(res).nse - per_catchment(pers).nse
    extra[-1].update({"dNSE_vs_persistence_median": round(d.median(), 3),
                      "beats_persistence_frac": round((d > 0).mean(), 3)})

print(f"=== lead {LEAD} day(s): standard card ===")
report(rows)
print("\n=== forecast-specific ===")
print(pd.DataFrame(extra).to_string(index=False))
out = Path(__file__).resolve().parent / "results" / f"forecast_skill_L{LEAD}.csv"
pd.DataFrame(rows).merge(pd.DataFrame(extra), on="model").to_csv(out, index=False)
print(f"\nwrote {out}")
