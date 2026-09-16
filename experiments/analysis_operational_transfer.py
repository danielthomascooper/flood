"""Operational-transfer test — Phase 7 finishing item.

The mixed-regime fine-tune learned from the GEFS v12 *reforecast*
(2000-2010: reanalysis initial conditions, 5 members). The test archive
holds both reforecast years (2010-10 -> 2019-12) and real-time
operational GEFS v12 (2020-09-23 -> 2022-09-30: operational analysis,
same 5-member subset of 31). Scoring every frozen-vs-fine-tuned pair on
each period separately asks whether the gain survives the switch to a
live feed. Same model version on both sides, so this tests initial
conditions / ensemble source, NOT NWP model upgrades.

Writes results/operational_transfer.csv.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import per_catchment

R = Path(__file__).resolve().parent / "results"
PERIODS = {"reforecast_2010_19": lambda d: np.asarray(d < "2020-01-01"),
           "operational_2020_22": lambda d: np.asarray(d >= "2020-09-23")}


def amax_mask(df):
    full = df[["gid", "obs"]].reset_index()
    full["wy"] = full.date.dt.year + (full.date.dt.month >= 10)
    n = full.groupby(["gid", "wy"]).obs.transform("size")
    am = full[n >= 350].groupby(["gid", "wy"]).obs.idxmax().values
    m = np.zeros(len(df), bool); m[am] = True
    return m


rows = []
# lead-1 LSTM ladders
runs = {"frozen_ens": "lstm_fc_ens_q_L1", "frozen_memmax": "lstm_fc_memmax_q_L1",
        "ft_ens": "lstm_ft_ens_q_L1", "ft_memmax": "lstm_ft_memmax_q_L1"}
D = {k: pd.read_parquet(R / v / "lstm_test_predictions.parquet") for k, v in runs.items()}
base = D["frozen_ens"]; am = amax_mask(base)
cov = base.covered.values & base.obs.notna().values
for per, f in PERIODS.items():
    sel = cov & f(base.index); sa = sel & am
    for k, d in D.items():
        obs = d.obs.values
        rows.append(dict(family="lstm_ladder_L1", model=k, period=per, n=int(sel.sum()), n_amax=int(sa.sum()),
                         amax_q99=(obs[sa] <= d.q99.values[sa]).mean(),
                         pooled_q99=(obs[sel] <= d.q99.values[sel]).mean(),
                         w90=float(np.median(d.q95.values[sel] - d.q05.values[sel]))))
# points, both families
pairs = {"lstm": [(L, R / f"lstm_fc_ens_L{L}/lstm_test_predictions.parquet",
                   R / f"lstm_ft_ens_L{L}/lstm_test_predictions.parquet") for L in (1, 2, 3)],
         "tree": [(L, R / f"forecast_ar_gefs_ens_L{L}.parquet",
                   R / f"forecast_fctrain_mixeds_L{L}.parquet") for L in (1, 2, 3)]}
for fam, lst in pairs.items():
    for L, pa, pb in lst:
        a, b = pd.read_parquet(pa), pd.read_parquet(pb)
        cov = a.covered.values & a.obs.notna().values
        for per, f in PERIODS.items():
            sel = cov & f(a.index)
            na = per_catchment(a[sel][["gid", "obs", "pred"]]).nse
            nb = per_catchment(b[sel][["gid", "obs", "pred"]]).nse
            rows.append(dict(family=f"{fam}_point_L{L}", model="frozen_vs_ft", period=per, n=int(sel.sum()),
                             frozen_nse=na.median(), ft_nse=nb.median(), dNSE=(nb - na).median(),
                             ft_better_frac=(nb > na).mean()))
df = pd.DataFrame(rows)
print(df.round(3).to_string(index=False))
df.to_csv(R / "operational_transfer.csv", index=False)
print("wrote operational_transfer.csv")
