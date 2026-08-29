"""Per-catchment scores for the tree's norm/log1p variants — Phase 2 C1.

The audit's live hypothesis: the LSTM's weak-catchment rescue is driven by
per-basin loss normalisation (its MSE on per-basin-standardised targets is
basin-NSE-like), not by learned memory. The tree-side test is to give the
tree the same treatment and look per catchment. hgb_targets.py already fit
`norm` (y / p_mean) and `log1p` but kept only pooled cards; this refits both
with the identical config and saves per-catchment NSE, then prints the
paired comparison against the raw tree and the 16-epoch LSTM on the exact
subsets where the LSTM's gains concentrate.

If norm/log1p recover most of the LSTM's gain on weak-tree catchments, Q1
resolves to "normalisation"; if they recover little, memory/architecture is
back on the table and the --seq 90 run (A1) becomes the decider.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import build_dataset, temporal_split
from evaluate import per_catchment

OUT = Path(__file__).resolve().parent / "results"
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

print("building dataset...", flush=True)
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
p_mean_tr, p_mean_te = Xtr["p_mean"].values, Xte["p_mean"].values

VARIANTS = {
    "norm":  (lambda y, pm: y / pm,      lambda p, pm: p * pm),
    "log1p": (lambda y, pm: np.log1p(y), lambda p, pm: np.expm1(p)),
}

pc = {}
for name, (fwd, inv) in VARIANTS.items():
    t0 = time.time()
    m = HistGradientBoostingRegressor(**BASE)
    m.fit(Xtr, fwd(ytr.values, p_mean_tr))
    pred = inv(m.predict(Xte), p_mean_te)
    res = pd.DataFrame({"gid": gte, "obs": yte.values,
                        "pred": np.clip(pred, 0, None).astype("float32")},
                       index=Xte.index)
    res.to_parquet(SCRATCH / f"targets_{name}.parquet")
    pc[name] = per_catchment(res)
    print(f"  {name}: fitted in {time.time()-t0:.0f}s "
          f"(median NSE {pc[name].nse.median():+.3f})", flush=True)

# raw tree + LSTM per-catchment scores already on disk, identical split
base = pd.read_csv(OUT / "groundwater_paired.csv", index_col=0)
lstm = pd.read_csv(OUT / "lstm_e16" / "lstm_per_catchment.csv", index_col=0)

tab = pd.DataFrame({
    "nse_raw": base.nse_raw,
    "nse_norm": pc["norm"].nse,
    "nse_log1p": pc["log1p"].nse,
    "nse_lstm": lstm.nse,
    "chalk": base.chalk,
    "gauge_name": base.gauge_name,
})
for c in ["norm", "log1p", "lstm"]:
    tab[f"d_{c}"] = tab[f"nse_{c}"] - tab.nse_raw
tab.to_csv(OUT / "target_transforms_per_catchment.csv")

# test-window flow std, for the variance stratification the audit used
flows = pd.read_parquet(Path(__file__).resolve().parent.parent
                        / "cache" / "daily_discharge_spec.parquet")
std = flows.loc["2010-10-01":].std()
std.index = std.index.astype(tab.index.dtype)
tab["low_var"] = tab.index.map(std) <= std.reindex(tab.index).quantile(0.25)

SUBSETS = {
    "all": tab.index == tab.index,
    "weak_tree (raw NSE<0.6)": tab.nse_raw < 0.6,
    "chalk": tab.chalk.astype(bool),
    "low_variance_quartile": tab.low_var,
}
print("\n=== paired NSE deltas vs raw tree (median [frac improved]) ===")
hdr = f"{'subset':26s}  n   " + "".join(f"{c:>18s}" for c in ["d_norm", "d_log1p", "d_lstm"])
print(hdr)
for name, m in SUBSETS.items():
    t = tab[m]
    cells = "".join(f"    {t[f'd_{c}'].median():+.3f} [{(t[f'd_{c}']>0).mean():.2f}]"
                    for c in ["norm", "log1p", "lstm"])
    print(f"{name:26s} {len(t):3d} {cells}")

print("\nfailed catchments (NSE<0): raw "
      f"{(tab.nse_raw<0).sum()}, norm {(tab.nse_norm<0).sum()}, "
      f"log1p {(tab.nse_log1p<0).sum()}, lstm {(tab.nse_lstm<0).sum()}")
frac = tab.d_lstm.where(tab.d_lstm.abs() > 1e-9)
recov = (tab.d_norm / frac).loc[tab.nse_raw < 0.6]
print(f"norm recovers a median {recov.median()*100:.0f}% of the LSTM's gain "
      f"on weak-tree catchments (log1p: "
      f"{(tab.d_log1p/frac).loc[tab.nse_raw < 0.6].median()*100:.0f}%)")
print(f"\nwrote {OUT/'target_transforms_per_catchment.csv'}")
