"""Refit the raw baseline tree and PERSIST its test predictions — Phase 2.

The baseline's prediction parquet previously lived only in a session scratch
dir and was lost to a reboot (the audit flagged exactly this). Identical
config and split to hgb_baseline.py / the `raw` variant everywhere else;
writes results/tree_test_predictions.parquet, the tree twin of the
lstm_test_predictions.parquet convention.
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

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

print("building dataset...", flush=True)
DATA, GID = build_dataset()
Xtr, ytr, gtr, Xte, yte, gte = temporal_split(DATA, GID)
t0 = time.time()
m = HistGradientBoostingRegressor(**BASE).fit(Xtr, ytr)
res = pd.DataFrame({"gid": gte, "obs": yte.values,
                    "pred": np.clip(m.predict(Xte), 0, None).astype("float32")},
                   index=Xte.index)
res.to_parquet(OUT / "tree_test_predictions.parquet")
print(f"fitted in {time.time()-t0:.0f}s", flush=True)
row, _ = evaluate(res, "tree_raw")
report([row])
print(f"wrote {OUT/'tree_test_predictions.parquet'}")
