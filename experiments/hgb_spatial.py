"""The ungauged-basin question: how much does the tree lose when it has never
seen the catchment?

Design: hold out 83 whole catchments (every 5th in area order, so the held-out
set spans the full size range), train on the remaining 333 over 1970-2010, and
evaluate the held-out catchments over 2010-2022 -- the same test window as the
temporal split, so the two experiments differ in exactly one thing: whether
the model saw that catchment's past. The gap between them is the price of
"ungauged", which is the setting FEH exists for and the one where Kratzert
et al. (2019) showed pooled ML beating locally calibrated models.

No per-catchment identifiers are in the features, so nothing needs removing;
the statics (area, climate, soils, geology, land cover) have to carry all
catchment identity on their own.

Run raw and log1p -- log1p so the comparison covers whichever target the
transform experiment favours.
"""
import sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import STATIC, build_dataset, good_catchments, TRAIN_END, TEST_START
from evaluate import evaluate, report

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)
SCRATCH = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT

BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)

GOOD = good_catchments()
by_area = sorted(GOOD, key=lambda g: STATIC.loc[g, "area"])
test_cats = set(by_area[2::5])              # area-stratified fifth
train_cats = [g for g in GOOD if g not in test_cats]
print(f"train catchments: {len(train_cats)}   held-out: {len(test_cats)}", flush=True)

print("building dataset...", flush=True)
t0 = time.time()
DATA, GID = build_dataset()
dates = DATA.index
feats = [c for c in DATA.columns if c != "y"]
in_test_cat = np.isin(GID, list(test_cats))
tr = np.asarray(dates <= TRAIN_END) & ~in_test_cat
te = np.asarray(dates >= TEST_START) & in_test_cat
Xtr, ytr = DATA.loc[tr, feats], DATA.loc[tr, "y"]
Xte, yte = DATA.loc[te, feats], DATA.loc[te, "y"]
gte = GID[te]
print(f"  {len(ytr):,} train / {len(yte):,} test rows in {time.time()-t0:.0f}s", flush=True)

VARIANTS = {
    "spatial_raw":   (lambda y: y,         lambda p: p),
    "spatial_log1p": (lambda y: np.log1p(y), lambda p: np.expm1(p)),
}

rows = []
for name, (fwd, inv) in VARIANTS.items():
    t0 = time.time()
    model = HistGradientBoostingRegressor(**BASE)
    model.fit(Xtr, fwd(ytr.values))
    pred = inv(model.predict(Xte))
    res = pd.DataFrame({"gid": gte, "obs": yte.values,
                        "pred": np.clip(pred, 0, None).astype("float32")},
                       index=Xte.index)
    res.to_parquet(SCRATCH / f"{name}.parquet")
    # no per-catchment train_max here: held-out catchments have no training data
    row, _ = evaluate(res, name)
    row["fit_s"] = round(time.time() - t0)
    rows.append(row)
    print(f"  {name}: {row['fit_s']}s (median NSE {row['median_NSE']:+.3f})", flush=True)

df = report(rows)
df.to_csv(OUT / "spatial_split.csv")
print(f"\nwrote {OUT/'spatial_split.csv'}")
