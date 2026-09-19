"""Phase 8b-2 - level ladder -> exceedance probability (tree, lead 1).

8b showed a POINT level forecast under-calls peaks (AMAX-day -0.22 m at
lead 1, 48% of q99 exceedances hit). The product has to be P(level >
threshold). This fits quantile trees (q50/q75/q90/q95/q99) on z-level in
the mixed regime, predicts each with ens-mean and member-max rain, turns
the ladder into P(exceed thr) by piecewise-linear interpolation of the
implied CDF, and scores the probability the way an alert would use it:
best-cut CSI, hit rate at a false-alarm budget, Brier score, reliability.

MODE: q50|q75|q90|q95|q99 (fit + predict, one per process; model saved
to results/models/) | score
Outputs: results/level_lq_<q>_L1.parquet, results/level_ladder_cards.csv,
results/level_ladder_reliability.csv
"""
import sys, time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import TRAIN_END, TEST_START
from level_common import stations, build, FC_START

OUT = Path(__file__).resolve().parent / "results"
MODELS = OUT / "models"
MODE = sys.argv[1] if len(sys.argv) > 1 else "score"
ALPHAS = {"q50": .50, "q75": .75, "q90": .90, "q95": .95, "q99": .99}
BASE = dict(max_iter=400, learning_rate=0.08, max_leaf_nodes=63,
            min_samples_leaf=100, l2_regularization=1.0,
            early_stopping=True, validation_fraction=0.1, random_state=0)
L = 1


def run(q):
    dst = OUT / f"level_lq_{q}_L1.parquet"
    if dst.exists():
        print(f"{q}: exists, skipping"); return
    gids, st = stations()
    DATA, GID = build(L, gids, st)
    aux = [c for c in DATA.columns if c.startswith(("fc_next", "mx_next"))] + ["target", "y_target"]
    cols = [c for c in DATA.columns if c not in aux]
    ok = DATA["target"].notna().values & DATA["h_now"].notna().values & DATA["y_now"].notna().values
    dates = DATA.index
    has_fc = DATA["fc_next1"].notna().values
    is_tr = np.asarray(dates <= TRAIN_END) & ok & (np.asarray(dates < FC_START) | has_fc)
    is_te = np.asarray(dates >= TEST_START) & ok

    def rain(mask, src):
        X = DATA.loc[mask, cols].copy()
        sub = DATA.loc[mask]
        fc = sub[f"{src}_next1"].values
        X["p_next1"] = np.where(np.isnan(fc), sub["p_next1"].values, fc) if src == "mixed" else fc
        return X

    Xtr = DATA.loc[is_tr, cols].copy()
    fc = DATA.loc[is_tr, "fc_next1"].values
    Xtr["p_next1"] = np.where(np.isnan(fc), DATA.loc[is_tr, "p_next1"].values, fc)
    MODELS.mkdir(exist_ok=True)
    mpath = MODELS / f"hgb_level_{q}_L1.joblib"
    if mpath.exists():
        m = joblib.load(mpath)
    else:
        t0 = time.time()
        m = HistGradientBoostingRegressor(**BASE, loss="quantile", quantile=ALPHAS[q]) \
            .fit(Xtr, DATA.loc[is_tr, "target"].values)
        joblib.dump(m, mpath)
        print(f"{q}: {is_tr.sum():,} train rows, fitted in {time.time()-t0:.0f}s", flush=True)
    idx = pd.DatetimeIndex(dates[is_te], name="date") + pd.Timedelta(days=L)
    out = pd.DataFrame({"gid": GID[is_te], "obs": DATA.loc[is_te, "target"].values,
                        "covered": has_fc[is_te]}, index=idx)
    out["pred_ens"] = m.predict(rain(is_te, "fc")).astype("float32")
    out["pred_memmax"] = m.predict(rain(is_te, "mx")).astype("float32")
    out.to_parquet(dst)
    st.to_csv(OUT / "level_fc_station_stats.csv")
    print(f"{q}: wrote {dst.name}", flush=True)


if MODE != "score":
    run(MODE); sys.exit(0)

# ---- score ---------------------------------------------------------------
st = pd.read_csv(OUT / "level_fc_station_stats.csv", index_col=0)
Q = {q: pd.read_parquet(OUT / f"level_lq_{q}_L1.parquet") for q in ALPHAS}
base = Q["q50"]; cov = base.covered.values
obs = base.obs.values[cov]; gid = base.gid.values[cov]
point = pd.read_parquet(OUT / "level_fc_L1.parquet")
assert len(point) == len(base) and (point.gid.values == base.gid.values).all()
alphas = np.array(list(ALPHAS.values()))


def ladder(src):                       # (n, 5) monotone quantile values
    M = np.stack([Q[q][f"pred_{src}"].values[cov] for q in ALPHAS], 1)
    return np.sort(M, axis=1)


def p_exceed(M, thr):
    """P(level > thr) from a 5-rung ladder: piecewise-linear CDF between rungs,
    linear tails: below q50 taper to F=0 at q50-(q75-q50)*2, above q99 to F=1 at
    q99+(q99-q95)*2."""
    lo = M[:, 0] - 2 * (M[:, 1] - M[:, 0]); hi = M[:, -1] + 2 * (M[:, -1] - M[:, -2])
    xs = np.column_stack([lo, M, hi]); ys = np.concatenate([[0.0], alphas, [1.0]])
    F = np.array([np.interp(t, x, ys) for t, x in zip(thr, xs)])
    return 1.0 - F


def score(prob, o, name, thr_name):
    cuts = np.linspace(0.05, 0.95, 19)
    best = max(((((o & (prob >= c)).sum()) / max(((o | (prob >= c)).sum()), 1), c) for c in cuts))
    # hit rate at the largest cut whose FAR <= 0.30
    hr_budget = 0.0
    for c in cuts[::-1]:
        f = prob >= c; hits = (o & f).sum(); fa = (~o & f).sum()
        if hits + fa and fa / (hits + fa) <= 0.30:
            hr_budget = hits / max(o.sum(), 1); break
    return dict(source=name, threshold=thr_name, n_events=int(o.sum()),
                best_CSI=best[0], best_cut=best[1], hit_at_FAR30=hr_budget,
                brier=float(np.mean((prob - o) ** 2)), clim_brier=float(np.mean((o.mean() - o) ** 2)))


cards, rel = [], []
for thr_name in ("q90", "q95", "q99"):
    thr = ((pd.Series(gid).map(st[thr_name]) - pd.Series(gid).map(st.mu)) / pd.Series(gid).map(st.sd)).values
    o = obs > thr
    ME, MM = ladder("ens"), ladder("memmax")
    comp = np.column_stack([ME[:, :2], MM[:, 2:]])      # q50/q75 ens, q90+ memmax
    sources = {"point_direct": (point.direct.values[cov] > thr).astype(float),
               "ladder_ens": p_exceed(ME, thr), "ladder_memmax": p_exceed(MM, thr),
               "ladder_composite": p_exceed(np.sort(comp, axis=1), thr)}
    for name, prob in sources.items():
        cards.append(score(prob, o, name, thr_name))
        if name != "point_direct":
            bins = np.clip((prob * 10).astype(int), 0, 9)
            for b in range(10):
                sel = bins == b
                if sel.sum() >= 50:
                    rel.append(dict(source=name, threshold=thr_name, bin=b / 10, n=int(sel.sum()),
                                    forecast_p=float(prob[sel].mean()), observed=float(o[sel].mean())))
df = pd.DataFrame(cards)
print(df.round(3).to_string(index=False))
df.to_csv(OUT / "level_ladder_cards.csv", index=False)
pd.DataFrame(rel).to_csv(OUT / "level_ladder_reliability.csv", index=False)
print("wrote level_ladder_cards.csv, level_ladder_reliability.csv")
