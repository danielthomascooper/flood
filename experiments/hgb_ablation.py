import time, numpy as np, pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
exec(open("/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/hgb_experiment.py").read().split("t0 = time.time()\nframes")[0])

t0=time.time(); frames,ids=[],[]
for gid in GOOD:
    X=features(gid).dropna(subset=["y"]); frames.append(X); ids.append(np.full(len(X),gid,dtype=np.int32))
DATA=pd.concat(frames); GID=np.concatenate(ids); del frames
dates=DATA.index; tr=dates<=TRAIN_END; te=dates>=TEST_START

MEMORY = [c for c in DATA.columns if c.startswith(("p_lag","p_sum","pet_mean","t_mean"))]
SETS = {
  "full  (7 lags + 7 rolling rain windows + PET/temp windows)": [c for c in DATA.columns if c!="y"],
  "no memory (same-day forcing + statics + season only)":       [c for c in DATA.columns if c!="y" and c not in MEMORY],
}
def nse_by_catchment(obs,pred,gid):
    d=pd.DataFrame({"gid":gid,"obs":obs,"pred":pred})
    def f(g):
        o,p=g.obs.values,g.pred.values; den=((o-o.mean())**2).sum()
        return 1-((o-p)**2).sum()/den if den>0 else np.nan
    return d.groupby("gid").apply(f,include_groups=False)

for name,feats in SETS.items():
    t0=time.time()
    m=HistGradientBoostingRegressor(max_iter=400,learning_rate=0.08,max_leaf_nodes=63,
        min_samples_leaf=100,l2_regularization=1.0,early_stopping=True,
        validation_fraction=0.1,random_state=0)
    m.fit(DATA.loc[tr,feats], DATA.loc[tr,"y"])
    p=m.predict(DATA.loc[te,feats])
    s=nse_by_catchment(DATA.loc[te,"y"].values,p,GID[np.asarray(te)])
    print(f"{name}\n   features={len(feats):2d}  fit={time.time()-t0:.0f}s  "
          f"median NSE={s.median():+.3f}  mean={s.mean():+.3f}  NSE<0: {int((s<0).sum())}")
