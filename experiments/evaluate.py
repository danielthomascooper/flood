"""Shared evaluation harness for CAMELS-GB streamflow models.

The single clearest result of the baseline run was that NSE hides everything
flood work cares about (+0.82 overall vs -0.81 on the top 1% of flows, same
predictions). So every model reports the same card, and the card separates
"fits ordinary days" from "fits floods":

  whole record   NSE, KGE, % catchments NSE<0
  the tail       top-1% NSE, top-1% bias, AMAX bias, q99 distribution bias
  the ceiling    largest value the model ever outputs; behaviour on days
                 exceeding the training-period maximum
  calibration    coverage = fraction of obs <= pred (reads ~0.5 for a mean
                 model, ~alpha for a quantile model)

Input format everywhere: a DataFrame indexed by date with columns
gid / obs / pred, plus optionally a per-catchment train_max Series.
"""
import numpy as np
import pandas as pd


# ---- per-catchment metrics -------------------------------------------------

def nse(obs, pred):
    d = ((obs - obs.mean()) ** 2).sum()
    return 1 - ((obs - pred) ** 2).sum() / d if d > 0 else np.nan


def kge(obs, pred):
    """Kling-Gupta efficiency (2009): 1 - sqrt((r-1)^2+(a-1)^2+(b-1)^2)."""
    if obs.std() == 0 or pred.std() == 0 or obs.mean() == 0:
        return np.nan
    r = np.corrcoef(obs, pred)[0, 1]
    a = pred.std() / obs.std()
    b = pred.mean() / obs.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (a - 1) ** 2 + (b - 1) ** 2)


def _top1(g):
    return g[g.obs >= g.obs.quantile(0.99)]


def per_catchment(res):
    """One row of metrics per catchment."""
    def one(g):
        o, p = g.obs.values, g.pred.values
        h = _top1(g)
        row = {
            "nse": nse(o, p),
            "kge": kge(o, p),
            "top1_nse": nse(h.obs.values, h.pred.values) if len(h) > 5 else np.nan,
            "top1_bias_pct": (h.pred.mean() / h.obs.mean() - 1) * 100 if len(h) else np.nan,
            "q99_dist_bias_pct": (np.quantile(p, 0.99) / np.quantile(o, 0.99) - 1) * 100
                                 if np.quantile(o, 0.99) > 0 else np.nan,
            "coverage": float((o <= p).mean()),
        }
        return pd.Series(row)
    return res.groupby("gid").apply(one, include_groups=False)


def amax_bias(res, min_days=350):
    """Median % error in the annual maximum, over catchment-years with enough
    data. This is the flood-frequency-relevant number: AMAX is what a GEV
    gets fitted to."""
    wy = res.index.year + (res.index.month >= 10).astype(int)
    g = res.assign(wy=wy).groupby(["gid", "wy"])
    ok = g.obs.count() >= min_days
    b = (g.pred.max() / g.obs.max() - 1) * 100
    return b[ok]


# ---- the card --------------------------------------------------------------

def evaluate(res, name, train_max=None):
    """Summarise one model into a single comparable row."""
    pc = per_catchment(res)
    ab = amax_bias(res)
    row = {
        "model": name,
        "n_catchments": len(pc),
        "median_NSE": pc.nse.median(),
        "median_KGE": pc.kge.median(),
        "pct_NSE_neg": (pc.nse < 0).mean() * 100,
        "top1_NSE": pc.top1_nse.median(),
        "top1_bias_pct": pc.top1_bias_pct.median(),
        "AMAX_bias_pct": ab.median(),
        "q99_dist_bias_pct": pc.q99_dist_bias_pct.median(),
        "pred_max": res.pred.max(),
        "obs_max": res.obs.max(),
        "coverage": pc.coverage.median(),
    }
    if train_max is not None:
        r2 = res.join(train_max.rename("train_max"), on="gid")
        beyond = r2[r2.obs > r2.train_max]
        row["days_beyond_train_max"] = len(beyond)
        row["pred_over_obs_beyond"] = (beyond.pred / beyond.obs).median() if len(beyond) else np.nan
    return row, pc


def report(rows, floatfmt="{:+.3f}"):
    """Print a comparison table across models."""
    df = pd.DataFrame(rows).set_index("model")
    with pd.option_context("display.width", 200, "display.max_columns", 30,
                           "display.float_format", lambda v: f"{v:+.3f}"):
        print(df.to_string())
    return df
