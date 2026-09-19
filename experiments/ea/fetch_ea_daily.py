"""Pull EA Hydrology API daily series for the CAMELS-GB catchments that have
an EA level station (219 of 416; Scotland/Wales are SEPA/NRW, not in this API).

The practical-forecast pivot: users think in LEVEL, and a daily MEAN is not a
level forecast. Per station this pulls the qualified daily series
  level_max  (the target: tomorrow's peak stage, m, station datum)
  level_min, flow_max, flow_mean  (peak/mean ratio; flow->level comparison route)
with the EA quality flag for each, 1950s -> present, open government licence.

Input catalogue: cache/ea/measures_level_stations.parquet (built from
/hydrology/id/stations?observedProperty=waterLevel). Output: one parquet per
NRFA id, cache/ea/daily/<nrfa>.parquet, indexed by date. Resume-safe.
"""
import io, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import good_catchments

OUT = ROOT / "cache/ea/daily"; OUT.mkdir(parents=True, exist_ok=True)
API = "https://environment.data.gov.uk/hydrology/id/measures/{}/readings.csv"
WANT = {("level", "maximum"): "level_max", ("level", "minimum"): "level_min",
        ("flow", "maximum"): "flow_max", ("flow", "mean"): "flow_mean"}

M = pd.read_parquet(ROOT / "cache/ea/measures_level_stations.parquet")
M = M[M.nrfa.isin(set(good_catchments())) & (M.period == 86400) & M.measure.str.endswith("qualified")]


def series(measure):
    for attempt in range(4):
        try:
            r = requests.get(API.format(measure), params={"_limit": 2000000}, timeout=600)
            r.raise_for_status()
            d = pd.read_csv(io.StringIO(r.text), usecols=["date", "value", "quality"], parse_dates=["date"])
            return d.drop_duplicates("date").set_index("date").sort_index()
        except Exception as e:
            err = e; time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"{measure}: {err}")


ids = sorted(M.nrfa.unique())
print(f"{len(ids)} catchments", flush=True)
for n, nrfa in enumerate(ids, 1):
    dst = OUT / f"{int(nrfa)}.parquet"
    if dst.exists():
        continue
    t0 = time.time(); cols = {}
    sub = M[M.nrfa == nrfa]
    # a gauge can map to >1 EA station: take the one with the most level_max data
    best = None
    for st, g in sub.groupby("station"):
        got = {}
        for (par, stat), name in WANT.items():
            mm = g[(g.parameter == par) & (g.stat == stat)].measure
            if len(mm):
                got[name] = series(mm.iloc[0])
        if "level_max" in got and (best is None or got["level_max"].value.notna().sum() > best[1]["level_max"].value.notna().sum()):
            best = (st, got)
    if best is None:
        print(f"[{n}/{len(ids)}] {int(nrfa)}: no level_max series", flush=True); continue
    st, got = best
    df = pd.concat({k: v.value for k, v in got.items()}, axis=1).astype("float32")
    for k, v in got.items():
        df[f"{k}_q"] = v.quality.reindex(df.index).astype("category")
    df.attrs["station"] = st
    df.to_parquet(dst)
    print(f"[{n}/{len(ids)}] {int(nrfa)} ({st[:8]}): {df.level_max.notna().sum():,} level days "
          f"{df.index.min().date()}..{df.index.max().date()} in {time.time()-t0:.0f}s", flush=True)
print(f"EA DAILY PULL DONE {pd.Timestamp.now()}", flush=True)
