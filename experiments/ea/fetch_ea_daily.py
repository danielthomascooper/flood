"""Pull EA Hydrology API daily series for the CAMELS-GB catchments that have
an EA level station (219 of 416; Scotland/Wales are SEPA/NRW, not in this API).

The practical-forecast pivot: users think in LEVEL, and a daily MEAN is not a
level forecast. Per station this pulls the qualified daily series
  level_max  (the target: tomorrow's peak stage, m, station datum)
  level_min, flow_max, flow_mean  (peak/mean ratio; flow->level comparison route)
with the EA quality flag for each, 1950s -> present, open government licence.

Input catalogue: cache/ea/measures_level_stations.parquet (built from
/hydrology/id/stations?observedProperty=waterLevel). Output: one parquet per
NRFA id, cache/ea/daily_wd/<nrfa>.parquet, indexed by WATER DAY (09:00 start). Resume-safe.
"""
import io, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import good_catchments

OUT = ROOT / "cache/ea/daily_wd"; OUT.mkdir(parents=True, exist_ok=True)
API = "https://environment.data.gov.uk/hydrology/id/measures/{}/readings.csv"
WANT = {("level", "maximum"): "level_max", ("level", "minimum"): "level_min",
        ("flow", "maximum"): "flow_max", ("flow", "mean"): "flow_mean"}

M = pd.read_parquet(ROOT / "cache/ea/measures_level_stations.parquet")
M = M[M.nrfa.isin(set(good_catchments())) & (M.period == 86400) & M.measure.str.endswith("qualified")]


def water_day(d, extreme):
    """EA daily statistics cover the WATER DAY 09:00 -> 09:00. Daily MEANS are
    stamped on the day itself, but daily MAX/MIN rows are stamped with the TIME
    THE EXTREME OCCURRED, so a peak at 05:00 on the 25th belongs to the water
    day that began on the 24th while its `date` field says the 25th. Keying on
    `date` mislabels ~15% of days and leaves a hole beside each. Assign each
    extreme to the water day of its timestamp; a reading at exactly 09:00 sits
    on the boundary - it starts its own day unless that day already has a later
    reading and the previous day is empty (a rising limb peaking at day end)."""
    if not extreme:
        return d.drop_duplicates("date").set_index("date")[["value", "quality"]].sort_index()
    d = d.sort_values("dateTime").reset_index(drop=True)
    wd = (d.dateTime - pd.Timedelta(hours=9)).dt.floor("D")
    on_boundary = (d.dateTime.dt.hour == 9) & (d.dateTime.dt.minute == 0)
    dup_next = wd.eq(wd.shift(-1))                       # same water day as the next row
    prev_empty = ~(wd - pd.Timedelta(days=1)).isin(set(wd))
    wd = wd.where(~(on_boundary & dup_next & prev_empty), wd - pd.Timedelta(days=1))
    d = d.assign(date=wd)
    # any residual duplicates: keep the more extreme value
    d = d.sort_values(["date", "value"], ascending=[True, extreme != "max"]).drop_duplicates("date")
    return d.set_index("date")[["value", "quality"]].sort_index()


def series(measure, extreme=None):
    for attempt in range(4):
        try:
            r = requests.get(API.format(measure), params={"_limit": 2000000}, timeout=600)
            r.raise_for_status()
            if not r.text.strip():                 # measure listed but holds no readings
                return None
            d = pd.read_csv(io.StringIO(r.text), usecols=["dateTime", "date", "value", "quality"],
                            parse_dates=["dateTime", "date"])
            return water_day(d, extreme)
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
                ser = series(mm.iloc[0], {"maximum": "max", "minimum": "min"}.get(stat))
                if ser is not None and len(ser):
                    got[name] = ser
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
