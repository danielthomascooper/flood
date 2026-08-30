"""Fetch ECMWF ensemble precipitation forecasts (TIGGE via ECDS) for the GB box.

Track A of Phase 7. Dataset `tigge-forecasts` on https://ecds.ecmwf.int
(needs ~/.cdsapirc and the TIGGE licence accepted). Requests are served
from tape, so requests are chunked (one month or one day per request) and
several run concurrently; each chunk is saved as GRIB and converted to the
same cube layout the GEFS fetcher writes, plus a member dimension:

  cache/nwp/tigge/ecmwf_pf_YYYYMM.nc   tp(init, member, lead_day, lat, lon) mm/day

Convention identical to fetch_gefs.py: the 00Z run of issue day t;
lead_day d = calendar-day total for t+d, obtained by differencing the
step-accumulated tp at 24d and 24(d+1) hours. Perturbed members only (the
control-forecast surface fields were lost for 710 dates in 2006-2019).

Usage: fetch_tigge.py START END [--chunk month|day] [--workers 4] [--leads 3]
"""
import argparse, calendar, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / "nwp" / "tigge"
OUT.mkdir(parents=True, exist_ok=True)


def request(year, month, days, leads):
    return {"origin": "ecmwf", "level_type": "single_level",
            "variable": ["total_precipitation"], "forecast_type": "perturbed_forecast",
            "year": [f"{year}"], "month": [f"{month:02d}"], "day": [f"{d:02d}" for d in days],
            "time": "00:00:00",
            "leadtime_hour": [str(24 * k) for k in range(1, leads + 2)],
            "grid": "0.5/0.5", "area": [61, -11, 49, 2], "data_format": "grib"}


def grib_to_cube(path, leads):
    ds = xr.open_dataset(path, engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds.tp                                   # kg m-2 == mm, accumulated from step 0
    if "time" not in tp.dims:                    # single date -> add init dim
        tp = tp.expand_dims("time")
    day = tp.diff("step")                        # lead days 1..leads
    day = day.rename({"time": "init", "number": "member", "step": "lead_day",
                      "latitude": "lat", "longitude": "lon"})
    day = day.assign_coords(lead_day=np.arange(1, leads + 1)).clip(min=0).astype("float32")
    return day.transpose("init", "member", "lead_day", "lat", "lon")


def fetch_chunk(year, month, days, leads, tag):
    import cdsapi
    grib = OUT / f"ecmwf_pf_{tag}.grib"
    nc = OUT / f"ecmwf_pf_{tag}.nc"
    if nc.exists():
        return tag, "exists"
    t0 = time.time()
    for attempt in range(3):
        try:
            cdsapi.Client(quiet=True).retrieve("tigge-forecasts",
                                               request(year, month, days, leads), str(grib))
            break
        except Exception as e:
            err = e; time.sleep(60 * (attempt + 1))
    else:
        return tag, f"FAILED ({err})"
    cube = grib_to_cube(grib, leads)
    cube.attrs.update(units="mm/day", source="ECMWF ENS via TIGGE/ECDS, 0.5 deg, pf members",
                      convention="lead_day d = calendar-day total for init+d, 00Z run")
    cube.to_dataset(name="tp").to_netcdf(nc, encoding={"tp": {"zlib": True, "complevel": 4}})
    grib.unlink()
    return tag, f"{cube.sizes['init']} inits x {cube.sizes['member']} members in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--chunk", choices=["month", "day"], default="month")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--leads", type=int, default=3)
    a = ap.parse_args()
    days = pd.date_range(a.start, a.end, freq="D")
    jobs = []
    if a.chunk == "month":
        for ym, chunk in days.to_series().groupby(days.strftime("%Y%m")):
            y, m = int(ym[:4]), int(ym[4:])
            jobs.append((y, m, [d.day for d in chunk], ym))
    else:
        for d in days:
            jobs.append((d.year, d.month, [d.day], d.strftime("%Y%m%d")))
    print(f"{len(jobs)} {a.chunk} requests, {a.workers} concurrent", flush=True)
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for tag, msg in ex.map(lambda j: fetch_chunk(*j[:3], a.leads, j[3]), jobs):
            print(f"  {tag}: {msg}", flush=True)


if __name__ == "__main__":
    main()
