"""Archive ECMWF open-data ENS precipitation for the GB box - daily.

ECMWF's real-time open data (CC BY 4.0, 0.25 deg, 50 perturbed members)
is the operational rain feed a practical product would run on, but ECMWF
keeps only a few days online and there is no public archive. This pulls
the 00Z ENS total precipitation at steps 24/48/72/96 h, crops the GB box
and writes one cube per issue day in the same layout as the TIGGE cubes:
  cache/nwp/ecmwf_od/ens_YYYYMMDD.nc   tp(init, member, lead_day, lat, lon) mm/day
Resume-safe (skips days that exist); intended to run from a daily timer.
Usage: fetch_ecmwf_opendata.py [YYYY-MM-DD ...]   (default: yesterday and today)
"""
import sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from ecmwf.opendata import Client

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / "nwp" / "ecmwf_od"
OUT.mkdir(parents=True, exist_ok=True)
LAT = slice(61, 49); LON = slice(-11, 2)          # lat descending in the grib
STEPS = [24, 48, 72, 96]


def fetch(day):
    tag = day.strftime("%Y%m%d")
    nc = OUT / f"ens_{tag}.nc"
    if nc.exists():
        return f"{tag}: exists"
    grib = OUT / f"ens_{tag}.grib2"
    t0 = time.time()
    try:
        Client(source="ecmwf").retrieve(
            date=day.strftime("%Y-%m-%d"), time=0, stream="enfo", type="pf",
            step=STEPS, param="tp", number=list(range(1, 51)), target=str(grib))
    except Exception as e:
        return f"{tag}: not available ({str(e)[:80]})"
    ds = xr.open_dataset(grib, engine="cfgrib", backend_kwargs={"indexpath": ""})
    tp = ds.tp.sel(latitude=LAT, longitude=LON) * 1000.0     # m -> mm, accumulated
    if "time" not in tp.dims:
        tp = tp.expand_dims("time")
    day_tot = tp.diff("step").rename({"time": "init", "number": "member", "step": "lead_day",
                                      "latitude": "lat", "longitude": "lon"})
    day_tot = day_tot.assign_coords(lead_day=np.arange(1, len(STEPS))).clip(min=0).astype("float32")
    day_tot = day_tot.transpose("init", "member", "lead_day", "lat", "lon")
    day_tot.attrs.update(units="mm/day", source="ECMWF open data ENS pf, 0.25 deg, 00Z",
                         convention="lead_day d = calendar-day total for init+d")
    day_tot.to_dataset(name="tp").to_netcdf(nc, encoding={"tp": {"zlib": True, "complevel": 4}})
    grib.unlink()
    return f"{tag}: {day_tot.sizes['member']} members x {day_tot.sizes['lead_day']} leads in {time.time()-t0:.0f}s"


if __name__ == "__main__":
    days = [pd.Timestamp(a) for a in sys.argv[1:]] or \
        [pd.Timestamp.utcnow().normalize() - pd.Timedelta(days=1), pd.Timestamp.utcnow().normalize()]
    for d in days:
        print(fetch(d), flush=True)
