"""Join GEFS c00 forecast rain onto catchments — Phase 7.

Nearest 0.25-degree cell to each gauge. Convention: the 00Z run of issue
day t gives lead_day d = the calendar-day rain total for t+d, so a row
indexed (gid, t) carries p_fc1..3 = forecast rain for t+1..t+3, all
available before the end of issue day t (conservative by ~24 h).

Output: cache/nwp/gefs_catchment_leads.parquet with index (gid, date)
and columns p_fc1, p_fc2, p_fc3 (mm/day, float32).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import read_attr, good_catchments

ds = xr.concat([xr.open_dataset(p).load() for p in
                sorted((ROOT / "cache/nwp").glob("gefs_c00_*.nc"))], dim="init")
print(f"{ds.sizes['init']} init days, "
      f"{str(ds.init.values[0])[:10]}..{str(ds.init.values[-1])[:10]}")

topo = read_attr("topographic").set_index("gauge_id")
gids = good_catchments()
la = topo.gauge_lat.reindex(gids).astype(float)
lo = topo.gauge_lon.reindex(gids).astype(float)

pts = ds.tp.sel(lat=xr.DataArray(la.values, dims="gid"),
                lon=xr.DataArray(lo.values, dims="gid"),
                method="nearest")  # (init, lead_day, gid)

frames = []
for i, L in enumerate(ds.lead_day.values):
    df = pd.DataFrame(pts.isel(lead_day=i).values, index=ds.init.values,
                      columns=gids).stack().rename(f"p_fc{L}")
    frames.append(df)
out = pd.concat(frames, axis=1).astype("float32")
out.index.names = ["date", "gid"]
out = out.swaplevel().sort_index()
out.to_parquet(ROOT / "cache/nwp/gefs_catchment_leads.parquet")
print(f"{len(out):,} (gid, day) rows -> cache/nwp/gefs_catchment_leads.parquet")
print(out.describe().round(2).to_string())
