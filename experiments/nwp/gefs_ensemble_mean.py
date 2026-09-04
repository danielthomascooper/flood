"""5-member GEFS ensemble, catchment-mean rain — Phase 7 upgrade 3.

Members c00 + p01-p04 (the full GEFS v12 reforecast ensemble), each put
through the same boundary-weighted catchment mean as upgrade 2, then
averaged across members. Output columns:

  p_fc1..3  ensemble-mean rain (mm/day)   — drop-in for hgb_forecast_gefs.py
  s_fc1..3  ensemble spread (member std)  — uncertainty feature, stage 2

Init days are aligned on the intersection across members (all five pulls
completed 136/136 monthly cubes, so this should be the full archive).

Output: cache/nwp/gefs_catchment_leads_ens.parquet, same (gid, date)
index as the c00 parquets, plus a committed copy under
experiments/results/nwp/.
"""
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
from common import good_catchments

MEMBERS = ["c00", "p01", "p02", "p03", "p04"]

shp = gpd.read_file(ROOT / "data/Catchment_Boundaries/"
                    "camels_gb_v2_catchment_boundaries.shp")
idcol = next(c for c in shp.columns if "id" in c.lower())
shp[idcol] = pd.to_numeric(shp[idcol], errors="coerce")
shp = shp.set_index(idcol).to_crs(27700)
gids = [g for g in good_catchments() if g in shp.index]
print(f"{len(gids)} catchments with boundaries (of {len(good_catchments())})")

W = None            # built once from the first member's grid
lead_days = None
per_member = []     # list of (init*lead*gid) DataFrames indexed (gid, date)

for m in MEMBERS:
    paths = sorted((ROOT / "cache/nwp").glob(f"gefs_{m}_*.nc"))
    ds = xr.concat([xr.open_dataset(p).load() for p in paths], dim="init")
    print(f"{m}: {len(paths)} cubes, {ds.sizes['init']} init days")

    if W is None:
        lats, lons = ds.lat.values, ds.lon.values
        lead_days = ds.lead_day.values
        cells = gpd.GeoDataFrame(
            {"cell": range(len(lats) * len(lons))},
            geometry=[box(lo - .125, la - .125, lo + .125, la + .125)
                      for la in lats for lo in lons], crs=4326).to_crs(27700)
        sidx = cells.sindex
        W = np.zeros((len(gids), len(cells)), dtype="float32")
        for i, g in enumerate(gids):
            poly = shp.geometry[g]
            cand = sidx.query(poly, predicate="intersects")
            a = cells.geometry.iloc[cand].intersection(poly).area.values
            if a.sum() == 0:
                raise SystemExit(f"catchment {g}: no cell overlap")
            W[i, cells.cell.iloc[cand].values] = a / a.sum()
        print(f"cells per catchment: median {int(np.median((W > 0).sum(1)))}, "
              f"max {(W > 0).sum(1).max()}")
    else:
        assert (ds.lat.values == lats).all() and (ds.lon.values == lons).all()

    nl = ds.sizes["lead_day"]
    tp = ds.tp.values.reshape(ds.sizes["init"] * nl, -1)   # (init*lead, cell)
    cm = (tp @ W.T).reshape(ds.sizes["init"], nl, len(gids))
    frames = [pd.DataFrame(cm[:, i, :], index=ds.init.values, columns=gids)
              .stack().rename(f"L{L}") for i, L in enumerate(ds.lead_day.values)]
    df = pd.concat(frames, axis=1).astype("float32")
    df.index.names = ["date", "gid"]
    per_member.append(df.swaplevel().sort_index())
    del ds, tp, cm

stack = pd.concat(per_member, axis=1, join="inner", keys=MEMBERS)
print(f"aligned rows: {len(stack):,} (member-wise inner join)")

out = {}
for L in lead_days:
    cols = stack.xs(f"L{L}", axis=1, level=1)          # (rows, members)
    out[f"p_fc{L}"] = cols.mean(axis=1)
    out[f"s_fc{L}"] = cols.std(axis=1, ddof=0)
out = pd.DataFrame(out).astype("float32").sort_index()

dst = ROOT / "cache/nwp/gefs_catchment_leads_ens.parquet"
out.to_parquet(dst)
copy = ROOT / "experiments/results/nwp/gefs_catchment_leads_ens.parquet"
out.to_parquet(copy)
print(f"{len(out):,} rows -> {dst.relative_to(ROOT)} (+ committed copy)")
print(out.describe().round(2).to_string())
