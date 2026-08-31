"""Catchment-mean GEFS rain — Phase 7 upgrade 2.

gefs_join.py took the single 0.25-degree cell nearest the *gauge* — the
outlet, not the catchment. Here each catchment's boundary polygon
(data/Catchment_Boundaries, reprojected to BNG for honest areas) is
intersected with the GEFS grid cells and the forecast rain is the
area-weighted mean over overlapping cells. Small catchments collapse to
one cell (same as before but centred correctly); large ones average many.

Output: cache/nwp/gefs_catchment_leads_mean.parquet, same shape as the
nearest-cell parquet (index (gid, date), columns p_fc1..3, mm/day).
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

ds = xr.concat([xr.open_dataset(p).load() for p in
                sorted((ROOT / "cache/nwp").glob("gefs_c00_*.nc"))], dim="init")
lats, lons = ds.lat.values, ds.lon.values

shp = gpd.read_file(ROOT / "data/Catchment_Boundaries/"
                    "camels_gb_v2_catchment_boundaries.shp")
idcol = next(c for c in shp.columns if "id" in c.lower())
shp[idcol] = pd.to_numeric(shp[idcol], errors="coerce")
shp = shp.set_index(idcol).to_crs(27700)
gids = [g for g in good_catchments() if g in shp.index]
print(f"{len(gids)} catchments with boundaries (of {len(good_catchments())})")

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

tp = ds.tp.values.reshape(ds.sizes["init"] * 3, -1)     # (init*lead, cell)
mean = tp @ W.T                                          # -> (init*lead, gid)
mean = mean.reshape(ds.sizes["init"], 3, len(gids))

frames = [pd.DataFrame(mean[:, i, :], index=ds.init.values, columns=gids)
          .stack().rename(f"p_fc{L}") for i, L in enumerate(ds.lead_day.values)]
out = pd.concat(frames, axis=1).astype("float32")
out.index.names = ["date", "gid"]
out = out.swaplevel().sort_index()
out.to_parquet(ROOT / "cache/nwp/gefs_catchment_leads_mean.parquet")
print(f"{len(out):,} rows -> cache/nwp/gefs_catchment_leads_mean.parquet")
