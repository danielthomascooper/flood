"""TIGGE ECMWF 50-member ensemble -> catchment-mean forecast rain (Phase 7e).

Reads every cube in cache/nwp/tigge (monthly and daily chunks, inits
de-duplicated), puts each member through the boundary-weighted catchment
mean used for GEFS (0.5-degree cells here), then reduces across members.
Coverage is whatever the drip has delivered so far; consumers score on
the `covered` rows only.

Outputs (index (gid, date), float32, same shape as the GEFS parquets so
hgb_forecast_*.py take them via --fcrain / argv unchanged):
  tigge_catchment_leads_mean.parquet   p_fc1..3 = member mean, s_fc1..3 = std
  tigge_catchment_leads_q{90,95,98}.parquet  p_fc1..3 = member percentile
  tigge_catchment_leads_max.parquet    p_fc1..3 = member max
in cache/nwp/ plus committed copies under experiments/results/nwp/.
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

paths = sorted((ROOT / "cache/nwp/tigge").glob("ecmwf_pf_*.nc"))
cubes = [xr.open_dataset(p)[["tp"]].load() for p in paths]
ds = xr.concat(cubes, dim="init").sortby("init")
_, keep = np.unique(ds.init.values, return_index=True)
ds = ds.isel(init=np.sort(keep))
print(f"{len(paths)} cubes -> {ds.sizes['init']} unique init days, "
      f"{ds.sizes['member']} members, {str(ds.init.values[0])[:10]}..{str(ds.init.values[-1])[:10]}")
lats, lons = ds.lat.values, ds.lon.values
half = float(abs(np.diff(lats)[0])) / 2

shp = gpd.read_file(ROOT / "data/Catchment_Boundaries/camels_gb_v2_catchment_boundaries.shp")
idcol = next(c for c in shp.columns if "id" in c.lower())
shp[idcol] = pd.to_numeric(shp[idcol], errors="coerce")
shp = shp.set_index(idcol).to_crs(27700)
gids = [g for g in good_catchments() if g in shp.index]
cells = gpd.GeoDataFrame({"cell": range(len(lats) * len(lons))},
                         geometry=[box(lo - half, la - half, lo + half, la + half)
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
print(f"cells per catchment: median {int(np.median((W > 0).sum(1)))}, max {(W > 0).sum(1).max()}")

ni, nm, nl = ds.sizes["init"], ds.sizes["member"], ds.sizes["lead_day"]
tp = ds.tp.values.reshape(ni * nm * nl, -1)              # (init*member*lead, cell)
cm = (tp @ W.T).reshape(ni, nm, nl, len(gids))           # catchment mean per member
del tp, cubes, ds["tp"]

def frame(arr3, prefix):                                 # arr3: (init, lead, gid)
    fr = [pd.DataFrame(arr3[:, i, :], index=ds.init.values, columns=gids)
          .stack().rename(f"{prefix}{L}") for i, L in enumerate(ds.lead_day.values)]
    df = pd.concat(fr, axis=1).astype("float32")
    df.index.names = ["date", "gid"]
    return df.swaplevel().sort_index()

outs = {
    "mean": pd.concat([frame(cm.mean(1), "p_fc"), frame(cm.std(1), "s_fc")], axis=1),
    "q90": frame(np.quantile(cm, 0.9, axis=1), "p_fc"),
    "q95": frame(np.quantile(cm, 0.95, axis=1), "p_fc"),
    "q98": frame(np.quantile(cm, 0.98, axis=1), "p_fc"),
    "max": frame(cm.max(1), "p_fc"),
}
for name, df in outs.items():
    for d in (ROOT / "cache/nwp", ROOT / "experiments/results/nwp"):
        df.to_parquet(d / f"tigge_catchment_leads_{name}.parquet")
    print(f"{name}: {len(df):,} rows; " + ", ".join(
        f"{c} mean {df[c].mean():.2f}" for c in df.columns if c.startswith("p_fc")))
