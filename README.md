# flood

Exploration of **CAMELS-GB v2** — 671 British river catchments with static
attributes, boundary polygons, and daily/hourly meteorology and streamflow.

Source: [EIDC 9a46d428-958f-4ac1-86eb-94eee70c0955](https://catalogue.ceh.ac.uk/documents/9a46d428-958f-4ac1-86eb-94eee70c0955)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name flood --display-name "flood (.venv)"
```

`requirements.txt` is unpinned. To reproduce the exact environment the notebooks
were executed in, use `requirements-lock.txt` instead — the notebooks rely on
pandas 3.x behaviour in a couple of places (`stack()` retains NaN, and the
`on_bad_lines` callable used to repair the hydrometry CSV).

The data itself is not in git — it is 10.6 GiB. See **Re-downloading** below.

## Run

```bash
.venv/bin/jupyter lab notebooks/
```

Select the **flood (.venv)** kernel.

| Notebook | What it does |
|---|---|
| `01_explore_camels_gb.ipynb` | Walks the data: inventory, maps, hydrological signatures, land cover, time series, record completeness, flood extremes, hourly vs daily, groundwater |
| `02_field_reference.ipynb` | Every field in every file — dtype, units, meaning, source, measured range and levels — plus cross-checks of the documentation against the data |

## Layout

```
data/                          1,436 files, 10.6 GiB (not in git)
  Catchment_Attributes/        9 CSVs — topography, climate, hydrology, soil,
                               land cover, hydrogeology, human influence,
                               hydrometry, groundwater wells
  Catchment_Boundaries/        shapefile, 671 polygons, EPSG:27700
  Catchment_Timeseries/
    hydro-meteorological/
      daily/                   671 files, 1970-10-01 → 2022-09-30 (754 MiB)
      hourly/                  671 files, 1990-10-01 → 2022-10-01 (9.9 GiB)
    groundwater/
      monthly/                 55 wells
      daily/                   23 wells
cache/                         Parquet caches built by the notebooks (not in git)
notebooks/
  01_explore_camels_gb.ipynb   inventory → attributes → maps → signatures →
                               time series → flood extremes
  02_field_reference.ipynb     field-by-field data dictionary + doc-vs-data checks
reference/
  camels_gb_v2_data_dictionary.csv   263 fields: description, unit, source
                                     (extracted from the publisher's doc)
  camels_gb_v2_field_profile.csv     the same, joined to measured dtype/range/levels
  camels_gb_v2_eidc_supp_info.docx   the publisher's supporting information
urls.txt / urls-aria.txt       download manifests (aria2 -i urls-aria.txt)
```

## Quirks in the published data

All found while building the notebooks; none is a download problem — every file
matches the EIDC catalogue byte for byte.

1. **`camels_gb_v2_hydrometry_attributes.csv` is malformed.** The file uses no
   quoting, and two rows (gauges 27038 and 42010) contain an unescaped comma in
   the final free-text comment column — 34 fields where the header declares 33.
   Pandas' default C parser aborts the whole file. The notebook's `read_attr`
   falls back to the python engine and folds the overflow back into the last
   column. No other file in the dataset is ragged.

2. **`discharge_spec` changes units between folders** — mm/**day** in the daily
   files, mm/**hour** in the hourly ones, under the same column name and with
   nothing in the CSVs to say so. Comparing them directly is silently wrong by a
   factor of 24. `discharge_vol` (m³/s) is consistent across both. It *is* in the
   publisher's supporting information; the notebooks demonstrate the 1/24 ratio
   rather than asking you to take it on trust.

3. **Two documentation errors**, found by checking the supporting information
   against the files (notebook 02, §12): the hourly `precipitation_cehgear`
   column is documented as running to 31 Dec 2019 but **ends 2016-12-31**, and the
   document states 56 monthly groundwater files where there are **55** (the
   attribute table agrees with the files, not the document).

4. **`gauge_lat` / `gauge_lon` are rounded to 2 decimal places** — about half a
   kilometre. `gauge_easting` / `gauge_northing` are exact metres; use those.

## Re-downloading

```bash
aria2c -i urls-aria.txt -c -x4 -j8
```

Verify against the source listing: every file should match the catalogue's byte
count (total 11,401,178,260 bytes across 1,436 files).
