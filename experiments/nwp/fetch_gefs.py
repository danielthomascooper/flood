"""Fetch GEFS v12 precipitation forecasts for the GB box — Phase 7 data.

Source: NOAA GEFS v12 reforecast (2000-01-01..2019-12-31, 00Z, members
c00,p01-p04) on s3://noaa-gefs-retrospective, and the operational GEFS v12
archive (2020-09-23.., 31 members) on s3://noaa-gefs-pds. Both public domain,
no registration. Files are global 0.25 deg GRIB2 with .idx sidecars, so each
6-hour accumulation record is one ~0.5 MB HTTP range request; only the GB box
(49.5-59.5N, 8W-2E) is kept.

Forecast-day convention (matches the hydrology target): the run initialised
at 00Z on issue day t is available during day t; its 24-48 h accumulation is
the calendar-day total for t+1 (lead_day=1), 48-72 h for t+2, 72-96 h for
t+3. lead_day=0 (0-24 h) is the same-day total, kept for reference. GB rain
days in HadUK are 09-09 UTC; the 9 h offset is accepted at daily scale.

Output: cache/nwp/gefs_{member}_{YYYYMM}.nc  dims (init, lead_day, lat, lon)
in mm/day. Resume-safe (skips months that exist), retries on HTTP errors.

Usage: fetch_gefs.py START END [--member c00] [--leads 3]
       e.g. fetch_gefs.py 2012-01-01 2012-01-31
"""
import argparse, time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache" / "nwp"
OUT.mkdir(parents=True, exist_ok=True)
REFO = "https://noaa-gefs-retrospective.s3.amazonaws.com/GEFSv12/reforecast"
OPER = "https://noaa-gefs-pds.s3.amazonaws.com"
LAT_N, LAT_S, LON_W, LON_E = 59.5, 49.5, 352.0, 2.0     # GB box, 0-360 lon


def idx_lines(url):
    r = requests.get(url + ".idx", timeout=60)
    r.raise_for_status()
    return r.text.strip().splitlines()


def get_range(url, start, end, tries=4):
    hdr = {"Range": f"bytes={start}-{end}" if end is not None else f"bytes={start}-"}
    for k in range(tries):
        try:
            r = requests.get(url, headers=hdr, timeout=180)
            if r.status_code in (200, 206):
                return r.content
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"failed {url} {hdr}")


_GRID = {}


def decode_gb(msg):
    """One GRIB2 message (bytes) -> (GB box array [lat, lon], lat, lon) via
    eccodes directly: milliseconds, no temp files."""
    import eccodes as ec
    h = ec.codes_new_from_message(msg)
    try:
        ni, nj = ec.codes_get(h, "Ni"), ec.codes_get(h, "Nj")
        vals = ec.codes_get_values(h).reshape(nj, ni).astype("float32")
        key = (ni, nj)
        if key not in _GRID:
            lat0 = ec.codes_get(h, "latitudeOfFirstGridPointInDegrees")
            dlat = ec.codes_get(h, "jDirectionIncrementInDegrees")
            lon0 = ec.codes_get(h, "longitudeOfFirstGridPointInDegrees")
            dlon = ec.codes_get(h, "iDirectionIncrementInDegrees")
            lat = lat0 - dlat * np.arange(nj)
            lon = (lon0 + dlon * np.arange(ni)) % 360
            li = np.where((lat <= LAT_N) & (lat >= LAT_S))[0]
            lo = np.where((lon >= LON_W) | (lon <= LON_E))[0]
            lon_w = ((lon[lo] + 180) % 360) - 180
            order = np.argsort(lon_w)
            _GRID[key] = (li, lo[order], lat[li], lon_w[order])
        li, lo, lat, lon = _GRID[key]
        return vals[np.ix_(li, lo)], lat, lon
    finally:
        ec.codes_release(h)


def six_hour_labels(lead_days, min_lead=0):
    out = {}
    for d in range(min_lead, lead_days + 1):
        h0 = 24 * d
        out[d] = [f"{h0+6*k}-{h0+6*k+6} hour acc fcst" for k in range(4)]
    return out


def fetch_day(init, member, lead_days, min_lead=0):
    ymd = init.strftime("%Y%m%d")
    labels = six_hour_labels(lead_days, min_lead)
    totals, lat, lon = {}, None, None
    if init <= pd.Timestamp("2019-12-31"):
        url = f"{REFO}/{init.year}/{ymd}00/{member}/Days%3A1-10/apcp_sfc_{ymd}00_{member}.grib2"
        lines = idx_lines(url)
        offs = [int(l.split(":")[1]) for l in lines] + [None]
        need = {lab: i for i, l in enumerate(lines)
                for d in labels for lab in labels[d] if f":{lab}:" in l}
        missing = [lab for d in labels for lab in labels[d] if lab not in need]
        if missing:
            raise KeyError(f"idx lacks {missing[:2]} for {ymd} {member}")
        # only the 6-h records are needed: fetch each individually (half the
        # bytes of the contiguous block; the line, not latency, is the limit)
        for d, labs in labels.items():
            acc = None
            for lab in labs:
                i = need[lab]
                msg = get_range(url, offs[i], (offs[i + 1] - 1) if offs[i + 1] else None)
                arr, lat, lon = decode_gb(msg)
                acc = arr if acc is None else acc + arr
            totals[d] = acc
    else:
        mem = "gec00" if member == "c00" else f"gep{member[1:]}"
        for d, labs in labels.items():
            acc = None
            for k in range(4):
                fh = 24 * d + 6 * (k + 1)
                url = f"{OPER}/gefs.{ymd}/00/atmos/pgrb2sp25/{mem}.t00z.pgrb2s.0p25.f{fh:03d}"
                lines = idx_lines(url)
                offs = [int(l.split(":")[1]) for l in lines] + [None]
                i = next(j for j, l in enumerate(lines) if ":APCP:" in l)
                msg = get_range(url, offs[i], (offs[i + 1] - 1) if offs[i + 1] else None)
                arr, lat, lon = decode_gb(msg)
                acc = arr if acc is None else acc + arr
            totals[d] = acc
    return totals, lat, lon


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("start"); ap.add_argument("end")
    ap.add_argument("--member", default="c00")
    ap.add_argument("--leads", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--min-lead", type=int, default=1,
                    help="first lead_day to keep (default 1: skip the same-day 0-24 h total)")
    ap.add_argument("--out", default=None, help="output dir (default cache/nwp)")
    a = ap.parse_args()
    global OUT
    if a.out:
        OUT = Path(a.out); OUT.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(a.start, a.end, freq="D")
    for ym, chunk in days.to_series().groupby(days.strftime("%Y%m")):
        out = OUT / f"gefs_{a.member}_{ym}.nc"
        if out.exists():
            print(f"{ym}: exists, skipping", flush=True); continue
        t0 = time.time(); cube, inits, lat, lon = [], [], None, None
        from concurrent.futures import ThreadPoolExecutor
        def one(init):
            for attempt in range(3):
                try:
                    return init, fetch_day(init, a.member, a.leads, a.min_lead)
                except Exception as e:
                    err = e; time.sleep(5 * (attempt + 1))
            print(f"  {init.date()}: FAILED ({err})", flush=True); return init, None
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            for init, res in ex.map(one, list(chunk)):
                if res is None: continue
                totals, lat, lon = res
                cube.append(np.stack([totals[d] for d in range(a.min_lead, a.leads + 1)]))
                inits.append(init)
        if not cube:
            continue
        da = xr.DataArray(np.stack(cube), dims=("init", "lead_day", "lat", "lon"),
                          coords={"init": inits, "lead_day": np.arange(a.min_lead, a.leads + 1),
                                  "lat": lat, "lon": lon},
                          attrs={"units": "mm/day", "member": a.member,
                                 "source": "NOAA GEFS v12 (reforecast <=2019, operational >=2020-09-23)",
                                 "convention": "lead_day d = calendar-day total for init+d, from 00Z run"})
        da.to_dataset(name="tp").to_netcdf(
            out, encoding={"tp": {"zlib": True, "complevel": 4, "dtype": "float32"}})
        print(f"{ym}: {len(inits)} inits, leads {a.min_lead}-{a.leads}, {time.time()-t0:.0f}s -> {out.name}", flush=True)


if __name__ == "__main__":
    main()
