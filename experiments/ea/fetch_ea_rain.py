"""Pull EA Hydrology API DAILY rainfall for every rainfall station - Phase 8c.

The research models are driven by HadUK-Grid observed rain, which is
published months in arrears. At issue time the observed-rain window has
to come from something live: the EA tipping-bucket network (995 stations,
15-min and daily, most open since the 1990s-2000s, same API as the level
data). This pulls the qualified daily totals (stamped 09:00 = the water
day ending then; alignment to HadUK is decided empirically at join time)
into one parquet per station: cache/ea/rain_daily/<station>.parquet with
columns value (mm), quality. Resume-safe.
"""
import io, json, sys, time
from pathlib import Path
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cache/ea/rain_daily"; OUT.mkdir(parents=True, exist_ok=True)
API = "https://environment.data.gov.uk/hydrology/id/measures/{}/readings.csv"
items = json.loads((ROOT / "cache/ea/stations_rainfall.json").read_text())["items"]
meta = []
for s in items:
    ms = [m for m in s.get("measures", []) if m.get("period") == 86400 and str(m.get("@id", "")).endswith("qualified")]
    if not ms:
        continue
    meta.append(dict(station=s["notation"], label=s.get("label"), lat=s.get("lat"), lon=s.get("long"),
                     opened=s.get("dateOpened"), measure=ms[0]["@id"].rsplit("/", 1)[-1]))
M = pd.DataFrame(meta)
for c in ("lat", "lon"):
    M[c] = pd.to_numeric(M[c].map(lambda v: v[0] if isinstance(v, list) else v), errors="coerce")
M.to_parquet(ROOT / "cache/ea/rain_stations.parquet")
print(f"{len(M)} rainfall stations with a daily qualified series", flush=True)
for n, r in enumerate(M.itertuples(), 1):
    dst = OUT / f"{r.station}.parquet"
    if dst.exists():
        continue
    t0 = time.time()
    for attempt in range(4):
        try:
            resp = requests.get(API.format(r.measure), params={"_limit": 2000000}, timeout=600)
            resp.raise_for_status()
            if not resp.text.strip():
                d = None; break
            d = pd.read_csv(io.StringIO(resp.text), usecols=["dateTime", "date", "value", "quality"],
                            parse_dates=["dateTime", "date"])
            break
        except Exception as e:
            err = e; time.sleep(30 * (attempt + 1))
    else:
        print(f"[{n}/{len(M)}] {r.station}: FAILED {err}", flush=True); continue
    if d is None or d.empty:
        print(f"[{n}/{len(M)}] {r.station}: empty", flush=True); continue
    d = d.drop_duplicates("date").set_index("date").sort_index()
    d[["value"]].astype("float32").assign(quality=d.quality.astype("category")).to_parquet(dst)
    print(f"[{n}/{len(M)}] {r.station}: {len(d):,} days {d.index.min().date()}..{d.index.max().date()} in {time.time()-t0:.0f}s", flush=True)
print(f"EA RAIN PULL DONE {pd.Timestamp.now()}", flush=True)
