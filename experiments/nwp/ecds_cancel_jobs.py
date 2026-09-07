"""Cancel every queued/running ECDS job on this account.

ECDS runs one request per account at a time; jobs submitted by a killed
fetcher keep their place in that slot server-side (a monthly TIGGE job can
sit 'running' for 5+ h before failing), so a restarted drip waits behind
ghosts. Run this at wrapper start. Reads ~/.cdsapirc.
"""
import re, sys
from pathlib import Path
import requests

cfg = dict(re.findall(r"^(\w+):\s*(.+)$", Path.home().joinpath(".cdsapirc").read_text(), re.M))
url, key = cfg["url"].rstrip("/"), cfg["key"]
H = {"PRIVATE-TOKEN": key}
jobs = requests.get(f"{url}/retrieve/v1/jobs", headers=H, params={"limit": 50}, timeout=60).json().get("jobs", [])
live = [j for j in jobs if j.get("status") in ("accepted", "running")]
for j in live:
    r = requests.delete(f"{url}/retrieve/v1/jobs/{j['jobID']}", headers=H, timeout=60)
    print(f"cancelled {j['jobID'][:8]} ({j['status']}): HTTP {r.status_code}")
print(f"{len(live)} live job(s) cancelled")
