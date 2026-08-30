#!/bin/bash
# GEFS control-member pull, restart-safe (monthly cubes skipped once written).
cd /home/habrt/source/flood
PY=.venv/bin/python
$PY -u experiments/nwp/fetch_gefs.py 2010-10-01 2019-12-31 --leads 3 --min-lead 1 --workers 4
$PY -u experiments/nwp/fetch_gefs.py 2020-09-23 2022-09-30 --leads 3 --min-lead 1 --workers 4
echo "GEFS pull complete"
