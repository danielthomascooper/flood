#!/bin/bash
# TIGGE drip-pull, restart-safe: monthly chunks are skipped once their .nc
# exists, so this script can die and restart at any point. Priority order:
# 1. the 2020-01..09 hole GEFS cannot fill; 2. the full 2010-2022 window.
cd /home/habrt/source/flood
PY=.venv/bin/python
$PY -u experiments/nwp/fetch_tigge.py 2020-01-01 2020-09-30 --chunk month --workers 2 --leads 3
$PY -u experiments/nwp/fetch_tigge.py 2010-10-01 2022-09-30 --chunk month --workers 2 --leads 3
echo "TIGGE pull complete"
