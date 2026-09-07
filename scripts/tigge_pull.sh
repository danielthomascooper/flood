#!/bin/bash
# TIGGE drip-pull, restart-safe: chunks are skipped once their .nc exists, so
# this script can die and restart at any point. Day-per-request per ECMWF
# advice (one MARS tape per job); months already pulled whole are skipped.
# Loops until nothing is missing, so chunks that fail on a transient ECDS
# outage are retried on the next pass instead of being silently dropped.
# On start, cancels ghost jobs left in the account's single ECDS slot by a
# previous process - otherwise the new pull queues behind them for hours.
cd /home/habrt/source/flood
PY=.venv/bin/python
FT=experiments/nwp/fetch_tigge.py
$PY experiments/nwp/ecds_cancel_jobs.py
while :; do
  $PY -u $FT 2020-01-01 2020-09-30 --chunk day --workers 1 --leads 3
  $PY -u $FT 2010-10-01 2022-09-30 --chunk day --workers 1 --leads 3
  left=$($PY $FT 2010-10-01 2022-09-30 --chunk day --check)
  echo "pass complete $(date): $left day-chunks still missing"
  [ "$left" -eq 0 ] && break
  sleep 600
done
echo "TIGGE pull complete"
