#!/bin/bash
# Phase 7e: TIGGE (partial drip) -> catchment parquets -> tree point runs -> ladder.
# Sequential on purpose (11 GB box). Resume-safe: every stage skips existing outputs.
cd /home/habrt/source/flood
PY=.venv/bin/python
set -o pipefail
echo "=== chain start $(date)"
[ -f cache/nwp/tigge_catchment_leads_max.parquet ] || $PY -u experiments/nwp/tigge_catchment.py || exit 1
for L in L1 L2 L3; do
  $PY -u experiments/hgb_forecast_gefs.py $L cache/nwp/tigge_catchment_leads_mean.parquet _tigge || exit 1
done
for L in L1 L2 L3; do
  $PY -u experiments/hgb_forecast_gefs.py $L cache/nwp/tigge_catchment_leads_max.parquet _tiggemax || exit 1
done
# ladder: no saved models yet -> refits once (saves joblib), predicts every rain source
for q in q50 q90 q95 q99; do
  mv -f experiments/results/forecast_${q}_L1.parquet experiments/results/forecast_${q}_L1.gefsonly.parquet 2>/dev/null
  $PY -u experiments/hgb_forecast_quantiles.py $q || exit 1
done
echo "=== CHAIN DONE $(date)"
