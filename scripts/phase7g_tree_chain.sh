#!/bin/bash
# Phase 7g tree chain: forecast-rain-trained variants (points L1-3), then the
# lead-1 ladders for the three forecast-trained regimes. Sequential (11 GB box).
cd /home/habrt/source/flood
PY=.venv/bin/python
echo "=== chain start $(date)"
for L in L1 L2 L3; do $PY -u experiments/hgb_forecast_fctrain.py $L || exit 1; done
for v in fc_2000 fcs_2000 mixeds; do
  for q in q50 q90 q95 q99; do $PY -u experiments/hgb_forecast_fctrain.py Q${v}_${q} || exit 1; done
done
echo "=== CHAIN DONE $(date)"
