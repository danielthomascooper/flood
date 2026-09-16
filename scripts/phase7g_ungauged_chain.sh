#!/bin/bash
# Ungauged forecasting (tree, mixed regime): points L1-3 for ung_mixed / ungnd_mixed,
# then the lead-1 ladder for ung_mixed. Sequential (11 GB box).
cd /home/habrt/source/flood
PY=.venv/bin/python
echo "=== ungauged chain start $(date)"
for L in U1 U2 U3; do $PY -u experiments/hgb_forecast_fctrain.py $L || exit 1; done
for q in q50 q90 q95 q99; do $PY -u experiments/hgb_forecast_fctrain.py Qung_mixed_${q} || exit 1; done
echo "=== UNGAUGED CHAIN DONE $(date)"
