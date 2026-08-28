# Session handoff — sweeps in flight (2026-08-28)

Written mid-session because the harness's permission classifier went down
(Bash blocked; Write/Agent still worked). Everything below lets a fresh
session continue without re-deriving anything.

## State

| Item | Status |
|---|---|
| Eval harness (`evaluate.py`, `common.py`) | committed & pushed (900dad0), validated |
| Target transforms (`hgb_targets.py` + results) | committed & pushed |
| Spatial split (`hgb_spatial.py` + results) | committed & pushed |
| Quantile sweep (`hgb_quantiles.py`) | **written, NOT yet run** |
| Chalk/GW audit (subagent) | launched; check outputs below |
| Chalk/GW sweep script | **not yet written** — design below |
| LSTM package (`lstm/train_lstm.py`, `lstm/README.md`) | written, not committed |
| Artifact correction | edited on disk, **not republished** |
| This file + lstm/ + hgb_quantiles.py | need commit & push |

## Immediate actions for the next session

1. **Run the quantile sweep** (~25 min, background):
   `.venv/bin/python experiments/hgb_quantiles.py <scratch-dir>`
   It reuses `targets_q99.parquet` from the scratch dir if present, else refits.
   Old scratch dir (files likely still on disk):
   `/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/`
   containing `targets_{raw,fine_leaf,log1p,norm,q99}.parquet`, `spatial_*.parquet`,
   `trainmax.parquet`, `hgb_final.parquet`.

2. **Check the groundwater audit output**: `cache/gw_levels_daily.parquet`
   (date × 55 well columns) and `cache/gw_well_match.parquet`
   (gauge_id, well_id, dist_km, aquifer, well_train_cov, well_test_cov).
   If missing, re-run the audit: build daily levels per well (daily file when
   present, else monthly resampled+ffill limit 40d; well ids UPPERCASE in
   attributes, lowercase in filenames), match each of the 416 modelled
   catchments (hydrometry `daily_flow_perc_complete >= 95`) to nearest well
   by easting/northing.

3. **Write + run `hgb_groundwater.py`** — design:
   - Features added to the standard set from `common.py`: `gw_z` (per-well
     z-score of level, **train-window mean/std only** — no test leakage) and
     `gw_z_d90` (90-day change in z). Catchments with no matched well within
     40 km get NaN — HistGradientBoosting handles NaN natively, one model
     covers everyone.
   - Two fits: real well assignment, and a **shuffled-well control**
     (permute assignments among matched catchments) to kill spurious gains.
   - No-GW control = the existing `targets_raw.parquet` (identical split).
   - Evaluate with the harness on subsets: all / chalk
     (`frac_high_perc >= 50` in hydrogeology attrs) / chalk with well ≤40 km
     / non-chalk / the 5 named failures (Lea Brook at Beck Bridge, Colne at
     Denham, Heacham at Heacham, Tilling Bourne at Shalford, Granta at
     Stapleford). Report paired per-catchment NSE deltas vs raw.
   - Success = chalk subset improves, non-chalk flat, shuffled control flat.

4. **Commit & push**: `experiments/hgb_quantiles.py`, `experiments/lstm/`,
   this file, quantile + GW results when done.

5. **Republish the corrected artifact.** The falsified §6 paragraph of
   "Trees on the Hydrograph" was replaced with the tested results. Corrected
   file: `<old-scratch-dir>/trees.html` (copy it somewhere durable first).
   Republish by passing `url: https://claude.ai/code/artifact/7e58bb2c-766f-4b41-8d4e-dea88b050f5c`
   from the new conversation.

6. **LSTM on the user's Intel Arc 140T box** (16 GB): everything in
   `experiments/lstm/README.md` — clone repo, torch XPU build, rsync the
   daily folder + attributes (~760 MiB), run, bring back
   `lstm_test_predictions.parquet` for scoring here.

## Established results (don't re-run)

Temporal split, 416 catchments, train ≤2010-09-30, test ≥2010-10-01.
All in `experiments/results/*.csv`.

- Tree baseline: median NSE **+0.820**, KGE +0.815, no drift over 12 test
  years; top-1% NSE **−0.811**, top-1% bias −23.9%, AMAX bias −17.4%;
  never predicts above 93 mm/day (obs max 244).
- Memory ablation: removing lags/rolling windows drops NSE to +0.349.
- Target transforms: log1p **worsens** the tail (−0.94); norm marginal
  (−0.71); fine leaves nothing. **Only quantile α=0.99 works** — coverage
  0.987 (nominal 0.990), predicts 1.21× obs on days beyond training max.
  On flood days the family's q99 runs at **1.89× its mean prediction** →
  the conditional mean is the wrong statistic for extremes; no mean-target
  fix exists. (This falsified the artifact's original §6 advice — hence
  the pending republish.)
- Spatial (ungauged) split, paired on the same 83 catchments: gauged
  +0.832 → ungauged +0.784 (median penalty **0.026**); worst failures all
  small chalk catchments → motivates the GW experiment.

## Environment gotchas (verified this session)

- Daily forcings: use `precipitation_haduk` / `pet_hydrope` /
  `temperature_haduk` — the cehgear/chess columns end 2019-12-31.
- `camels_gb_v2_hydrometry_attributes.csv` has 2 ragged rows; read via
  `common.read_attr`.
- Never feed hydrologic signatures (q_mean, baseflow_index, Q5/Q95…) as
  features — they are computed from the target.
- pandas 3.x: `groupby.apply(..., include_groups=False)`.
