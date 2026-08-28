# Session handoff (2026-08-28, updated evening)

Written mid-session because the harness's permission classifier went down
(Bash blocked; Write/Agent still worked). Everything below lets a fresh
session continue without re-deriving anything.

## State

| Item | Status |
|---|---|
| Eval harness (`evaluate.py`, `common.py`) | committed & pushed (900dad0), validated |
| Target transforms (`hgb_targets.py` + results) | committed & pushed |
| Spatial split (`hgb_spatial.py` + results) | committed & pushed |
| Quantile sweep (`hgb_quantiles.py` + `results/quantile_sweep.csv`) | **run**, results below |
| Chalk/GW audit (`cache/build_gw_cache.py`) | **run** — caches in `cache/gw_*.parquet` |
| Chalk/GW experiment (`hgb_groundwater.py` + `results/groundwater_*.csv`) | **run** — null result, see below |
| LSTM package (`lstm/train_lstm.py`, `lstm/README.md`) | committed (c71d1bb); not yet run on GPU box |
| Artifact correction | **republished** (section-6-corrected) |

## Immediate actions for the next session

Items 1–5 below are DONE as of the evening session; kept for the record.
Remaining: item 6 (LSTM on the GPU box), then score it with the harness
against `results/`. Optional: GW features in the ungauged split (see the
groundwater null result below).

1. ~~**Run the quantile sweep**~~ DONE. (~25 min, background):
   `.venv/bin/python experiments/hgb_quantiles.py <scratch-dir>`
   It reuses `targets_q99.parquet` from the scratch dir if present, else refits.
   Old scratch dir (files likely still on disk):
   `/tmp/claude-1000/-home-habrt-source-flood/c7aff7d7-24a1-490c-b77c-e9fed94eb3a6/scratchpad/`
   containing `targets_{raw,fine_leaf,log1p,norm,q99}.parquet`, `spatial_*.parquet`,
   `trainmax.parquet`, `hgb_final.parquet`.

2. ~~**Check the groundwater audit output**~~ DONE.: `cache/gw_levels_daily.parquet`
   (date × 55 well columns) and `cache/gw_well_match.parquet`
   (gauge_id, well_id, dist_km, aquifer, well_train_cov, well_test_cov).
   If missing, re-run the audit: build daily levels per well (daily file when
   present, else monthly resampled+ffill limit 40d; well ids UPPERCASE in
   attributes, lowercase in filenames), match each of the 416 modelled
   catchments (hydrometry `daily_flow_perc_complete >= 95`) to nearest well
   by easting/northing.

3. ~~**Write + run `hgb_groundwater.py`**~~ DONE — design as built:
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

4. ~~**Commit & push**~~ DONE: `experiments/hgb_quantiles.py`, `experiments/lstm/`,
   this file, quantile + GW results when done.

5. ~~**Republish the corrected artifact.**~~ DONE. The falsified §6 paragraph of
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

- Quantile sweep (α = 0.05…0.99, identical split): every quantile
  calibrated within ~1.5 pp of nominal (0.05→0.064, 0.50→0.501,
  0.95→0.936, 0.99→0.983 pooled). 50% interval covers 49.3% at median
  width 0.24 mm/day; 90% covers 87.5% at 0.64 mm/day. **q50 as a point
  forecast is worse on floods than the mean** (top-1% NSE −1.66 vs −0.81,
  AMAX bias −30.5%). 28.6% of rows had crossing quantiles before the
  monotone sort — independently fitted quantiles are not a free
  distribution.
- Groundwater audit: 55 wells (23 daily, 32 monthly), 29 usable with
  ≥50% coverage in both windows. Nearest *usable* well ≤40 km matches
  147/416 catchments (median 19.8 km), 14/23 chalk. Granta at Stapleford
  is 42 km from its nearest usable well → unmatched. The naive
  nearest-well match hands Lea Brook and Granta a well with 1.9% training
  coverage — hence the usability filter in `hgb_groundwater.py`.
- **Groundwater features (temporal split): null.** Adding the nearest
  usable well's z-scored level (+90-day change, lagged 1 day) changes
  paired per-catchment NSE by a median of −0.001 (all), −0.003 (chalk),
  −0.000 (chalk with a well), −0.000 (non-chalk). The chalk subset's median
  NSE rises 0.728→0.754 with real wells — but rises to 0.751 with
  **shuffled** wells too, and the five spatial-split failures gain +0.045
  real vs +0.035 shuffled. Whatever moves is a regional wetness signal the
  365-day rainfall window already carries, not the catchment's aquifer.
  Tail metrics unchanged (top-1% NSE −0.815 vs −0.811). Caveats: only 14
  chalk catchments have a usable well ≤40 km (low power), and this is the
  *gauged* temporal split — the chalk failure that motivated the test was
  in the *ungauged* split, which is the natural follow-up
  (`hgb_spatial.py` + GW features) if the question is worth another run.

## Environment gotchas (verified this session)

- Daily forcings: use `precipitation_haduk` / `pet_hydrope` /
  `temperature_haduk` — the cehgear/chess columns end 2019-12-31.
- `camels_gb_v2_hydrometry_attributes.csv` has 2 ragged rows; read via
  `common.read_attr`.
- Never feed hydrologic signatures (q_mean, baseflow_index, Q5/Q95…) as
  features — they are computed from the target.
- pandas 3.x: `groupby.apply(..., include_groups=False)`.
- The box has 11 GB RAM; one `build_dataset()` + fit peaks ~9.5 GB. Run
  experiments **one at a time**.
- Fits take ~6 min each; launch multi-fit scripts with `setsid nohup … &`
  so the harness's 10-minute Bash timeout can't kill them mid-run.
