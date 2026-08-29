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

## Phase 2 (2026-08-29): mechanism + peak diagnosis, two boxes in parallel

Written after a three-way review (evidence audit, ideation over unused
data assets, literature check). Two open questions, one per box; neither
box waits on the other after the enabling commit that adds `--seq` and
`--head quantile` to `train_lstm.py`.

**Q1 (Arc box): is the LSTM's win memory or per-basin loss
normalisation?**  **Q2 (CPU box): is the peak failure a statistic
problem, an information problem, or an observation problem?**

Literature stakes (checked 2026-08-29): Lees et al. 2021 report chalk as
the LSTM *failure* mode on CAMELS-GB, so a validated chalk win would
invert published results — but only with the mechanism nailed. The
field's standard reviewer-ask for our evidence is a distributional head
(Klotz et al. 2022 CMAL); nobody has probed LSTM state against the v2
groundwater wells (Lees et al. 2022 probed soil moisture/snow only).

### Arc box queue (pull first; run in order)

- A1 **Horizon test** (~50 min):
  `python experiments/lstm/train_lstm.py --seq 90 --out experiments/results/lstm_seq90`
  If chalk/weak-catchment gains survive a 90-day window, "memory beyond
  the tree's rolling windows" is dead and normalisation is the story.
- A2 **Quantile head** (~1.5 h):
  `python experiments/lstm/train_lstm.py --head quantile --epochs 16 --out experiments/results/lstm_q`
  Joint monotone pinball ladder (q05..q99; cannot cross by
  construction; `pred` = q50). First model combining the two proven
  wins (LSTM skill + calibrated quantiles). Read AMAX-day coverage of
  q99 and the q50 point card. NOTE: the head is new code smoke-tested
  only without torch — watch epoch 0; if loss is NaN or val NSE ≪ 0,
  report back rather than burning epochs.
- A3 (conditional on A1 keeping the gains) **More memory**: `--seq 730`
  — tests whether extra horizon adds anything at all.
- Commit each run's parquet + manifest + log (same pattern as before),
  push. Keep runs sequential — parallel runs share the GPU and gain no
  throughput (measured earlier).

**Arc box status (2026-08-29, evening): A1, A2, A3 all DONE, committed
and pushed** — `results/lstm_seq90/`, `results/lstm_q/`,
`results/lstm_seq730/`, cards/calibration/gate tables in `results/`,
findings in `lstm/README.md` ("Phase 2, Arc box"). One-line verdicts:
A1 weak-tree gain keeps 87% at 90 days, chalk keeps 47%; A3 730 days adds
+0.013 on chalk (noise) → Gate 1 = normalisation + a 90–365-day chalk
component, nothing beyond 365. A2 quantile head: never crosses, sharper
than the tree ladder at nominal pooled calibration, q99 covers 85% of
AMAX days; q50 is a worse flood point forecast than the mean (−23.5%).
Also fixed: `--seq` was not reaching spawned DataLoader workers on
Windows/macOS (71153d0).

### CPU box queue (11 GB — one job at a time; use setsid nohup)

- C1 **Per-catchment transform refits** (~15 min): refit `norm` and
  `log1p` (exact `hgb_targets.py` config), save per-catchment NSE →
  `results/target_transforms_per_catchment.csv`, compare their
  weak-tree/chalk gains against the LSTM's. The tree-side half of Q1.
- C2 **Quantile rerun with persistence** (~30 min): rerun
  `hgb_quantiles.py`, keep `quantile_predictions.parquet` in
  `results/`, add per-catchment coverage and AMAX-day-only coverage of
  q95/q99 (pooled 0.983 can hide per-basin miscalibration exactly where
  the envelope would be used).
- C3 **Free analyses** (no fits, from files already in `results/` +
  `cache/` + attribute tables):
  (a) AMAX event anatomy — per test AMAX event, decompose error into
      timing offset (±3-day peak match), matched-peak amplitude, and
      5-day event volume; tree vs LSTM.
  (b) Rating-curve noise floor — stratify AMAX bias by hydrometry q99
      uncertainty bounds, extrapolation duration, and obs peak vs
      `max_gauging_flow` (peaks beyond the highest gauged flow are
      themselves extrapolations).
  (c) Free 4-member LSTM ensemble (mean and upper member) from the four
      prediction parquets already committed.
  (d) Bankfull-exceedance skill — POD/FAR/CSI against `bankfull_flow`
      (318 gauges), triggers = raw pred and quantile levels; floods as
      the threshold decision they operationally are.
- C4 (optional) **Spatial fold rotation** (5 fits ≈ 30 min): offsets
  0–4 of the stratified fold → a distribution on the 0.026 ungauged
  penalty instead of one number.

### Decision gates — CLOSED 2026-08-29, all Phase 2 experiments done

- **Gate 1 verdict: normalisation, plus a small within-year chalk
  component.** Tree log1p recovers 74% of the LSTM's weak-catchment gain
  (C1); a 90-day window keeps 87% of it while chalk halves (A1: 90→365
  days adds +0.057 median NSE on chalk, 91% improving, vs +0.004
  elsewhere); 730 days adds nothing anywhere (A3). "Multi-year aquifer
  memory" is dead; "months 4–12 matter on chalk" is the surviving,
  properly controlled residue. The cell-state-vs-wells probe is still
  defensible but the effect it would explain is ~0.06 NSE on 23
  catchments, not the headline.
- **Gate 2 verdict: primarily statistic + information; observation
  secondary.** Statistic: q50-as-point under-predicts peaks for tree and
  LSTM alike (A2 replicates the tree result). Information: even
  calibrated quantile envelopes miss AMAX days at the same rate for both
  model classes — tree q99 covers 82.9% (C2), LSTM q99 84.7% (A2), both
  nominal 99% — so on ~15% of annual maxima the forcings do not flag the
  day as extreme; a better representation did not fix it. Observation:
  real but not dominant — no AMAX-bias gradient across rating-uncertainty
  quartiles, median rating half-width ±10.6% (≈ half the bias), 178/416
  test AMAX beyond the largest gauged flow, and daily averaging clips
  instantaneous peaks ~20% (C3b + cache audit).
- **C4 (fold rotation): the 0.026 penalty was the median hiding the
  story.** All 416 catchments held out once: median +0.030 (folds
  +0.026…+0.046) but mean +0.283, q90 +0.303, ungauged failures 25 vs 11
  gauged. Chalk penalty +0.097 vs non-chalk +0.027 — and the worst
  ungauged failures (Law Brook, Ver, Mimram, Burn, Pang…) are classic
  chalk streams sitting *below* the frac_high_perc≥50 flag: the binary
  chalk definition under-counts groundwater-dominated catchments.
  `results/spatial_folds_per_catchment.csv`.
- **Cheap tail win from C3c:** the 4-member ensemble upper member gets
  AMAX bias −14.3% and top-1% NSE −0.29, the best tail numbers of any
  model, for free. Bankfull threshold skill (C3d): LSTM CSI 0.691, FAR
  0.14 — as an alarm the models beat what amplitude metrics suggest.

### Next big build (chosen by Gate 2)

The residual is information at the daily scale: both model classes'
envelopes miss the same ~15% of AMAX days. The two candidate attacks:
**neighbour-gauge nowcasting** (same-day observed flow at nearby gauges —
directly observes the flood wave the forcings miss; also exactly what an
ungauged site lacks, tying into C4's chalk-penalty finding) and the
**hourly pilot** (tests whether sub-daily rain intensity is the missing
signal and how much daily averaging understates true peaks). Nowcasting
is the cheaper first step (tree fits on cache/daily_discharge_spec.parquet
features, CPU-only); hourly is the deeper one (~GPU-day + 10 GB
transfer). Also on the table: seed-ensemble upper member as a standing
cheap baseline for any tail claim, and the wells cell-state probe scoped
to the (now smaller) chalk question.

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

- LSTM (run on the Arc box, scored here; cards in
  `results/lstm_cards.csv`): median NSE +0.855 (16 ep) vs tree +0.820,
  no failed catchments, but flood peaks no better (AMAX bias −19 to −22%
  vs −17%, seed spread as large as the gap). Chalk catchments improve
  +0.728 → +0.907, reproducibly across all four runs.
  **CORRECTION (2026-08-29 audit)** of the interpretation committed in
  52c2488 ("multi-year aquifer memory"): (a) `train_lstm.py` uses
  stateless 365-day windows — the same horizon as the tree's `p_sum365`,
  so multi-year memory is architecturally impossible; (b) the LSTM
  rescues *all* weak-tree catchments, not chalk specifically (non-chalk
  with tree NSE<0.6 gain +0.45 median; delta correlates ≈−0.6 with flow
  variance; variance-matched chalk residual is not significant,
  p≈0.11); (c) the five "spatial-split failure" catchments have
  frac_high_perc 5.8–33 — none is chalk under this repo's own ≥50
  definition; (d) per-basin loss normalisation is the live alternative
  driver: the tree's own norm/log1p variants already cut failed
  catchments from 2.6% to 1.2%/0.5% (`target_transforms.csv`). Also
  noted by the audit: the 16-epoch pick was made on test cards, and
  quantile calibration was only ever computed pooled. Phase 2 below
  exists to settle this.

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
