# CAMELS-GB flood modelling — what we know (consolidated 2026-08-30)

Every number below is seed-replicated, control-validated, or explicitly
caveated; the two adversarial reviews' corrections are already applied.
Write-ups: `docs/where_the_floods_went.html` (findings),
`docs/floods_in_plain_words.html` (primer + definitions),
`docs/trees_on_the_hydrograph.html` (original essay, corrected twice).

**Setting.** CAMELS-GB v2, 416 catchments (≥95% complete), train
1970–2010 / test 2010–2022, HadUK-Grid + Hydro-PE forcings, no
discharge-derived features. Every model reports the same card: median
NSE/KGE, top-1% NSE, AMAX bias, q99 distribution bias, coverage.

**Standing results.**
1. Tree baseline +0.820 NSE but AMAX bias −17.4%, top-1% NSE −0.81,
   never predicts above 93 mm/d (obs max 244). Not a tuning artifact.
2. Point statistics vs distributions: mean-regression targets cannot fix
   the tail (log1p worse; q99/mean = 1.89× on flood days); the median of
   a calibrated ladder is a worse point forecast for peaks than the mean
   (4 replications — arithmetic of right skew, not a discovery).
3. Pooled ladder calibration is nominal (q99 0.983) but AMAX-day coverage
   is 0.83 (no donors) / 0.90 (donors). **Oracle:** a perfectly calibrated
   q99 of this sharpness covers only ~0.88 of AMAX days (selection on the
   realisation), so 0.90 is at the ceiling of the daily feature set. Say
   "exhausts the daily feature set", never "limit is in the data".
   Date-clustered 95% CI on coverage: [0.883, 0.907].
4. LSTM (+0.855) beats the tree mostly via per-basin loss normalisation
   (tree log1p recovers 74%; 90-day window keeps 87%); a small chalk-only
   90–365-day component (+0.057) survives; 730 days adds nothing. The
   published "multi-year aquifer memory" claim was wrong and corrected.
5. Nearest-well groundwater covariates: null (shuffled control matches).
6. Neighbour-gauge nowcasting (same-day + lag-1 flow at 3 nearest gauges):
   tree +0.882, top-1% −0.10, AMAX −9.0%; shuffled donors flat; survives
   dropping the nearest donor (88%); flat in distance to 43 km. **Zero
   lead time** (lag-1 keeps 14%); zero-parameter donor floor NSE 0.715.
7. LSTM + donors, 3 seeds: median NSE +0.910 (0.906–0.914), top-1% NSE
   +0.136 (0.085–0.201, positive every seed), AMAX −10.1% (ties the donor
   tree). Seed 0 (−6.6%) was the outlier.
8. Ungauged (5 rotated folds): median penalty +0.030 but heavy-tailed
   (mean +0.283, chalk +0.097). With donors on both sides the penalty is
   +0.048 (chalk +0.132) — NOT closed; ungauged-with-donors matches
   gauged-without. Geology-similar donors halve chalk penalty (+0.047).
9. Missed events: the two donor ladders each miss ~10% of AMAX days, only
   44% overlapping; 225 missed by both — summer 2×, 55% with same-day
   rain ≤ its q90. Hourly pilot (60 outcome-selected catchments, 2 seeds):
   recovers 64% (±1 d, same comparator) vs a 25% diversity baseline;
   width-null 4.1%; rain-only recovers the same share → **the recovery is
   hourly rainfall; donors are point skill**.
10. Observation floor: rating half-width ±10.6% at q99; 178/416 test AMAX
    beyond the largest gauged flow; daily means understate hourly peaks
    (median 1.42×). Bankfull alarm skill: LSTM CSI 0.69, nowcast POD 0.80.

**What no model here does: forecast.** All are simulations/nowcasts
(inputs complete only at end of day t). Own-flow autoregression was
deliberately withheld. Phase 6 below opens that question.

**Traps.** cehgear/chess daily columns end 2019; hourly cehgear ends
~2017–19 and gradgb starts ~2006–08 (blend); hourly discharge is mm/h
(daily mm/day); /tmp scratch dies on reboot — persist to results/;
11 GB RAM — one experiment at a time, `setsid nohup` for long runs;
Arc box must write UTF-8.

---

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

### Nowcasting result (2026-08-29, hgb_nowcast.py) — the Gate 2 build, run

Same-day + lag-1 observed flow at the 3 nearest usable gauges (median
13.2 km; donor-q95-scaled; 628-donor pool), shuffled-donor control.

- **Temporal (gauged): the peak problem largely yields.** Median NSE
  +0.882 (best of any model, LSTM included), top-1% NSE −0.811 → −0.099,
  AMAX bias −17.4% → −9.0%, q99 distribution bias −5.4%. The shuffled
  control is exactly flat (ΔNSE −0.000, AMAX −19.7%): the gain is the
  local flood wave, not regional wetness. Gate 2's "information" residual
  is confirmed and addressed — the missing flood-day information exists
  in real time at the neighbouring gauges.
- **Spatial (ungauged): the median site is rescued, the pathological
  ones are not.** Median ungauged penalty +0.030 → −0.006 (an ungauged
  site with neighbours now matches a gauged site without them), but
  chalk only improves +0.097 → +0.077, ungauged failures stay ~25, and
  some groundwater catchments get *worse* (Law Brook −22.6 → −31.8,
  Mimram −6.7 → −14.2): where a catchment behaves unlike its
  neighbours, donors actively mislead. The ungauged-chalk problem is
  now the one clearly unsolved thing in the project.

### Phase 3 (2026-08-29 pm): the synthesis, two boxes in parallel

Everything proven so far in one place: nowcast donors fix flood-day
information (temporal), quantile heads fix the statistic, the LSTM's
normalisation wins ordinary days. Phase 3 combines them and attacks the
one unsolved thing (ungauged chalk).

**Arc box queue** (pull first — train_lstm.py now has `--donors K`):

- B1 **LSTM + donors, point** (~1.5 h):
  `python experiments/lstm/train_lstm.py --donors 3 --epochs 16 --out experiments/results/lstm_nowcast`
  Does the LSTM still add anything over the nowcast tree (+0.882) once
  both see the neighbours? Donor build loads all 671 daily files
  (~2-3 min) and appends 6 q95-scaled columns per basin.
- B2 **LSTM + donors, quantile head** (~1.5 h):
  `python experiments/lstm/train_lstm.py --donors 3 --head quantile --epochs 16 --out experiments/results/lstm_qnow`
  The full synthesis. The one number that matters: AMAX-day q99 coverage
  (0.847 without donors; nominal 0.99).
- Commit each run's parquet + manifest + log; push. UTF-8 writes only.

**Arc box status (2026-08-29, late): B1 and B2 DONE, committed and
pushed** — `results/lstm_nowcast/`, `results/lstm_qnow/`, cards in
`results/lstm_nowcast_cards.csv`, paired `results/lstm_nowcast_paired.csv`,
calibration `results/lstm_qnow_calibration.csv`; findings in
`lstm/README.md` ("Phase 3, Arc box"). B1: median NSE +0.914, AMAX bias
−6.6%, top-1% NSE +0.201, beats the nowcast tree on 80% of catchments —
donors and normalisation are additive; best point model in the project.
B2: AMAX-day q99 coverage 0.892 (vs 0.847 without donors; tree ladder +
donors 0.896) — the envelope gain from donors is the same in both model
classes and both leave the same ~10% of annual maxima unflagged; q50 AMAX
bias −15.0% vs the mean head's −6.6% (q50-vs-mean replicates a 4th time).

**CPU box queue** (one at a time):

- C1 **Tree quantile ladder + donors** (~40 min, `hgb_quantiles_nowcast.py`):
  the tree twin of B2 — does AMAX-day q99 coverage climb from 0.829
  toward 0.99, and does the ladder sharpen? Writes
  `results/quantile_nowcast_calibration.csv` with the no-donor column
  alongside.
- C2 **Similarity-restricted donors for ungauged chalk** (~35 min,
  `hgb_nowcast_similar.py`): donors filtered to |frac_high_perc − target|
  ≤ 20 before taking the nearest 3 (gauge-free similarity, so the setting
  stays honestly ungauged). Success = chalk penalty and the pathological
  worseners (Law Brook, Mimram) improve vs the nearest-donor run.

Shared donor code now lives in `experiments/nowcast_common.py`.

**Phase 3 CPU results (2026-08-29 pm):**

- **C1: donors move flood-day calibration a long way but do not close
  it.** AMAX-day q99 coverage 0.829 → 0.896 (q95 0.630 → 0.745), ladder
  sharper (90% width 0.64 → 0.54 mm/day at unchanged pooled coverage).
  ~10% of annual maxima remain unflagged even with the neighbours'
  flood wave — the remaining candidates are sub-daily rain intensity
  (hourly pilot) and irreducible event surprise. And for the third time,
  q50-as-point is worse on peaks (AMAX −16.4%) than the mean model of
  the same feature set (−9.0%).
- **C2: geology-similar donors halve the remaining ungauged-chalk
  penalty.** Median chalk penalty raw +0.097 → nearest +0.077 → similar
  +0.047, ungauged failures 25/26/23 (best), non-chalk unharmed. The
  filter only changes 80/416 donor sets and costs ~1 km of median
  distance. The two pathological catchments (Law Brook, Mimram) are
  dampened but still worse than no donors at all.
- **C3 (donor-trust features): a tail fix, not a median fix.** Adding
  per-donor gauge-free dissimilarity (geology gap, log-area gap,
  distance) to the similar-donor folds drops chalk ungauged failures to
  2 (best of all variants) and improves every pathological catchment
  (Mimram −10.5 → −7.2, Ver/Burn → −2.8), but the median chalk penalty
  regresses to +0.085 vs the hard filter's +0.047 — the extra columns
  dilute the typical catchment. **Recommended ungauged config stays
  `similar` (hard geology filter)**; trust features are worth revisiting
  only as a per-catchment fallback. Nowcast diagnostics also added to
  the C3 CSVs: best event anatomy across the board (matched-peak 0.855,
  5-day volume 0.927), bankfull POD 0.797 (LSTM keeps best CSI 0.691).

### Phase 4 (2026-08-29 eve): the last frontier + the write-up

The missing ~10% is now characterised (`analysis_missed_amax.py`):
tree and LSTM ladders each miss ~10.4/10.8% of AMAX days but only 44%
of those events overlap — 225 events (4.5%) are missed by BOTH, summer
2× over-represented, and 55% of them have same-day rain below its own
q90: invisible in daily rainfall by construction. (Corollary: a
max-of-both-envelopes union already covers ~95.5% of AMAX days.)

**Arc box queue — the hourly pilot** (pull first):

- A0 **Transfer** (~3.1 GB): 198 hourly files (60 pilot catchments =
  48 most-missed + 12 zero-miss controls, plus each target's 3 donors):
  `rsync -a --files-from=experiments/results/hourly_pilot_files.txt main-machine:~/source/flood/data/Catchment_Timeseries/hydro-meteorological/hourly/ ./data/Catchment_Timeseries/hydro-meteorological/hourly/`
- A1 **Hourly LSTM, mse head** (~10 epochs; expect slower epochs than
  daily): `python experiments/lstm/train_lstm_hourly.py --out experiments/results/lstm_hourly`
  Hourly rain = gradgb.fillna(cehgear) — **hourly forcing trap**: cehgear
  dies ~2017-19, gradgb starts ~2006-08, neither spans both windows.
  336-hour windows, hourly donor flows (same-hour + 24h lag), per-basin
  normalisation, same split boundary.
- A2 **Hourly quantile head**:
  `python experiments/lstm/train_lstm_hourly.py --head quantile --out experiments/results/lstm_hourly_q`
- Commit each run's parquets (hourly + the daily-mean aggregation the
  script writes) + manifest + log; push. UTF-8 only.
- Scoring happens on the main machine: daily-agg card vs the daily
  models on the same 60 catchments; hourly AMAX capture; and the named
  225 both-missed events (`results/missed_amax_events.csv`) — the pilot
  succeeds if the hourly model's envelope covers a material share of
  exactly those events.

**Arc box status (2026-08-30, 00:15): A0, A1, A2 all DONE, committed and
pushed** — `results/lstm_hourly/`, `results/lstm_hourly_q/` (slim hourly
ladder 98 MB + full ladder in the daily aggregate; the 222 MB full hourly
file is gitignored). Rain-gap fix in `train_lstm_hourly.py` (fb661a8).
Local sanity on the 60 pilot catchments: A1 daily-agg NSE +0.869 =
daily donor LSTM; A2 daily-agg q99 covers **76.6%** of the 145 pilot
both-missed events (hourly q99 vs hourly peak: 69.0%), 94.2% of all
pilot AMAX days. Success criterion met, pending the rigorous scoring here;
caveat: hourly rain and hourly donors are confounded in this pilot.
Findings in `lstm/README.md` ("Phase 4, Arc box").

**CPU box / this machine:** the rigorous write-up of all findings as an
artifact (published: "Where the Floods Went"), plus scoring the pilot.

**Phase 4 verdict (2026-08-30): the hourly pilot succeeds on its named
criterion.** Of the 145 both-missed AMAX events inside the 60 pilot
catchments, the hourly quantile model's envelope covers ~72–77% at the
daily-mean level (71.7% by the main-machine replication matching on the
local obs-max day, 76.6% by the Arc box's exact-date match) — events
both daily ladders missed 100% of by construction. At the instantaneous
hourly peak, coverage is 51–69% (matching-dependent), and 90–94% over
all pilot AMAX days. Daily-aggregate point skill is at parity with the
daily donor LSTM (median NSE +0.869 vs +0.867). So the residual was
real information sitting below the daily sampling rate — mostly
recoverable, at the price of hourly data. Caveats: hourly rain and
hourly donor flows are confounded in this pilot (both changed at once);
hourly discharge is mm/h (daily aggregates are 24× smaller than the
daily files — a units trap for any cross-comparison); post-2016 gradgb
outages are zero-filled with a rain_gap flag column. The project's
question ladder is, at this point, answered at every rung.

### Phase 7 data (2026-08-30): archived rainfall forecasts for GB

Phase 6 showed the rainfall forecast is the binding constraint. Survey of
archived NWP precipitation forecasts covering 2010-10 → 2022-09 (research
agent, 99 sources, verified locally where possible):

| Source | Window | Grid | Members | Licence | Access | Verdict |
|---|---|---|---|---|---|---|
| **ECMWF TIGGE via ECDS** | 2006-10 → now, full window | 0.5°/0.25° on request | 50+1 (ECMWF), UKMO 17+1, NCEP 30+1 | CC BY 4.0 (ECMWF/UKMO/NCEP) | free ECMWF account + licence click; `cdsapi` ≥0.7.7; server-side `area`+`grid` → **~1 GB** for the window | **best data**; risks: tape queue (hours/date), one Apr-2026 non-Member-State rejection report, 710 lost `cf/sfc` dates 2006–19 (use pf members). Old `ecmfwapi` WebAPI decommissioned 27 May 2026. |
| **NOAA GEFS v12 reforecast (AWS)** | 2000-01 → 2019-12 | 0.25° native | c00 + p01–p04 | public domain | none; `.idx` byte ranges, **proven from this box** | consistent model; 23 GB transfer (c00, lead 1) / 61 GB (leads ≤3); UK crop kept ≈ MB |
| GEFS v12 operational (AWS) | 2020-09-23 → now | 0.25° | 31 | public domain | same mechanics (`pgrb2sp25`, APCP 6-h) | fills 2020-10 → 2022-09; **gap 2020-01 → 09** (v11 at 1° only) |
| Met Office MOGREPS-UK legacy S3 | 2013-01 → 2017-01 | 2.2 km | 12 × 4/day | non-commercial | whole files, ~12 GB/day | case studies only |
| Met Office UKV/Global via CEDA | 2016-03+ / 2012+ | 1.5 km / 17–25 km | det. | NERC academic/govt only | JASMIN | not for a personal/commercial user |
| ECMWF open data, DataHub, ERA5 "forecast" tp, S2S, EFAS/GloFAS reforecasts | none usable in window | | | | | excluded |

Precedents for the observed-vs-forecast rain mismatch: Nearing 2024
(ERA5-Land imputation), Hunt 2022 (accept the bias), Sharma 2023
(residual postprocessor on GEFSRv2), **Taccari 2026 (pre-train on
observed rain, fine-tune on the forecast archive — the cleanest template
for us)**.

**Decision — two tracks:**
- Track A (user): register at https://ecds.ecmwf.int, accept the TIGGE
  licence, put the token in `~/.cdsapirc`, and test ONE date
  (`tigge-forecasts`, origin ecmwf, total_precipitation, pf, area
  [61,-11,49,2], grid 0.5/0.5, leadtime 24/48/72/96). If it returns, the
  whole window is ~1 GB in one-date-per-request jobs.
- Track B (this box, no registration): `experiments/nwp/fetch_gefs.py`
  pulls GEFS v12 c00 (+ members) by byte range, crops the GB box, writes
  monthly netCDF cubes (init × lead_day × lat × lon, mm/day) to
  `cache/nwp/`. Convention: 00Z run of issue day t; lead_day 1 = 24–48 h
  = calendar-day total for t+1. Resume-safe. Smoke-tested on Jan 2012.

**Resolution caveat** (any global NWP): 0.25° ≈ 450 km² at 55°N, 3× the
median catchment; orographic and convective rain under-resolved. The
tree's perfect-rain run (+0.859 / −28%) is the ceiling no archive reaches.

**Track A status (2026-08-30 night):** licence accepted, one-date test
verified (50 members, 0.5°, tp in mm; spatial r = 0.94 vs GEFS control
at lead 1). ECDS serialises requests per user (no concurrency speedup),
so the full window is a ~1–2 week drip. It runs as a **systemd user
service** (`scripts/tigge-pull.service` installed at
`~/.config/systemd/user/`, linger enabled): survives reboots, connection
loss and crashes; monthly chunks are skipped once their `.nc` exists.
Priority order: the 2020-01→09 hole first, then the whole 2010–2022
window. Log: `cache/nwp/tigge/pull.log`; manage with
`systemctl --user {status,stop,start} tigge-pull`.

**Track B status: COMPLETE 2026-08-31 03:23** (136/136 cubes, 0 errors). Was —
control member, leads 1–3, 2010-10-01 → 2019-12-31 then 2020-09-23 →
2022-09-30, ~4 MB/day at this box's 0.55 MB/s line ≈ 9–10 h, into
`cache/nwp/gefs_c00_YYYYMM.nc` (zlib, ~1 MB/month). Monthly cubes are
resume-safe — rerun the same command to fill gaps.

**Arc box (optional acceleration, no code needed):** if its line is
faster, run the same fetcher for the perturbed members and commit the
cubes (they are small — put them under `experiments/results/nwp/` via
`--out`, since `cache/` is gitignored):
`python experiments/nwp/fetch_gefs.py 2010-10-01 2019-12-31 --member p01 --leads 3 --out experiments/results/nwp`
(needs `pip install eccodes cfgrib xarray netCDF4 requests`). Members
p02–p04 likewise if time allows; ensemble spread is the payoff.

**Phase 7 model — DONE 2026-08-31 (tree ladder, GEFS c00).** GEFS pull
finished clean (136/136 monthly cubes, 0 errors, 7 h). Build:
`nwp/gefs_join.py` (nearest 0.25° cell to gauge_lat/lon →
`cache/nwp/gefs_catchment_leads.parquet`, p_fc1..3 per gid×issue-day) +
`hgb_forecast_gefs.py` (per-lead subprocesses; trains ar_perfect on
*observed* rain path p_next1..L — train years predate the archive — then
predicts test twice: observed rain = ceiling, GEFS rain substituted =
real skill; 2020-01→09 hole left NaN, trees route it; `covered` column
marks GEFS rows).

GEFS c00 rain quality at the gauge cell (30-catchment sample): r = 0.56
/ 0.50 / 0.43 vs HadUK at leads 1/2/3, dry bias −19% (2.6 vs 3.2
mm/day).

Verdict (median NSE; floor = Phase 6 ar_donor, ceiling = perfect rain
path — ceiling now fit at every lead, a new fact):

| lead | persistence | floor | **GEFS** | ceiling | gap recovered |
|---|---|---|---|---|---|
| 1 | 0.538 | 0.778 | **0.803** | 0.859 | 31% |
| 2 | 0.227 | 0.485 | **0.543** | 0.834 | 17% |
| 3 | 0.078 | 0.390 | **0.457** | 0.821 | 16% |

Two headline facts. (1) The perfect-rain ceiling barely decays with lead
(0.859→0.821): 1–3-day flow forecastability is almost entirely rain
knowledge. (2) Real 2010s-era control-member NWP already moves every
lead off the floor and beats persistence in 85–88% of catchments —
"lead ≥ 2 is near-blind" is softened, not overturned: peak-day bias
L1 −29.9% (nearly the ceiling's −27.9%) but −58% / −65% at L2/L3
(ceiling holds ~−30%); own-max ±1d timing 46% / 27% / 19% vs ceiling's
~52%. Timing-blind AMAX quirk: GEFS shows −8.3% at L1, *better* than
the ceiling's −15% — noisy rain inflates the year-max; trust the
peak-day numbers (`forecast_gefs_skill_L{1,2,3}.csv`).

Upgrades that should close more of the gap, in expected-value order:
TIGGE 50-member ensemble mean (pull running), catchment-mean rain via
Catchment_Boundaries instead of gauge-point cell, member spread as an
uncertainty feature, then the LSTM Taccari-style fine-tune.

**Upgrade 2 — catchment-mean rain (CPU box, 2026-08-31):**
`nwp/gefs_catchment_mean.py` intersects each boundary polygon with the
0.25° grid in BNG (median 4 cells, max 40) →
`cache/nwp/gefs_catchment_leads_mean.parquet`. Rain r vs HadUK rises
0.563→0.578 / 0.499→0.512 / 0.428→0.440, dry bias −19%→−15%.
`hgb_forecast_gefs.py` now takes `[MODE] [fc_parquet] [suffix]`.
Verdict of the `_mean` rerun: median NSE 0.803→0.806 / 0.543→0.551 /
0.457→0.466 at leads 1/2/3 (paired dNSE +0.001/+0.003/+0.004, better
in 59–65% of catchments); peak-day bias −29.0/−57.2/−63.9 (each ~1 pp
better than nearest-cell). Small but uniformly positive — the
catchment-mean parquet is now the standard forecast-rain input
(committed at experiments/results/nwp/; skill CSVs
forecast_gefs_mean_skill_L*.csv). The rain, not its spatial sampling,
remains the binding constraint — the ensemble mean is the next lever.

### ARC BOX — Phase 7 LSTM with forecast rain (pass-off, 2026-08-31)

`git pull`, then run these two on the Arc box (each writes its own
--out; move/rename the parquets before the next run):

```
# 1. ceiling: observed future rain in the fc channels end-to-end
python experiments/lstm/train_lstm.py --lead 1 --autoreg --fcrain perfect \
  --out experiments/results/lstm_fc_perfect_L1
# 2. real forecast: same training, GEFS catchment-mean rain substituted
#    into the genuinely-future window steps at test time
python experiments/lstm/train_lstm.py --lead 1 --autoreg \
  --fcrain experiments/results/nwp/gefs_catchment_leads_mean.parquet \
  --out experiments/results/lstm_fc_gefs_L1
```

How the patch works (train_lstm.py `--fcrain`, committed + smoke-tested
on CPU): for lead L it appends L dynamic channels, channel k holding
rain(τ+k) at step τ — observed wherever τ+k ≤ t (known by issue day),
and only the last k steps of each window (genuinely future) are
overwritten at test time with the forecast issued on day τ. Training
years predate the GEFS archive so training always uses observed rain
(the same train-on-obs / drive-with-forecast surrogate as the tree).
Missing-forecast rows (2020-01→09 hole) keep observed rain and are
marked `covered=False` in the output parquet — score honest subsets
with it. NOTE a committed copy of the mean parquet lives at
`experiments/results/nwp/gefs_catchment_leads_mean.parquet` (cache/ is
gitignored) — point --fcrain there on the Arc box. Commit both prediction parquets; the CPU
box scores them with analysis_forecast_skill.py (LEAD 1). Comparators:
tree gefs_L1 0.803, tree ceiling 0.859, Phase 6 LSTM autoreg L1 0.813.
If time allows: --lead 2 and --lead 3 pairs likewise (bump --epochs if
val NSE is still climbing). Write results into this section, UTF-8.

**ARC BOX STATUS — Phase 7 LSTM DONE (2026-09-02, all six runs committed
and pushed: 7fc860b L1, 8fadbc0 L2, + the L3 commit).** Parquets +
manifests + logs in `results/lstm_fc_{perfect,gefs}_L{1,2,3}`; cards
`results/lstm_p7_cards.csv`, paired + peak timing
`results/lstm_p7_vs_persistence.csv` (covered-only rows included).
Findings in `lstm/README.md` ("Phase 7, Arc box"). Notes: 16 epochs per
run (matched to the Phase 6 budget, not the snippet's default 8, so the
rain effect is not confounded with training length); each GEFS run
reuses its ceiling twin's checkpoint and runs inference only — training
is identical by construction (obs rain only; epoch-0 vals matched
exactly), which makes the ceiling-vs-GEFS comparison exactly paired.

| lead | persistence | LSTM no rain fc | tree+GEFS | **LSTM+GEFS** | LSTM ceiling | tree ceiling |
|---|---|---|---|---|---|---|
| 1 | 0.538 | 0.813 | 0.806 | **0.858** | 0.901 | 0.859 |
| 2 | 0.227 | (0.485 tree) | 0.551 | **0.644** | 0.880 | 0.834 |
| 3 | 0.077 | 0.388 | 0.466 | **0.544** | 0.874 | 0.821 |

Verdicts: (1) the LSTM perfect-rain ceiling is nearly lead-invariant
(0.901/0.880/0.874) — all lead decay in no-rain-forecast models was rain
ignorance; its flood-day skill barely decays (peak-day −25/−24/−22%,
top-1% NSE −0.2/−0.3/−0.2). (2) LSTM + GEFS equals the tree's
perfect-rain ceiling at lead 1 and beats the tree on identical rain by
+0.05–0.09 at every lead; peak-day −25% at L1 (better than the tree
ceiling's −28%) but −52/−59% at L2/3. (3) The remaining gap (0.04/0.24/
0.33 NSE) is rain-forecast error — the measured prize for the TIGGE
ensemble mean + spread and the Taccari-style fine-tune. Suggested next
Arc runs once TIGGE lands: same pairs with ensemble-mean rain + a spread
channel, and a lead-1 quantile head on GEFS rain for the operational
ladder.

### Phase 6 (2026-08-30): from simulation to forecasting

Nothing built so far forecasts: every model's inputs are complete only at
the end of the target day. Phase 6 moves the target into the future. Issue
day = t, target = flow at t+L. Inputs = everything known at end of t:
forcings and rolling windows to t (rain on t included), optionally the
target's own past flow (autoregression — deliberately withheld until now)
and donors at t. Skill is judged against **persistence** (flow(t+L) =
flow(t)), the bar every river forecaster must beat.

**Arc box queue** (pull first — train_lstm.py now has `--lead L` and
`--autoreg`; both compose with `--donors` and `--head quantile`):

- F1 **1-day-ahead LSTM forecaster** (~1.5 h):
  `python experiments/lstm/train_lstm.py --lead 1 --autoreg --donors 3 --epochs 16 --out experiments/results/lstm_fc1`
  Window ends at t, target t+1, own normalised flow as an input channel,
  donors at ≤t. Parquet rows are dated at the TARGET day, so evaluate.py
  scores it directly; persistence is obs shifted from the daily flow
  matrix (CPU-side scoring script coming).
- F2 **Same with the quantile head** (~1.5 h):
  `... --lead 1 --autoreg --donors 3 --head quantile --epochs 16 --out experiments/results/lstm_fc1_q`
  The useful operational object: a calibrated 1-day-ahead flood bound.
- F3 **3-day lead** (~1.5 h): `--lead 3 --autoreg --donors 3 --epochs 16 --out experiments/results/lstm_fc3`
  How fast skill decays with lead.
- Optional F4: `--lead 1 --autoreg --epochs 16` (no donors) to price the
  neighbours at lead 1 for the LSTM as well.
- Commit parquets + manifests + logs; push. UTF-8 only. Val curves in the
  commit message as before.

**CPU box queue:**

- C1 `hgb_forecast.py` (running): tree at L=1 — persistence, weather-only,
  + own flow (ar), + donors (ar_donor), + actual rain on t+1 (ar_perfect
  = perfect-rainfall-forecast ceiling); ar_donor at L=2, 3. Standard card
  + paired skill vs persistence + AMAX bias by lead.
- C2 DONE — Arc F1–F4 scored with `analysis_forecast_skill.py` (fan-out
  bug fixed: never `.loc` on the non-unique date index); every Arc
  number reproduces. `results/forecast_skill_L1.csv`, `_L3.csv`.

**Phase 6 verdict (2026-08-30) — forecasting, both boxes:**

| lead 1 day | median NSE | beats persistence | peak-day bias | own-max ±1 d |
|---|---|---|---|---|
| persistence | +0.538 | — | −54% | — |
| tree, own flow + donors | +0.778 | 87% | −40% | 50% |
| LSTM, own flow + donors (F1) | +0.811 | 94% | −34% | 52% |
| LSTM, own flow, no donors (F4) | **+0.813** | 96% | −34% | 53% |
| LSTM quantile q50 (F2) | +0.806 | 96% | −35% | 55% |
| tree + actual next-day rain (ceiling) | +0.859 | 87% | −28% | 57% |
| lead 3: tree / LSTM | +0.390 / +0.388 | 91% / 95% | −77% / −76% | 3% / 5% |

1. A real 1-day-ahead forecaster exists: LSTM with the basin's own past
   flow, +0.81 median NSE, better than persistence on 94–96% of
   catchments, no failed catchments. The normalisation advantage over
   the tree (+0.03) survives into forecasting.
2. **Donors add nothing at any lead ≥ 1** in either family (F4 edges F1;
   tree +0.001). The nowcasting result is strictly a same-day result.
3. **Floods a day ahead are 34–40% low on the peak day** for every model
   without a rainfall forecast; the observed next-day rain (a perfect
   forecast) takes the tree to +0.859 / −28%. The binding constraint is
   the rainfall forecast, not the model — next step is NWP-driven inputs.
4. At 3 days nothing sees a flood (−76%, timing hits 3–5%); the LSTM and
   tree tie exactly there — lead-3 skill is entirely rain-forecast-bound.
5. The 1-day quantile ladder is the operational object: pooled q99 0.992,
   90% width 0.45 mm/day (sharper than the simulation ladder's 0.54),
   zero crossings, and **q99 ≥ the observed annual peak on 79% of AMAX
   days one day ahead** (simulation ladder: 85%).

Known scope: no NWP rainfall forecasts here — "weather" variants use rain
up to t only; ar_perfect uses the *observed* rain on t+1 as the ceiling.

**Phase 6 CPU results (tree, `hgb_forecast.py`; cards in
`results/forecast_cards.csv`):**

| lead 1 day | median NSE | top-1% NSE | AMAX bias (year-max) | peak-day bias | own-max ±1 d |
|---|---|---|---|---|---|
| persistence | +0.538 | −3.25 | 0% (artifact) | **−54%** | 100% (artifact) |
| weather only | +0.702 | −2.58 | −22.5% | −43% | 47% |
| + own flow (ar) | +0.779 | −2.10 | −19.0% | −40% | 50% |
| + donors | +0.778 | −2.15 | −18.9% | −40% | 50% |
| + actual rain on t+1 | **+0.859** | −0.59 | −15.0% | **−28%** | 57% |
| ar_donor, lead 2 | +0.485 | −5.6 | −47% | −73% | 20% |
| ar_donor, lead 3 | +0.390 | −6.6 | −54% | −77% | 3% |

Readings: (i) own-flow autoregression is worth +0.08 NSE over
weather-only and is the main forecast ingredient; (ii) **donors add
nothing at lead 1** (+0.001) — consistent with Phase 5 C1; (iii) the
**rainfall forecast is the binding constraint**: the observed next-day
rain adds +0.08 NSE and takes peak-day bias from −40% to −28%, restoring
roughly the simulation LSTM's skill (+0.859 vs +0.855); (iv) beyond
lead 1 nothing sees a flood without rain forecasts. (v) **Metric
warning:** `evaluate.amax_bias` compares the year's max prediction with
the year's max observation regardless of timing, so persistence scores
0% while being a day late on every peak; for forecasts report peak-day
bias (pred on the observed AMAX day / obs) and timing, as above —
`analysis_forecast_skill.py` does this for any forecast parquet.

**Arc box status (2026-08-30, 15:00): F1, F2, F3, F4 all DONE, committed and
pushed** (55d67a0, 9aff85a, d49b053, 918ea82 and the F4 commit). Parquets + manifests
+ logs in `results/lstm_fc1`, `lstm_fc1_q`, `lstm_fc3`, `lstm_fc1_nodonor`;
cards `results/lstm_forecast_cards.csv`, paired skill + peak timing
`results/lstm_forecast_vs_persistence.csv`, F2 ladder
`results/lstm_fc1_q_cards.csv` / `lstm_fc1_q_calibration.csv`. Findings in
`lstm/README.md` ("Phase 6, Arc box"), with the C1 tree rows alongside.
Local persistence scoring reproduces C1's bar exactly (+0.538 / +0.077).

- F1 (lead 1, own flow + donors): median NSE **+0.811** vs persistence
  +0.538 and tree ar_donor +0.778 — the LSTM keeps its +0.03 simulation
  margin over the tree a day ahead; paired +0.253, better on 94%; top-1%
  NSE −1.51 (persistence −3.25, tree −2.15); AMAX bias −17.4%; peak-day
  bias −34% (tree −40%); own AMAX within ±1 day 52% (tree 50%).
- F2 (quantile head): q50 +0.806 ties F1; the ladder is near-nominal
  pooled (q50 .517, q95 .963, q99 .992) and 17% sharper than the A2
  simulation ladder, but q99 clears the observed annual peak on 79% of
  catchment-years vs A2's 85% — a day of lead costs the envelope at the
  top.
- F3 (lead 3): +0.388 vs persistence +0.077 — ties the tree's +0.390;
  flood-blind exactly as the tree is (AMAX −54%, peak-day −76%, own AMAX
  within ±1 day 4.9%, median lag +2 d). Lead 3 is rain-forecast-bound.
- F4 (lead 1, no donors): **+0.813** matches or edges F1 (+0.811) on every
  column (top-1% NSE -1.34 vs -1.51, AMAX -16.4% vs -17.4%, own-peak timing
  53% vs 52%, better than persistence on 96%): donors add nothing at lead 1
  for the LSTM, exactly as for the tree (C1 +0.001). Own flow at t already
  carries what the neighbours know; in simulation donors were a proxy for
  the withheld own flow. The operational forecaster needs no gauge network.

Verdict for C2: at lead 1 the LSTM is the better point forecaster by the
same margin as in simulation and its quantile ladder is the operational
object; beyond lead 1 model class stops mattering. Compare forecasters on
top-1% NSE, peak-day bias and AMAX timing — never on `AMAX_bias_pct`,
where persistence is 0% by construction.

### Phase 5 review synthesis (2026-08-30) — corrections to the record

Two adversarial reviews (methodology; conclusions) reported. Their most
damaging findings were independently verified on this machine and are
CORRECT. Claims change as follows:

- **Hourly pilot headline CORRECTED.** The 72–77% scored the hourly
  model against its own (hourly-aggregated) obs record, which on the
  145 event days is median 85.5% of the daily-file obs the daily models
  were judged on. Same-comparator rescore (q99×24 vs daily-file obs):
  **51.7% exact-date, 64.1% ±1 day**. And "0% by construction" was a
  rigged baseline — the weaker no-donor daily ladders already cover
  10.3%/20.7% (union **24.8%**) of those events through model
  diversity. Defensible claim: **64% vs a 25% third-model baseline** —
  a 2.6× recovery, still decisively real (the C2 width-null of 4.1%
  shows it is not envelope inflation), but not 77-vs-0.
- **"Nowcasting closes the ungauged penalty" RETRACTED.** That compared
  ungauged-with-donors to gauged-WITHOUT-donors. Fair comparison (both
  with donors): median penalty **+0.048**, chalk **+0.132**. Donors
  lift both settings; the penalty is roughly unchanged. True statement:
  an ungauged site with donors matches a gauged site without them.
- **Operational framing tightened**: zero lead time (C1: lag-1 donors
  keep 14% of the gain); a zero-parameter nearest-donor rescale already
  achieves NSE 0.715 and beats the full raw tree in 34% of catchments —
  the floor any donor model must beat, to be reported alongside.
  Nestedness itself is secondary (C1 drop-nearest keeps 88%; gains
  uncorrelated with donor distance/correlation).
- **"The limit is in the data" REFRAMED** (both reviews converged on
  this independently): AMAX days are selected on the realisation, so a
  perfectly calibrated q99 of this sharpness would cover ~0.88 (C2
  oracle) — observed 0.896 is AT the oracle; the ladder sits at the
  information ceiling of the daily feature set (not "the data"), which
  hourly inputs raise. Also report date-clustered CIs ([0.883, 0.907])
  and event-level (non-)overlap alongside any convergence claim.
- Smaller mandated caveats: donor pool uses test-window availability
  (future information in donor selection); AMAX-bias median hides a
  near-zero mean (−1.0%) with compensating errors; top-1% NSE is a
  fragile subset metric; the tree ladder's sort-fix makes its q99 a
  max-of-six-fits (flatters it vs the LSTM's monotone head); hourly
  rain-gap rows (14%) sit inside all pilot numbers; per-catchment
  AMAX coverage is unvalidatable at n=12 events (p10 = 0.74).
- Survives as-is (conclusions review): the Gate 1 chalk correction
  (called "the methodological high point"), the GW shuffled null, the
  flat shuffled-donor control, pooled/per-catchment ladder calibration,
  the 1.42× daily-clipping measurement, the missed-event anatomy, the
  bankfull threshold-skill reframe, and the fold-rotation heavy-tail
  finding.
- **Arc hardening DONE (H1–H4), verified on this machine:**
  * Seed replication of LSTM+donors: median NSE +0.910 (0.906–0.914) —
    robust; top-1% NSE +0.136 (0.085–0.201) — positive on every seed,
    the sign claim survives; **AMAX bias −10.1% mean (−6.6 / −11.8 /
    −11.8) — seed 0 was the optimistic outlier and the advantage over
    the nowcast tree (−9.0%) is NOT robust: a tie.** Retract "−6.6%,
    inside rating uncertainty". Val curves peak at ep 10–11 (+0.88),
    consistent with the fixed 16-epoch budget. Defensible headline model
    = the 3-seed prediction mean (+0.917 / +0.185 / −10.1%).
  * Hourly deconfound (`--donors 0`): rain-only recovers the same share
    of the 145 missed events as with donors (75.2% vs 76.6%,
    own-comparator; second hourly seed 75.2%) at lower sharpness; donors
    buy point skill (q50 daily-agg NSE +0.88 vs +0.77, ~40× the seed
    spread). **Event recovery is hourly rainfall; donors are point
    skill.** Exactly what the missed-event anatomy (55% sub-q90 daily
    rain) predicted. `results/hourly_deconfound.csv`.
  * Still optional: a held-back sub-window (last 4 water years)
    rescoring of the final models only.

### Phase 5 (2026-08-30): hardening + adversarial review

Before the paper: make the headline numbers robust and stress-test the
reasoning. Two hostile-reviewer passes (methodology; conclusions) are
running on the CPU box — their findings may append runs to this list.

**Arc box queue (start immediately; independent of the review):**

- H1/H2 **Seed replication of the headline model** (~1.5 h each). The
  +0.914 / top-1% +0.201 / AMAX −6.6% synthesis is a single seed, and
  the audit showed LSTM tail metrics carry seed spread comparable to
  model gaps:
  `python experiments/lstm/train_lstm.py --donors 3 --epochs 16 --seed 1 --out experiments/results/lstm_nowcast_s1`
  `python experiments/lstm/train_lstm.py --donors 3 --epochs 16 --seed 2 --out experiments/results/lstm_nowcast_s2`
  Also copy each run's per-epoch val NSE line into the commit message —
  we need the val curves to defend the fixed 16-epoch budget (the
  original 8-vs-16 choice was made on test cards; the defence is
  "fixed pre-registered budget, val curve consistent with it").
- H3 **Hourly deconfound** (~1 h): the pilot changed hourly rain AND
  hourly donors at once; this splits the attribution:
  `python experiments/lstm/train_lstm_hourly.py --head quantile --donors 0 --out experiments/results/lstm_hourly_q_nodonor`
  (train_lstm_hourly.py now takes --donors; 0 = rain-only.) The number
  to report: daily-agg q99 coverage of the 145 pilot both-missed events
  (was 72–77% with donors; the rain-only number attributes the recovery).
- H4 **Second hourly seed** (added by review): rerun A2's config with
  `--seed 1 --out experiments/results/lstm_hourly_q_s1` after H3.
- Commit parquets + manifests + logs as usual; push. UTF-8 only.

**Arc box status (2026-08-30 am): H1, H2 DONE and committed; H3 running.**
Seed 0 was the tail outlier: seeds 1/2 give median NSE +0.910/+0.906,
top-1% NSE +0.122/+0.085, AMAX bias −11.8%/−11.8% (seed 0: +0.914 /
+0.201 / −6.6%). Robust vs the nowcast tree: ordinary days (+0.02–0.03,
better on 76–80%) and top-1% NSE (all seeds positive vs −0.10). Not
robust: AMAX bias (seed mean −10.1% ties the tree's −9.0%). Headline to
quote: 3-seed ensemble mean +0.917 / +0.185 / −10.1%
(`results/lstm_nowcast_seeds_cards.csv`; val curves in commit messages).
H3 DONE: rain-only hourly ladder covers 75.2% of the 145 pilot both-missed events on daily aggregates (with donors 76.6%), 70.3% at hourly peak (69.0%), at lower sharpness (q99/obs 1.34 vs 1.21) — the event recovery is attributable to hourly rain, not donors; donors buy q50 skill (+0.880 vs +0.772). `results/hourly_deconfound.csv`. All three Phase 5 Arc runs committed and pushed. H4 DONE (second hourly seed): daily-agg event coverage 75.2% (seed 0 76.6%), hourly peak 66.2% (69.0%), q50 NSE +0.883 (+0.880) — the 1.4 pp seed spread equals the donors-vs-rain-only gap, so the envelope tie is within noise while the point-skill gap (+0.88 vs +0.77) is ~40× it. `results/hourly_deconfound.csv` has all three rows. Arc queue empty (2026-08-30 08:00).

**CPU box queue:**

- C1 **Lead time + nestedness** (`hgb_nowcast_hardening.py`, 2 fits):
  (a) donors at lag-1/lag-2 only — what one day of lead time costs the
  nowcast result (any operational claim must state this); (b) donors at
  ranks 2–4 with the nearest excluded — the nearest donor is the one
  most likely up/downstream on the same river, where its flow partially
  IS the target's flow; the network-information claim must survive
  dropping it.
- C2 Adversarial-review fixes (`analysis_review_fixes.py`) — DONE, and
  every result strengthens the study:
  (a) **daily-null for the hourly claim**: a daily q99 inflated to match
  the hourly model's overall AMAX coverage captures only **4.1%** of the
  145 both-missed events (hourly: 72–77%) — the hourly recovery is
  information, not envelope width;
  (b) **oracle coverage**: because AMAX days are selected on the
  realisation, a PERFECTLY calibrated q99 of this ladder's sharpness
  would cover only ~0.88 of them (0.81–0.90 across tail assumptions) —
  observed 0.896 is at/above the oracle, so the daily ladder was never
  miscalibrated on floods; it sits at its information ceiling, and
  hourly data raises the ceiling. §3/§7 of the write-up need this
  reframe ("coverage" → "event capture"; drop the 0.99 target);
  (c) date-clustered 95% CI on AMAX coverage [0.883, 0.907] (742
  distinct event dates — report this everywhere coverage is compared);
  (d) nowcast gain is FLAT in donor distance (medians 0.056–0.058 by
  quartile out to 43 km, Spearman −0.004): the transferability curve.

**Phase 5 CPU results (C1, hgb_nowcast_hardening.py):**
- **Nestedness: the claim survives.** Donors at ranks 2–4 (nearest
  excluded) retain 88% of the nowcast NSE gain (paired median +0.050 vs
  +0.057) and AMAX bias −10.4% (full nowcast −9.0%, raw −17.4%). The
  gain is network information, not self-measurement through a nested
  gauge.
- **Lead time: the value is same-day.** Lag-1/2-only donors retain 14%
  of the NSE gain and AMAX −16.2% — with one day of lead the donor
  advantage essentially vanishes at daily resolution. All operational
  framing must say nowcasting, not forecasting; the flood-day
  information problem returns as soon as lead time is required.

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
