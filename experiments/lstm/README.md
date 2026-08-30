# Regional LSTM on the GPU box

The clean tree-vs-LSTM comparison: identical 416 catchments, identical
1970–2010 / 2010–2022 temporal split, identical HadUK-Grid / Hydro-PE
forcings as `experiments/hgb_baseline.py`. The run produces one parquet of
test predictions that `experiments/evaluate.py` scores on the main machine —
so only that small file needs to travel back.

## Setup (Intel Arc 140T)

```bash
git clone https://github.com/danielthomascooper/flood && cd flood
python3 -m venv .venv && source .venv/bin/activate
# PyTorch with Intel XPU support (torch >= 2.5):
pip install torch --index-url https://download.pytorch.org/whl/xpu
pip install pandas pyarrow scikit-learn
python -c "import torch; print(torch.xpu.is_available())"   # expect True
```

(NVIDIA box instead: plain `pip install torch` — the script auto-detects
cuda → xpu → cpu.)

## Data needed (~760 MiB, not the full 10.6 GiB)

Only two folders:

```
data/Catchment_Attributes/                                  (~1 MiB)
data/Catchment_Timeseries/hydro-meteorological/daily/       (754 MiB)
```

Either rsync them from the main machine:

```bash
rsync -a main-machine:~/source/flood/data/Catchment_Attributes ./data/
rsync -a main-machine:~/source/flood/data/Catchment_Timeseries/hydro-meteorological/daily \
      ./data/Catchment_Timeseries/hydro-meteorological/
```

or re-download with `aria2c -i urls-aria.txt -c -x4 -j8` after filtering
`urls-aria.txt` to those two path prefixes.

## Run

```bash
python experiments/lstm/train_lstm.py --out experiments/results
```

Defaults: 365-day sequences, hidden 128, batch 256, 8 epochs × 1500 batches
(~3M training sequences drawn from the 5.5M available). Checkpoints every
epoch to `experiments/results/lstm_checkpoint.pt`; rerunning the same
command resumes. Watch `val NSE(norm)` — it should climb past +0.6 within a
few epochs; if it plateaus early, add `--epochs 16`.

Rough expectations on an Arc 140T (16 GB): a few minutes per epoch, then
10–20 min of test inference — a coffee-length run end to end. If XPU memory
complains, drop `--batch 128`.

## Bring the result back

Copy these to the main machine's `experiments/results/`:

```
lstm_test_predictions.parquet    # gid / obs / pred, indexed by date
lstm_manifest.json               # config + split provenance
```

Then score it against the tree:

```python
import sys; sys.path.insert(0, "experiments")
import pandas as pd
from evaluate import evaluate, report
res = pd.read_parquet("experiments/results/lstm_test_predictions.parquet")
report([evaluate(res, "lstm")[0]])
```

The numbers to beat, from `experiments/results/`: median NSE +0.820
(tree, gauged), top-1% NSE −0.811, AMAX bias −17.4%. The interesting
question is not the headline NSE but whether learned state narrows the
flood-day gap the tree could not: top-1% NSE and AMAX bias are the columns
to watch.

## Results so far (Arc Pro 140T, 2026-08-28/29)

All runs: hidden 128, batch 256, 1500 batches/epoch, lr 1e-3. Cards from
`experiments/evaluate.py`, collected in `results/lstm_cards.csv`; each run's
parquet, manifest, per-catchment metrics and training log sit in
`results/lstm_<run>/` (the 8-epoch seed-0 baseline is in `results/` itself).

| model | median NSE | %NSE<0 | top-1% NSE | top-1% bias | AMAX bias | q99 bias | coverage |
|---|---|---|---|---|---|---|---|
| tree (hgb raw) | +0.820 | 2.6 | −0.811 | −23.9% | −17.4% | −11.5% | 0.560 |
| lstm 8 ep, seed 0 | +0.852 | 0.0 | −0.630 | −25.1% | −19.4% | −14.5% | 0.563 |
| lstm 16 ep, seed 0 | **+0.855** | 0.0 | −0.572 | −23.9% | −19.2% | −14.2% | 0.553 |
| lstm 8 ep, seed 1 | +0.847 | 0.0 | −0.730 | −25.9% | −21.6% | −16.1% | 0.580 |
| lstm 8 ep, seed 0, `--tail-weight 1` | +0.840 | 0.0 | **−0.521** | −24.3% | −19.5% | −12.6% | 0.635 |

Val NSE(norm) per epoch is in each `lstm_train.log` (seed 0: +0.74 after
one epoch, +0.82 at 8, +0.83 at 16). ~5.5 min/epoch and ~5 min test
inference when the GPU is not shared; three runs in parallel each take ~3x
longer, so there is no throughput gain from overlapping them.

What it says:

- **Ordinary days: LSTM wins, robustly.** +0.03 median NSE over the tree and
  no failed catchments, reproduced across seeds (±0.005).
- **Flood peaks: no better than the tree.** AMAX bias −19 to −22% vs the
  tree's −17%; seed-to-seed spread on the tail metrics (~2 points of AMAX
  bias, ~0.1 of top-1% NSE) is as large as the LSTM–tree gap, so treat the
  two as tied there. Learned state fixes ordinary-day dynamics, not the
  systematic peak under-prediction.
- **The weak-catchment problem is gone — chalk included, but not chalk
  specifically.** Paired per-catchment against the tree (16 ep, seed 0):
  all 23 chalk catchments improve (+0.728 → +0.907) and the biggest wins
  are chalk/groundwater rivers (Law Brook −5.89 → +0.63, Mimram, Ver),
  with worst losses anywhere only ~−0.09. **Correction (2026-08-29
  audit)** of the "multi-year aquifer memory" reading committed earlier:
  the windows here are stateless 365-day sequences (the tree's `p_sum365`
  horizon — multi-year memory is impossible); the gain concentrates on
  *all* low-NSE / low-variance catchments (non-chalk with tree NSE<0.6
  gain +0.45 median; variance-matched chalk residual p≈0.11); and
  per-basin target normalisation — which makes the loss basin-NSE-like —
  is the live alternative driver. The `--seq 90` run and the tree
  per-catchment norm refit (PLAN.md Phase 2) are the discriminating
  tests.
- **More epochs help a little** everywhere (16 ep is best on every column
  except coverage) and the val curve was still creeping up; worth 32 if
  the LSTM is pursued.
- **Tail weighting (α=1) moves the wrong knob:** best top-1% NSE and q99
  bias of any model, but AMAX bias unchanged, median NSE −0.012 and coverage
  up to 0.635 — it inflates moderately high days rather than the annual
  peaks. A steeper weight, or weighting AMAX days specifically, would be
  the next test, at further ordinary-day cost.

`--tail-weight ALPHA` weights each sample's squared error by
`1 + ALPHA * max(y_norm, 0)` (normalised by the weight sum); 0 = plain MSE.
To extend a finished run, copy its `lstm_checkpoint.pt` into a new `--out`
directory and rerun with a larger `--epochs`; the script resumes from it.

## Phase 2, Arc box (2026-08-29)

**A1 horizon test, `--seq 90`** (8 ep, seed 0; `results/lstm_seq90/`,
paired numbers in `results/lstm_seq_paired.csv`, cards in
`results/lstm_seq_cards.csv`). 90-day windows train 4x faster (~90 s/epoch)
and reach val NSE(norm) +0.803 vs +0.824 at 365 days.

| paired NSE gain vs tree | seq 365 | seq 90 | survives |
|---|---|---|---|
| all 416 | +0.023 | +0.018 | 78% |
| weak-tree (NSE<0.6, n=35) | +0.426 | +0.372 | 87% |
| chalk (n=23) | +0.098 | +0.046 | 47% |

Paired seq90 − seq365: −0.005 overall, −0.045 weak-tree, **−0.057 chalk**.
Reading: the weak-catchment rescue is mostly *not* horizon — it survives
a 90-day window, consistent with per-basin normalisation being the driver
(Gate 1 tree-side refit will confirm). Chalk is the exception: half its
gain needs the 90–365-day range, i.e. the LSTM extracts more from the
same horizon than the tree's `p_sum180/365` do. Card-wise the 90-day model
is still ahead of the tree (median NSE +0.840, no failed catchments) with
the same peak under-prediction (AMAX bias −19.8%). A3 (`--seq 730`) is
queued to test whether horizon beyond 365 days adds anything for chalk.

**A2 quantile head, `--head quantile --epochs 16`** (seed 0;
`results/lstm_q/`, cards `results/lstm_q_cards.csv`, calibration
`results/lstm_q_calibration.csv`). Joint monotone pinball ladder
q05/q25/q50/q75/q95/q99; zero crossing rows (the tree sweep had 28.6%).
Trains as easily as the MSE head (q50 val NSE(norm) +0.750 after one epoch,
+0.835 peak, +0.831 final).

| point forecast | median NSE | top-1% NSE | AMAX bias | coverage |
|---|---|---|---|---|
| tree mean | +0.820 | −0.811 | −17.4% | 0.560 |
| lstm mse (16 ep) | +0.855 | −0.572 | −19.2% | 0.553 |
| lstm q50 | **+0.858** | −0.672 | −23.5% | 0.456 |
| lstm q95 as point | +0.643 | −0.505 | +22.0% | 0.954 |

Calibration (fraction of obs ≤ quantile), nominal → pooled / per-catchment
median / **AMAX days**: q50 0.50 → 0.444 / 0.456 / 0.08; q95 0.95 →
0.935 / 0.954 / 0.63; q99 0.99 → 0.985 / 0.992 / **0.85**. Median
interval widths 0.22 mm/day (50%) and 0.54 (90%) vs the tree sweep's
0.24 / 0.64 — sharper at the same pooled calibration. On AMAX days q99
sits at a median 1.31× obs and q50 at 0.68× obs.

Reading for Gate 2: (a) the median is a worse flood point forecast than
the mean, as it was for the tree (−23.5% vs −19.2% AMAX bias) — the
statistic, not the model family, drives peak bias; (b) pooled calibration
hides flood-day miscalibration exactly as suspected — q99 is nominal
pooled and per-catchment (p10 0.966) but misses 15% of annual maxima and
13% of top-1% days; (c) the lower half of the ladder is biased low
(q25 covers 0.185, q50 0.444) while the upper half is on target, so the
model's spread is right but its centre is low on ordinary days too.

**A3 horizon test, `--seq 730`** (8 ep, seed 0; `results/lstm_seq730/`).
Card: median NSE +0.857, KGE +0.835 (best of any run on both), AMAX bias
−20.8%, top-1% NSE −0.560. Paired seq730 − seq365 at equal epochs:
**+0.002 all, +0.013 chalk, +0.017 weak-tree**, 730 better on 55% of
catchments — inside the seed spread (±0.005). The 90→365 step was worth
+0.057 on chalk; 365→730 is worth nothing measurable.

### Gate 1 verdict (Arc side; full table `results/gate1_memory_vs_normalisation.csv`)

| paired NSE gain vs raw tree | weak-tree (35) | chalk (23) |
|---|---|---|
| tree `norm` (C1) | +0.185 | +0.051 |
| tree `log1p` (C1) | +0.268 | +0.079 |
| lstm seq 90 | +0.372 | +0.046 |
| lstm seq 365 (8 ep) | +0.426 | +0.098 |
| lstm seq 730 | +0.458 | +0.097 |
| lstm seq 365 (16 ep) | +0.457 | +0.122 |

- The weak-catchment rescue is **loss scaling**: the tree's `log1p`
  recovers 74% of it (C1) and the LSTM keeps 87% of it with only 90 days
  of input (A1). What remains over `log1p` (~+0.1–0.2 on weak catchments)
  is the LSTM being a better function of the same inputs, not memory.
- Chalk has a real but small horizon component in the **90–365-day**
  range (+0.05), which the tree's `log1p` transform already captures most
  of (+0.079 vs +0.098). Nothing beyond 365 days (A3).
- "Memory beyond the tree's rolling windows" is dead; the cell-state vs
  groundwater-well probe is not worth building on this evidence. The
  honest chalk claim is: *per-basin loss normalisation plus a 365-day
  window fixes the chalk failure; the tree gets most of the way there
  with `log1p`.*

## Phase 3, Arc box (2026-08-29 evening)

**B1 LSTM + 3 nearest-gauge donors, point head, `--donors 3 --epochs 16`**
(`results/lstm_nowcast/`; cards `results/lstm_nowcast_cards.csv`, paired
`results/lstm_nowcast_paired.csv`). Donor build takes 20 s on this box
(median donor distance 13.2 km, as in `hgb_nowcast.py`; no self-donors).
Val NSE(norm) +0.832 after one epoch (the no-donor runs took 16 epochs to
get there), +0.883 final.

| | tree raw | tree nowcast | lstm (no donors) | **lstm + donors** |
|---|---|---|---|---|
| median NSE | +0.820 | +0.882 | +0.855 | **+0.914** |
| median KGE | +0.815 | +0.859 | +0.822 | **+0.891** |
| % catchments NSE<0 | 2.6 | 1.9 | 0 | **0** |
| top-1% NSE | −0.811 | −0.099 | −0.572 | **+0.201** |
| top-1% bias | −23.9% | −14.1% | −23.9% | **−10.6%** |
| AMAX bias | −17.4% | −9.0% | −19.2% | **−6.6%** |
| q99 dist bias | −11.5% | −5.4% | −14.2% | **−2.2%** |

Paired per-catchment vs the nowcast tree: +0.021 median NSE, LSTM better
on **80%** of catchments; chalk +0.079 (96% better); weak-tree +0.313
(94% better). Without donors the LSTM was *behind* the nowcast tree
(−0.025, better on 37%). So the two wins are additive: donors supply the
flood-day information, normalisation supplies the weak-catchment fit,
and the combination is the first model in the project with a positive
top-1% NSE and single-digit AMAX bias.

**B2 LSTM + donors, quantile head, `--donors 3 --head quantile --epochs 16`**
(`results/lstm_qnow/`; cards `results/lstm_nowcast_cards.csv`, calibration
`results/lstm_qnow_calibration.csv`). Zero crossing rows; q50 val NSE(norm)
+0.880 final (+0.884 peak), pinball loss 0.041 vs 0.047 without donors.

AMAX-day coverage (fraction of annual maxima ≤ quantile; nominal 0.99 / 0.95):

| ladder | q99 on AMAX days | q95 on AMAX days | q99 pooled | 90% width mm/day |
|---|---|---|---|---|
| tree, no donors | 0.829 | 0.630 | 0.983 | 0.64 |
| lstm, no donors (A2) | 0.847 | 0.625 | 0.985 | 0.54 |
| tree + donors (CPU C1) | 0.896 | 0.745 | 0.983 | 0.54 |
| **lstm + donors (B2)** | **0.892** | 0.725 | 0.986 | **0.52** |

Point forecasts: q50 median NSE +0.909 (mean head B1 +0.914), top-1% NSE
−0.010, AMAX bias **−15.0%** vs the mean head's −6.6% — the
median-under-shoots-peaks pattern replicates a fourth time (tree q50
−30.5%, tree nowcast q50 −16.4%, LSTM q50 −23.5%, this −15.0%). Paired vs
the nowcast tree: +0.018 median, better on 77% of catchments; chalk
+0.086 (100%); weak-tree +0.338. On AMAX days q50 sits at 0.80× obs
(0.68× without donors) and q99 at 1.36×.

Reading: donors lift flood-day envelope coverage by ~4.5 pp in *both*
model classes and to the *same* level (~0.89–0.90); the LSTM ladder is
marginally sharper but no better on the tail. The LSTM's advantage over
the tree is entirely in the centre and in weak/chalk catchments. The
~10% of annual maxima that neither calibrated ladder flags, even with
same-day neighbour flows, is model-class-independent — consistent with
the Gate 2 candidates that are not daily-scale modelling problems
(sub-daily rain intensity, peaks beyond the gauged range).

### Phase 3 Arc summary

Best point model in the project: B1 (`lstm_nowcast`), median NSE +0.914,
AMAX bias −6.6%, top-1% NSE +0.201, no failed catchments. Best envelope:
either donor ladder (tree or LSTM, ~0.89 AMAX-day q99 coverage); use the
LSTM one if the point forecast matters too, since it comes from the same
model. Timings on the Arc: donor build 20 s, ~4.5–6 min/epoch, 16 epochs
+ inference ≈ 1.5 h per run.

## Phase 4, Arc box — hourly pilot (2026-08-29 night)

Data: 198 hourly files (3.0 GB) fetched from CEH with parallel curl. Blended
rain `gradgb.fillna(cehgear)` still has 1,203 NaN hours per gauge (77
product-wide gradgb outages in 2017–22, identical across gauges); these are
zero-filled and the hourly parquet carries `rain_gap` = NaN rain hours inside
the 336-h window (14% of test rows > 0) so the scorer can flag them.
8.75M train / 6.28M test hourly windows; ~4m45s per epoch, ~17 min inference.

**A1 hourly, mse head** (`results/lstm_hourly/`, 10 ep): val NSE(norm) +0.719
after one epoch, +0.811 peak, +0.810 final. Local sanity card on the 60 pilot
catchments (the rigorous scoring is on the main machine):

| model (60 pilot catchments) | median NSE | top-1% NSE | AMAX bias | q99 bias |
|---|---|---|---|---|
| tree nowcast | +0.831 | −0.745 | −23.1% | −12.9% |
| daily LSTM + donors (B1) | +0.867 | −0.183 | −16.9% | −7.8% |
| **hourly LSTM, daily-mean aggregate** | **+0.869** | −0.212 | **−15.7%** | −12.1% |
| hourly LSTM, hourly resolution | +0.841 | −0.128 | −21.0% | −10.6% |

Hourly AMAX / daily-mean AMAX: median **1.42×**, q90 2.29× — daily averaging
understates true peaks by ~40%. On daily aggregates the hourly model is at
parity with the daily donor LSTM; the pilot's real question (does the hourly
*envelope* cover the 225 both-missed events, 145 of which are on pilot
catchments) waits for A2's quantile head.

**A2 hourly, quantile head** (`results/lstm_hourly_q/`, 10 ep): q50 val
NSE(norm) +0.806 peak, +0.805 final. The full hourly ladder parquet is
222 MB (over GitHub's limit, gitignored); the committed
`lstm_hourly_test_predictions_slim.parquet` (98 MB) keeps
`obs / pred(=q50) / q95 / q99 / rain_gap`, and `lstm_hourly_daily_agg.parquet`
(12 MB) carries the full six-quantile ladder as daily means.

Local sanity numbers on the 60 pilot catchments (rigorous scoring on the
main machine):

- Daily-aggregate q50 card: median NSE **+0.880** (mse head +0.869, daily
  donor LSTM +0.867), AMAX bias −18.0% (q50 pattern again), 1 failed
  catchment.
- **Both-missed events** (145 of the 225 are on pilot catchments; by
  construction 0% were inside either daily ladder's q99):
  daily-aggregate hourly-model q99 ≥ daily obs on **76.6%** of them;
  hourly q99 (max over the day) ≥ the hourly peak on **69.0%**, median
  q99/peak 1.21. Over *all* pilot-catchment AMAX days the hourly q99
  covers the hourly peak on 94.2%.

Reading: the pilot's success criterion is met — most of the events that
were invisible to both daily model classes fall inside the hourly
model's nominal-99% envelope. Caveat the main-machine scoring should
separate: the hourly model sees both hourly rain *and* hourly donor
flows, so this does not by itself attribute the gain to rain intensity
vs. finer-resolution neighbour flows; an hourly run with daily-mean rain
(or without donors) is the discriminating follow-up. The 14% of test
rows with `rain_gap > 0` are included above.

## Phase 5, Arc box — hardening (2026-08-30)

**H1/H2 seed replication of the headline donor model** (`--donors 3
--epochs 16`, seeds 1 and 2; `results/lstm_nowcast_s1/`, `results/lstm_nowcast_s2/`;
cards `results/lstm_nowcast_seeds_cards.csv`; per-epoch val curves in each
run's log and commit message).

| | tree nowcast | seed 0 | seed 1 | seed 2 | 3-seed mean | 3-seed ensemble |
|---|---|---|---|---|---|---|
| median NSE | +0.882 | +0.914 | +0.910 | +0.906 | +0.910 | **+0.917** |
| median KGE | +0.859 | +0.891 | +0.879 | +0.871 | +0.880 | +0.886 |
| % catchments NSE<0 | 1.9 | 0 | 0 | 0 | 0 | 0 |
| top-1% NSE | −0.099 | +0.201 | +0.122 | +0.085 | +0.136 | +0.185 |
| top-1% bias | −14.1% | −10.6% | −14.4% | −14.5% | −13.2% | −12.9% |
| AMAX bias | −9.0% | −6.6% | −11.8% | −11.8% | −10.1% | −10.1% |
| q99 dist bias | −5.4% | −2.2% | −7.1% | −7.9% | −5.7% | −5.6% |
| paired vs nowcast tree | — | +0.021 (80%) | +0.020 (80%) | +0.017 (76%) | | |

Val NSE(norm) curves are near-identical across seeds (peaks +0.883 /
+0.883 / +0.881 at epochs 11–15; the 16-epoch budget was fixed in advance
and every curve is still flat-to-rising at the end).

**Verdict.** Seed 0 was the tail outlier; seeds 1 and 2 agree with each
other to within 0.04 top-1% NSE and 0.0 pts AMAX bias. Robust claims:
the ordinary-day win over the nowcast tree (+0.02–0.03 median NSE, better
on 76–80% of catchments, no failed catchments) and the top-1% NSE win (all
seeds positive vs −0.10). **Not robust:** the AMAX-bias advantage — the
seed mean (−10.1%) ties the nowcast tree (−9.0%). The defensible headline
is the 3-seed ensemble mean: median NSE +0.917, top-1% NSE +0.185, AMAX
bias −10.1%. Any tail claim in the write-up should quote the seed range,
not seed 0.

**H3 hourly deconfound, `--head quantile --donors 0`** (rain-only hourly
pilot; `results/lstm_hourly_q_nodonor/`, slim ladder + daily aggregate
committed as for A2; comparison table `results/hourly_deconfound.csv`).
q50 val NSE(norm) +0.725 peak, +0.705 final (donor version +0.805).

| 60 pilot catchments, 145 both-missed events | hourly + donors (A2) | **rain only (H3)** |
|---|---|---|
| daily-agg q99 ≥ obs on the event day | 76.6% | **75.2%** |
| hourly q99 (max over day) ≥ hourly peak | 69.0% | **70.3%** |
| all pilot AMAX days, hourly q99 ≥ peak | 94.2% | 96.7% |
| median q99 / obs on the events | 1.21 | 1.34 |
| daily-agg q50 median NSE | +0.880 | +0.772 |
| daily-agg q50 AMAX bias | −18.0% | −12.6% |

**Attribution.** The pilot's event recovery is a property of hourly
*rain*, not of hourly donor flows: without any neighbour-gauge input the
envelope still covers three-quarters of the events every daily model
missed, at slightly lower sharpness (q99 sits 1.34× obs instead of
1.21×). Donors buy point skill (q50 NSE +0.11) and a tighter ladder, not
event coverage. This is the cleaner claim for the write-up — it does not
depend on the donor network being available at an ungauged site.

*Comparator caveat (from the Phase 5 review synthesis, commit 5c10420):*
the 72–77% and 75% figures above are scored against the hourly product's
own observations, which run at a median 85.5% of the daily-file obs on
those event days; scored like-for-like against the daily obs the donor
pilot covers 51.7% exact / 64.1% ±1 day, against a 24.8% no-donor
ladder-union baseline — a ~2.6× recovery, not "77 vs 0". The H3
*comparison* (rain-only vs donors) is unaffected because both runs use
the same scoring; the absolute level should be quoted from the review's
same-comparator numbers.

**H4 second hourly seed, `--head quantile --seed 1`** (added by the review;
`results/lstm_hourly_q_s1/`, slim ladder + daily aggregate committed; third
row of `results/hourly_deconfound.csv`). q50 val NSE(norm) +0.818 peak,
+0.808 final (seed 0: +0.806 / +0.805).

| 60 pilot catchments, 145 both-missed events | donors, seed 0 (A2) | **donors, seed 1 (H4)** | rain only (H3) |
|---|---|---|---|
| daily-agg q99 ≥ obs on the event day | 76.6% | **75.2%** | 75.2% |
| hourly q99 (max over day) ≥ hourly peak | 69.0% | **66.2%** | 70.3% |
| all pilot AMAX days, hourly q99 ≥ peak | 94.2% | 92.9% | 96.7% |
| median q99 / obs on the events | 1.21 | 1.19 | 1.34 |
| daily-agg q50 median NSE | +0.880 | **+0.883** | +0.772 |
| daily-agg q50 AMAX bias | −18.0% | −18.8% | −12.6% |

The hourly pilot replicates across seeds to within 1.5 pp on daily-agg
event coverage and 3 pp at hourly peak; the seed spread (1.4 pp) equals
the donors-vs-rain-only gap (1.4 pp), so the equal event coverage of the
rain-only and donor ladders is a within-noise tie, whereas the q50 skill
gap (+0.88 vs +0.77) is ~40× the seed spread. Both readings of H3 stand:
event recovery is hourly rain; point skill is donors. (Comparator caveat
above applies to the absolute coverage levels.)

## Phase 6, Arc box — forecasting (2026-08-30)

`train_lstm.py --lead L --autoreg`: the window ends at issue day *t*, the
target is flow at *t+L*, the basin's own normalised observed flow (≤ *t*)
is an input channel, donors are at ≤ *t*, and the train/test split is on
the target date. Output rows are dated at the target day, so `evaluate.py`
scores them unchanged; the local scoring script builds persistence
(flow(*t+L*) = flow(*t*)) from the daily files for the same rows.

**The bar: persistence** on the 416-catchment test set — lead 1: median
NSE **+0.538**, top-1% NSE −3.25, AMAX bias 0.0% (the peak simply arrives a
day late); lead 3: median NSE **+0.077**, 43% of catchments negative.
Persistence is unbeatable on AMAX *bias* by construction, so the forecast
question is timing and flood-day skill (top-1% NSE), not bias.

**F1 1-day-ahead LSTM, `--lead 1 --autoreg --donors 3 --epochs 16`**
(`results/lstm_fc1/`; cards `results/lstm_forecast_cards.csv`, paired
`results/lstm_forecast_vs_persistence.csv`). Val NSE(norm) +0.693 after one
epoch, +0.764 peak, +0.759 final — ~0.12 below the lead-0 nowcast, the
price of the day of lead.

| lead 1, 416 catchments | persistence | F1 |
|---|---|---|
| median NSE / KGE | +0.538 / +0.769 | **+0.811 / +0.820** |
| % catchments NSE<0 | 2.6 | 0 |
| top-1% NSE | −3.25 | −1.51 |
| top-1% bias | −37.7% | −32.1% |
| AMAX bias | 0.0% (by construction) | −17.4% |
| pred max, mm/day (obs 244) | 244 | 93 |
| paired vs persistence | — | +0.253 median, better on 94% |
| bias on the observed AMAX day | −100% + a day-late peak | −34.3% |
| model's own AMAX within ±1 day of obs | — | 52.2% (median lag 0) |

Reading: a day ahead, own flow + donors + forcings recover essentially the
tree's *simulation* skill on ordinary days (+0.811 vs +0.820) and beat
persistence almost everywhere. On flood days the forecaster looks like the
raw daily tree: −17% AMAX bias and a 93 mm/day ceiling. Persistence's
0% AMAX bias is not skill — it is the peak arriving a day late — so the
flood-day comparison to make is top-1% NSE (−1.5 vs −3.3) and, once F2
lands, calibrated envelope coverage. Timing is the encouraging part: on
the observed annual-max day the forecast issued the day before carries
−34% bias (persistence, by definition, carries the previous day's flow),
and the model's own annual maximum falls within a day of the true one in
52% of catchment-years with a median lag of 0 days — it sees the peak
coming, it just under-calls its size.
