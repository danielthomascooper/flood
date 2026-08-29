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
