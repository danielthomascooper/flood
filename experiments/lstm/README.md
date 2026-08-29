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
