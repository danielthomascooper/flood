"""Regional LSTM on CAMELS-GB v2 daily data -- the clean comparison the tree
benchmark needs: identical 416 catchments, identical temporal split, identical
HadUK-Grid / Hydro-PE forcings. Writes test predictions in the exact format
experiments/evaluate.py scores, so results transfer between machines as one
parquet file.

Designed to run on a separate GPU box (Intel Arc via PyTorch XPU, NVIDIA via
CUDA) or CPU. Device is auto-detected: cuda > xpu > cpu.

Setup on the GPU machine (see experiments/lstm/README.md):
    pip install torch --index-url https://download.pytorch.org/whl/xpu   # Intel Arc
    pip install pandas pyarrow scikit-learn
Data needed: data/Catchment_Attributes/ and
data/Catchment_Timeseries/hydro-meteorological/daily/ (754 MiB).

Run:
    python experiments/lstm/train_lstm.py --out experiments/results
Resume-safe: checkpoints each epoch to <out>/lstm_checkpoint.pt.

Architecture follows the standard regional setup (Kratzert et al. 2019;
Lees et al. 2021): 365-day forcing sequences, statics concatenated to every
timestep, per-basin target normalisation so the MSE behaves like a basin-NSE
loss. Deliberately plain -- one LSTM layer, no attention, no embeddings --
because the question is whether learned state beats hand-built rolling
windows, not to win a leaderboard.
"""
import argparse, json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import STATIC, good_catchments, DAILY, TRAIN_END, TEST_START

SEQ = 365
FORCINGS = ["precipitation_haduk", "pet_hydrope", "temperature_haduk"]


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_basins(gauges):
    """Per-basin forcing/target arrays on a common daily index."""
    frames = {}
    for gid in gauges:
        f = sorted(DAILY.glob(f"*_{gid}_*.csv"))[0]
        d = pd.read_csv(f, parse_dates=["date"], na_values=["NaN"],
                        usecols=["date"] + FORCINGS + ["discharge_spec"]
                        ).set_index("date")
        frames[gid] = d.astype("float32")
    return frames


class Windows(torch.utils.data.Dataset):
    """(basin, t) pairs; __getitem__ slices the 365-day window ending at t."""

    def __init__(self, X, Y, S, index):
        self.X, self.Y, self.S, self.index = X, Y, S, index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        b, t = self.index[i]
        x = self.X[b][t - SEQ + 1:t + 1]                      # (365, n_forcing)
        s = np.broadcast_to(self.S[b], (SEQ, self.S[b].shape[0]))
        return np.concatenate([x, s], axis=1), self.Y[b][t], b


class LSTMModel(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.4):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1500, help="batches per epoch")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tail-weight", type=float, default=0.0,
                    help="alpha in per-sample MSE weight 1 + alpha*max(y_norm, 0); "
                         "0 = plain MSE. Upweights high-flow days.")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {dev}", flush=True)

    gauges = good_catchments()
    print(f"loading {len(gauges)} basins...", flush=True)
    frames = load_basins(gauges)

    # --- normalisation, train-window statistics only -------------------------
    train_end = pd.Timestamp(TRAIN_END)
    cat_tr = pd.concat([d.loc[:train_end, FORCINGS] for d in frames.values()])
    f_mean = cat_tr.mean().to_numpy("float32")
    f_std = cat_tr.std().to_numpy("float32") + 1e-6

    stat = STATIC.loc[gauges].copy()
    stat = stat.fillna(stat.median())
    s_z = ((stat - stat.mean()) / (stat.std() + 1e-6)).astype("float32")

    X, Y, S, y_stats = {}, {}, {}, {}
    tr_index, va_index, te_index, dates = [], [], [], {}
    val_start = train_end - pd.DateOffset(years=3)
    for b, d in frames.items():
        X[b] = ((d[FORCINGS].to_numpy("float32") - f_mean) / f_std)
        y = d["discharge_spec"].to_numpy("float32")
        ytr = d.loc[:train_end, "discharge_spec"]
        mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-6
        y_stats[b] = (mu, sd)
        Y[b] = (y - mu) / sd
        S[b] = s_z.loc[b].to_numpy("float32")
        dates[b] = d.index
        ok = ~np.isnan(y)
        for t in range(SEQ - 1, len(d)):
            if not ok[t]:
                continue
            dt = d.index[t]
            if dt <= train_end:
                (va_index if dt > val_start else tr_index).append((b, t))
            elif dt >= pd.Timestamp(TEST_START):
                te_index.append((b, t))
    print(f"windows: {len(tr_index):,} train / {len(va_index):,} val / "
          f"{len(te_index):,} test", flush=True)

    model = LSTMModel(len(FORCINGS) + s_z.shape[1], args.hidden).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    def lossf(pred, y):
        se = (pred - y) ** 2
        if args.tail_weight > 0:
            w = 1 + args.tail_weight * torch.clamp(y, min=0)
            return (w * se).sum() / w.sum()
        return se.mean()
    ckpt = out / "lstm_checkpoint.pt"
    start_ep = 0
    if ckpt.exists():
        state = torch.load(ckpt, map_location=dev)
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        start_ep = state["epoch"] + 1
        print(f"resumed from epoch {start_ep}", flush=True)

    def run_eval(index, sample=50_000):
        idx = [index[i] for i in rng.choice(len(index),
                                            min(sample, len(index)), replace=False)]
        dl = torch.utils.data.DataLoader(Windows(X, Y, S, idx), batch_size=1024,
                                         num_workers=args.workers)
        model.eval(); preds, obs = [], []
        with torch.no_grad():
            for xb, yb, _ in dl:
                preds.append(model(xb.to(dev)).cpu().numpy()); obs.append(yb.numpy())
        model.train()
        p, o = np.concatenate(preds), np.concatenate(obs)
        return 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()

    tr_ds = Windows(X, Y, S, tr_index)
    for ep in range(start_ep, args.epochs):
        sampler = torch.utils.data.RandomSampler(
            tr_ds, replacement=True, num_samples=args.steps * args.batch)
        dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch,
                                         sampler=sampler, num_workers=args.workers)
        t0, running = time.time(), 0.0
        for i, (xb, yb, _) in enumerate(dl):
            opt.zero_grad()
            loss = lossf(model(xb.to(dev)), yb.to(dev))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
        vnse = run_eval(va_index)
        print(f"epoch {ep}: loss {running/args.steps:.4f}  "
              f"val NSE(norm) {vnse:+.3f}  {time.time()-t0:.0f}s", flush=True)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "epoch": ep}, ckpt)

    # --- full test inference, harness format ---------------------------------
    print("test inference...", flush=True)
    dl = torch.utils.data.DataLoader(Windows(X, Y, S, te_index), batch_size=1024,
                                     num_workers=args.workers)
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _, _ in dl:
            preds.append(model(xb.to(dev)).cpu().numpy())
    p_norm = np.concatenate(preds)

    gid = np.array([b for b, _ in te_index], dtype=np.int32)
    mu = np.array([y_stats[b][0] for b, _ in te_index], dtype=np.float32)
    sd = np.array([y_stats[b][1] for b, _ in te_index], dtype=np.float32)
    obs = np.array([Y[b][t] for b, t in te_index], dtype=np.float32) * sd + mu
    idx = pd.DatetimeIndex([dates[b][t] for b, t in te_index], name="date")
    res = pd.DataFrame({"gid": gid, "obs": obs,
                        "pred": np.clip(p_norm * sd + mu, 0, None)}, index=idx)
    res.to_parquet(out / "lstm_test_predictions.parquet")
    (out / "lstm_manifest.json").write_text(json.dumps(
        {**vars(args), "device": str(dev), "n_basins": len(gauges),
         "train_end": TRAIN_END, "test_start": TEST_START, "seq": SEQ}, indent=2))
    print(f"wrote {out/'lstm_test_predictions.parquet'} ({len(res):,} rows)")
    print("score it back on the main machine with experiments/evaluate.py")


if __name__ == "__main__":
    main()
