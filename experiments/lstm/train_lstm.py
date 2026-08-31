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
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import STATIC, good_catchments, DAILY, TRAIN_END, TEST_START

SEQ = 365  # overridden by --seq
FORCINGS = ["precipitation_haduk", "pet_hydrope", "temperature_haduk"]
ALPHAS = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)  # --head quantile ladder


def to_quantiles(raw):
    """Monotone ladder: first output is q05, the rest are positive increments,
    so the quantiles cannot cross (unlike the tree sweep's 28.6% of rows)."""
    return torch.cat([raw[:, :1],
                      raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), 1)], 1)


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


def donor_features(gauges, k, train_end):
    """Same-day + lag-1 observed flow at each basin's k nearest usable gauges
    (scaled by the donor's own train-window q95), the feature set that halved
    the tree's AMAX bias in hgb_nowcast.py. Donor pool is every gauge in the
    daily folder with >=50% coverage in both windows; a basin is never its
    own donor. Gaps ffilled up to 7 days then set to 0."""
    from common import read_attr
    cols = {}
    for f in sorted(DAILY.glob("*.csv")):
        gid = int(re.search(r"_(\d+)_\d{8}-\d{8}\.csv$", f.name).group(1))
        cols[gid] = pd.read_csv(f, parse_dates=["date"], na_values=["NaN"],
                                usecols=["date", "discharge_spec"]
                                ).set_index("date")["discharge_spec"].astype("float32")
    F = pd.DataFrame(cols)
    tr = F.loc[:train_end]
    usable = F.columns[(tr.notna().mean() >= 0.5)
                       & (F.loc[pd.Timestamp(TEST_START):].notna().mean() >= 0.5)]
    D0 = (F[usable] / tr[usable].quantile(0.95)).astype("float32")

    topo = read_attr("topographic").set_index("gauge_id")
    ex = topo.gauge_easting.astype(float)
    ny = topo.gauge_northing.astype(float)
    dx = ex.loc[gauges].values[:, None] - ex.loc[usable].values[None, :]
    dy = ny.loc[gauges].values[:, None] - ny.loc[usable].values[None, :]
    Dk = np.sqrt(dx ** 2 + dy ** 2)
    self_col = {g: j for j, g in enumerate(usable)}
    out = {}
    for i, g in enumerate(gauges):
        row = Dk[i].copy()
        if g in self_col:
            row[self_col[g]] = np.inf
        donors = usable[np.argsort(row)[:k]]
        d0 = D0[donors]
        feats = pd.concat([d0, d0.shift(1)], axis=1)
        out[g] = feats.ffill(limit=7).fillna(0.0).astype("float32")
    return out


class Windows(torch.utils.data.Dataset):
    """(basin, t) pairs; __getitem__ slices the 365-day window ending at t.

    Future-rain channels (--fcrain): channel k (of L, the last L dynamic
    columns, first of them at column fc0) holds rain(tau+k) at timestep tau
    -- observed rain wherever tau+k <= t (known by issue day), and for the
    last k steps of the window (tau+k > t, genuinely future) the value from
    FC: the forecast issued on day tau. FC is None for the perfect-rain
    ceiling, where the observed value simply stays in place."""

    def __init__(self, X, Y, S, index, seq=SEQ, lead=0, FC=None, fc0=0):
        self.X, self.Y, self.S, self.index = X, Y, S, index
        self.seq = seq  # carried on the instance: spawned DataLoader workers
                        # (Windows/macOS) re-import this module and would
                        # otherwise see the default SEQ, not --seq
        self.lead = lead
        self.FC, self.fc0 = FC, fc0

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        b, t = self.index[i]
        x = self.X[b][t - self.seq + 1:t + 1]                 # (seq, n_forcing)
        s = np.broadcast_to(self.S[b], (self.seq, self.S[b].shape[0]))
        arr = np.concatenate([x, s], axis=1)
        if self.FC is not None:
            for k in range(1, self.lead + 1):
                arr[self.seq - k:, self.fc0 + k - 1] = \
                    self.FC[b][t - k + 1:t + 1, k - 1]
        return arr, self.Y[b][t + self.lead], b


class LSTMModel(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.4, n_out=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_out))

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
    ap.add_argument("--seq", type=int, default=365,
                    help="input window length in days (the model sees nothing "
                         "beyond this horizon)")
    ap.add_argument("--head", choices=["mse", "quantile"], default="mse",
                    help="quantile = joint monotone pinball head over "
                         f"{ALPHAS}; pred column is q50, ladder columns "
                         "q05..q99 added to the parquet. Ignores --tail-weight.")
    ap.add_argument("--tail-weight", type=float, default=0.0,
                    help="alpha in per-sample MSE weight 1 + alpha*max(y_norm, 0); "
                         "0 = plain MSE. Upweights high-flow days. mse head only.")
    ap.add_argument("--donors", type=int, default=0,
                    help="K nearest-gauge nowcast features (same-day + lag-1 "
                         "q95-scaled observed flow per donor); 0 = off")
    ap.add_argument("--lead", type=int, default=0,
                    help="forecast lead in days: the window ends at day t and "
                         "the target is flow at t+lead. 0 = simulation (default)")
    ap.add_argument("--basins", type=int, default=0,
                    help="use only the first N gauges (0 = all); smoke tests")
    ap.add_argument("--fcrain", default="",
                    help="future-rain channels for --lead L runs. 'perfect' = "
                         "observed rain on t+1..t+L (the ceiling, trained and "
                         "evaluated on observed); a parquet path (e.g. cache/"
                         "nwp/gefs_catchment_leads_mean.parquet, columns "
                         "p_fc1..3 indexed (gid,date)) = train on observed, "
                         "evaluate with the forecast substituted into the "
                         "genuinely-future steps; rows whose issue day has no "
                         "forecast keep observed rain and are marked "
                         "covered=False in the output parquet")
    ap.add_argument("--autoreg", action="store_true",
                    help="add the target's own normalised observed flow as an "
                         "input channel at every step of the window (known up "
                         "to day t; NaN -> 0). The persistence information a "
                         "real forecaster would use first.")
    args = ap.parse_args()
    global SEQ
    SEQ = args.seq

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {dev}", flush=True)

    gauges = good_catchments()
    if args.basins:
        gauges = gauges[:args.basins]
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

    donor = donor_features(gauges, args.donors, train_end) if args.donors else None
    if donor is not None:
        print(f"donor features: {args.donors} nearest gauges "
              f"(2x{args.donors} columns per basin)", flush=True)

    fc_tab = None
    if args.fcrain:
        assert args.lead > 0, "--fcrain needs --lead >= 1"
        if args.fcrain != "perfect":
            fc_tab = pd.read_parquet(args.fcrain)
            print(f"forecast rain: {args.fcrain}", flush=True)

    X, Y, S, y_stats, FC, COV = {}, {}, {}, {}, {}, {}
    tr_index, va_index, te_index, dates = [], [], [], {}
    val_start = train_end - pd.DateOffset(years=3)
    for b, d in frames.items():
        X[b] = ((d[FORCINGS].to_numpy("float32") - f_mean) / f_std)
        if donor is not None:
            X[b] = np.concatenate(
                [X[b], donor[b].reindex(d.index).fillna(0.0).to_numpy("float32")],
                axis=1)
        y = d["discharge_spec"].to_numpy("float32")
        ytr = d.loc[:train_end, "discharge_spec"]
        mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-6
        y_stats[b] = (mu, sd)
        Y[b] = (y - mu) / sd
        if args.autoreg:
            X[b] = np.concatenate(
                [X[b], np.nan_to_num(Y[b], nan=0.0)[:, None].astype("float32")], axis=1)
        if args.fcrain:  # future-rain channels, LAST dynamic columns (fc0)
            obs_n = np.stack(
                [(d["precipitation_haduk"].shift(-k).to_numpy("float32")
                  - f_mean[0]) / f_std[0] for k in range(1, args.lead + 1)], 1)
            obs_n = np.nan_to_num(obs_n, nan=0.0)
            X[b] = np.concatenate([X[b], obs_n], axis=1)
            if fc_tab is not None:
                g = fc_tab.loc[b].reindex(d.index)
                fcv = (g[[f"p_fc{k}" for k in range(1, args.lead + 1)]]
                       .to_numpy("float32") - f_mean[0]) / f_std[0]
                COV[b] = ~np.isnan(fcv[:, 0])
                FC[b] = np.where(np.isnan(fcv), obs_n, fcv).astype("float32")
        S[b] = s_z.loc[b].to_numpy("float32")
        dates[b] = d.index
        ok = ~np.isnan(y)
        L = args.lead
        for t in range(SEQ - 1, len(d) - L):
            if not ok[t + L]:
                continue
            dt = d.index[t + L]          # split on the TARGET date
            if dt <= train_end:
                (va_index if dt > val_start else tr_index).append((b, t))
            elif dt >= pd.Timestamp(TEST_START):
                te_index.append((b, t))
    print(f"windows: {len(tr_index):,} train / {len(va_index):,} val / "
          f"{len(te_index):,} test", flush=True)

    n_out = len(ALPHAS) if args.head == "quantile" else 1
    n_dyn = next(iter(X.values())).shape[1]
    model = LSTMModel(n_dyn + s_z.shape[1], args.hidden, n_out=n_out).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    alphas_t = torch.tensor(ALPHAS, device=dev)
    def lossf(raw, y):
        if args.head == "quantile":
            e = y.unsqueeze(1) - to_quantiles(raw)
            return torch.maximum(alphas_t * e, (alphas_t - 1) * e).mean()
        se = (raw - y) ** 2
        if args.tail_weight > 0:
            w = 1 + args.tail_weight * torch.clamp(y, min=0)
            return (w * se).sum() / w.sum()
        return se.mean()
    def point(raw):
        """Point forecast in normalised space: q50 for the quantile head."""
        return to_quantiles(raw)[:, 2] if args.head == "quantile" else raw
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
        dl = torch.utils.data.DataLoader(Windows(X, Y, S, idx, SEQ, lead=args.lead), batch_size=1024,
                                         num_workers=args.workers)
        model.eval(); preds, obs = [], []
        with torch.no_grad():
            for xb, yb, _ in dl:
                preds.append(point(model(xb.to(dev))).cpu().numpy()); obs.append(yb.numpy())
        model.train()
        p, o = np.concatenate(preds), np.concatenate(obs)
        return 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()

    tr_ds = Windows(X, Y, S, tr_index, SEQ, lead=args.lead)
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
    dl = torch.utils.data.DataLoader(
        Windows(X, Y, S, te_index, SEQ, lead=args.lead,
                FC=FC if fc_tab is not None else None,
                fc0=n_dyn - args.lead),
        batch_size=1024, num_workers=args.workers)
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _, _ in dl:
            r = model(xb.to(dev))
            preds.append((to_quantiles(r) if args.head == "quantile" else r).cpu().numpy())
    p_norm = np.concatenate(preds)

    gid = np.array([b for b, _ in te_index], dtype=np.int32)
    mu = np.array([y_stats[b][0] for b, _ in te_index], dtype=np.float32)
    sd = np.array([y_stats[b][1] for b, _ in te_index], dtype=np.float32)
    L = args.lead
    obs = np.array([Y[b][t + L] for b, t in te_index], dtype=np.float32) * sd + mu
    idx = pd.DatetimeIndex([dates[b][t + L] for b, t in te_index], name="date")
    if args.head == "quantile":
        q = np.clip(p_norm * sd[:, None] + mu[:, None], 0, None)
        res = pd.DataFrame({"gid": gid, "obs": obs, "pred": q[:, 2]}, index=idx)
        for a, col in zip(ALPHAS, q.T):
            res[f"q{int(a*100):02d}"] = col
    else:
        res = pd.DataFrame({"gid": gid, "obs": obs,
                            "pred": np.clip(p_norm * sd + mu, 0, None)}, index=idx)
    if fc_tab is not None:
        res["covered"] = [COV[b][t] for b, t in te_index]
    res.to_parquet(out / "lstm_test_predictions.parquet")
    (out / "lstm_manifest.json").write_text(json.dumps(
        {**vars(args), "device": str(dev), "n_basins": len(gauges),
         "train_end": TRAIN_END, "test_start": TEST_START, "seq": SEQ}, indent=2))
    print(f"wrote {out/'lstm_test_predictions.parquet'} ({len(res):,} rows)")
    print("score it back on the main machine with experiments/evaluate.py")


if __name__ == "__main__":
    main()
