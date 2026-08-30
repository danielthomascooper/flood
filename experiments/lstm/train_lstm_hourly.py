"""Hourly LSTM pilot — Phase 4 A1 (runs on the GPU box).

The question, made precise by analysis_missed_amax.py: both daily model
classes with donor gauges leave ~10% of AMAX days outside a nominal-99%
envelope; 225 events are missed by BOTH, summer events twice
over-represented, and 55% of them have same-day rain below its own q90 —
the flood is invisible in daily rainfall. Does hourly rain intensity (and
hourly donor flow) make those events visible?

Pilot set: results/hourly_pilot_catchments.txt (48 most-missed + 12
zero-miss controls); transfer list including each target's 3 donors:
results/hourly_pilot_files.txt (198 files, ~3.1 GB):

    rsync -a --files-from=experiments/results/hourly_pilot_files.txt \
      main-machine:~/source/flood/data/Catchment_Timeseries/hydro-meteorological/hourly/ \
      ./data/Catchment_Timeseries/hydro-meteorological/hourly/

HOURLY FORCING TRAP (verified): hourly cehgear ends ~2017-2019, gradgb
starts ~2006-2008 — neither spans train+test, so rain =
gradgb.fillna(cehgear). Same split boundary as everywhere (train
<=2010-09-30, giving 1990-2010 hourly; test >=2010-10-01).

Model: same recipe that won at daily scale — per-basin target
normalisation, statics on every step, donor flows (same-hour + 24h lag,
donor-train-q95-scaled) — at hourly resolution, 336-hour windows.

Run (mse head first; quantile after):
    python experiments/lstm/train_lstm_hourly.py --out experiments/results/lstm_hourly
    python experiments/lstm/train_lstm_hourly.py --head quantile --out experiments/results/lstm_hourly_q

Writes hourly test predictions (gid/obs/pred[,q05..q99]) plus a daily-mean
aggregation of the same parquet for scoring against the daily models on
the main machine. Evaluate there with evaluate.py and
analysis_missed_amax.py's event list.
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import ROOT, STATIC, read_attr, TRAIN_END, TEST_START

HOURLY = ROOT / "data/Catchment_Timeseries/hydro-meteorological/hourly"
RESULTS = ROOT / "experiments/results"
ALPHAS = (0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
K = 3


def to_quantiles(raw):
    return torch.cat([raw[:, :1],
                      raw[:, :1] + torch.cumsum(nn.functional.softplus(raw[:, 1:]), 1)], 1)


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_hourly(gid, cols):
    m = sorted(HOURLY.glob(f"*_{gid}_*.csv"))
    if not m:
        raise FileNotFoundError(f"no hourly file for {gid} — check the rsync")
    d = pd.read_csv(m[0], parse_dates=["date"], na_values=["NaN"],
                    usecols=["date"] + cols).set_index("date")
    return d.astype("float32")


class LSTMModel(nn.Module):
    def __init__(self, n_in, hidden=128, dropout=0.4, n_out=1):
        super().__init__()
        self.lstm = nn.LSTM(n_in, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_out))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1]).squeeze(-1)


class Windows(torch.utils.data.Dataset):
    def __init__(self, X, Y, S, index, seq):
        self.X, self.Y, self.S, self.index, self.seq = X, Y, S, index, seq

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        b, t = self.index[i]
        x = self.X[b][t - self.seq + 1:t + 1]
        s = np.broadcast_to(self.S[b], (self.seq, self.S[b].shape[0]))
        return np.concatenate([x, s], axis=1), self.Y[b][t], b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="experiments/results/lstm_hourly")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seq", type=int, default=336, help="window length in hours")
    ap.add_argument("--head", choices=["mse", "quantile"], default="mse")
    ap.add_argument("--donors", type=int, default=3,
                    help="hourly donor gauges per target; 0 = rain-only "
                         "(the deconfound run: splits the pilot's gain "
                         "between hourly rain and hourly donor flow)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = pick_device()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"device: {dev}", flush=True)

    targets = [int(x) for x in
               (RESULTS / "hourly_pilot_catchments.txt").read_text().split()]
    topo = read_attr("topographic").set_index("gauge_id")
    ex, ny = topo.gauge_easting.astype(float), topo.gauge_northing.astype(float)

    # donor pool = every gauge with an hourly file on this box
    have = {int(re.search(r"_(\d+)_\d{8}-\d{8}\.csv$", f.name).group(1))
            for f in HOURLY.glob("*.csv")}
    print(f"loading {len(have)} hourly files "
          f"({len(targets)} targets + donors)...", flush=True)
    t0 = time.time()
    rain, flow, gap = {}, {}, {}
    for g in sorted(have):
        d = load_hourly(g, ["precipitation_cehgear", "precipitation_gradgb",
                            "discharge_spec"])
        r = d.precipitation_gradgb.fillna(d.precipitation_cehgear)
        # 77 product-wide gradgb outages after cehgear ends (1,203 h in
        # 2017-22, median 4 h, max 383 h) would otherwise poison ~15% of
        # test windows with NaN. Fill with 0 and report the per-window gap
        # count in the output so the scorer can flag those rows.
        gap[g] = r.isna().to_numpy()
        rain[g] = r.fillna(0.0)
        flow[g] = d.discharge_spec
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    train_end = pd.Timestamp(TRAIN_END) + pd.Timedelta(hours=23)
    fq95 = {g: max(float(s.loc[:train_end].quantile(0.95)), 1e-3)
            for g, s in flow.items()}

    def donors_of(g):
        if args.donors == 0:
            return []
        others = [h for h in have if h != g]
        dist = [(np.hypot(ex[g] - ex[h], ny[g] - ny[h]), h) for h in others
                if h in ex.index]
        return [h for _, h in sorted(dist)[:args.donors]]

    stat = STATIC.loc[targets].fillna(STATIC.median())
    s_z = ((stat - stat.mean()) / (stat.std() + 1e-6)).astype("float32")

    rains = pd.concat([r.loc[:train_end] for g, r in rain.items() if g in targets])
    r_mean, r_std = float(rains.mean()), float(rains.std()) + 1e-6

    X, Y, S, y_stats, dates = {}, {}, {}, {}, {}
    tr_index, va_index, te_index = [], [], []
    val_start = pd.Timestamp(TRAIN_END) - pd.DateOffset(years=3)
    for g in targets:
        idx = flow[g].index
        hod = idx.hour.values
        doy = idx.dayofyear.values
        cols = [((rain[g].values - r_mean) / r_std)]
        for d in donors_of(g):
            nb = (flow[d] / fq95[d]).reindex(idx)
            cols.append(nb.ffill(limit=168).fillna(0.0).values)
            cols.append(nb.shift(24).ffill(limit=168).fillna(0.0).values)
        cols += [np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
                 np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)]
        X[g] = np.column_stack(cols).astype("float32")
        y = flow[g].values.astype("float32")
        ytr = flow[g].loc[:train_end]
        mu, sd = float(ytr.mean()), float(ytr.std()) + 1e-6
        y_stats[g] = (mu, sd)
        Y[g] = (y - mu) / sd
        S[g] = s_z.loc[g].to_numpy("float32")
        dates[g] = idx
        ok = ~np.isnan(y) & ~np.isnan(X[g]).any(axis=1)
        for t in range(args.seq - 1, len(idx)):
            if not ok[t]:
                continue
            dt = idx[t]
            if dt <= train_end:
                (va_index if dt > val_start else tr_index).append((g, t))
            elif dt >= pd.Timestamp(TEST_START):
                te_index.append((g, t))
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
        return ((raw - y) ** 2).mean()

    def point(raw):
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
        dl = torch.utils.data.DataLoader(Windows(X, Y, S, idx, args.seq),
                                         batch_size=1024, num_workers=args.workers)
        model.eval(); preds, obs = [], []
        with torch.no_grad():
            for xb, yb, _ in dl:
                preds.append(point(model(xb.to(dev))).cpu().numpy())
                obs.append(yb.numpy())
        model.train()
        p, o = np.concatenate(preds), np.concatenate(obs)
        return 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()

    tr_ds = Windows(X, Y, S, tr_index, args.seq)
    for ep in range(start_ep, args.epochs):
        sampler = torch.utils.data.RandomSampler(
            tr_ds, replacement=True, num_samples=args.steps * args.batch)
        dl = torch.utils.data.DataLoader(tr_ds, batch_size=args.batch,
                                         sampler=sampler, num_workers=args.workers)
        t0, running = time.time(), 0.0
        for xb, yb, _ in dl:
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

    print("test inference...", flush=True)
    dl = torch.utils.data.DataLoader(Windows(X, Y, S, te_index, args.seq),
                                     batch_size=1024, num_workers=args.workers)
    model.eval(); preds = []
    with torch.no_grad():
        for xb, _, _ in dl:
            r = model(xb.to(dev))
            preds.append((to_quantiles(r) if args.head == "quantile" else r)
                         .cpu().numpy())
    p_norm = np.concatenate(preds)

    gid = np.array([b for b, _ in te_index], dtype=np.int32)
    mu = np.array([y_stats[b][0] for b, _ in te_index], dtype=np.float32)
    sd = np.array([y_stats[b][1] for b, _ in te_index], dtype=np.float32)
    obs = np.array([Y[b][t] for b, t in te_index], dtype=np.float32) * sd + mu
    idx = pd.DatetimeIndex([dates[b][t] for b, t in te_index], name="date")
    gcum = {b: np.concatenate([[0], np.cumsum(gap[b])]) for b in targets}
    wgap = np.array([gcum[b][t + 1] - gcum[b][t + 1 - args.seq] for b, t in te_index],
                    dtype=np.int16)
    if args.head == "quantile":
        q = np.clip(p_norm * sd[:, None] + mu[:, None], 0, None)
        res = pd.DataFrame({"gid": gid, "obs": obs, "pred": q[:, 2]}, index=idx)
        for a, col in zip(ALPHAS, q.T):
            res[f"q{int(a*100):02d}"] = col
    else:
        res = pd.DataFrame({"gid": gid, "obs": obs,
                            "pred": np.clip(p_norm * sd + mu, 0, None)}, index=idx)
    res["rain_gap"] = wgap          # NaN rain hours inside the window (0 = clean)
    res.to_parquet(out / "lstm_hourly_test_predictions.parquet")
    daily = res.groupby([res.gid, res.index.floor("D")]).mean()
    daily.index.names = ["gid", "date"]
    daily.reset_index("gid").to_parquet(out / "lstm_hourly_daily_agg.parquet")
    (out / "lstm_manifest.json").write_text(json.dumps(
        {**vars(args), "device": str(dev), "n_basins": len(targets),
         "train_end": TRAIN_END, "test_start": TEST_START}, indent=2))
    print(f"wrote {out}/lstm_hourly_test_predictions.parquet "
          f"({len(res):,} rows) + lstm_hourly_daily_agg.parquet")
    print("score on the main machine: evaluate.py on the daily agg; "
          "hourly AMAX capture vs the missed-events list")


if __name__ == "__main__":
    main()
