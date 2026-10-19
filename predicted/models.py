"""The six prediction models behind Predicted.

Every model answers the same question for a set of rows (one row = one stock at
one completed 5-minute candle):
    ret_30m, prob_30m  -> log return to the close 30 minutes later, and P(up)
    ret_eod, prob_eod  -> log return to the day's 15:25 close, and P(up)
Some models also give their own 80% band (lo_*, hi_*).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

from . import config
from .features import FEATURES, H

TARGET_COL = {"30m": "y30", "eod": "yeod"}
OUT_COLS = ["ret_30m", "prob_30m", "ret_eod", "prob_eod", "lo_30m", "hi_30m", "lo_eod", "hi_eod"]


def _empty(idx) -> pd.DataFrame:
    return pd.DataFrame(np.nan, index=idx, columns=OUT_COLS)


def split_days(days: list, n_val: int = 5):
    days = sorted(days)
    return days[:-n_val], days[-n_val:]


class Clipper:
    """Learns 0.5% / 99.5% limits on training rows, then clips features (outlier control)."""

    def fit(self, X: pd.DataFrame):
        self.lo, self.hi = X.quantile(0.005), X.quantile(0.995)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X.clip(self.lo, self.hi, axis=1)

    def to_dict(self):
        return {"lo": self.lo.to_dict(), "hi": self.hi.to_dict()}

    @classmethod
    def from_dict(cls, d):
        c = cls()
        c.lo, c.hi = pd.Series(d["lo"], dtype=float), pd.Series(d["hi"], dtype=float)
        return c


# ---------------------------------------------------------------- Naive
class NaiveModel:
    """'The price won't change.' Direction = whichever way moved more often in training."""
    key = "naive"

    def fit(self, panel, fit_days, val_days):
        m = panel.date.isin(list(fit_days) + list(val_days))
        self.up = {h: float((panel.loc[m, c].dropna() > 0).mean()) for h, c in TARGET_COL.items()}
        return self

    def predict(self, panel, idx):
        out = _empty(idx)
        out["ret_30m"], out["ret_eod"] = 0.0, 0.0
        out["prob_30m"], out["prob_eod"] = self.up["30m"], self.up["eod"]
        return out

    def save(self, d):
        json.dump(self.up, open(os.path.join(d, "naive.json"), "w"))

    def load(self, d):
        self.up = json.load(open(os.path.join(d, "naive.json")))
        return self


# ---------------------------------------------------------------- LightGBM
LGB_PARAMS = dict(learning_rate=0.03, num_leaves=15, min_child_samples=40, feature_fraction=0.8,
                  bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, seed=config.SEED,
                  num_threads=0)
CLIP_FEATURES = [f for f in FEATURES if f != "sym_code"]


class LightGBMModel:
    """Gradient-boosted trees. Huber-loss regressor for the size of the move + classifier for P(up)."""
    key = "lightgbm"

    def _fit_one(self, Xtr, ytr, Xva, yva, kind):
        import lightgbm as lgb
        if kind == "reg":
            delta = float(np.median(np.abs(ytr)) * 1.5) or 1e-4
            params = dict(LGB_PARAMS, objective="huber", alpha=delta)
        else:
            params = dict(LGB_PARAMS, objective="binary")
        dtr = lgb.Dataset(Xtr, ytr, categorical_feature=["sym_code"], free_raw_data=False)
        dva = lgb.Dataset(Xva, yva, categorical_feature=["sym_code"], reference=dtr)
        return lgb.train(params, dtr, num_boost_round=600, valid_sets=[dva],
                         callbacks=[lgb.early_stopping(50, verbose=False)])

    def _X(self, panel, rows, h):
        X = panel.loc[rows, FEATURES].copy()
        X[CLIP_FEATURES] = self.clip[h].transform(X[CLIP_FEATURES])
        X["sym_code"] = X["sym_code"].astype(int)
        return X

    def fit(self, panel, fit_days, val_days):
        self.reg, self.clf, self.clip = {}, {}, {}
        for h, yc in TARGET_COL.items():
            tr = panel.index[panel.date.isin(fit_days) & panel[yc].notna()]
            va = panel.index[panel.date.isin(val_days) & panel[yc].notna()]
            self.clip[h] = Clipper().fit(panel.loc[tr, CLIP_FEATURES])
            Xtr, Xva = self._X(panel, tr, h), self._X(panel, va, h)
            ytr, yva = panel.loc[tr, yc], panel.loc[va, yc]
            self.reg[h] = self._fit_one(Xtr, ytr, Xva, yva, "reg")
            self.clf[h] = self._fit_one(Xtr, (ytr > 0).astype(int), Xva, (yva > 0).astype(int), "clf")
        return self

    def predict(self, panel, idx):
        out = _empty(idx)
        for h in TARGET_COL:
            X = self._X(panel, idx, h)
            out[f"ret_{h}"] = self.reg[h].predict(X, num_iteration=self.reg[h].best_iteration)
            out[f"prob_{h}"] = self.clf[h].predict(X, num_iteration=self.clf[h].best_iteration)
        return out

    def importance(self) -> dict:
        imp = sum(pd.Series(self.reg[h].feature_importance("gain"), index=FEATURES) for h in TARGET_COL)
        return (imp / imp.sum()).sort_values(ascending=False).head(15).round(4).to_dict()

    def save(self, d):
        for h in TARGET_COL:
            self.reg[h].save_model(os.path.join(d, f"lgb_reg_{h}.txt"), num_iteration=self.reg[h].best_iteration)
            self.clf[h].save_model(os.path.join(d, f"lgb_clf_{h}.txt"), num_iteration=self.clf[h].best_iteration)
        json.dump({h: self.clip[h].to_dict() for h in TARGET_COL}, open(os.path.join(d, "lgb_clip.json"), "w"))

    def load(self, d):
        import lightgbm as lgb
        self.reg = {h: lgb.Booster(model_file=os.path.join(d, f"lgb_reg_{h}.txt")) for h in TARGET_COL}
        self.clf = {h: lgb.Booster(model_file=os.path.join(d, f"lgb_clf_{h}.txt")) for h in TARGET_COL}
        clips = json.load(open(os.path.join(d, "lgb_clip.json")))
        self.clip = {h: Clipper.from_dict(clips[h]) for h in TARGET_COL}
        return self


# ---------------------------------------------------------------- LSTM
def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


class LSTMModel:
    """Deep learning: reads the last 24 candles (2 hours) and predicts both horizons at once.

    Multi-task (30-min return, 30-min direction, EOD return, EOD direction), robust
    (median/IQR) scaling, Huber loss for returns, early stopping, 3 seeds averaged.
    """
    key = "lstm"
    seeds = (0, 1, 2)
    epochs = 60

    def _net(self, n_feat):
        torch, nn = _torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(n_feat, 48, batch_first=True)
                self.drop = nn.Dropout(0.3)
                self.head = nn.Sequential(nn.Linear(48, 32), nn.ReLU(), nn.Linear(32, 4))

            def forward(self, x):
                o, _ = self.lstm(x)
                return self.head(self.drop(o[:, -1]))
        return Net()

    def _matrix(self, panel):
        X = self.clip.transform(panel[CLIP_FEATURES])
        X = ((X - self.med) / self.iqr).fillna(0.0).clip(-10, 10)
        onehot = np.eye(len(config.STOCKS))[panel["sym_code"].astype(int).values]
        return np.hstack([X.values, onehot]).astype(np.float32)

    def _seq(self, X, rows):
        return np.stack([X[i - config.SEQ_LEN + 1:i + 1] for i in rows]).astype(np.float32)

    def _rows(self, panel, mask):
        ok = mask & (panel["seq_pos"] >= config.SEQ_LEN - 1)
        return np.where(ok.values)[0]

    def fit(self, panel, fit_days, val_days):
        torch, nn = _torch()
        fit_m = panel.date.isin(fit_days)
        self.clip = Clipper().fit(panel.loc[fit_m, CLIP_FEATURES])
        Xf = self.clip.transform(panel.loc[fit_m, CLIP_FEATURES])
        self.med = Xf.median()
        self.iqr = (Xf.quantile(0.75) - Xf.quantile(0.25)).replace(0, 1).fillna(1)
        Y = panel[["y30", "yeod"]].values
        self.scale = np.nanstd(Y[fit_m.values], axis=0)
        X = self._matrix(panel)
        has_y = ~np.all(np.isnan(Y), axis=1)
        i_fit = self._rows(panel, fit_m & has_y)
        i_val = self._rows(panel, panel.date.isin(val_days) & has_y)
        Xf_t, Yf_t = torch.tensor(self._seq(X, i_fit)), torch.tensor(Y[i_fit] / self.scale, dtype=torch.float32)
        Xv_t, Yv_t = torch.tensor(self._seq(X, i_val)), torch.tensor(Y[i_val] / self.scale, dtype=torch.float32)
        huber = nn.HuberLoss(reduction="none", delta=1.0)
        bce = nn.BCEWithLogitsLoss(reduction="none")

        def loss_fn(out, y):
            mask = ~torch.isnan(y)
            y0 = torch.nan_to_num(y)
            l = (huber(out[:, [0, 2]], y0) + bce(out[:, [1, 3]], (y0 > 0).float())) * mask
            return l.sum() / mask.sum().clamp(min=1)

        self.n_feat = X.shape[1]
        self.states = []
        for s in self.seeds:
            torch.manual_seed(config.SEED + s)
            np.random.seed(config.SEED + s)
            net = self._net(self.n_feat)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
            best, best_state, bad = 1e9, None, 0
            for _ in range(self.epochs):
                net.train()
                perm = torch.randperm(len(Xf_t))
                for b in range(0, len(Xf_t), 256):
                    j = perm[b:b + 256]
                    opt.zero_grad()
                    l = loss_fn(net(Xf_t[j]), Yf_t[j])
                    l.backward()
                    nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    opt.step()
                net.eval()
                with torch.no_grad():
                    vl = loss_fn(net(Xv_t), Yv_t).item()
                if vl < best - 1e-4:
                    best, best_state, bad = vl, {k: v.clone() for k, v in net.state_dict().items()}, 0
                else:
                    bad += 1
                    if bad >= 8:
                        break
            self.states.append(best_state)
        return self

    def predict(self, panel, idx):
        torch, _ = _torch()
        out = _empty(idx)
        rows = panel.index.get_indexer(idx)
        ok = panel["seq_pos"].values[rows] >= config.SEQ_LEN - 1
        if not ok.any():
            return out
        X = self._matrix(panel)
        xs = torch.tensor(self._seq(X, rows[ok]))
        preds = []
        for st in self.states:
            net = self._net(self.n_feat)
            net.load_state_dict(st)
            net.eval()
            with torch.no_grad():
                preds.append(net(xs).numpy())
        o = np.mean(preds, axis=0)
        sel = np.array(idx)[ok]
        out.loc[sel, "ret_30m"] = o[:, 0] * self.scale[0]
        out.loc[sel, "prob_30m"] = 1 / (1 + np.exp(-o[:, 1]))
        out.loc[sel, "ret_eod"] = o[:, 2] * self.scale[1]
        out.loc[sel, "prob_eod"] = 1 / (1 + np.exp(-o[:, 3]))
        return out

    def save(self, d):
        torch, _ = _torch()
        torch.save({"states": self.states, "n_feat": self.n_feat, "scale": self.scale.tolist(),
                    "med": self.med.to_dict(), "iqr": self.iqr.to_dict(), "clip": self.clip.to_dict()},
                   os.path.join(d, "lstm.pt"))

    def load(self, d):
        torch, _ = _torch()
        s = torch.load(os.path.join(d, "lstm.pt"), weights_only=False)
        self.states, self.n_feat, self.scale = s["states"], s["n_feat"], np.array(s["scale"])
        self.med, self.iqr = pd.Series(s["med"], dtype=float), pd.Series(s["iqr"], dtype=float)
        self.clip = Clipper.from_dict(s["clip"])
        return self


# ---------------------------------------------------------------- Chronos-2
QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def prob_up_from_quantiles(qvals: np.ndarray) -> float:
    """P(return > 0) read off a forecast's quantiles by linear interpolation."""
    qvals = np.maximum.accumulate(np.asarray(qvals, dtype=float))
    if 0 <= qvals[0]:
        return 0.95
    if 0 >= qvals[-1]:
        return 0.05
    cdf0 = float(np.interp(0.0, qvals, QUANTILES))
    return 1.0 - cdf0


class Chronos2Model:
    """Amazon Chronos-2 (Hugging Face amazon/chronos-2), zero-shot: no training on our data.

    Input: the last 512 candles of the gap-removed log price. Output: quantile forecasts.
    """
    key = "chronos2"
    context = 512

    def __init__(self, pipeline=None):
        self.pipe = pipeline

    def fit(self, *a, **k):
        return self

    def _load(self):
        if self.pipe is None:
            from chronos import Chronos2Pipeline
            import torch
            self.pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2",
                                                         device_map="cuda" if torch.cuda.is_available() else "cpu")
        return self.pipe

    def predict(self, panel, idx, horizons=("30m", "eod"), batch=64):
        import torch
        pipe = self._load()
        out = _empty(idx)
        rows = panel.index.get_indexer(idx)
        adj, pos, left = panel["adj"].values, panel["seq_pos"].values, panel["candles_left"].values
        steps_needed = {i: max(H if "30m" in horizons else 1, int(left[i]) if "eod" in horizons else 1) for i in rows}
        order = sorted(rows, key=lambda i: steps_needed[i])
        for b in range(0, len(order), batch):
            chunk = order[b:b + batch]
            plen = max(1, max(steps_needed[i] for i in chunk))
            ctx = [torch.tensor(adj[i - min(pos[i], self.context - 1):i + 1], dtype=torch.float32) for i in chunk]
            q, _ = pipe.predict_quantiles(ctx, prediction_length=plen, quantile_levels=QUANTILES)
            for i, qq in zip(chunk, q):
                f = qq[0].numpy() - adj[i]           # (steps, 9) as returns from now
                lab = panel.index[i]
                for h in horizons:
                    step = H if h == "30m" else int(left[i])
                    if step < 1 or (h == "30m" and left[i] < H):
                        continue
                    qs = np.sort(f[step - 1])   # guard against crossed quantiles
                    out.loc[lab, f"ret_{h}"] = qs[4]
                    out.loc[lab, f"lo_{h}"] = qs[0]
                    out.loc[lab, f"hi_{h}"] = qs[-1]
                    out.loc[lab, f"prob_{h}"] = prob_up_from_quantiles(qs)
        return out

    def save(self, d):
        pass

    def load(self, d):
        return self


# ---------------------------------------------------------------- Kronos
class KronosModel:
    """Kronos (github.com/shiyu-coder/Kronos): a foundation model pre-trained on candlesticks.

    Reads full OHLCV candles and generates the next candles; we read off the predicted close.
    """
    key = "kronos"
    context = 400

    def __init__(self, predictor=None):
        self.pred = predictor

    def fit(self, *a, **k):
        return self

    def _load(self):
        if self.pred is None:
            sys.path.insert(0, os.path.abspath(config.KRONOS_REPO_DIR))
            from model import Kronos, KronosTokenizer, KronosPredictor
            tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            self.pred = KronosPredictor(mdl, tok, max_context=512)
        return self.pred

    def predict(self, panel, idx, horizons=("30m", "eod"), batch=16):
        pred = self._load()
        out = _empty(idx)
        rows = [i for i in panel.index.get_indexer(idx) if panel["seq_pos"].values[i] >= 60]
        left = panel["candles_left"].values
        dt = panel["datetime"].dt.tz_localize(None)
        jobs = []   # (row, horizon, steps)
        for i in rows:
            if "30m" in horizons and left[i] >= H:
                jobs.append((i, "30m", H))
            if "eod" in horizons and left[i] >= 1:
                jobs.append((i, "eod", int(left[i])))
        by_len: dict[tuple, list] = {}
        for i, h, n in jobs:
            ctx_len = int(min(self.context, panel["seq_pos"].values[i] + 1))
            by_len.setdefault((n, ctx_len), []).append((i, h))
        for (n, ctx_len), items in by_len.items():
            for b in range(0, len(items), batch):
                chunk = items[b:b + batch]
                dfl, xts, yts = [], [], []
                for i, _ in chunk:
                    ctx = panel.iloc[i - ctx_len + 1:i + 1]
                    dfl.append(ctx[["open", "high", "low", "close", "volume"]].reset_index(drop=True))
                    xts.append(dt.iloc[i - ctx_len + 1:i + 1].reset_index(drop=True))
                    yts.append(pd.Series(pd.date_range(dt.iloc[i] + pd.Timedelta(minutes=5), periods=n, freq="5min")))
                res = pred.predict_batch(dfl, xts, yts, pred_len=n, T=1.0, top_p=0.9,
                                         sample_count=config.KRONOS_SAMPLES, verbose=False)
                for (i, h), r in zip(chunk, res):
                    out.loc[panel.index[i], f"ret_{h}"] = float(np.log(r["close"].iloc[-1] / panel["close"].values[i]))
        return out

    def save(self, d):
        pass

    def load(self, d):
        return self


def ensemble(lgb_out: pd.DataFrame, lstm_out: pd.DataFrame) -> pd.DataFrame:
    """Average of LightGBM and LSTM (falls back to whichever exists)."""
    out = _empty(lgb_out.index)
    for c in ["ret_30m", "prob_30m", "ret_eod", "prob_eod"]:
        out[c] = pd.concat([lgb_out[c], lstm_out[c]], axis=1).mean(axis=1)
    return out
