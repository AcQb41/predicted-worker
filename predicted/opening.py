"""Tomorrow's opening price.

Forecasts the gap from the last close to the next session's opening price, for the four
stocks, with six models. Two stages a day:

  evening   after India's close (~15:50 IST). Uses only what is known at the close.
  morning   pre-open (~08:35-09:10 IST). Adds the overnight US session and today's Asian
            opens (Nikkei, Hang Seng, KOSPI), USD/INR and Brent. Research on opening gaps
            (Nifty and global indices) finds most of the gap is explained by what happens
            abroad overnight, which is why the morning update is the one to watch.

Training uses years of DAILY Yahoo data (the 5-minute history is only ~60 days, i.e.
~60 gaps per stock, far too few). Scoring uses the real 09:15 candle open in the database.

Models (keys as in the database):
  naive      "opens where it closed" (gap = 0); direction = training up-rate
  linear     ridge regression + logistic regression on standardised features
  lightgbm   gradient-boosted trees (Huber regressor + classifier)
  ensemble   average of linear and lightgbm
  chronos2   Amazon Chronos-2, zero-shot, on the gap history with the features as covariates
  kronos     Kronos-small, zero-shot, on daily OHLCV candles; reads off the next candle's open
"""
from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
import sys
import warnings
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .models import QUANTILES, prob_up_from_quantiles

log = logging.getLogger("predicted.open")
warnings.filterwarnings("ignore", message=".*eval_set.*")   # LightGBM >= 4.6 deprecation notice

# ---------------------------------------------------------------- settings
INDIA = {"NIFTY50": "^NSEI", "BANKNIFTY": "^NSEBANK", "INDIAVIX": "^INDIAVIX"}
GLOBALS = {
    "SPX": "^GSPC",      # S&P 500: its session runs while India is closed
    "NDX": "^IXIC",      # Nasdaq Composite
    "N225": "^N225",     # Nikkei 225: opens 05:30 IST
    "HSI": "^HSI",       # Hang Seng: opens 07:00 IST
    "KOSPI": "^KS11",    # KOSPI: opens 05:30 IST
    "USDINR": "INR=X",
    "BRENT": "BZ=F",
}
HISTORY_YEARS = float(os.environ.get("OPEN_HISTORY_YEARS", "6"))
BACKTEST_DAYS = int(os.environ.get("OPEN_BACKTEST_DAYS", "250"))   # ~1 year of test days
BLOCK_DAYS = int(os.environ.get("OPEN_BLOCK_DAYS", "25"))          # walk-forward block size
KRONOS_EVERY = int(os.environ.get("OPEN_KRONOS_EVERY", "5"))       # Kronos backtest: every Nth test day
CHRONOS_CONTEXT = 256
KRONOS_CONTEXT = 400
MODEL_DIR = os.environ.get("PREDICTED_OPEN_MODEL_DIR", "models_open")
EVENING_MADE_AT = time(15, 30)     # evening forecasts are stamped at the close
MORNING_MADE_AT = time(8, 45)      # pre-open forecasts are stamped 08:45 IST
MORNING_CUTOFF = time(9, 14)       # never forecast the open once the market is about to open

EVENING_FEATURES = [
    "r1", "intraday", "prev_gap", "r5", "r20", "range", "vol20", "close_loc", "vratio",
    "gap_mean5", "gap_std20", "beta60",
    "nifty_r1", "nifty_intraday", "nifty_gap", "bank_r1", "vix_lvl", "vix_chg",
    "dow", "days_gap", "sym_code",
]
MORNING_EXTRA = [
    "spx_day", "spx_cc", "ndx_day", "us_sessions",
    "n225_gap", "hsi_gap", "kospi_gap", "asia_mean",
    "fx_chg", "brent_chg", "beta_x_us", "beta_x_asia",
]
STAGE_FEATURES = {"evening": EVENING_FEATURES, "morning": EVENING_FEATURES + MORNING_EXTRA}
STAGE_KEY = {"morning": "open", "evening": "open_evening"}     # keys in model_versions.backtest
CHRONOS_COVS = {
    "evening": ["r1", "intraday", "nifty_r1", "vix_chg"],
    "morning": ["r1", "intraday", "nifty_r1", "vix_chg", "spx_day", "ndx_day", "n225_gap", "hsi_gap"],
}
TRAINED = ("naive", "linear", "lightgbm")
OUT = ["ret", "prob", "lo", "hi"]


def enabled_models() -> list[str]:
    return ["naive", "linear", "lightgbm", "ensemble"] + (["chronos2"] if config.ENABLE_CHRONOS else []) \
        + (["kronos"] if config.ENABLE_KRONOS else [])


# ---------------------------------------------------------------- data
EMPTY_DAILY = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], dtype=float)


def tidy_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """yfinance daily frame -> index = date (naive), columns open/high/low/close/volume."""
    if raw is None or len(raw) == 0:
        return EMPTY_DAILY.copy()
    raw = raw.copy()
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.rename(columns=str.lower)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in df:
            df[c] = np.nan if c != "volume" else 0.0
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df[["open", "high", "low", "close", "volume"]]
    idx = pd.to_datetime(df.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx.normalize()
    df = df.dropna(subset=["open", "close"])
    df = df[(df["open"] > 0) & (df["close"] > 0)]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    return df[~df.index.duplicated(keep="last")].sort_index()


def fetch_daily(now=None) -> dict[str, pd.DataFrame]:
    """Years of daily candles for the stocks, Indian indices and global markets (split- and
    dividend-adjusted, so corporate actions don't show up as fake gaps)."""
    import yfinance as yf

    now = now or config.now_ist()
    start = (now - timedelta(days=int(365.25 * HISTORY_YEARS))).date().isoformat()
    out = {}
    for name, tk in {**config.STOCKS, **INDIA, **GLOBALS}.items():
        try:
            raw = yf.download(tk, start=start, interval="1d", auto_adjust=True, progress=False, threads=False)
            out[name] = tidy_daily(raw)
        except Exception as e:                       # one missing market must not stop the run
            log.warning("daily %s (%s) failed: %r", name, tk, e)
            out[name] = EMPTY_DAILY.copy()
        log.info("daily %-10s %5d rows, last %s", name, len(out[name]),
                 out[name].index[-1].date() if len(out[name]) else "-")
    return out


def bars_from_candles(candles: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Daily bars built from our own 5-minute candles (used when Yahoo's daily bar is late)."""
    if candles is None or candles.empty:
        return {}
    c = candles.copy()
    c["ts"] = pd.to_datetime(c["ts"], utc=True)
    c["d"] = c["ts"].dt.tz_convert(config.IST).dt.tz_localize(None).dt.normalize()
    c = c.sort_values(["symbol", "ts"])
    g = c.groupby(["symbol", "d"])
    bars = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(), "low": g["low"].min(),
                         "close": g["close"].last(), "volume": g["volume"].sum().astype(float)})
    return {s: b.droplevel(0) for s, b in bars.groupby(level=0)}


def add_recent_bars(daily: dict, candle_bars: dict, since: pd.Timestamp) -> dict:
    """Append days (>= since) that Yahoo's daily series doesn't have yet."""
    out = dict(daily)
    for s, b in candle_bars.items():
        cur = out.get(s, EMPTY_DAILY)
        extra = b[(b.index >= since) & (~b.index.isin(cur.index))]
        if len(extra):
            out[s] = pd.concat([cur, extra]).sort_index()
            log.info("%s: added %d day(s) from our 5-minute candles", s, len(extra))
    return out


# ---------------------------------------------------------------- features
def _log(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.log(np.asarray(a, dtype=float) / np.asarray(b, dtype=float))


def _window_sum(series: pd.Series, start, end) -> np.ndarray:
    """Sum of `series` over dates d with start <= d < end, for arrays of start/end dates."""
    if series is None or series.empty:
        return np.zeros(len(start))
    s = series.dropna().sort_index()
    dates = s.index.values
    cs = np.concatenate([[0.0], np.cumsum(s.values)])
    a = np.searchsorted(dates, np.asarray(start, dtype="datetime64[ns]"), side="left")
    b = np.searchsorted(dates, np.asarray(end, dtype="datetime64[ns]"), side="left")
    return cs[b] - cs[a]


def _window_count(series: pd.Series, start, end) -> np.ndarray:
    return _window_sum(pd.Series(1.0, index=series.dropna().index) if series is not None and len(series) else None,
                       start, end)


def _last_before(series: pd.Series, when) -> np.ndarray:
    """Value of the last observation strictly before each date in `when` (NaN if none)."""
    if series is None or series.empty:
        return np.full(len(when), np.nan)
    s = series.dropna().sort_index()
    i = np.searchsorted(s.index.values, np.asarray(when, dtype="datetime64[ns]"), side="left") - 1
    v = np.where(i >= 0, s.values[np.clip(i, 0, None)], np.nan)
    return v.astype(float)


def _at_or_before(series: pd.Series, when) -> np.ndarray:
    if series is None or series.empty:
        return np.full(len(when), np.nan)
    s = series.dropna().sort_index()
    i = np.searchsorted(s.index.values, np.asarray(when, dtype="datetime64[ns]"), side="right") - 1
    return np.where(i >= 0, s.values[np.clip(i, 0, None)], np.nan).astype(float)


def _gap_on(df: pd.DataFrame, when) -> np.ndarray:
    """An Asian index's opening gap on each date (0 when it didn't trade that day)."""
    if df is None or df.empty:
        return np.zeros(len(when))
    g = pd.Series(_log(df["open"], df["close"].shift(1)), index=df.index)
    return g.reindex(pd.DatetimeIndex(when)).fillna(0.0).values


def next_weekday(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.Timestamp(d).normalize() + pd.Timedelta(days=1)
    while d.weekday() >= 5:
        d += pd.Timedelta(days=1)
    return d


def build_table(daily: dict, forecast_day=None) -> pd.DataFrame:
    """One row per (stock, D) where D is a session date: the features known before D's open
    and the target y = ln(open_D / close_P), P = the previous session.

    forecast_day: also add a row for this date (no target yet). Indian series must only
    contain sessions before it.
    """
    fday = pd.Timestamp(forecast_day).normalize() if forecast_day is not None else None
    nifty, bank, vix = (daily.get(k, EMPTY_DAILY) for k in ("NIFTY50", "BANKNIFTY", "INDIAVIX"))
    n_r1 = pd.Series(_log(nifty["close"], nifty["close"].shift(1)), index=nifty.index)
    frames = []
    for code, sym in enumerate(config.STOCKS):
        d = daily.get(sym, EMPTY_DAILY)
        if fday is not None:
            d = d[d.index < fday]
        if len(d) < 30:
            log.warning("%s: only %d daily bars, skipped", sym, len(d))
            continue
        o, h, l, c, v = (d[k] for k in ("open", "high", "low", "close", "volume"))
        r1 = pd.Series(_log(c, c.shift(1)), index=d.index)
        gap = pd.Series(_log(o, c.shift(1)), index=d.index)          # the gap OF that day
        with np.errstate(divide="ignore", invalid="ignore"):
            vr = np.log(v / v.rolling(20, min_periods=10).mean())
        nr = n_r1.reindex(d.index)
        cov = r1.rolling(60, min_periods=30).cov(nr)
        var = nr.rolling(60, min_periods=30).var()
        pf = pd.DataFrame({                                          # features as of session P
            "base": c,
            "r1": r1,
            "intraday": _log(c, o),
            "prev_gap": gap,
            "r5": _log(c, c.shift(5)),
            "r20": _log(c, c.shift(20)),
            "range": _log(h, l),
            "vol20": r1.rolling(20, min_periods=10).std(),
            "close_loc": ((c - l) / (h - l)).where(h > l, 0.5),
            "vratio": vr.replace([np.inf, -np.inf], np.nan),
            "gap_mean5": gap.rolling(5, min_periods=3).mean(),
            "gap_std20": gap.rolling(20, min_periods=10).std(),
            "beta60": (cov / var).clip(-1, 4),
        }, index=d.index)
        dates = list(d.index)
        P = dates[:-1]
        D = dates[1:]
        if fday is not None and fday > dates[-1]:
            P, D = P + [dates[-1]], D + [fday]
        P, D = pd.DatetimeIndex(P), pd.DatetimeIndex(D)
        t = pf.reindex(P).reset_index(drop=True)
        t.insert(0, "symbol", sym)
        t.insert(1, "date", D)
        t.insert(2, "prev_date", P)
        opn = o.reindex(D).values
        t["open_actual"] = opn
        t["y"] = _log(opn, t["base"].values)
        # Indian market context at P
        t["nifty_r1"] = _at_or_before(n_r1, P)
        t["nifty_intraday"] = _at_or_before(pd.Series(_log(nifty["close"], nifty["open"]), index=nifty.index), P)
        t["nifty_gap"] = _at_or_before(pd.Series(_log(nifty["open"], nifty["close"].shift(1)), index=nifty.index), P)
        t["bank_r1"] = _at_or_before(pd.Series(_log(bank["close"], bank["close"].shift(1)), index=bank.index), P)
        t["vix_lvl"] = np.log(_at_or_before(vix["close"], P))
        t["vix_chg"] = _at_or_before(pd.Series(_log(vix["close"], vix["close"].shift(1)), index=vix.index), P)
        t["dow"] = D.weekday
        t["days_gap"] = (D - P).days
        t["sym_code"] = code
        # Overnight: US sessions dated P .. D-1 run entirely between India's close P and open D
        spx, ndx = daily.get("SPX", EMPTY_DAILY), daily.get("NDX", EMPTY_DAILY)
        t["spx_day"] = _window_sum(pd.Series(_log(spx["close"], spx["open"]), index=spx.index), P, D)
        t["spx_cc"] = _window_sum(pd.Series(_log(spx["close"], spx["close"].shift(1)), index=spx.index), P, D)
        t["ndx_day"] = _window_sum(pd.Series(_log(ndx["close"], ndx["open"]), index=ndx.index), P, D)
        t["us_sessions"] = _window_count(spx["close"], P, D)
        # Asian opens on day D (all before 07:00 IST)
        for k, col in (("N225", "n225_gap"), ("HSI", "hsi_gap"), ("KOSPI", "kospi_gap")):
            t[col] = _gap_on(daily.get(k, EMPTY_DAILY), D)
        t["asia_mean"] = t[["n225_gap", "hsi_gap", "kospi_gap"]].mean(axis=1)
        for k, col in (("USDINR", "fx_chg"), ("BRENT", "brent_chg")):
            s = daily.get(k, EMPTY_DAILY)["close"] if k in daily else None
            t[col] = _log(_last_before(s, D), _last_before(s, P))
        t["beta_x_us"] = t["beta60"].fillna(1.0) * t["spx_day"]
        t["beta_x_asia"] = t["beta60"].fillna(1.0) * t["asia_mean"]
        frames.append(t)
    if not frames:
        return pd.DataFrame()
    tab = pd.concat(frames, ignore_index=True)
    feats = list(dict.fromkeys(EVENING_FEATURES + MORNING_EXTRA))
    tab[feats] = tab[feats].replace([np.inf, -np.inf], np.nan)
    return tab.sort_values(["date", "symbol"]).reset_index(drop=True)


# ---------------------------------------------------------------- trained models
def _y_train(y: pd.Series) -> pd.Series:
    lo, hi = y.quantile(0.005), y.quantile(0.995)
    return y.clip(lo, hi)


class OpenNaive:
    key = "naive"

    def fit(self, tr, feats):
        y = tr["y"].dropna()
        self.up = float((y > 0).mean()) if len(y) else 0.5
        return self

    def predict(self, X, feats):
        return pd.DataFrame({"ret": 0.0, "prob": self.up, "lo": np.nan, "hi": np.nan}, index=X.index)


class OpenLinear:
    """Ridge regression for the size of the gap, logistic regression for its direction."""
    key = "linear"

    def fit(self, tr, feats):
        from sklearn.linear_model import LogisticRegression, RidgeCV
        X = tr[feats].astype(float)
        self.lo, self.hi = X.quantile(0.005), X.quantile(0.995)
        X = X.clip(self.lo, self.hi, axis=1)
        self.med = X.median().fillna(0.0)
        X = X.fillna(self.med)
        self.mu, self.sd = X.mean(), X.std().replace(0, 1).fillna(1)
        Z = ((X - self.mu) / self.sd).values
        y = _y_train(tr["y"])
        self.reg = RidgeCV(alphas=np.logspace(-1, 4, 16)).fit(Z, y.values)
        m = tr["y"].values != 0
        self.clf = LogisticRegression(C=0.05, max_iter=2000).fit(Z[m], (tr["y"].values[m] > 0).astype(int))
        return self

    def _Z(self, X, feats):
        X = X[feats].astype(float).clip(self.lo, self.hi, axis=1).fillna(self.med)
        return ((X - self.mu) / self.sd).values

    def predict(self, X, feats):
        Z = self._Z(X, feats)
        return pd.DataFrame({"ret": self.reg.predict(Z), "prob": self.clf.predict_proba(Z)[:, 1],
                             "lo": np.nan, "hi": np.nan}, index=X.index)

    def importance(self, feats) -> dict:
        w = pd.Series(np.abs(self.reg.coef_), index=feats)
        return (w / w.sum()).sort_values(ascending=False).head(8).round(4).to_dict()


class OpenLightGBM:
    """Small gradient-boosted trees: Huber loss for the size, a classifier for the direction."""
    key = "lightgbm"

    def fit(self, tr, feats):
        import lightgbm as lgb
        tr = tr.sort_values("date")
        X, y = tr[feats].astype(float), _y_train(tr["y"])
        cut = int(len(tr) * 0.85)
        common = dict(n_estimators=600, learning_rate=0.02, num_leaves=7, min_child_samples=40,
                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1)
        delta = float(max(y.std(), 1e-4) * 1.5)
        stop = [lgb.early_stopping(50, verbose=False)]
        self.reg = lgb.LGBMRegressor(objective="huber", alpha=delta, **common)
        self.reg.fit(X.iloc[:cut], y.iloc[:cut], eval_set=[(X.iloc[cut:], y.iloc[cut:])], callbacks=stop)
        m = (tr["y"] != 0).values
        yc = (tr["y"].values > 0).astype(int)
        Xm, ym = X[m], yc[m]
        cm = int(len(Xm) * 0.85)
        self.clf = lgb.LGBMClassifier(objective="binary", **common)
        self.clf.fit(Xm.iloc[:cm], ym[:cm], eval_set=[(Xm.iloc[cm:], ym[cm:])], callbacks=stop)
        return self

    def predict(self, X, feats):
        Xf = X[feats].astype(float)
        return pd.DataFrame({"ret": self.reg.predict(Xf), "prob": self.clf.predict_proba(Xf)[:, 1],
                             "lo": np.nan, "hi": np.nan}, index=X.index)

    def importance(self, feats) -> dict:
        w = pd.Series(self.reg.booster_.feature_importance("gain"), index=feats)
        return (w / max(w.sum(), 1e-12)).sort_values(ascending=False).head(8).round(4).to_dict()


MODEL_CLASSES = {"naive": OpenNaive, "linear": OpenLinear, "lightgbm": OpenLightGBM}


def ensemble(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(np.nan, index=a.index, columns=OUT)
    for c in ("ret", "prob"):
        out[c] = pd.concat([a[c], b[c]], axis=1).mean(axis=1)
    return out


# ---------------------------------------------------------------- zero-shot models
class OpenChronos2:
    """Chronos-2 on each stock's gap history; the features enter as covariates whose value
    for the day being forecast is known (in-context regression, no training)."""
    key = "chronos2"

    def __init__(self, pipeline=None):
        self.pipe = pipeline

    def _load(self):
        if self.pipe is None:
            from chronos import Chronos2Pipeline
            import torch
            self.pipe = Chronos2Pipeline.from_pretrained(
                "amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
        return self.pipe

    def predict(self, table: pd.DataFrame, rows: pd.Index, stage: str, batch: int = 128) -> pd.DataFrame:
        pipe = self._load()
        covs = CHRONOS_COVS[stage]
        out = pd.DataFrame(np.nan, index=rows, columns=OUT)
        hist = {s: g.dropna(subset=["y"]).sort_values("date") for s, g in table.groupby("symbol")}
        inputs, labels = [], []
        for i in rows:
            r = table.loc[i]
            g = hist.get(r["symbol"])
            if g is None:
                continue
            g = g[g["date"] < r["date"]].tail(CHRONOS_CONTEXT)
            if len(g) < 30:
                continue
            inputs.append({
                "target": np.clip(g["y"].values, -0.2, 0.2).astype(np.float32),
                "past_covariates": {c: g[c].fillna(0.0).values.astype(np.float32) for c in covs},
                "future_covariates": {c: np.array([0.0 if pd.isna(r[c]) else r[c]], dtype=np.float32) for c in covs},
            })
            labels.append(i)
        for b in range(0, len(inputs), batch):
            q, _ = pipe.predict_quantiles(inputs[b:b + batch], prediction_length=1, quantile_levels=QUANTILES)
            for lab, qq in zip(labels[b:b + batch], q):
                qs = np.sort(np.asarray(qq).reshape(-1, len(QUANTILES))[0])
                out.loc[lab] = [qs[4], prob_up_from_quantiles(qs), qs[0], qs[-1]]
        return out


class OpenKronos:
    """Kronos-small on daily OHLCV candles: generate the next daily candle, read its open."""
    key = "kronos"

    def __init__(self, predictor=None):
        self.pred = predictor

    def _load(self):
        if self.pred is None:
            sys.path.insert(0, os.path.abspath(config.KRONOS_REPO_DIR))
            from model import Kronos, KronosPredictor, KronosTokenizer
            tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            self.pred = KronosPredictor(mdl, tok, max_context=512)
        return self.pred

    def predict(self, table: pd.DataFrame, rows: pd.Index, daily: dict, batch: int = 16) -> pd.DataFrame:
        pred = self._load()
        out = pd.DataFrame(np.nan, index=rows, columns=OUT)
        groups: dict[int, list] = {}
        for i in rows:
            r = table.loc[i]
            d = daily.get(r["symbol"], EMPTY_DAILY)
            ctx = d[d.index <= r["prev_date"]].tail(KRONOS_CONTEXT)
            if len(ctx) < 60:
                continue
            groups.setdefault(len(ctx), []).append((i, ctx, r["date"], r["base"]))
        for n, items in groups.items():
            for b in range(0, len(items), batch):
                chunk = items[b:b + batch]
                dfl = [c[["open", "high", "low", "close", "volume"]].reset_index(drop=True) for _, c, _, _ in chunk]
                xts = [pd.Series(c.index) for _, c, _, _ in chunk]
                yts = [pd.Series([pd.Timestamp(dd)]) for _, _, dd, _ in chunk]
                res = pred.predict_batch(dfl, xts, yts, pred_len=1, T=1.0, top_p=0.9,
                                         sample_count=config.KRONOS_SAMPLES, verbose=False)
                for (i, _, _, base), r in zip(chunk, res):
                    out.loc[i, "ret"] = float(np.log(float(r["open"].iloc[0]) / base))
        return out


# ---------------------------------------------------------------- backtest
def _pred_cols(stage, m):
    return {c: f"{stage}|{m}|{c}" for c in OUT}


def walk_forward(table: pd.DataFrame, daily: dict, models: list[str], chronos=None, kronos=None) -> pd.DataFrame:
    """Expanding-window test over the last BACKTEST_DAYS sessions, in blocks of BLOCK_DAYS.
    Trained models only ever see sessions before the block they are tested on."""
    lab = table.dropna(subset=["y"])
    dates = sorted(lab["date"].unique())
    n_test = min(BACKTEST_DAYS, max(0, len(dates) - 120))
    test_dates = dates[len(dates) - n_test:]
    bt = lab[lab["date"].isin(test_dates)].copy()
    if bt.empty:
        return bt
    bt["fold"] = 0
    blocks = [test_dates[i:i + BLOCK_DAYS] for i in range(0, len(test_dates), BLOCK_DAYS)]
    for k, block in enumerate(blocks, 1):
        tr = lab[lab["date"] < block[0]]
        te_idx = bt.index[bt["date"].isin(block)]
        bt.loc[te_idx, "fold"] = k
        for stage, feats in STAGE_FEATURES.items():
            outs = {}
            for m in TRAINED:
                outs[m] = MODEL_CLASSES[m]().fit(tr, feats).predict(bt.loc[te_idx], feats)
            outs["ensemble"] = ensemble(outs["linear"], outs["lightgbm"])
            for m, o in outs.items():
                for c, col in _pred_cols(stage, m).items():
                    bt.loc[te_idx, col] = o[c].values
        log.info("open backtest block %d/%d: train to %s, test %s..%s",
                 k, len(blocks), pd.Timestamp(tr["date"].max()).date(), pd.Timestamp(block[0]).date(),
                 pd.Timestamp(block[-1]).date())
    if "chronos2" in models and chronos is not None:
        for stage in STAGE_FEATURES:
            try:
                o = chronos.predict(table, bt.index, stage)
                for c, col in _pred_cols(stage, "chronos2").items():
                    bt[col] = o[c].values
            except Exception as e:
                log.warning("Chronos-2 open backtest (%s) skipped: %r", stage, e)
    if "kronos" in models and kronos is not None:
        try:
            pick = set(test_dates[::KRONOS_EVERY])
            idx = bt.index[bt["date"].isin(pick)]
            o = kronos.predict(table, idx, daily)
            for stage in STAGE_FEATURES:           # Kronos only sees candles: same in both stages
                for c, col in _pred_cols(stage, "kronos").items():
                    bt[col] = np.nan
                    bt.loc[idx, col] = o[c].values
        except Exception as e:
            log.warning("Kronos open backtest skipped: %r", e)
    return bt


def _metrics(q: pd.DataFrame, stage: str, m: str):
    rc, pc = f"{stage}|{m}|ret", f"{stage}|{m}|prob"
    nc = f"{stage}|naive|prob"
    if rc not in q:
        return None
    q = q.dropna(subset=[rc, "y"])
    q = q[q["y"] != 0]
    if len(q) < 10:
        return None
    has_prob = pc in q and q[pc].notna().all()
    pred_up = (q[pc] >= 0.5) if has_prob else (q[rc] > 0)
    actual_up = q["y"] > 0
    correct = pred_up == actual_up
    naive_up = q[nc] >= 0.5 if nc in q else pd.Series(actual_up.mean() >= 0.5, index=q.index)
    naive_acc = float((naive_up == actual_up).mean())
    n, k = len(q), int(correct.sum())
    rel = np.abs(np.exp(q[rc]) - np.exp(q["y"])) / np.exp(q["y"])
    rel_naive = np.abs(1 - np.exp(q["y"])) / np.exp(q["y"])
    out = {
        "n": n,
        "dir_acc": k / n,
        "balanced_acc": 0.5 * (float(pred_up[actual_up].mean()) + float((~pred_up[~actual_up]).mean()))
        if actual_up.any() and (~actual_up).any() else None,
        "p_vs_naive": float(stats.binomtest(k, n, min(max(naive_acc, 1e-6), 0.999), alternative="greater").pvalue)
        if m != "naive" else None,
        "auc": None,
        "mape_pct": float(rel.mean() * 100),
        "skill_vs_naive_pct": float((1 - rel.sum() / max(rel_naive.sum(), 1e-12)) * 100),
        "ic": float(stats.spearmanr(q[rc], q["y"]).correlation) if m != "naive" and q[rc].nunique() > 1 else None,
    }
    if has_prob and m != "naive" and actual_up.nunique() == 2:
        from sklearn.metrics import roc_auc_score
        out["auc"] = float(roc_auc_score(actual_up, q[pc]))
    if m != "naive":
        if stage == "evening":
            # Buy (or short) at the close when the forecast beats costs, exit at the next open.
            sig = np.sign(q[rc]) * (q[rc].abs() > config.COST_ROUND_TRIP)
            pnl = sig * (np.exp(q["y"]) - 1) - sig.abs() * config.COST_ROUND_TRIP
            nt = int((sig != 0).sum())
            out.update(bt_trades=nt, bt_win_rate=float((pnl[sig != 0] > 0).mean()) if nt else None,
                       bt_return_pct=float(pnl.sum() * 100 / max(1, q["symbol"].nunique())))
        else:
            # The pre-open forecast is made after the close: you can't trade the gap with it.
            out.update(bt_trades=None, bt_win_rate=None, bt_return_pct=None)
    return out


def score(bt: pd.DataFrame, models: list[str]) -> dict:
    """{model: {"open": {...}, "open_evening": {...}, "resid_q": {...}, "fold_ic": {...}}}"""
    res = {}
    for m in models:
        r = {"resid_q": {}, "fold_ic": {}}
        for stage, key in STAGE_KEY.items():
            per = {}
            a = _metrics(bt, stage, m)
            if a:
                per["ALL"] = a
            for sym, g in bt.groupby("symbol"):
                mm = _metrics(g, stage, m)
                if mm:
                    per[sym] = mm
            if per:
                r[key] = per
            rc = f"{stage}|{m}|ret"
            src = rc if rc in bt and bt[rc].notna().sum() > 30 else f"{stage}|naive|ret"
            if src in bt:
                e = (bt["y"] - bt[src]).dropna()
                if len(e) > 30:
                    r["resid_q"][key] = [float(e.quantile(0.1)), float(e.quantile(0.9))]
            if rc in bt and m != "naive":
                ics = {}
                for k, g in bt.dropna(subset=[rc, "y"]).groupby("fold"):
                    v = stats.spearmanr(g[rc], g["y"]).correlation if len(g) > 20 and g[rc].nunique() > 1 else np.nan
                    ics[str(k)] = None if pd.isna(v) else round(float(v), 4)
                r["fold_ic"][key] = ics
        res[m] = r
    return res


# ---------------------------------------------------------------- jobs
def _version_label(now) -> str:
    return now.strftime("%Y-%m-%d %H:%M IST") + " open"


def train(store, daily: dict, now, backtest: bool = True, publish: bool = True) -> dict:
    """Backtest, fit the trained models on all history, save them, publish model versions."""
    from .jobs import run_logged

    models = enabled_models()
    with run_logged(store, "train_open") as run:
        table = build_table(daily)
        lab = table.dropna(subset=["y"])
        if lab.empty:
            raise RuntimeError("no daily history to train on (Yahoo daily download failed?)")
        scores = {}
        if backtest:
            chronos = OpenChronos2() if "chronos2" in models else None
            kronos = OpenKronos() if "kronos" in models else None
            bt = walk_forward(table, daily, models, chronos, kronos)
            scores = score(bt, models)
        tmp = MODEL_DIR + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        fitted = {}
        for stage, feats in STAGE_FEATURES.items():
            for m in TRAINED:
                fitted[(stage, m)] = MODEL_CLASSES[m]().fit(lab, feats)
        pickle.dump(fitted, open(os.path.join(tmp, "fitted.pkl"), "wb"))
        version = _version_label(now)
        first, last = pd.Timestamp(lab["date"].min()).date(), pd.Timestamp(lab["date"].max()).date()
        params = {
            "naive": {"up_rate": fitted[("morning", "naive")].up},
            "linear": {"model": "ridge + logistic", "top_features": fitted[("morning", "linear")].importance(
                STAGE_FEATURES["morning"])},
            "lightgbm": {"num_leaves": 7, "learning_rate": 0.02, "top_features": fitted[("morning", "lightgbm")]
                         .importance(STAGE_FEATURES["morning"])},
            "ensemble": {"members": ["linear", "lightgbm"]},
            "chronos2": {"hf_model": "amazon/chronos-2", "context_days": CHRONOS_CONTEXT, "zero_shot": True,
                         "covariates": CHRONOS_COVS},
            "kronos": {"hf_model": "NeoQuasar/Kronos-small", "context_days": KRONOS_CONTEXT, "zero_shot": True},
        }
        meta = {"version": version, "trained_at": now.isoformat(), "models": {}}
        for m in models:
            vid = None
            if publish:
                vid = store.create_model_version({
                    "model_key": m, "version": version, "train_start": str(first), "train_end": str(last),
                    "n_train": int(len(lab)),
                    "params": {"scope": "open", "stages": list(STAGE_KEY.values()), **params.get(m, {})},
                    "backtest": scores.get(m, {}),
                })
            meta["models"][m] = {"version_id": vid, "resid_q": scores.get(m, {}).get("resid_q", {})}
        json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1, default=str)
        shutil.rmtree(MODEL_DIR, ignore_errors=True)
        shutil.move(tmp, MODEL_DIR)
        summary = {m: {k: scores[m][k]["ALL"] for k in STAGE_KEY.values() if "ALL" in scores.get(m, {}).get(k, {})}
                   for m in scores}
        run.details.update({"version": version, "rows": int(len(lab)), "first": str(first), "last": str(last),
                            "summary": summary})
        run.message = f"trained open models {', '.join(models)} on {len(lab)} gaps ({first} to {last})"
        log.info(run.message)
        return summary


def _load():
    meta_p, fit_p = os.path.join(MODEL_DIR, "meta.json"), os.path.join(MODEL_DIR, "fitted.pkl")
    if not (os.path.exists(meta_p) and os.path.exists(fit_p)):
        return None, None
    return json.load(open(meta_p)), pickle.load(open(fit_p, "rb"))


def predict(store, stage: str = "auto", retrain: bool = True) -> dict:
    """Forecast the next session's opening price for every stock and model."""
    from .jobs import run_logged

    now = config.now_ist()
    t = now.time().replace(tzinfo=None)
    if stage == "auto":
        stage = "evening" if t >= time(12, 0) else "morning" if t < MORNING_CUTOFF else "none"
    if stage == "none" or (stage == "morning" and t >= MORNING_CUTOFF):
        msg = f"{now:%H:%M} IST: too late for a pre-open forecast and too early for the evening one; nothing to do"
        log.info(msg)
        return {"skipped": msg}

    today = pd.Timestamp(now.date())
    daily = fetch_daily(now)
    try:
        candles = store.get_candles(list(config.ALL_SYMBOLS), now - timedelta(days=7))
        daily = add_recent_bars(daily, bars_from_candles(candles), today - pd.Timedelta(days=6))
    except Exception as e:
        log.warning("could not read recent candles: %r", e)

    if stage == "evening":
        last = {s: daily.get(s, EMPTY_DAILY).index.max() for s in config.STOCKS}
        if not any(pd.notna(d) and d == today for d in last.values()):
            msg = "no session today (holiday or data not in yet): no evening forecast"
            log.info(msg)
            return {"skipped": msg}
        fday = next_weekday(today)
        made_at = datetime.combine(now.date(), EVENING_MADE_AT, tzinfo=config.IST)
        if retrain:
            train(store, daily, now)
    else:
        fday = today
        made_at = datetime.combine(now.date(), MORNING_MADE_AT, tzinfo=config.IST)
        for k in list(config.STOCKS) + list(INDIA):        # India: only sessions before today
            if k in daily:
                daily[k] = daily[k][daily[k].index < today]

    meta, fitted = _load()
    if meta is None:
        log.info("no saved open models: training now (no backtest)")
        train(store, daily, now, backtest=False, publish=False)
        meta, fitted = _load()

    with run_logged(store, "predict_open") as run:
        table = build_table(daily, fday)
        rows = table[table["date"] == fday]
        if rows.empty:
            raise RuntimeError("could not build the forecast rows (missing daily data)")
        feats = STAGE_FEATURES[stage]
        outs = {m: fitted[(stage, m)].predict(rows, feats) for m in TRAINED}
        outs["ensemble"] = ensemble(outs["linear"], outs["lightgbm"])
        if config.ENABLE_CHRONOS and "chronos2" in meta["models"]:
            try:
                outs["chronos2"] = OpenChronos2().predict(table, rows.index, stage)
            except Exception as e:
                log.warning("Chronos-2 skipped: %r", e)
        if config.ENABLE_KRONOS and "kronos" in meta["models"]:
            try:
                outs["kronos"] = OpenKronos().predict(table, rows.index, daily)
            except Exception as e:
                log.warning("Kronos skipped: %r", e)

        key = STAGE_KEY[stage]
        target = datetime.combine(fday.date(), time(9, 15), tzinfo=config.IST)
        preds = []
        for i in rows.index:
            r = rows.loc[i]
            base = float(r["base"])
            for m, o in outs.items():
                ret = o.loc[i, "ret"]
                if pd.isna(ret):
                    continue
                prob, lo, hi = o.loc[i, "prob"], o.loc[i, "lo"], o.loc[i, "hi"]
                if pd.isna(lo) or pd.isna(hi):
                    rq = (meta["models"].get(m, {}).get("resid_q") or {}).get(key) \
                        or (meta["models"].get("naive", {}).get("resid_q") or {}).get(key)
                    if rq:
                        lo, hi = ret + rq[0], ret + rq[1]
                direction = ("up" if prob >= 0.5 else "down") if not pd.isna(prob) else ("up" if ret > 0 else "down")
                preds.append({
                    "symbol": r["symbol"], "model_key": m, "horizon": "open",
                    "made_at": made_at.isoformat(), "target_ts": target.isoformat(),
                    "base_price": base, "pred_return": float(ret), "pred_price": base * float(np.exp(ret)),
                    "pred_direction": direction, "prob_up": None if pd.isna(prob) else float(prob),
                    "band_low": None if pd.isna(lo) else base * float(np.exp(lo)),
                    "band_high": None if pd.isna(hi) else base * float(np.exp(hi)),
                    "model_version_id": meta["models"].get(m, {}).get("version_id"),
                })
        n = store.upsert_predictions(preds) if preds else 0
        resolved = store.resolve()
        label = "evening estimate" if stage == "evening" else "pre-open update"
        run.message = f"{n} opening-price forecasts ({label}) for {fday:%a %d %b}; {resolved} resolved"
        run.details.update({"stage": stage, "for": str(fday.date()), "models": sorted(outs),
                            "inputs": rows[["symbol", "base"] + (MORNING_EXTRA if stage == "morning" else [])]
                            .round(5).to_dict("records")})
        log.info(run.message)
        return {"stage": stage, "for": str(fday.date()), "predictions": n, "resolved": resolved}
