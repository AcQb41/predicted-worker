"""Cleaning, outlier handling, features and targets.

Same logic as the Colab notebook, extended to several stocks at once and to
market context (Nifty 50, Nifty Bank, India VIX).
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from . import config

H = config.HORIZON_30M_CANDLES

STOCK_FEATURES = [
    "r1", "ret_3", "ret_6", "ret_12", "ret_24", "vol_12", "vol_36", "vol_ratio",
    "rsi_14", "macd_hist", "boll_z", "ema_gap", "gap_today", "ret_from_open",
    "pos_in_range", "day_range", "vwap_dist", "vol_z", "vol_cum_ratio", "body",
    "upper_wick", "lower_wick", "prev_day_ret", "prev_day_range", "minute_of_day",
    "candles_left", "tod_sin", "tod_cos", "weekday", "spike_flag",
]
MARKET_FEATURES = ["nifty_ret_6", "nifty_ret_from_open", "bank_ret_6", "vix_level", "vix_chg_12", "rel_ret_6"]
FEATURES = STOCK_FEATURES + MARKET_FEATURES + ["sym_code"]


def clean_symbol(c: pd.DataFrame, now) -> pd.DataFrame:
    """Rebuild a full 09:15-15:25 grid per day; fill missing candles; fix zero volume."""
    c = c.sort_values("ts").copy()
    c["datetime"] = pd.to_datetime(c["ts"], utc=True).dt.tz_convert(config.IST)
    c["date"] = c["datetime"].dt.date
    now = pd.Timestamp(now).tz_convert(config.IST)
    out = []
    for d, g in c.groupby("date"):
        g = g.set_index("datetime")
        day0 = g.index[0].normalize()
        start = day0 + pd.Timedelta(hours=9, minutes=15)
        end_full = day0 + pd.Timedelta(hours=15, minutes=25)
        finished = (d < now.date()) or (now >= end_full + pd.Timedelta(minutes=5))
        complete = finished and g.index[-1] >= end_full - pd.Timedelta(minutes=15)
        end = end_full if complete else g.index[-1]
        g = g.reindex(pd.date_range(start, end, freq="5min"))
        g["was_missing"] = g["close"].isna().astype(int)
        g["close"] = g["close"].ffill().bfill()
        for col in ["open", "high", "low"]:
            g[col] = g[col].fillna(g["close"])
        v = g["volume"].replace(0, np.nan)
        g["vol_imputed"] = v.isna().astype(int)
        g["volume"] = v.ffill().bfill().fillna(0)
        g["date"] = d
        g["day_complete"] = int(complete)
        out.append(g[["open", "high", "low", "close", "volume", "was_missing", "vol_imputed", "date", "day_complete"]])
    df = pd.concat(out)
    df.index.name = "datetime"
    return df.reset_index()


def _rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / (dn + 1e-12))


def stock_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lc = np.log(df["close"])
    first = df["date"] != df["date"].shift(1)
    prev_close = df["close"].shift(1)

    # Overnight gap: the biggest outliers. Kept as its own feature, removed from indicators.
    gap = np.nan_to_num(np.where(first, np.log(df["open"] / prev_close), 0.0))
    df["gap_today"] = pd.Series(gap, index=df.index).groupby(df["date"]).transform("sum")

    r1 = pd.Series(np.nan_to_num(np.where(first, np.log(df["close"] / df["open"]), lc.diff())), index=df.index)
    # Spike candles: robust z-score (median / MAD) above 5 -> flagged and clipped.
    med = r1.rolling(300, min_periods=50).median()
    mad = (r1 - med).abs().rolling(300, min_periods=50).median() * 1.4826
    rz = (r1 - med) / (mad + 1e-9)
    df["spike_flag"] = (rz.abs() > 5).astype(int)
    r1c = r1.clip(med - 5 * mad, med + 5 * mad).fillna(r1)
    df["r1"] = r1c

    adj = r1c.cumsum()          # gap-removed, spike-clipped log price
    df["adj"] = adj
    for k in [3, 6, 12, 24]:
        df[f"ret_{k}"] = adj - adj.shift(k)
    df["vol_12"] = r1c.rolling(12).std()
    df["vol_36"] = r1c.rolling(36).std()
    df["vol_ratio"] = df["vol_12"] / (df["vol_36"] + 1e-9)

    p = np.exp(adj)
    df["rsi_14"] = _rsi(p, 14)
    macd = p.ewm(span=12).mean() - p.ewm(span=26).mean()
    df["macd_hist"] = (macd - macd.ewm(span=9).mean()) / p
    ma20, sd20 = p.rolling(20).mean(), p.rolling(20).std()
    df["boll_z"] = (p - ma20) / (sd20 + 1e-9)
    df["ema_gap"] = np.log(p / p.ewm(span=48).mean())

    g = df.groupby("date")
    day_open = g["open"].transform("first")
    df["ret_from_open"] = np.log(df["close"] / day_open)
    hi, lo = g["high"].cummax(), g["low"].cummin()
    df["pos_in_range"] = (df["close"] - lo) / (hi - lo + 1e-9)
    df["day_range"] = np.log(hi / lo)
    tp = (df["high"] + df["low"] + df["close"]) / 3
    cumv = df["volume"].groupby(df["date"]).cumsum().replace(0, np.nan)
    vwap = (tp * df["volume"]).groupby(df["date"]).cumsum() / cumv
    df["vwap_dist"] = np.log(df["close"] / vwap)

    lv = np.log1p(df["volume"])
    df["vol_z"] = (lv - lv.rolling(36).mean()) / (lv.rolling(36).std() + 1e-9)
    day_vol = df.groupby("date")["volume"].sum().shift(1)
    df["vol_cum_ratio"] = g["volume"].cumsum() / (df["date"].map(day_vol) + 1)

    rng = (df["high"] - df["low"]).replace(0, np.nan)
    df["body"] = ((df["close"] - df["open"]) / rng).fillna(0)
    df["upper_wick"] = ((df["high"] - df[["open", "close"]].max(axis=1)) / rng).fillna(0)
    df["lower_wick"] = ((df[["open", "close"]].min(axis=1) - df["low"]) / rng).fillna(0)

    daily = g.agg(d_open=("open", "first"), d_close=("close", "last"), d_hi=("high", "max"), d_lo=("low", "min"))
    daily["prev_day_ret"] = np.log(daily.d_close / daily.d_open).shift(1)
    daily["prev_day_range"] = np.log(daily.d_hi / daily.d_lo).shift(1)
    df = df.merge(daily[["prev_day_ret", "prev_day_range"]], left_on="date", right_index=True, how="left")

    mins = (df["datetime"].dt.hour * 60 + df["datetime"].dt.minute) - (9 * 60 + 15)
    df["minute_of_day"] = mins
    df["candles_left"] = (370 - mins) // 5
    df["tod_sin"] = np.sin(2 * np.pi * mins / 375)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 375)
    df["weekday"] = df["datetime"].dt.weekday
    return df


def targets(df: pd.DataFrame) -> pd.DataFrame:
    """30-minute and end-of-day log returns (only inside the same day)."""
    df = df.copy()
    same_day = df["date"].shift(-H) == df["date"]
    df["y30"] = np.where(same_day, np.log(df["close"].shift(-H) / df["close"]), np.nan)
    day_close = df.groupby("date")["close"].transform("last")
    is_last = df["date"] != df["date"].shift(-1)
    ok = (df["day_complete"] == 1) & (~is_last)
    df["yeod"] = np.where(ok, np.log(day_close / df["close"]), np.nan)
    return df


def market_context(candles: pd.DataFrame, now) -> pd.DataFrame:
    """Index features by timestamp. Missing index data -> NaN (models handle it)."""
    cols = {"datetime": []}
    frames = []
    for sym, prefix in [("NIFTY50", "nifty"), ("BANKNIFTY", "bank"), ("INDIAVIX", "vix")]:
        c = candles[candles.symbol == sym]
        if c.empty:
            continue
        d = clean_symbol(c, now)[["datetime", "date", "close"]]
        lc = np.log(d["close"])
        f = pd.DataFrame({"datetime": d["datetime"]})
        if prefix == "vix":
            f["vix_level"] = d["close"]
            f["vix_chg_12"] = lc - lc.groupby(d["date"]).shift(12)
        else:
            f[f"{prefix}_ret_6"] = lc - lc.groupby(d["date"]).shift(6)
            if prefix == "nifty":
                f["nifty_ret_from_open"] = lc - lc.groupby(d["date"]).transform("first")
        frames.append(f.set_index("datetime"))
    if not frames:
        return pd.DataFrame(columns=["datetime"])
    return pd.concat(frames, axis=1).reset_index()


def build_panel(candles: pd.DataFrame, now=None) -> pd.DataFrame:
    """All stocks, cleaned, with features and targets, sorted by symbol then time."""
    now = pd.Timestamp(now or config.now_ist())
    mkt = market_context(candles, now)
    parts = []
    for code, sym in enumerate(config.STOCKS):
        c = candles[candles.symbol == sym]
        if len(c) < 100:
            continue
        df = targets(stock_features(clean_symbol(c, now)))
        df["symbol"] = sym
        df["sym_code"] = code
        parts.append(df)
    if not parts:
        raise RuntimeError("Not enough candles to build features (need at least 100 per stock).")
    panel = pd.concat(parts, ignore_index=True)
    if len(mkt.columns) > 1:
        panel = panel.merge(mkt, on="datetime", how="left")
    for col in MARKET_FEATURES:
        if col not in panel:
            panel[col] = np.nan
    panel["rel_ret_6"] = panel["ret_6"] - panel["nifty_ret_6"]
    panel = panel.sort_values(["symbol", "datetime"]).reset_index(drop=True)
    panel["seq_pos"] = panel.groupby("symbol").cumcount()     # position inside its own stock's history
    panel["ts_utc"] = panel["datetime"].dt.tz_convert("UTC")
    return panel


def target_ts(made_at: pd.Timestamp, horizon: str) -> pd.Timestamp:
    """Start time of the candle whose close the prediction is about."""
    if horizon == "30m":
        return made_at + timedelta(minutes=5 * H)
    ist = made_at.tz_convert(config.IST)
    return (ist.normalize() + pd.Timedelta(hours=15, minutes=25)).tz_convert("UTC")
