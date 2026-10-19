"""Getting candles: from the seed CSV files and from Yahoo Finance (yfinance)."""
from __future__ import annotations

import os
import re
from datetime import timedelta

import pandas as pd

from . import config


def session_filter(df: pd.DataFrame, now=None) -> pd.DataFrame:
    """Keep NSE session candles (09:15 to 15:25 starts, weekdays) that have finished."""
    if df.empty:
        return df
    ts = pd.to_datetime(df["ts"], utc=True)
    ist = ts.dt.tz_convert(config.IST)
    minutes = ist.dt.hour * 60 + ist.dt.minute
    first = config.SESSION_FIRST_CANDLE[0] * 60 + config.SESSION_FIRST_CANDLE[1]
    last = config.SESSION_LAST_CANDLE[0] * 60 + config.SESSION_LAST_CANDLE[1]
    keep = (minutes >= first) & (minutes <= last) & (minutes % 5 == 0) & (ist.dt.second == 0) & (ist.dt.weekday < 5)
    now = pd.Timestamp(now or config.now_ist()).tz_convert("UTC")
    keep &= ts + timedelta(minutes=config.CANDLE_MINUTES) <= now   # candle must be complete
    out = df[keep.values].copy()
    out["ts"] = ts[keep.values]
    return out


def tidy(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Normalise columns, fix impossible high/low, drop broken rows."""
    df = df.rename(columns=str.lower).copy()
    df["symbol"] = symbol
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    return df[["symbol", "ts", "open", "high", "low", "close", "volume"]].drop_duplicates(["symbol", "ts"])


def symbol_from_filename(path: str) -> str:
    """'JIOFIN_5min.csv', '1792..._JIOFIN_5min.csv' -> 'JIOFIN'."""
    name = os.path.basename(path).upper()
    for sym in sorted(config.ALL_SYMBOLS, key=len, reverse=True):
        if re.search(rf"(^|[^A-Z]){sym}([^A-Z]|$)", name):
            return sym
    raise ValueError(f"Can't tell which stock {path} is. Put the symbol in the file name, e.g. JIOFIN_5min.csv")


def load_seed_csv(path: str, now=None) -> pd.DataFrame:
    """Read a CSV made by the Colab downloader (datetime, open, high, low, close, volume)."""
    sym = symbol_from_filename(path)
    raw = pd.read_csv(path)
    tcol = "datetime" if "datetime" in raw.columns else raw.columns[0]
    raw["ts"] = pd.to_datetime(raw[tcol], utc=True)
    return session_filter(tidy(raw, sym), now)


def fetch_yahoo(symbol: str, days: int, now=None) -> pd.DataFrame:
    """Last `days` days of 5-minute candles from Yahoo (max ~60 days for 5m)."""
    import yfinance as yf

    ticker = config.ALL_SYMBOLS[symbol]
    days = max(1, min(int(days), 59))
    raw = yf.download(ticker, period=f"{days}d", interval="5m", auto_adjust=False,
                      progress=False, threads=False)
    if raw is None or raw.empty:
        return pd.DataFrame(columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw.reset_index()
    tcol = "Datetime" if "Datetime" in raw.columns else raw.columns[0]
    idx = pd.to_datetime(raw[tcol])
    raw["ts"] = (idx.dt.tz_localize("UTC") if idx.dt.tz is None else idx).dt.tz_convert("UTC")
    return session_filter(tidy(raw, symbol), now)
