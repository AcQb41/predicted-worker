"""Central settings for the Predicted worker.

Everything that changes between environments comes from environment variables
(GitHub Actions secrets and variables). Everything else lives here.
"""
from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# ---- Symbols -------------------------------------------------------------
# Internal symbol -> Yahoo Finance ticker. Must match the `symbols` table in Lovable.
STOCKS: dict[str, str] = {
    "PCJEWELLER": "PCJEWELLER.NS",
    "JIOFIN": "JIOFIN.NS",
    "SUZLON": "SUZLON.NS",
    "JSWENERGY": "JSWENERGY.NS",
    "OPTIEMUS": "OPTIEMUS.NS"
}
INDICES: dict[str, str] = {
    "NIFTY50": "^NSEI",
    "BANKNIFTY": "^NSEBANK",
    "INDIAVIX": "^INDIAVIX",
}
ALL_SYMBOLS: dict[str, str] = {**STOCKS, **INDICES}

# ---- Market session (IST) --------------------------------------------------
SESSION_FIRST_CANDLE = (9, 15)   # first 5-minute candle starts 09:15
SESSION_LAST_CANDLE = (15, 25)   # last 5-minute candle starts 15:25
CANDLE_MINUTES = 5
CANDLES_PER_DAY = 75

# ---- Prediction setup --------------------------------------------------------
HORIZON_30M_CANDLES = 6          # 6 x 5 min = 30 minutes
HORIZONS = ("30m", "eod")
MODEL_KEYS = ("naive", "lightgbm", "lstm", "ensemble", "chronos2", "kronos")
MODEL_LABELS = {
    "naive": "Naive (no change)",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
    "ensemble": "Ensemble (LightGBM + LSTM)",
    "chronos2": "Chronos-2",
    "kronos": "Kronos",
}


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _int(name: str, default: int) -> int:
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


# ---- Environment ---------------------------------------------------------------
API_URL = os.environ.get("PREDICTED_API_URL", "")            # worker-api edge function URL
WORKER_TOKEN = os.environ.get("PREDICTED_WORKER_TOKEN", "")  # shared secret, same value in Lovable
ANON_KEY = os.environ.get("PREDICTED_ANON_KEY", "")          # optional: Lovable publishable key
MODEL_DIR = os.environ.get("PREDICTED_MODEL_DIR", "models")
KRONOS_REPO_DIR = os.environ.get("KRONOS_REPO_DIR", "Kronos")

ENABLE_LSTM = _flag("ENABLE_LSTM", True)
ENABLE_CHRONOS = _flag("ENABLE_CHRONOS", True)
ENABLE_KRONOS = _flag("ENABLE_KRONOS", True)

TRAIN_LOOKBACK_DAYS = _int("TRAIN_LOOKBACK_DAYS", 250)      # trading history used for training
BACKTEST_FOLDS = _int("BACKTEST_FOLDS", 8)                  # walk-forward test blocks (5 days each)
BACKTEST_BLOCK_DAYS = 5
MIN_TRAIN_DAYS = 25
CHRONOS_BACKTEST_EVERY = _int("CHRONOS_BACKTEST_EVERY", 3)  # score every Nth test candle
KRONOS_BACKTEST_EVERY = _int("KRONOS_BACKTEST_EVERY", 12)
KRONOS_SAMPLES = _int("KRONOS_SAMPLES", 3)
SEQ_LEN = 24                                                # LSTM look-back: 24 candles = 2 hours
COST_ROUND_TRIP = 0.0005                                    # 0.05% per trade in backtests
SEED = 42


def now_ist() -> datetime:
    """Current time in IST. PREDICTED_NOW (ISO string) overrides it for tests and replays."""
    fake = os.environ.get("PREDICTED_NOW")
    if fake:
        dt = datetime.fromisoformat(fake)
        return dt.astimezone(IST) if dt.tzinfo else dt.replace(tzinfo=IST)
    return datetime.now(IST)
