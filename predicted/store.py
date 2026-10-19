"""How the worker reads and writes Predicted's database.

Lovable Cloud does not give out database keys, so the worker never connects to
Postgres directly. It calls the `worker-api` edge function over HTTPS with a
shared secret (PREDICTED_WORKER_TOKEN). `LocalStore` implements the same
interface on local files, for tests.
"""
from __future__ import annotations

import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from . import config

CANDLE_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def _clean_json(obj):
    """Replace NaN/inf with None and numpy types with Python types (JSON has no NaN)."""
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        f = float(obj)
        return None if (math.isnan(f) or math.isinf(f)) else f
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp, datetime)):
        return pd.Timestamp(obj).tz_convert("UTC").isoformat() if pd.Timestamp(obj).tzinfo else pd.Timestamp(obj).isoformat()
    return obj


def candles_to_rows(df: pd.DataFrame, source: str) -> list[dict]:
    out = df[CANDLE_COLS].copy()
    out["ts"] = pd.to_datetime(out["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["volume"] = out["volume"].fillna(0).round().astype("int64")
    out["source"] = source
    return _clean_json(out.to_dict("records"))


def rows_to_candles(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CANDLE_COLS)
    df = pd.DataFrame(rows)[CANDLE_COLS]
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df["volume"] = df["volume"].fillna(0).astype(float)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


class ApiStore:
    """Talks to the `worker-api` edge function."""

    def __init__(self, url: str | None = None, token: str | None = None, anon_key: str | None = None):
        self.url = url or config.API_URL
        self.token = token or config.WORKER_TOKEN
        self.anon_key = anon_key if anon_key is not None else config.ANON_KEY
        if not self.url or not self.token:
            raise RuntimeError("PREDICTED_API_URL and PREDICTED_WORKER_TOKEN must be set")

    def _call(self, action: str, **payload):
        headers = {"Content-Type": "application/json", "x-worker-token": self.token}
        if self.anon_key:
            headers["apikey"] = self.anon_key
            headers["Authorization"] = f"Bearer {self.anon_key}"
        body = json.dumps(_clean_json({"action": action, **payload}))
        last_err = None
        for attempt in range(4):
            try:
                r = requests.post(self.url, data=body, headers=headers, timeout=120)
                if r.status_code >= 500 or r.status_code == 429:
                    raise requests.HTTPError(f"{r.status_code}: {r.text[:300]}")
                if r.status_code >= 400:
                    raise RuntimeError(f"worker-api {action} failed {r.status_code}: {r.text[:500]}")
                return r.json()
            except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as e:
                last_err = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"worker-api {action} failed after retries: {last_err}")

    # -- candles --
    def get_candles(self, symbols: list[str], since: datetime) -> pd.DataFrame:
        rows: list[dict] = []
        for sym in symbols:  # one symbol per call keeps responses small
            res = self._call("get_candles", symbols=[sym], since=pd.Timestamp(since).tz_convert("UTC").isoformat())
            rows.extend(res.get("rows", []))
        return rows_to_candles(rows)

    def upsert_candles(self, df: pd.DataFrame, source: str, mode: str) -> int:
        rows = candles_to_rows(df, source)
        n = 0
        for i in range(0, len(rows), 1000):
            res = self._call("upsert_candles", rows=rows[i:i + 1000], mode=mode)
            n += int(res.get("count", 0))
        return n

    # -- predictions / models / runs --
    def upsert_predictions(self, rows: list[dict]) -> int:
        n = 0
        for i in range(0, len(rows), 500):
            n += int(self._call("upsert_predictions", rows=rows[i:i + 500]).get("count", 0))
        return n

    def create_model_version(self, row: dict) -> int:
        return int(self._call("create_model_version", row=row)["id"])

    def start_run(self, job: str) -> int:
        return int(self._call("log_run", row={"job": job, "status": "running"})["id"])

    def finish_run(self, run_id: int, status: str, message: str = "", details: dict | None = None) -> None:
        self._call("log_run", id=run_id, row={"status": status, "message": message[:2000], "details": details or {}})

    def resolve(self) -> int:
        return int(self._call("resolve").get("resolved", 0))

    def status(self) -> dict:
        return self._call("status")


class LocalStore:
    """File-based stand-in for the database, used by tests and dry runs."""

    def __init__(self, path: str):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self._candles_file = os.path.join(path, "candles.csv")
        self._json_file = os.path.join(path, "db.json")
        self.candles = pd.read_csv(self._candles_file, parse_dates=["ts"]) if os.path.exists(self._candles_file) else pd.DataFrame(columns=CANDLE_COLS + ["source"])
        self.db = json.load(open(self._json_file)) if os.path.exists(self._json_file) else {"predictions": [], "model_versions": [], "runs": []}

    def _save(self):
        self.candles.to_csv(self._candles_file, index=False)
        json.dump(self.db, open(self._json_file, "w"), default=str)

    def get_candles(self, symbols, since):
        c = self.candles
        if c.empty:
            return rows_to_candles([])
        c = c[c.symbol.isin(symbols) & (pd.to_datetime(c.ts, utc=True) >= pd.Timestamp(since).tz_convert("UTC"))]
        return rows_to_candles(c.to_dict("records"))

    def upsert_candles(self, df, source, mode):
        rows = pd.DataFrame(candles_to_rows(df, source))
        if rows.empty:
            return 0
        rows["ts"] = pd.to_datetime(rows["ts"], utc=True)
        cur = self.candles.copy()
        if not cur.empty:
            cur["ts"] = pd.to_datetime(cur["ts"], utc=True)
        both = pd.concat([cur, rows] if mode == "overwrite" else [rows, cur])
        before = len(cur)
        self.candles = both.drop_duplicates(["symbol", "ts"], keep="last").sort_values(["symbol", "ts"]).reset_index(drop=True)
        self._save()
        return len(rows) if mode == "overwrite" else len(self.candles) - before

    def upsert_predictions(self, rows):
        rows = _clean_json(rows)
        key = lambda r: (r["symbol"], r["model_key"], r["horizon"], r["made_at"])
        existing = {key(r): r for r in self.db["predictions"]}
        for r in rows:
            r = {**r, "status": "open"}
            existing[key(r)] = {**existing.get(key(r), {}), **r}
        self.db["predictions"] = list(existing.values())
        self._save()
        return len(rows)

    def create_model_version(self, row):
        row = _clean_json(dict(row))
        row["id"] = len(self.db["model_versions"]) + 1
        self.db["model_versions"].append(row)
        self._save()
        return row["id"]

    def start_run(self, job):
        rid = len(self.db["runs"]) + 1
        self.db["runs"].append({"id": rid, "job": job, "status": "running"})
        self._save()
        return rid

    def finish_run(self, run_id, status, message="", details=None):
        for r in self.db["runs"]:
            if r["id"] == run_id:
                r.update(status=status, message=message, details=_clean_json(details or {}))
        self._save()

    def resolve(self, now: datetime | None = None):
        """Same rule as the SQL function resolve_predictions()."""
        now = pd.Timestamp(now or config.now_ist()).tz_convert("UTC")
        c = self.candles.copy()
        c["ts"] = pd.to_datetime(c["ts"], utc=True)
        n = 0
        for p in self.db["predictions"]:
            if p.get("status") != "open":
                continue
            made, target = pd.Timestamp(p["made_at"]), pd.Timestamp(p["target_ts"])
            if now < target + timedelta(minutes=6):
                continue
            m = c[(c.symbol == p["symbol"]) & (c.ts <= target) & (c.ts > made) & (c.ts >= target - timedelta(minutes=10))]
            if m.empty:
                if now > target + timedelta(days=1):
                    p["status"] = "expired"
                continue
            close = float(m.sort_values("ts").close.iloc[-1])
            p.update(actual_price=close, actual_return=math.log(close / p["base_price"]),
                     abs_error=abs(p["pred_price"] - close),
                     direction_correct=None if close == p["base_price"] else ((p["pred_direction"] == "up") == (close > p["base_price"])),
                     status="resolved", resolved_at=now.isoformat())
            n += 1
        self._save()
        return n

    def status(self):
        c = self.candles
        return {"candles": {s: int((c.symbol == s).sum()) for s in config.ALL_SYMBOLS}}


def get_store():
    local = os.environ.get("PREDICTED_LOCAL_STORE")
    return LocalStore(local) if local else ApiStore()
