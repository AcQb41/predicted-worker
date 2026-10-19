"""The jobs GitHub Actions runs: seed, backfill, train (nightly), predict (every 5 min)."""
from __future__ import annotations

import glob
import json
import logging
import os
import platform
import shutil
import traceback
from datetime import timedelta

import numpy as np
import pandas as pd

from . import config
from .backtest import eod_bucket, run_walk_forward, score
from .features import build_panel, target_ts
from .marketdata import fetch_yahoo, load_seed_csv
from .models import (Chronos2Model, KronosModel, LightGBMModel, LSTMModel, NaiveModel, ensemble,
                     split_days)

log = logging.getLogger("predicted")


def _enabled() -> dict:
    return {"lstm": config.ENABLE_LSTM, "chronos2": config.ENABLE_CHRONOS, "kronos": config.ENABLE_KRONOS}


class run_logged:
    """Context manager: writes a row to worker_runs (running -> ok / error)."""

    def __init__(self, store, job):
        self.store, self.job, self.details, self.message = store, job, {}, ""

    def __enter__(self):
        try:
            self.id = self.store.start_run(self.job)
        except Exception as e:
            log.warning("could not log run start: %r", e)
            self.id = None
        return self

    def __exit__(self, et, ev, tb):
        status = "ok" if et is None else "error"
        msg = self.message if et is None else f"{et.__name__}: {ev}\n{''.join(traceback.format_tb(tb))[-1500:]}"
        if self.id is not None:
            try:
                self.store.finish_run(self.id, status, msg, self.details)
            except Exception as e:
                log.warning("could not log run end: %r", e)
        return False


# ------------------------------------------------------------------ seed
def seed(store, paths: list[str]) -> dict:
    """Load the downloader CSV files into the database (never overwrites existing candles)."""
    files = [p for pat in paths for p in glob.glob(pat)]
    if not files:
        raise SystemExit(f"No CSV files matched {paths}")
    out = {}
    with run_logged(store, "seed") as run:
        for f in files:
            df = load_seed_csv(f)
            n = store.upsert_candles(df, source="csv", mode="fill")
            out[os.path.basename(f)] = {"symbol": df.symbol.iloc[0] if len(df) else None, "rows": len(df), "inserted": n}
            log.info("seeded %s: %d rows (%d new)", f, len(df), n)
        run.details, run.message = out, f"{len(files)} files"
    return out


# ------------------------------------------------------------------ backfill
def backfill(store, days: int = 5, only_missing_history: bool = False) -> dict:
    """Fill gaps from Yahoo via yfinance. 'fill' mode: never overwrites live candles."""
    now = config.now_ist()
    out = {}
    have = {}
    if only_missing_history:
        since = now - timedelta(days=90)
        cur = store.get_candles(list(config.ALL_SYMBOLS), since)
        have = cur.groupby("symbol").size().to_dict() if len(cur) else {}
    for sym in config.ALL_SYMBOLS:
        d = 59 if have.get(sym, 0) < 1500 and only_missing_history else days
        try:
            df = fetch_yahoo(sym, d, now)
            n = store.upsert_candles(df, source="backfill", mode="fill") if len(df) else 0
            out[sym] = {"days": d, "rows": len(df), "inserted": n}
        except Exception as e:
            out[sym] = {"days": d, "error": repr(e)[:200]}
            log.warning("backfill %s failed: %r", sym, e)
    log.info("backfill: %s", out)
    return out


# ------------------------------------------------------------------ train (nightly)
def train(store, skip_backfill: bool = False) -> dict:
    """Backfill, walk-forward backtest, train final models, publish model versions."""
    now = config.now_ist()
    with run_logged(store, "train") as run:
        if not skip_backfill:
            run.details["backfill"] = backfill(store, days=5, only_missing_history=True)
        candles = store.get_candles(list(config.ALL_SYMBOLS), now - timedelta(days=int(config.TRAIN_LOOKBACK_DAYS * 1.5)))
        panel = build_panel(candles, now)
        enabled = _enabled()
        complete_days = sorted(panel.loc[panel.day_complete == 1, "date"].unique())
        log.info("panel: %d rows, %d complete days, stocks=%s", len(panel), len(complete_days), sorted(panel.symbol.unique()))

        bt = run_walk_forward(panel, enabled)
        scores = score(bt)

        # Final models on all complete days (last 5 days = early-stopping check).
        fit_days, val_days = split_days(complete_days)
        tmp = config.MODEL_DIR + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        naive = NaiveModel().fit(panel, fit_days, val_days)
        naive.save(tmp)
        lgb = LightGBMModel().fit(panel, fit_days, val_days)
        lgb.save(tmp)
        if enabled["lstm"]:
            LSTMModel().fit(panel, fit_days, val_days).save(tmp)

        version = now.strftime("%Y-%m-%d %H:%M IST")
        n_train = int(panel[panel.date.isin(complete_days)].y30.notna().sum())
        params = {
            "naive": {"up_rate": naive.up},
            "lightgbm": {"top_features": lgb.importance(), "learning_rate": 0.03, "num_leaves": 15},
            "lstm": {"seq_len": config.SEQ_LEN, "hidden": 48, "seeds": 3},
            "ensemble": {"members": ["lightgbm", "lstm"]},
            "chronos2": {"hf_model": "amazon/chronos-2", "context": 512, "zero_shot": True},
            "kronos": {"hf_model": "NeoQuasar/Kronos-small", "tokenizer": "NeoQuasar/Kronos-Tokenizer-base",
                       "context": 400, "samples": config.KRONOS_SAMPLES, "zero_shot": True},
        }
        active = ["naive", "lightgbm"] + (["lstm", "ensemble"] if enabled["lstm"] else []) \
            + (["chronos2"] if enabled["chronos2"] else []) + (["kronos"] if enabled["kronos"] else [])
        meta = {"version": version, "trained_at": now.isoformat(), "models": {}, "env": platform.python_version()}
        for m in active:
            row = {"model_key": m, "version": version, "train_start": str(complete_days[0]),
                   "train_end": str(complete_days[-1]), "n_train": n_train, "params": params.get(m, {}),
                   "backtest": scores.get(m, {})}
            vid = store.create_model_version(row)
            meta["models"][m] = {"version_id": vid, "resid_q": scores.get(m, {}).get("resid_q", {})}
        json.dump(meta, open(os.path.join(tmp, "meta.json"), "w"), indent=1, default=str)
        shutil.rmtree(config.MODEL_DIR, ignore_errors=True)
        shutil.move(tmp, config.MODEL_DIR)

        summary = {m: {h: scores[m][h]["ALL"] for h in ("30m", "eod") if h in scores.get(m, {}) and "ALL" in scores[m][h]}
                   for m in scores}
        run.details.update({"version": version, "days": len(complete_days), "rows": len(panel), "summary": summary})
        run.message = f"trained {', '.join(active)} on {len(complete_days)} days"
        log.info("trained version %s", version)
        return summary


# ------------------------------------------------------------------ predict (every 5 min)
def _load_models(store):
    meta_path = os.path.join(config.MODEL_DIR, "meta.json")
    if not os.path.exists(meta_path):
        log.info("no saved models found; training now")
        train(store)
    meta = json.load(open(meta_path))
    d = config.MODEL_DIR
    models = {"naive": NaiveModel().load(d), "lightgbm": LightGBMModel().load(d)}
    if "lstm" in meta["models"] and os.path.exists(os.path.join(d, "lstm.pt")):
        models["lstm"] = LSTMModel().load(d)
    return models, meta


def _band(meta, m, h, cl):
    rq = meta["models"].get(m, {}).get("resid_q") or meta["models"].get("naive", {}).get("resid_q", {})
    key = "30m" if h == "30m" else f"eod_{eod_bucket(cl)}"
    return rq.get(key) or [np.nan, np.nan]


def predict(store, refresh_today: bool = True) -> dict:
    """Predict from the latest completed candle of each stock, for every model and both horizons."""
    now = config.now_ist()
    with run_logged(store, "predict") as run:
        if refresh_today:
            run.details["backfill"] = backfill(store, days=1)
        candles = store.get_candles(list(config.ALL_SYMBOLS), now - timedelta(days=45))
        panel = build_panel(candles, now)
        latest = panel.groupby("symbol").tail(1)
        today = now.date()
        latest = latest[latest.date == today]
        if latest.empty:
            run.message = "no candle from today yet (market closed or data delayed)"
            log.info(run.message)
            return {"predictions": 0}
        idx = latest.index
        models, meta = _load_models(store)
        enabled = _enabled()
        outs = {m: mdl.predict(panel, idx) for m, mdl in models.items()}
        if "lstm" in outs:
            outs["ensemble"] = ensemble(outs["lightgbm"], outs["lstm"])
        if enabled["chronos2"] and "chronos2" in meta["models"]:
            try:
                outs["chronos2"] = Chronos2Model().predict(panel, idx)
            except Exception as e:
                log.warning("Chronos-2 skipped: %r", e)
        if enabled["kronos"] and "kronos" in meta["models"]:
            try:
                outs["kronos"] = KronosModel().predict(panel, idx)
            except Exception as e:
                log.warning("Kronos skipped: %r", e)

        rows = []
        for i in idx:
            r = panel.loc[i]
            made_at = r["ts_utc"]
            base = float(r["close"])
            cl = int(r["candles_left"])
            for h in ("30m", "eod"):
                if (h == "30m" and cl < 6) or (h == "eod" and cl < 1):
                    continue
                for m, o in outs.items():
                    ret = o.loc[i, f"ret_{h}"]
                    if pd.isna(ret):
                        continue
                    prob = o.loc[i, f"prob_{h}"]
                    lo, hi = o.loc[i, f"lo_{h}"], o.loc[i, f"hi_{h}"]
                    if pd.isna(lo) or pd.isna(hi):
                        q10, q90 = _band(meta, m, h, cl)
                        lo, hi = ret + q10, ret + q90
                    direction = ("up" if prob >= 0.5 else "down") if not pd.isna(prob) else ("up" if ret > 0 else "down")
                    rows.append({
                        "symbol": r["symbol"], "model_key": m, "horizon": h,
                        "made_at": made_at.isoformat(), "target_ts": target_ts(made_at, h).isoformat(),
                        "base_price": base, "pred_return": float(ret), "pred_price": base * float(np.exp(ret)),
                        "pred_direction": direction, "prob_up": None if pd.isna(prob) else float(prob),
                        "band_low": None if pd.isna(lo) else base * float(np.exp(lo)),
                        "band_high": None if pd.isna(hi) else base * float(np.exp(hi)),
                        "model_version_id": meta["models"].get(m, {}).get("version_id"),
                    })
        n = store.upsert_predictions(rows) if rows else 0
        resolved = store.resolve()
        as_of = latest["datetime"].max()
        run.message = f"{n} predictions as of {as_of:%H:%M} IST; {resolved} resolved"
        run.details.update({"as_of": str(as_of), "models": sorted(outs), "stocks": sorted(latest.symbol)})
        log.info(run.message)
        return {"predictions": n, "resolved": resolved}
