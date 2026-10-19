"""Walk-forward backtest: train on the past, test on the next 5 days, repeat.

Produces the 'Backtest' scores shown on the Models page and the error bands
used around every live prediction.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from . import config
from .models import (Chronos2Model, KronosModel, LightGBMModel, LSTMModel, NaiveModel,
                     TARGET_COL, ensemble, split_days)

log = logging.getLogger("predicted")

EOD_BUCKETS = [(1, 19, "late"), (20, 49, "mid"), (50, 75, "early")]


def eod_bucket(candles_left: int) -> str:
    for lo, hi, name in EOD_BUCKETS:
        if lo <= candles_left <= hi:
            return name
    return "late"


def make_folds(days: list, n_folds: int, block: int, min_train: int):
    days = sorted(days)
    starts = list(range(len(days) - block, min_train - 1, -block))[:n_folds]
    return [(days[:s], days[s:s + block]) for s in sorted(starts)]


def run_walk_forward(panel: pd.DataFrame, enabled: dict) -> pd.DataFrame:
    """Predictions for every test row of every fold, one column block per model."""
    days = sorted(panel.loc[panel.day_complete == 1, "date"].unique())
    folds = make_folds(days, config.BACKTEST_FOLDS, config.BACKTEST_BLOCK_DAYS, config.MIN_TRAIN_DAYS)
    if not folds:
        raise RuntimeError(f"Need at least {config.MIN_TRAIN_DAYS + config.BACKTEST_BLOCK_DAYS} complete days for a backtest; have {len(days)}.")
    parts = []
    for k, (train_days, test_days) in enumerate(folds, 1):
        fit_days, val_days = split_days(train_days)
        test_idx = panel.index[panel.date.isin(test_days) & (panel.y30.notna() | panel.yeod.notna())]
        res = panel.loc[test_idx, ["symbol", "datetime", "date", "close", "y30", "yeod", "candles_left"]].copy()
        res["fold"] = k
        naive = NaiveModel().fit(panel, fit_days, val_days)
        outs = {"naive": naive.predict(panel, test_idx)}
        lgb = LightGBMModel().fit(panel, fit_days, val_days)
        outs["lightgbm"] = lgb.predict(panel, test_idx)
        if enabled.get("lstm"):
            outs["lstm"] = LSTMModel().fit(panel, fit_days, val_days).predict(panel, test_idx)
            outs["ensemble"] = ensemble(outs["lightgbm"], outs["lstm"])
        for m, o in outs.items():
            for c in o.columns:
                res[f"{m}__{c}"] = o[c]
        parts.append(res)
        log.info("fold %d/%d: train %s..%s test %s..%s (%d rows)", k, len(folds), train_days[0], train_days[-1], test_days[0], test_days[-1], len(test_idx))
    bt = pd.concat(parts)

    # Pretrained models: no training, so they are scored on a sample of the same test rows.
    if enabled.get("chronos2"):
        try:
            sample = bt.index[::config.CHRONOS_BACKTEST_EVERY]
            o = Chronos2Model().predict(panel, sample)
            for c in o.columns:
                bt.loc[sample, f"chronos2__{c}"] = o[c]
        except Exception as e:  # never let an optional model break the run
            log.warning("Chronos-2 backtest skipped: %r", e)
    if enabled.get("kronos"):
        try:
            sample = bt.index[::config.KRONOS_BACKTEST_EVERY]
            o = KronosModel().predict(panel, sample, horizons=("30m",))
            for c in o.columns:
                bt.loc[sample, f"kronos__{c}"] = o[c]
        except Exception as e:
            log.warning("Kronos backtest skipped: %r", e)
    return bt


def _metrics(q: pd.DataFrame, m: str, h: str) -> dict | None:
    rc, pc = f"{m}__ret_{h}", f"{m}__prob_{h}"
    y = TARGET_COL[h]
    if rc not in q:
        return None
    q = q.dropna(subset=[rc, y])
    q = q[q[y] != 0]
    if len(q) < 30:
        return None
    has_prob = pc in q and q[pc].notna().all()
    pred_up = (q[pc] >= 0.5) if has_prob else (q[rc] > 0)
    actual_up = q[y] > 0
    correct = pred_up == actual_up
    naive_up = q["naive__prob_" + h] >= 0.5 if "naive__prob_" + h in q else pd.Series(actual_up.mean() >= 0.5, index=q.index)
    naive_acc = float((naive_up == actual_up).mean())
    n, k = len(q), int(correct.sum())
    pred_price = q["close"] * np.exp(q[rc])
    true_price = q["close"] * np.exp(q[y])
    rel_err = (pred_price - true_price).abs() / true_price
    rel_naive = (q["close"] - true_price).abs() / true_price
    out = {
        "n": n,
        "dir_acc": k / n,
        "balanced_acc": 0.5 * (float(pred_up[actual_up].mean()) + float((~pred_up[~actual_up]).mean())) if actual_up.any() and (~actual_up).any() else None,
        "p_vs_naive": float(stats.binomtest(k, n, min(max(naive_acc, 1e-6), 0.999), alternative="greater").pvalue) if m != "naive" else None,
        "auc": float(roc_auc_score(actual_up, q[pc])) if has_prob and actual_up.nunique() == 2 and m != "naive" else None,
        "mape_pct": float(rel_err.mean() * 100),
        "skill_vs_naive_pct": float((1 - rel_err.sum() / rel_naive.sum()) * 100),
        "ic": float(stats.spearmanr(q[rc], q[y]).correlation) if m != "naive" and q[rc].nunique() > 1 else None,
    }
    if m != "naive":
        out.update(_trading(q, rc, y, h))
    return out


def _trading(q, rc, y, h) -> dict:
    """Simple strategy: trade when the predicted move beats costs; non-overlapping trades."""
    if h == "30m":
        q = q[q["datetime"].dt.minute.isin([15, 45])]
    else:
        q = q[q["datetime"].dt.strftime("%H:%M") == "10:15"]
    sig = np.sign(q[rc]) * (q[rc].abs() > config.COST_ROUND_TRIP)
    pnl = sig * (np.exp(q[y]) - 1) - sig.abs() * config.COST_ROUND_TRIP
    n = int((sig != 0).sum())
    return {"bt_trades": n, "bt_win_rate": float((pnl[sig != 0] > 0).mean()) if n else None,
            "bt_return_pct": float(pnl.sum() * 100 / max(1, q["symbol"].nunique()))}


def score(bt: pd.DataFrame) -> dict:
    """{model: {horizon: {'ALL': metrics, 'JIOFIN': metrics, ...}, 'resid_q': {...}}}"""
    models = sorted({c.split("__")[0] for c in bt.columns if "__" in c})
    result = {}
    for m in models:
        result[m] = {}
        for h in TARGET_COL:
            per = {}
            allm = _metrics(bt, m, h)
            if allm:
                per["ALL"] = allm
            for sym, g in bt.groupby("symbol"):
                mm = _metrics(g, m, h)
                if mm:
                    per[sym] = mm
            if per:
                result[m][h] = per
        result[m]["resid_q"] = residual_bands(bt, m)
        fold_ic = {}
        for h in TARGET_COL:
            rc, y = f"{m}__ret_{h}", TARGET_COL[h]
            if rc in bt and m != "naive":
                ics = bt.dropna(subset=[rc, y]).groupby("fold").apply(
                    lambda g: stats.spearmanr(g[rc], g[y]).correlation if len(g) > 30 and g[rc].nunique() > 1 else np.nan,
                    include_groups=False)
                fold_ic[h] = {str(k): (None if pd.isna(v) else round(float(v), 4)) for k, v in ics.items()}
        result[m]["fold_ic"] = fold_ic
    return result


def residual_bands(bt: pd.DataFrame, m: str) -> dict:
    """10th / 90th percentile of (actual - predicted) return, for the 80% band."""
    out = {}
    src = m if f"{m}__ret_30m" in bt and bt[f"{m}__ret_30m"].notna().sum() > 50 else "naive"
    q = bt.dropna(subset=[f"{src}__ret_30m", "y30"])
    if len(q):
        r = q["y30"] - q[f"{src}__ret_30m"]
        out["30m"] = [float(r.quantile(0.1)), float(r.quantile(0.9))]
    src = m if f"{m}__ret_eod" in bt and bt[f"{m}__ret_eod"].notna().sum() > 50 else "naive"
    q = bt.dropna(subset=[f"{src}__ret_eod", "yeod"])
    for lo, hi, name in EOD_BUCKETS:
        g = q[(q.candles_left >= lo) & (q.candles_left <= hi)]
        if len(g) > 20:
            r = g["yeod"] - g[f"{src}__ret_eod"]
            out[f"eod_{name}"] = [float(r.quantile(0.1)), float(r.quantile(0.9))]
    return out
