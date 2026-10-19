"""Replays one trading day candle by candle against a local store and checks that
predictions are made at the right times and resolved against the right prices.

Run:  PREDICTED_LOCAL_STORE=/tmp/x PREDICTED_MODEL_DIR=/tmp/m ENABLE_CHRONOS=0 ENABLE_KRONOS=0 \
      python tests/replay_test.py 2026-10-16
"""
import logging
import math
import os
import shutil
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted import config, jobs  # noqa: E402
from predicted.marketdata import load_seed_csv  # noqa: E402
from predicted.store import LocalStore  # noqa: E402

logging.basicConfig(level=logging.WARNING)
day = sys.argv[1] if len(sys.argv) > 1 else "2026-10-16"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = os.environ["PREDICTED_LOCAL_STORE"] + "_replay"
shutil.rmtree(path, ignore_errors=True)
store = LocalStore(path)

full = pd.concat([load_seed_csv(os.path.join(os.environ.get("SEED_DIR", os.path.join(root, "data/seed")), f"{s}_5min.csv"), now=pd.Timestamp("2030-01-01", tz="UTC"))
                  for s in config.STOCKS])
full["ts"] = pd.to_datetime(full["ts"], utc=True)
ist_day = pd.Timestamp(day, tz=config.IST)
times = pd.date_range(ist_day + pd.Timedelta(hours=10), ist_day + pd.Timedelta(hours=15, minutes=25), freq="25min")

for t in times:
    store.candles = full[full.ts < t.tz_convert("UTC")].assign(source="csv")
    os.environ["PREDICTED_NOW"] = (t + pd.Timedelta(minutes=1)).isoformat()
    config_now = config.now_ist()
    jobs.predict(store, refresh_today=False)

store.candles = full[full.ts < (ist_day + pd.Timedelta(days=1)).tz_convert("UTC")].assign(source="csv")
store.resolve(now=ist_day + pd.Timedelta(hours=16))

p = pd.DataFrame(store.db["predictions"])
print("predictions:", len(p), "| status:", p.status.value_counts().to_dict())
print(p.groupby(["horizon"]).made_at.nunique().rename("distinct made_at per horizon").to_dict())

# Check every resolved prediction against the source data
idx = full.set_index(["symbol", "ts"])["close"]
bad = 0
for r in p[p.status == "resolved"].itertuples():
    want = idx[(r.symbol, pd.Timestamp(r.target_ts))]
    ok = math.isclose(r.actual_price, want, rel_tol=1e-9)
    dir_ok = r.direction_correct == ((r.pred_direction == "up") == (want > r.base_price)) if want != r.base_price else r.direction_correct is None
    bad += (not ok) or (not dir_ok)
print("resolution mismatches:", bad)
latest_made = p.made_at.max()
print("30m predictions after 14:55 (should be 0):",
      int(((pd.to_datetime(p.made_at).dt.tz_convert(config.IST).dt.strftime("%H:%M") > "14:55") & (p.horizon == "30m")).sum()))
acc = p[p.status == "resolved"].groupby(["horizon", "model_key"]).direction_correct.mean().round(3)
print(acc.to_string())
assert bad == 0
print("REPLAY OK")
