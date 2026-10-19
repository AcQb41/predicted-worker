"""Command line: python -m predicted <command>

  seed "data/seed/*.csv"   load the downloader CSV files into the database
  backfill [--days N]      fill gaps from Yahoo (default 5 days; 59 max)
  train                    nightly: backfill, walk-forward backtest, train, publish versions
  predict                  every 5 minutes: predict from the latest candle, resolve old predictions
  status                   print what the database holds
"""
from __future__ import annotations

import argparse
import json
import logging

from . import jobs
from .store import get_store


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(prog="predicted")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("seed")
    s.add_argument("paths", nargs="+")
    b = sub.add_parser("backfill")
    b.add_argument("--days", type=int, default=5)
    b.add_argument("--history", action="store_true", help="pull 59 days for any symbol with little history")
    t = sub.add_parser("train")
    t.add_argument("--skip-backfill", action="store_true")
    pr = sub.add_parser("predict")
    pr.add_argument("--no-refresh", action="store_true", help="don't pull today's candles from Yahoo first")
    sub.add_parser("status")
    a = p.parse_args()

    store = get_store()
    if a.cmd == "seed":
        out = jobs.seed(store, a.paths)
    elif a.cmd == "backfill":
        out = jobs.backfill(store, a.days, only_missing_history=a.history)
    elif a.cmd == "train":
        out = jobs.train(store, skip_backfill=a.skip_backfill)
    elif a.cmd == "predict":
        out = jobs.predict(store, refresh_today=not a.no_refresh)
    else:
        out = store.status()
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
