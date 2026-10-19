"""Tomorrow's-open pipeline test on SYNTHETIC markets (no network needed).

The fake data has a planted rule: a stock's opening gap = its beta x the overnight US day
return + part of the Nikkei's opening gap + noise. If features are aligned correctly (no look-
ahead, no off-by-one), the PRE-OPEN models must find it and the EVENING models must not.
Then runs the real evening and morning jobs on a local file store and checks scoring.

Run: PREDICTED_LOCAL_STORE=/tmp/x PREDICTED_OPEN_MODEL_DIR=/tmp/mo ENABLE_CHRONOS=0 ENABLE_KRONOS=0 \
     python tests/open_test.py
"""
import os
import shutil
import sys

import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted import config, opening  # noqa: E402
from predicted.store import LocalStore  # noqa: E402

warnings.filterwarnings("ignore", category=Warning, module="lightgbm")
rng = np.random.default_rng(7)
U = lambda col: pd.to_datetime(col, utc=True)  # noqa: E731
END = pd.Timestamp("2026-10-16")
all_days = pd.bdate_range("2021-01-01", END)


def series(days, drift=0.0, vol=0.01, start=100.0, gaps=None, intraday=None):
    """Daily OHLC where close->open is `gaps` and open->close is `intraday`."""
    n = len(days)
    g = gaps if gaps is not None else rng.normal(0, vol / 3, n)
    d = intraday if intraday is not None else rng.normal(drift, vol, n)
    o, c = np.empty(n), np.empty(n)
    prev = start
    for i in range(n):
        o[i] = prev * np.exp(g[i])
        c[i] = o[i] * np.exp(d[i])
        prev = c[i]
    hi = np.maximum(o, c) * np.exp(np.abs(rng.normal(0, vol / 2, n)))
    lo = np.minimum(o, c) * np.exp(-np.abs(rng.normal(0, vol / 2, n)))
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c,
                         "volume": rng.integers(1e5, 1e6, n).astype(float)}, index=days)


# Markets with their own holidays
us_days = all_days[rng.random(len(all_days)) > 0.04]
jp_days = all_days[rng.random(len(all_days)) > 0.06]
in_days = all_days[rng.random(len(all_days)) > 0.05]
in_days = in_days.union(pd.DatetimeIndex([END]))
spx = series(us_days, vol=0.01, start=4000)
spx_day = pd.Series(np.log(spx.close / spx.open), index=us_days)
n225 = series(jp_days, vol=0.01, start=30000)
# Nikkei's gap reacts to the previous US day
us_prev = spx_day.reindex(jp_days, method="ffill").shift(1).fillna(0).values
n225 = series(jp_days, vol=0.01, start=30000, gaps=0.5 * us_prev + rng.normal(0, 0.003, len(jp_days)))

daily = {"SPX": spx, "NDX": series(us_days, vol=0.012, start=12000), "N225": n225,
         "HSI": series(jp_days, vol=0.012, start=20000), "KOSPI": series(jp_days, vol=0.01, start=2500),
         "USDINR": series(all_days, vol=0.002, start=83), "BRENT": series(all_days, vol=0.02, start=80),
         "NIFTY50": series(in_days, vol=0.008, start=20000), "BANKNIFTY": series(in_days, vol=0.01, start=45000),
         "INDIAVIX": series(in_days, vol=0.03, start=14)}

# Stocks: gap_D = beta * sum(US day returns over [P, D)) + 0.3 * Nikkei gap on D + noise
nk_gap = pd.Series(np.log(n225.open / n225.close.shift(1)).values, index=jp_days)
P = in_days[:-1]
D = in_days[1:]
us_sum = opening._window_sum(spx_day, P, D)
nk_on_D = nk_gap.reindex(D).fillna(0).values
betas = {"PCJEWELLER": 1.6, "JIOFIN": 1.1, "SUZLON": 1.4, "JSWENERGY": 0.9}
for k, (sym, b) in enumerate(betas.items()):
    gaps = np.concatenate([[0.0], b * us_sum + 0.3 * nk_on_D + rng.normal(0, 0.004, len(D))])
    daily[sym] = series(in_days, vol=0.015, start=[20, 300, 40, 500][k], gaps=gaps)

# ---- 1) alignment check on the backtest
os.environ.setdefault("OPEN_BACKTEST_DAYS", "250")
table = opening.build_table(daily)
assert set(table.symbol) == set(config.STOCKS), "all four stocks present"
lab = table.dropna(subset=["y"])
assert np.allclose(lab.y.values, np.log(lab.open_actual / lab.base).values), "target = ln(open_D / close_P)"
bt = opening.walk_forward(table, daily, ["naive", "linear", "lightgbm", "ensemble"])
sc = opening.score(bt, ["naive", "linear", "lightgbm", "ensemble"])
for m in ["linear", "lightgbm", "ensemble"]:
    mo, ev = sc[m]["open"]["ALL"], sc[m]["open_evening"]["ALL"]
    print(f"{m:9s} pre-open: dir {mo['dir_acc']:.3f} skill {mo['skill_vs_naive_pct']:6.1f}% ic {mo['ic']:.3f} | "
          f"evening: dir {ev['dir_acc']:.3f} skill {ev['skill_vs_naive_pct']:6.1f}% ic {ev['ic']:.3f}")
    assert mo["dir_acc"] > 0.75 and mo["skill_vs_naive_pct"] > 30, f"{m}: pre-open should find the planted rule"
    assert abs(ev["ic"]) < 0.15 and ev["dir_acc"] < 0.58, f"{m}: evening must NOT see overnight data (leak!)"
lin = opening.OpenLinear().fit(lab, opening.STAGE_FEATURES["morning"])
print("linear top features:", lin.importance(opening.STAGE_FEATURES["morning"]))
assert {"beta_x_us", "spx_day"} & set(list(lin.importance(opening.STAGE_FEATURES["morning"]))[:3])
assert sc["naive"]["open"]["ALL"]["skill_vs_naive_pct"] == 0.0
print("alignment check OK (pre-open learns the overnight rule, evening can't)")

# ---- 2) the real jobs on a local store
store_dir = os.environ["PREDICTED_LOCAL_STORE"] + "_open"
shutil.rmtree(store_dir, ignore_errors=True)
shutil.rmtree(opening.MODEL_DIR, ignore_errors=True)
store = LocalStore(store_dir)
import predicted.store as st  # noqa: E402
st.get_store = lambda: store
cut = lambda d, last: {k: v[v.index <= last] for k, v in d.items()}  # noqa: E731

# Friday 16 Oct, 15:50 IST: evening estimate for Monday 19 Oct (needs Friday's bar)
os.environ["PREDICTED_NOW"] = "2026-10-16T15:50:00+05:30"
opening.fetch_daily = lambda now=None: cut(daily, END)
out = opening.predict(store, "auto")
print("evening:", out)
assert out["stage"] == "evening" and out["for"] == "2026-10-19" and out["predictions"] == 16
p = pd.DataFrame(store.db["predictions"])
assert set(p.model_key) == {"naive", "linear", "lightgbm", "ensemble"} and (p.horizon == "open").all()
assert (U(p.made_at) == pd.Timestamp("2026-10-16T10:00Z")).all() and (U(p.target_ts) == pd.Timestamp("2026-10-19T03:45Z")).all()
assert (p.band_low < p.pred_price).all() and (p.pred_price < p.band_high).all()
assert len(store.db["model_versions"]) == 4 and all(v["params"]["scope"] == "open" for v in store.db["model_versions"])

# Monday 19 Oct, 08:40 IST: pre-open update. US Friday session + Monday's Asian opens are in.
mon = pd.Timestamp("2026-10-19")
us_fri = spx.loc[[END]]
asia_mon = {k: pd.concat([daily[k], series(pd.DatetimeIndex([mon]), vol=0.01, start=float(daily[k].close.iloc[-1]))])
            for k in ("N225", "HSI", "KOSPI")}
live = {**cut(daily, END), **asia_mon}
# A partial Indian bar for "today" must be ignored by the morning stage
live["JIOFIN"] = pd.concat([live["JIOFIN"], series(pd.DatetimeIndex([mon]), start=999)])
opening.fetch_daily = lambda now=None: live
os.environ["PREDICTED_NOW"] = "2026-10-19T08:40:00+05:30"
out = opening.predict(store, "auto")
print("morning:", out)
assert out["stage"] == "morning" and out["for"] == "2026-10-19" and out["predictions"] == 16
p = pd.DataFrame(store.db["predictions"])
assert len(p) == 32 and set(U(p.made_at)) == {pd.Timestamp("2026-10-16T10:00Z"), pd.Timestamp("2026-10-19T03:15Z")}
morning = p[U(p.made_at) == pd.Timestamp("2026-10-19T03:15Z")]
evening = p[U(p.made_at) == pd.Timestamp("2026-10-16T10:00Z")]
assert np.allclose(sorted(morning.base_price), sorted(evening.base_price)), "same base: Friday's close"

# 09:20 IST: too late, must refuse
os.environ["PREDICTED_NOW"] = "2026-10-19T09:20:00+05:30"
assert "skipped" in opening.predict(store, "morning")

# Scoring: the 09:15 candle's OPEN on Monday, resolved after 09:21
cand = []
for sym in config.STOCKS:
    base = float(evening[evening.symbol == sym].base_price.iloc[0])
    for k, mins in enumerate([0, 5]):
        ts = pd.Timestamp("2026-10-19 09:15", tz=config.IST) + pd.Timedelta(minutes=mins)
        o = base * (1.01 if k == 0 else 1.02)
        cand.append({"symbol": sym, "ts": ts.tz_convert("UTC"), "open": o, "high": o * 1.01, "low": o * 0.99,
                     "close": o, "volume": 1000})
store.upsert_candles(pd.DataFrame(cand), source="live", mode="fill")
assert store.resolve(pd.Timestamp("2026-10-19 09:18", tz=config.IST)) == 0, "not before 09:21"
n = store.resolve(pd.Timestamp("2026-10-19 09:30", tz=config.IST))
assert n == 32, n
p = pd.DataFrame(store.db["predictions"])
assert np.allclose(p.actual_return, np.log(1.01)), "scored against the 09:15 open, not a close"
print("scoring OK:", p.groupby("model_key").direction_correct.mean().round(2).to_dict())
print("OPEN TEST OK")
