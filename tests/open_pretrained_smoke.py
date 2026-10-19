"""Chronos-2 / Kronos wrappers for tomorrow's open, with tiny RANDOM-weight models of the real
architectures (no download). Checks plumbing and shapes, not forecast quality.

Run: KRONOS_REPO_DIR=/path/to/Kronos OPEN_BACKTEST_DAYS=40 python tests/open_pretrained_smoke.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted import config, opening  # noqa: E402

torch.manual_seed(0)
rng = np.random.default_rng(1)
days = pd.bdate_range("2024-01-01", "2026-10-16")


def bars(n0):
    c = n0 * np.exp(np.cumsum(rng.normal(0, 0.01, len(days))))
    o = c * np.exp(rng.normal(0, 0.004, len(days)))
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995,
                         "close": c, "volume": rng.integers(1e5, 1e6, len(days)).astype(float)}, index=days)


daily = {k: bars(100) for k in list(config.STOCKS) + list(opening.INDIA) + list(opening.GLOBALS)}
table = opening.build_table(daily, opening.next_weekday(days[-1]))
live = table.index[table.date == table.date.max()]
assert len(live) == 4

from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model as C2, Chronos2Pipeline  # noqa: E402
qs = [0.01, 0.05] + [round(0.1 * i, 1) for i in range(1, 10)] + [0.95, 0.99]
cfg = Chronos2CoreConfig(d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
                         chronos_config=dict(context_length=512, input_patch_size=16, output_patch_size=16,
                                             input_patch_stride=16, quantiles=qs, use_reg_token=True,
                                             use_arcsinh=True, max_output_patches=64))
ch = opening.OpenChronos2(pipeline=Chronos2Pipeline(model=C2(cfg).eval()))
for stage in ("evening", "morning"):
    o = ch.predict(table, live, stage)
    assert o["ret"].notna().all() and o["prob"].between(0, 1).all() and (o["lo"] <= o["hi"]).all(), o
    print("Chronos-2", stage, o.round(4).to_dict("records")[0])

sys.path.insert(0, os.path.abspath(config.KRONOS_REPO_DIR))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
                      ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4,
                      beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0,
             attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
kr = opening.OpenKronos(predictor=KronosPredictor(mdl.eval(), tok.eval(), device="cpu", max_context=512))
o = kr.predict(table, live, daily)
assert o["ret"].notna().all() and np.isfinite(o["ret"]).all(), o
print("Kronos", o["ret"].round(4).tolist())

models = ["naive", "linear", "lightgbm", "ensemble", "chronos2", "kronos"]
bt = opening.walk_forward(table, daily, models, ch, kr)
sc = opening.score(bt, models)
for m in models:
    assert "open" in sc[m] and "open_evening" in sc[m], m
    print(f"{m:9s} n={sc[m]['open']['ALL']['n']:4d} fold_ic keys={list(sc[m]['fold_ic'])}")
print("OPEN PRETRAINED SMOKE OK")
