"""Smoke test for the Chronos-2 and Kronos wrappers using tiny RANDOM-weight versions
of the real model classes (no download needed). Checks shapes, horizons and plumbing,
not forecast quality.

Run: KRONOS_REPO_DIR=/path/to/Kronos python tests/pretrained_smoke_test.py
"""
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predicted import config  # noqa: E402
from predicted.features import build_panel  # noqa: E402
from predicted.marketdata import load_seed_csv  # noqa: E402
from predicted.models import Chronos2Model, KronosModel  # noqa: E402

torch.manual_seed(0)
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
now = pd.Timestamp("2026-10-19 10:51", tz=config.IST)
candles = pd.concat([load_seed_csv(os.path.join(os.environ.get("SEED_DIR", os.path.join(root, "data/seed")), f"{s}_5min.csv"), now) for s in config.STOCKS])
panel = build_panel(candles, now)
idx_live = panel.groupby("symbol").tail(1).index                     # like the predict job
idx_bt = panel.index[panel.date.astype(str) == "2026-10-16"][::7]    # like the backtest

# ---- Chronos-2 with a tiny random model of the real architecture
from chronos.chronos2 import Chronos2CoreConfig, Chronos2Model as C2, Chronos2Pipeline  # noqa: E402
qs = [0.01, 0.05] + [round(0.1 * i, 1) for i in range(1, 10)] + [0.95, 0.99]
cfg = Chronos2CoreConfig(d_model=64, d_kv=16, d_ff=128, num_layers=2, num_heads=4,
                         chronos_config=dict(context_length=512, input_patch_size=16, output_patch_size=16,
                                             input_patch_stride=16, quantiles=qs, use_reg_token=True,
                                             use_arcsinh=True, max_output_patches=64))
pipe = Chronos2Pipeline(model=C2(cfg).eval())
for name, idx in [("live", idx_live), ("backtest", idx_bt)]:
    out = Chronos2Model(pipeline=pipe).predict(panel, idx)
    late = panel.loc[idx, "candles_left"] < 6
    assert out.loc[~late, "ret_30m"].notna().all(), "30m missing"
    assert out.loc[late, "ret_30m"].isna().all(), "30m should be empty after 14:55"
    has_eod = panel.loc[idx, "candles_left"] >= 1
    assert out.loc[has_eod, "ret_eod"].notna().all() and out.loc[has_eod, "prob_eod"].between(0, 1).all()
    assert out.loc[~has_eod, "ret_eod"].isna().all(), "no EOD prediction from the 15:25 candle"
    assert (out.loc[has_eod, "lo_eod"] <= out.loc[has_eod, "hi_eod"]).all()
    print(f"Chronos-2 {name}: {len(idx)} rows OK", out[["ret_30m", "prob_30m", "ret_eod"]].describe().loc[["count", "mean"]].round(5).to_dict())

# ---- Kronos with a tiny random model of the real architecture
sys.path.insert(0, os.path.abspath(config.KRONOS_REPO_DIR))
from model import Kronos, KronosPredictor, KronosTokenizer  # noqa: E402
tok = KronosTokenizer(d_in=6, d_model=32, n_heads=2, ff_dim=64, n_enc_layers=1, n_dec_layers=1,
                      ffn_dropout_p=0.0, attn_dropout_p=0.0, resid_dropout_p=0.0, s1_bits=4, s2_bits=4,
                      beta=0.05, gamma0=1.0, gamma=1.1, zeta=0.05, group_size=4)
mdl = Kronos(s1_bits=4, s2_bits=4, n_layers=1, d_model=32, n_heads=2, ff_dim=64, ffn_dropout_p=0.0,
             attn_dropout_p=0.0, resid_dropout_p=0.0, token_dropout_p=0.0, learn_te=True)
pred = KronosPredictor(mdl.eval(), tok.eval(), device="cpu", max_context=512)
out = KronosModel(predictor=pred).predict(panel, idx_live)
assert out["ret_30m"].notna().all() and out["ret_eod"].notna().all()
print("Kronos live:", out[["ret_30m", "ret_eod"]].round(5).to_dict())
out = KronosModel(predictor=pred).predict(panel, idx_bt, horizons=("30m",))
assert out["ret_eod"].isna().all()
print("Kronos backtest rows with 30m:", int(out["ret_30m"].notna().sum()), "of", len(idx_bt))
print("PRETRAINED SMOKE OK")
