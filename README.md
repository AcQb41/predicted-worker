# predicted-worker

The model worker for **Predicted**, an intraday prediction dashboard for four NSE stocks
(PC Jeweller, Jio Financial Services, Suzlon Energy, JSW Energy). The dashboard and its
database live in Lovable. This repository holds the models and the schedules that run them on
GitHub Actions.

| Schedule | Workflow | What it does |
| --- | --- | --- |
| Every 5 min, 09:22 to 15:27 IST, Mon to Fri | `predict.yml` | Tops up today's candles from Yahoo, predicts 30 min ahead and to end of day with every model, and scores old predictions |
| 16:10 IST, Mon to Fri | `nightly.yml` | Fills data gaps, runs the walk-forward backtest, retrains, publishes new model versions |
| Run once by hand | `setup.yml` | Checks the connection, pulls 59 days of index history, trains the first models |
| 1st of each month | `keepalive.yml` | One tiny commit, so GitHub keeps the schedules switched on |

## Models

| Key | Model | Trained on our data? |
| --- | --- | --- |
| `naive` | "Price won't change" baseline | Only its up/down base rate |
| `lightgbm` | Gradient-boosted trees (Huber regressor + classifier) | Yes, nightly |
| `lstm` | PyTorch LSTM, 24-candle look-back, multi-task, 3 seeds | Yes, nightly |
| `ensemble` | Average of LightGBM and LSTM | Via its members |
| `chronos2` | Amazon Chronos-2 (`amazon/chronos-2`) | No, zero-shot |
| `kronos` | Kronos-small (`NeoQuasar/Kronos-small`, github.com/shiyu-coder/Kronos, pinned commit) | No, zero-shot |

## Setup

1. Create the repository (public is recommended: Actions minutes are free and unlimited for public repositories; a private repository's 2,000 free minutes a month run out in about a week at 5-minute runs).
2. Upload everything in this folder, keeping the `.github/workflows` folder.
3. **Settings → Secrets and variables → Actions → New repository secret**:
   - `PREDICTED_API_URL`: the URL of the `worker-api` function from Lovable
   - `PREDICTED_WORKER_TOKEN`: the same long random value you saved in Lovable as `WORKER_TOKEN`
   - `PREDICTED_ANON_KEY` (optional): your Lovable project's publishable key
4. Optional **Variables** (same page, Variables tab): `ENABLE_CHRONOS`, `ENABLE_KRONOS`, `ENABLE_LSTM` set to `0` to switch a model off.
5. **Actions** tab → enable workflows → **First-time setup** → Run workflow.
6. From the next market day, `predict.yml` and `nightly.yml` run on their own.

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
export PREDICTED_API_URL=... PREDICTED_WORKER_TOKEN=...
python -m predicted status
python -m predicted train
python -m predicted predict
```

Dry run without Lovable: set `PREDICTED_LOCAL_STORE=./localdb` to use a local file store.

## Tests

| Test | Command |
| --- | --- |
| Replay a trading day and check scoring | `SEED_DIR=... PREDICTED_LOCAL_STORE=/tmp/db PREDICTED_MODEL_DIR=/tmp/m ENABLE_CHRONOS=0 ENABLE_KRONOS=0 python tests/replay_test.py 2026-10-16` (train first with the same settings) |
| Chronos-2 / Kronos wrappers (random weights) | `SEED_DIR=... KRONOS_REPO_DIR=./Kronos python tests/pretrained_smoke_test.py` |
| Database migration (Postgres via PGlite) | `npm i @electric-sql/pglite && MIGRATION=lovable/migration.sql node tests/sql_test.mjs` |
| Edge functions | `deno test -A tests/functions_test.ts` |
| Worker ↔ worker-api contract | `SEED_DIR=... bash tests/contract_test.sh` |

## The `lovable/` folder

The exact database migration, edge functions, schedules and function settings that the
Lovable build prompts ask Lovable to create. Attach these files to the Lovable prompts.

Research tool, not investment advice.
