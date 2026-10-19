PREDICTED: project knowledge (follow on every request)

WHAT IT IS
Predicted is a personal research dashboard for four NSE stocks: PCJEWELLER (PC Jeweller), JIOFIN (Jio Financial Services), SUZLON (Suzlon Energy), JSWENERGY (JSW Energy). It shows what six models predict for the price 30 minutes ahead and at the day's close (15:25 candle close), and how often each model has been right. Nifty 50 (NIFTY50), Nifty Bank (BANKNIFTY) and India VIX (INDIAVIX) are stored as market context only.

WHO DOES WHAT
- The app NEVER runs models. A Python worker on GitHub Actions trains and predicts, and writes through the edge function worker-api. The app only reads.
- Edge functions: ingest-live (Yahoo candles every 5 min via pg_cron + Refresh button), worker-api (worker only, x-worker-token = secret WORKER_TOKEN), import-candles (CSV import from the Data page, x-admin-passcode = secret ADMIN_PASSCODE). All three have verify_jwt = false and do their own auth. Use the attached code exactly; do not rewrite their logic.
- Browser code never inserts, updates or deletes table rows. RLS: public read on all tables, no write policies.

DATA (Lovable Cloud Postgres; schema = attached migration.sql, do not change it without being asked)
- candles(symbol, ts, open, high, low, close, volume, source, updated_at). ts = candle START in UTC. Session candles 09:15 to 15:25 IST.
- predictions(id, symbol, model_key, horizon '30m'|'eod', made_at, target_ts, base_price, pred_return, pred_price, pred_direction 'up'|'down', prob_up, band_low, band_high, model_version_id, created_at, status 'open'|'resolved'|'expired', actual_price, actual_return, abs_error, direction_correct, resolved_at).
- model_versions(id, model_key, version, trained_at, train_start, train_end, n_train, params jsonb, backtest jsonb). backtest = {"30m": {"ALL": metrics, "<SYMBOL>": metrics}, "eod": {...}, "resid_q": {...}, "fold_ic": {"30m": {"1": ic, ...}, "eod": {...}}}. metrics keys: n, dir_acc, balanced_acc, p_vs_naive, auc, mape_pct, skill_vs_naive_pct, ic, bt_trades, bt_win_rate, bt_return_pct (fractions 0-1 for acc/auc; percentages for *_pct).
- worker_runs(id, job, status 'running'|'ok'|'error'|'skipped', started_at, finished_at, message, details jsonb).
- Views: latest_predictions, model_scoreboard(symbol incl. 'ALL', model_key, horizon, n, dir_acc, balanced_acc, mae_inr, mape_pct, skill_vs_naive_pct, ic, last_made_at), model_daily_scores(trade_date, model_key, horizon, n, dir_acc, mape_pct), data_health(symbol, name, kind, sort_order, candles, first_ts, last_ts, trading_days, candles_today), latest_model_versions.

MODELS (key, label, colour; same colour everywhere)
naive "Naive (baseline)" #94A3B8 dashed; lightgbm "LightGBM" #2563EB; lstm "LSTM" #7C3AED; ensemble "Ensemble" #0891B2; chronos2 "Chronos-2" #EA580C; kronos "Kronos" #DB2777; linear "Linear" #65A30D (tomorrow's open only). Show models that have data; never invent rows.

DISPLAY RULES
- All times in IST (Asia/Kolkata), 24-hour "10:45", dates "Mon 19 Oct".
- Prices "₹216.40". Moves from log returns: pct = (Math.exp(r) - 1) * 100, shown with sign and 2 decimals. Probabilities as whole %.
- Up/down never by colour alone: "▲ Up" green #059669, "▼ Down" red #DC2626.
- Market open = Mon-Fri 09:15-15:30 IST. Refetch every 60 s while open; no polling when closed.
- 30m forecasts only exist up to the 14:55 candle; EOD up to the 15:20 candle. Say so instead of showing empty cards.
- "Edge" badge for a model+horizon only if its latest backtest (stock or ALL) has skill_vs_naive_pct > 0 AND balanced_acc > 0.52 AND p_vs_naive < 0.05, and, when it has 200+ live predictions in model_scoreboard, live skill_vs_naive_pct > 0. Else "No edge". If no model has Edge: amber "No reliable signal: models do not beat 'no change' yet".
- Naive is always shown as the reference ("Baseline"), muted, pinned last in forecast tables and first in leaderboards.
- Stale banner when market open and newest prediction created_at older than 15 min.
- Footer on every page: "Research tool, not investment advice. Data: Yahoo Finance, may be delayed."

TOMORROW'S OPEN (horizon 'open'; schema add-on = attached migration_open.sql)
- Forecast of the next session's opening price. target_ts = next session 09:15 IST; scored on the OPEN of the first 09:15 candle at or after it (holidays roll to the next session). base_price = last close.
- Two stages, told apart by made_at in IST: 15:30 = "Evening estimate" (after the close, only India data); 08:45 = "Pre-open update" (adds overnight US session + Asian opens). Hour >= 12 means evening.
- Models: naive, linear, lightgbm, ensemble (= linear + lightgbm), chronos2, kronos. LSTM has no opening forecast.
- Backtest JSON keys: backtest.open (pre-open) and backtest.open_evening (evening); fold_ic.open / fold_ic.open_evening. Pre-open rows have no bt_* trading numbers (the gap can't be traded with a forecast made after the close).
- Live scores: view open_scoreboard(symbol incl. 'ALL', model_key, stage 'evening'|'pre_open', n, dir_acc, balanced_acc, mae_inr, mape_pct, skill_vs_naive_pct, ic, last_made_at).
- Edge for an open forecast uses the matching stage: backtest.open + open_scoreboard pre_open, or backtest.open_evening + evening. Same thresholds.
- Open forecasts change only at ~15:50 and ~08:35/08:55 IST: never part of the 15-minute stale check.
- worker_runs jobs 'train_open' and 'predict_open' belong to the opening-price workflow.

LOOK (light theme)
Background #F8FAFC; cards #FFFFFF, border #E2E8F0, radius 12px, subtle shadow; text #0F172A, muted #64748B; accent #4F46E5; warning #D97706. Font Inter, 14px base, tabular numbers. Clean, calm, data-first; no gradients, no dark mode, no stock photos. shadcn/ui components, Recharts for charts, TanStack Query for data, Papa Parse for CSV.

PAGES
/ Overview (stock strip, price chart with predicted-vs-actual overlay, "Next 30 minutes", "End of day" and "Tomorrow's open" forecast cards, today's scorecard) · /models (live vs backtest leaderboard, accuracy-by-day chart, IC-per-block chart, model cards) · /history (filterable prediction ledger, CSV export) · /data (data health, job cards and log, CSV import with passcode, Refresh). Full spec: attached PREDICTED_SPEC.md, section 6.
