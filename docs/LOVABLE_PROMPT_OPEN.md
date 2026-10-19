Add a "Tomorrow's open" forecast to Predicted. Attached: migration_open.sql (database change) and index.ts (the worker-api code, for reference). Follow the project knowledge.

1. DATABASE
Run the attached migration_open.sql exactly as written, as ONE migration, after the existing schema. Don't change it. It adds horizon 'open', model 'linear', job types 'train_open' and 'predict_open', scoring against the 09:15 candle's opening price, the view open_scoreboard, and a new latest_model_versions that merges the intraday and opening-price backtests.

2. WORKER-API ENDPOINT (/api/public/worker-api)
Make exactly two changes and nothing else:
- In the model key allow-list, add "linear". It is used both when validating predictions and in create_model_version.
- In the horizon check, allow "open" as well as "30m" and "eod".
Keep the token check, the field lists and all other logic the same. The attached index.ts shows the change.

3. OVERVIEW: a third forecast card, "Tomorrow's open"
- Place it next to "Next 30 minutes" and "End of day", for the selected stock. Data: latest_predictions where horizon = 'open'.
- The header has two lines:
  - The stage, from made_at in IST: hour 12 or later means "Evening estimate (after the close)"; otherwise "Pre-open update (before 09:15)".
  - "For <weekday dd Mon> 09:15", from target_ts.
- Show one row per model, using the same layout as the other cards: predicted opening price, change vs last close (base_price) as a signed %, ▲ Up / ▼ Down, probability, 80% range, and Edge / No edge.
- Edge uses the matching stage, with the same thresholds as the other cards:
  - Pre-open: latest_model_versions.backtest.open, plus open_scoreboard where stage = 'pre_open'.
  - Evening: backtest.open_evening, plus stage = 'evening'.
- Models on this card: naive (as Baseline, last), linear, lightgbm, ensemble, chronos2 and kronos. LSTM has no opening forecast; don't show it here.
- Card states:
  - Before the open (target_ts still in the future): show the forecast. Add the muted note "Evening estimates use only India's close; the pre-open update adds the overnight US session and today's Asian opens."
  - After the open (target_ts passed): switch to the result. Show "Opened at ₹X" (actual_price once resolved) and ✓/✗ per model. Add the note "Next forecast after today's close, about 15:50 IST."
  - No 'open' predictions yet: "First forecast after the next close, about 15:50 IST."
- Stock strip cards: add a third chip, "Open: n▼ n▲". It counts the latest 'open' predictions by direction for that stock, without naive.
- The stale banner ignores horizon 'open'.

4. MODELS PAGE
- Add "Tomorrow's open" to the horizon toggle, next to "30 min" and "End of day".
- When "Tomorrow's open" is selected, show a stage switch: "Pre-open update" (default) | "Evening estimate".
  - Live table: open_scoreboard filtered by stage and the stock selector (symbol 'ALL' for all stocks).
  - Backtest table: latest_model_versions.backtest.open for pre-open, or .open_evening for evening, with the same columns as today. The trading columns are empty for pre-open; show "n/a" with the tooltip "A forecast made after the close can't be used to trade the opening gap."
  - IC-per-block chart: backtest.fold_ic.open or .open_evening.
  - Accuracy-by-day chart: model_daily_scores where horizon = 'open'.
- Model cards: add a "Linear" card (colour #65A30D): "Ridge + logistic regression on daily features; tomorrow's open only". On every card, add a "Tomorrow's open" section from params.open: version, training range, rows and top features if present.

5. HISTORY PAGE
- Add "Tomorrow's open" to the horizon filter.
- For 'open' rows, add a Stage column (Evening / Pre-open) and show the target as a date ("Mon 20 Oct open").

6. DATA PAGE
- Add a job card, "Last opening-price run": the newest worker_runs row where job is 'train_open' or 'predict_open', showing its status, time and message.

Don't change any other page, logic or style. Show models that have data; never invent rows.
