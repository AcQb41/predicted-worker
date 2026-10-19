# Predicted — Business & Technical Requirements

Oct 19, 2026 · @Akshay

## How to use this document

Predicted is built in two parts. Lovable builds the dashboard and its database. A small GitHub repository, `predicted-worker`, runs the prediction models on a schedule and writes the results into that database.

| Part | Where it goes | What you do with it |
| --- | --- | --- |
| Knowledge text (section 7, step 0; `docs/LOVABLE_KNOWLEDGE.md`) | Lovable → Knowledge | Paste once; Lovable follows it on every prompt (Knowledge holds about 10,000 characters, so it carries the rules, not the whole spec) |
| This document, exported as Markdown (`docs/PREDICTED_SPEC.md` in the zip) | Lovable chat, attached to Prompt 1 | The full spec Lovable builds from |
| Code files in `lovable/` (migration, functions, schedules) | Lovable chat, attached to Prompts 1 and 2 | Exact code Lovable must use, so nothing is reinvented |
| Section 7 prompts | Lovable chat | Send one at a time, in order; check each before the next |
| `predicted-worker.zip` | GitHub, as a new repository | The Python models and the schedules that run them |
| Section 8 | You | Manual GitHub steps: account, repository, secrets, first run |

**Build order**

1. Create the Lovable project **Predicted** and paste the knowledge text (section 7, step 0).
2. Send Prompts 1 and 2 (database, functions, secrets, schedule). Note the `worker-api` URL Lovable gives you.
3. Send Prompt 3 (Data page), then import your four CSV files on that page.
4. Set up GitHub (section 8) and run **First-time setup** once.
5. Send Prompts 4 to 7 (Overview, Models, History, polish).
6. On the next market day, go through the acceptance checklist in section 9.

## 1. Business requirements

Predicted shows, for four NSE stocks, what six models expect the price to be **30 minutes from now** and **at today's close**, and how often each model has been right. It is a personal research tool built on 5-minute candles: 60 days of history plus live data collected from now on.

**Goal.** Make it easy to compare prediction models honestly, on the same stocks and the same moments, with live results recorded before the outcome is known.

**Users.** Akshay (owner, admin) and anyone he shares the link with (read-only viewers).

**Stocks.** PC Jeweller (`PCJEWELLER`), Jio Financial Services (`JIOFIN`), Suzlon Energy (`SUZLON`), JSW Energy (`JSWENERGY`). Nifty 50, Nifty Bank and India VIX are collected as market context, not predicted.

**In scope**

| ID | Requirement |
| --- | --- |
| BR-1 | Store 5-minute OHLCV candles: the four historical CSV files, then live candles from Yahoo Finance every 5 minutes during market hours |
| BR-2 | Predict, from the latest finished candle, the price in 30 minutes and at the 15:25 candle's close, with direction (up/down), probability of up, and an 80% range |
| BR-3 | Run six models side by side: Naive baseline, LightGBM, LSTM, Ensemble, Chronos-2, Kronos |
| BR-4 | Show the predictions on refresh, with no waiting: the page only reads stored results |
| BR-5 | Record every prediction before its outcome and score it automatically once the target candle closes (forward-test ledger) |
| BR-6 | Compare models on live results and on a walk-forward backtest, always against the Naive baseline |
| BR-7 | Say plainly when no model beats the baseline ("No reliable signal") |
| BR-8 | Retrain nightly on all data collected so far; keep every model version and its scores |
| BR-9 | Show data health and job history so gaps and failures are visible |
| BR-10 | Light theme, works on laptop and phone |

**Out of scope.** Placing trades, broker connections, user accounts beyond the admin passcode, more stocks (the design allows adding them later), push alerts.

**Success criteria**

- Each market day, all four stocks get predictions from every enabled model at least every 10 minutes between 09:30 and 15:20 IST.
- 95% of predictions are scored within 10 minutes of their target time.
- The dashboard loads in under 2 seconds on a normal connection.
- After 20 market days, the Models page shows which models, if any, beat the Naive baseline on live data.

**Constraints and disclaimers**

- Data comes from Yahoo Finance's unofficial chart API. It can be delayed, change or stop without notice; the worker fills gaps from Yahoo again each night.
- Short-horizon price moves are mostly noise. In the first backtest on these four stocks (59 days, 6 test blocks), no model beat the baseline with statistical confidence. The app must show this honestly.
- Every page footer: "Research tool, not investment advice."

## 2. Architecture

The dashboard never runs a model. Lovable Cloud stores everything and collects live candles; a Python worker on GitHub Actions does all the modelling and writes results back through one secured function. Lovable Cloud does not give out database keys or allow outside database connections ([Supabase FAQ](https://supabase.com/docs/guides/troubleshooting/cant-access-supabase-project-lovable-cloud.md)), so that function, `worker-api`, is the worker's only way in.

&#91;embedded content: Predicted architecture · 3 functions, 2 schedules, 1 database\]

Live candles reach the database two ways (`ingest-live` and the worker's top-up); predictions and model versions arrive only over HTTPS, with a token, through `worker-api`.

| Component | Runs on | Job | Schedule |
| --- | --- | --- | --- |
| React dashboard | Lovable hosting | Reads tables and views; never writes except through functions | On load, and every 60 s while the market is open |
| Postgres database | Lovable Cloud | Candles, predictions, model versions, job log, scoring views | Always |
| `ingest-live` function | Lovable Cloud | Pulls finished 5-minute candles from Yahoo for 7 symbols, upserts them, scores due predictions | pg\_cron, 1 minute after each candle closes, Mon to Fri; also the Refresh button |
| `worker-api` function | Lovable Cloud | The worker's door: read candles, write predictions, model versions and job logs; token-protected | Called by the worker |
| `import-candles` function | Lovable Cloud | Inserts candles from CSV files uploaded on the Data page; passcode-protected | On demand |
| `predict.yml` | GitHub Actions | Predicts from the latest candle with all models, both horizons | Every 5 min, 09:22 to 15:27 IST |
| `nightly.yml` | GitHub Actions | Fills gaps, walk-forward backtest, retrains, publishes model versions | 16:10 IST, Mon to Fri |
| Yahoo Finance | External | 5-minute candles (`/v8/finance/chart`, interval 5m) | Polled |
| Hugging Face, GitHub | External | Chronos-2 and Kronos weights; Kronos code at a pinned commit | Downloaded by the worker, cached |

**Why it is split this way**

- Lovable's backend functions are short-lived TypeScript and cannot run PyTorch, LightGBM or the pretrained models. GitHub Actions gives a full Linux machine with Python.
- Reading stored predictions makes refresh instant (BR-4); predictions for a closed candle never change, so computing them once is enough.
- Two independent data paths make the data sturdier: Lovable collects candles every 5 minutes, and the worker tops up gaps from Yahoo each run and each night. Inserts from the worker never overwrite live candles.

**Timing on a market day (IST)**

1. 09:15 to 09:20: the first candle forms.
2. 09:21: `ingest-live` stores it.
3. 09:22: `predict.yml` starts (GitHub may start it a few minutes late), predicts from the 09:15 candle and writes 24 to 48 predictions (4 stocks × up to 6 models × 2 horizons).
4. Every 5 minutes after that, the same cycle. At each ingest, predictions whose target candle has closed are scored.
5. 30-minute predictions stop after the 14:55 candle; end-of-day predictions stop after the 15:20 candle.
6. 16:10: the nightly run retrains on everything collected, including today.

## 3. Data model

Five tables and five views, created by one migration (`lovable/migration.sql` in the zip, which is the source of truth; the copy below matches it). Everyone can read; nothing is written from the browser. All writes go through the three functions, which use the service role. Tested on Postgres 17 (PGlite): resolution, a missing target candle, expiry, views and read-only access for anonymous users.

| Table / view | One row per | Written by | Used on |
| --- | --- | --- | --- |
| `symbols` | Stock or index (7 rows, seeded) | Migration | Everywhere |
| `candles` | Symbol × 5-minute candle; `ts` = candle start (UTC) | `ingest-live` (overwrite), `import-candles` and worker (insert-only) | Overview chart, Data |
| `predictions` | Symbol × model × horizon × `made_at`; outcome filled in later | Worker; `resolve_predictions()` | Overview, History, Models |
| `model_versions` | Model × nightly training run, with backtest scores (JSON) | Worker | Models |
| `worker_runs` | Job run (ingest, import, seed, backfill, train, predict) | Functions and worker | Data, header status |
| `latest_predictions` (view) | Newest prediction per stock × model × horizon |  | Overview |
| `model_scoreboard` (view) | Live scores, last 30 days, per stock and `ALL` |  | Models, Overview |
| `model_daily_scores` (view) | Live accuracy per day |  | Models chart |
| `data_health` (view) | Symbol: candle count, first and last candle, candles today |  | Data |
| `latest_model_versions` (view) | Newest version per model |  | Models, Overview |

**Key rules**

- Prices are rupees; returns are log returns (`ln(new / old)`); the UI shows them as percentages: `(exp(r) − 1) × 100`.
- `made_at` is the start of the last finished candle the prediction used. `target_ts` is the start of the candle whose close is predicted: `made_at + 30 min` for `30m`; the 15:25 IST candle for `eod`.
- `resolve_predictions()` scores a prediction 6 minutes after its target candle starts, using that candle or, if missing, the latest candle up to 10 minutes earlier (still after `made_at`). Predictions with no data after a day become `expired`. A price that did not move gives `direction_correct = null` and is left out of accuracy.
- Skill vs naive = `1 − Σ (model % error) / Σ (naive % error)` on the same stocks and moments. Positive means better than "no change".

### Migration SQL

```sql
-- Predicted: database schema (Lovable Cloud / Postgres)
-- Run as ONE migration. Safe to read top to bottom: tables, security, functions, views.

-- ============================================================ symbols
create table if not exists public.symbols (
  symbol        text primary key,                 -- internal code, e.g. 'JIOFIN'
  yahoo_ticker  text not null unique,             -- e.g. 'JIOFIN.NS', '^NSEI'
  name          text not null,
  kind          text not null check (kind in ('stock', 'index')),
  active        boolean not null default true,
  sort_order    integer not null default 0
);

insert into public.symbols (symbol, yahoo_ticker, name, kind, sort_order) values
  ('PCJEWELLER', 'PCJEWELLER.NS', 'PC Jeweller',            'stock', 1),
  ('JIOFIN',     'JIOFIN.NS',     'Jio Financial Services', 'stock', 2),
  ('SUZLON',     'SUZLON.NS',     'Suzlon Energy',          'stock', 3),
  ('JSWENERGY',  'JSWENERGY.NS',  'JSW Energy',             'stock', 4),
  ('NIFTY50',    '^NSEI',         'Nifty 50',               'index', 5),
  ('BANKNIFTY',  '^NSEBANK',      'Nifty Bank',             'index', 6),
  ('INDIAVIX',   '^INDIAVIX',     'India VIX',              'index', 7)
on conflict (symbol) do nothing;

-- ============================================================ candles (5-minute OHLCV)
-- ts = candle START time (UTC). The 09:15 IST candle covers 09:15:00-09:19:59.
create table if not exists public.candles (
  symbol      text not null references public.symbols(symbol),
  ts          timestamptz not null,
  open        double precision not null check (open > 0),
  high        double precision not null check (high > 0),
  low         double precision not null check (low > 0),
  close       double precision not null check (close > 0),
  volume      bigint not null default 0 check (volume >= 0),
  source      text not null check (source in ('csv', 'live', 'backfill')),
  updated_at  timestamptz not null default now(),
  primary key (symbol, ts),
  check (high >= low)
);
create index if not exists candles_ts_idx on public.candles (ts desc);

-- ============================================================ model versions
create table if not exists public.model_versions (
  id           bigint generated always as identity primary key,
  model_key    text not null check (model_key in ('naive','lightgbm','lstm','ensemble','chronos2','kronos')),
  version      text not null,                     -- e.g. '2026-10-19 16:15 IST'
  trained_at   timestamptz not null default now(),
  train_start  date,
  train_end    date,
  n_train      integer,
  params       jsonb not null default '{}'::jsonb,
  backtest     jsonb not null default '{}'::jsonb, -- {"30m":{"ALL":{...},"JIOFIN":{...}},"eod":{...},"resid_q":{...},"fold_ic":{...}}
  unique (model_key, version)
);
create index if not exists model_versions_key_idx on public.model_versions (model_key, trained_at desc);

-- ============================================================ predictions (forward-test ledger)
-- Written BEFORE the outcome is known; resolved later by resolve_predictions().
create table if not exists public.predictions (
  id                bigint generated always as identity primary key,
  symbol            text not null references public.symbols(symbol),
  model_key         text not null check (model_key in ('naive','lightgbm','lstm','ensemble','chronos2','kronos')),
  horizon           text not null check (horizon in ('30m', 'eod')),
  made_at           timestamptz not null,          -- start of the last completed candle used
  target_ts         timestamptz not null,          -- start of the candle whose close is predicted
  base_price        double precision not null,     -- close of the made_at candle
  pred_return       double precision not null,     -- log return
  pred_price        double precision not null,
  pred_direction    text not null check (pred_direction in ('up', 'down')),
  prob_up           double precision check (prob_up between 0 and 1),
  band_low          double precision,              -- 80% range
  band_high         double precision,
  model_version_id  bigint references public.model_versions(id),
  created_at        timestamptz not null default now(),
  status            text not null default 'open' check (status in ('open', 'resolved', 'expired')),
  actual_price      double precision,
  actual_return     double precision,
  abs_error         double precision,
  direction_correct boolean,                       -- null when the price did not move
  resolved_at       timestamptz,
  unique (symbol, model_key, horizon, made_at)
);
create index if not exists predictions_open_idx on public.predictions (target_ts) where status = 'open';
create index if not exists predictions_lookup_idx on public.predictions (symbol, horizon, made_at desc);
create index if not exists predictions_resolved_idx on public.predictions (made_at desc) where status = 'resolved';

-- ============================================================ job log
create table if not exists public.worker_runs (
  id           bigint generated always as identity primary key,
  job          text not null check (job in ('ingest', 'seed', 'import', 'backfill', 'train', 'predict')),
  status       text not null check (status in ('running', 'ok', 'error', 'skipped')),
  started_at   timestamptz not null default now(),
  finished_at  timestamptz,
  message      text,
  details      jsonb not null default '{}'::jsonb
);
create index if not exists worker_runs_job_idx on public.worker_runs (job, started_at desc);

-- ============================================================ security
-- Anyone with the app link can READ. Nobody writes from the browser:
-- all writes go through edge functions using the service role.
alter table public.symbols        enable row level security;
alter table public.candles        enable row level security;
alter table public.model_versions enable row level security;
alter table public.predictions    enable row level security;
alter table public.worker_runs    enable row level security;

create policy "public read symbols"        on public.symbols        for select to anon, authenticated using (true);
create policy "public read candles"        on public.candles        for select to anon, authenticated using (true);
create policy "public read model_versions" on public.model_versions for select to anon, authenticated using (true);
create policy "public read predictions"    on public.predictions    for select to anon, authenticated using (true);
create policy "public read worker_runs"    on public.worker_runs    for select to anon, authenticated using (true);

-- ============================================================ resolve predictions
-- Fills in the actual price once the target candle has closed.
-- Uses the latest candle in (target_ts - 10 min, target_ts], after made_at, so one
-- missing candle doesn't block scoring. Expires predictions with no data after 1 day.
create or replace function public.resolve_predictions()
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  n integer;
begin
  with hit as (
    select p.id, c.close
    from public.predictions p
    join lateral (
      select c.close
      from public.candles c
      where c.symbol = p.symbol
        and c.ts <= p.target_ts
        and c.ts > p.made_at
        and c.ts >= p.target_ts - interval '10 minutes'
      order by c.ts desc
      limit 1
    ) c on true
    where p.status = 'open'
      and now() >= p.target_ts + interval '6 minutes'
  )
  update public.predictions p
     set actual_price      = hit.close,
         actual_return     = ln(hit.close / p.base_price),
         abs_error         = abs(p.pred_price - hit.close),
         direction_correct = case when hit.close = p.base_price then null
                                  else (p.pred_direction = 'up') = (hit.close > p.base_price) end,
         status            = 'resolved',
         resolved_at       = now()
    from hit
   where p.id = hit.id;
  get diagnostics n = row_count;

  update public.predictions
     set status = 'expired', resolved_at = now()
   where status = 'open' and now() > target_ts + interval '1 day';

  return n;
end;
$$;
revoke all on function public.resolve_predictions() from public, anon, authenticated;
grant execute on function public.resolve_predictions() to service_role;

-- ============================================================ views for the dashboard
-- Newest prediction per stock x model x horizon.
create or replace view public.latest_predictions
with (security_invoker = on) as
select distinct on (p.symbol, p.model_key, p.horizon) p.*
from public.predictions p
order by p.symbol, p.model_key, p.horizon, p.made_at desc;

-- Live forward-test scores over the last 30 days; symbol = 'ALL' row per model.
-- Skill uses percentage errors so cheap and expensive stocks count equally.
create or replace view public.model_scoreboard
with (security_invoker = on) as
with r as (
  select p.symbol, p.model_key, p.horizon, p.made_at, p.pred_return, p.actual_return,
         p.direction_correct, p.abs_error, p.actual_price,
         p.abs_error / p.actual_price                 as rel_err,
         n.abs_error / n.actual_price                 as naive_rel_err
  from public.predictions p
  left join public.predictions n
    on n.symbol = p.symbol and n.horizon = p.horizon and n.made_at = p.made_at
   and n.model_key = 'naive' and n.status = 'resolved'
  where p.status = 'resolved'
    and p.made_at >= now() - interval '30 days'
)
select coalesce(symbol, 'ALL')                                         as symbol,
       model_key,
       horizon,
       count(*)::integer                                               as n,
       avg(direction_correct::int)::double precision                   as dir_acc,
       ((avg(direction_correct::int) filter (where actual_return > 0)
        + avg(direction_correct::int) filter (where actual_return < 0)) / 2)::double precision as balanced_acc,
       avg(abs_error)                                                  as mae_inr,
       avg(rel_err) * 100                                              as mape_pct,
       case when sum(naive_rel_err) > 0
            then (1 - sum(rel_err) filter (where naive_rel_err is not null) / sum(naive_rel_err)) * 100
       end                                                             as skill_vs_naive_pct,
       corr(pred_return, actual_return)                                as ic,
       max(made_at)                                                    as last_made_at
from r
group by grouping sets ((symbol, model_key, horizon), (model_key, horizon));

-- Daily live scores, for the accuracy-over-time chart.
create or replace view public.model_daily_scores
with (security_invoker = on) as
select (p.made_at at time zone 'Asia/Kolkata')::date as trade_date,
       p.model_key,
       p.horizon,
       count(*)::integer                              as n,
       avg(p.direction_correct::int)::double precision as dir_acc,
       avg(p.abs_error / p.actual_price) * 100        as mape_pct
from public.predictions p
where p.status = 'resolved'
group by 1, 2, 3;

-- Data health per symbol, for the Data page.
create or replace view public.data_health
with (security_invoker = on) as
select s.symbol, s.name, s.kind, s.sort_order,
       count(c.ts)::integer                                                 as candles,
       min(c.ts)                                                            as first_ts,
       max(c.ts)                                                            as last_ts,
       count(distinct (c.ts at time zone 'Asia/Kolkata')::date)::integer    as trading_days,
       count(*) filter (where (c.ts at time zone 'Asia/Kolkata')::date
                              = (now() at time zone 'Asia/Kolkata')::date)::integer as candles_today
from public.symbols s
left join public.candles c on c.symbol = s.symbol
group by s.symbol, s.name, s.kind, s.sort_order;

-- Latest model version per model (with its backtest JSON).
create or replace view public.latest_model_versions
with (security_invoker = on) as
select distinct on (model_key) *
from public.model_versions
order by model_key, trained_at desc;
```

### Scheduled jobs SQL

Run after the functions exist. Replace `<INGEST_LIVE_URL>` with the `ingest-live` URL (Cloud → Edge functions → Copy URL).

```sql
select cron.schedule(
  'predicted-ingest-live',
  '1-56/5 3-10 * * 1-5',   -- UTC; 1 minute after each candle closes; function skips outside 09:15-15:45 IST
  $$
  select net.http_post(
    url     := '<INGEST_LIVE_URL>',
    headers := '{"Content-Type": "application/json"}'::jsonb,
    body    := '{}'::jsonb
  );
  $$
);

select cron.schedule(
  'predicted-cleanup-runs',
  '30 20 * * *',           -- 02:00 IST: keep 30 days of job log
  $$ delete from public.worker_runs where started_at < now() - interval '30 days'; $$
);
```

## 4. Backend functions

Three Deno edge functions, each self-contained (`lovable/functions/<name>/index.ts` in the zip). All three were type-checked with Deno 2, their parsing and validation logic is unit-tested, and `worker-api` was run as a real server against the Python worker for seed → train → predict.

| Function | Called by | Auth | Input | Output |
| --- | --- | --- | --- | --- |
| `ingest-live` | pg\_cron every 5 min; Refresh button | None (writes only market data; throttled to once a minute; skips outside 09:15 to 15:45 IST unless `{"force": true}`) | `{}` or `{"force": true}` | `{status, total, errors, details}` |
| `worker-api` | GitHub worker | Header `x-worker-token` = secret `WORKER_TOKEN` | `{"action": ..., ...}` (actions below) | JSON per action |
| `import-candles` | Data page CSV import | Header `x-admin-passcode` = secret `ADMIN_PASSCODE` | `{rows, symbol, file_name, last_batch}`, at most 2000 rows | `{received, inserted, skipped, rejected}` |

**`worker-api` actions:** `get_candles` (symbols, since; pages past the 1000-row API limit), `upsert_candles` (rows, mode `fill` = insert missing only), `upsert_predictions`, `create_model_version`, `log_run` (insert, or update by id), `resolve`, `status`.

**Secrets to add in Lovable Cloud** (Cloud → Secrets)

| Secret | Value | Also stored in |
| --- | --- | --- |
| `WORKER_TOKEN` | A long random string, e.g. 48 characters from a password manager | GitHub secret `PREDICTED_WORKER_TOKEN` |
| `ADMIN_PASSCODE` | A passcode you will type on the Data page to import CSVs | Nowhere else |

`SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` are provided to functions automatically.

**Function settings** (`supabase/config.toml`): the three functions authenticate themselves, so the platform's login check is off for them.

```toml
[functions.ingest-live]
verify_jwt = false

[functions.worker-api]
verify_jwt = false

[functions.import-candles]
verify_jwt = false
```

### ingest-live/index.ts

```typescript
// ingest-live: pulls the latest finished 5-minute candles from Yahoo Finance for every
// active symbol, upserts them, then resolves predictions whose target candle has closed.
// Called by pg_cron every 5 minutes on weekdays, and by the dashboard's "Refresh data" button.
// verify_jwt = false (see supabase/config.toml); it only ever writes market data.
import { createClient } from "npm:@supabase/supabase-js@2.45.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};

const IST_OFFSET_MIN = 330;               // UTC+05:30
const FIRST_CANDLE_MIN = 9 * 60 + 15;     // 09:15 IST
const LAST_CANDLE_MIN = 15 * 60 + 25;     // 15:25 IST
const THROTTLE_SECONDS = 60;              // ignore repeat calls within a minute

export type Candle = {
  symbol: string; ts: string; open: number; high: number; low: number; close: number;
  volume: number; source: "live";
};

/** Minutes since midnight IST and weekday (0 = Sunday) for a UTC epoch in ms. */
export function istClock(ms: number): { minutes: number; weekday: number } {
  const d = new Date(ms + IST_OFFSET_MIN * 60_000);
  return { minutes: d.getUTCHours() * 60 + d.getUTCMinutes(), weekday: d.getUTCDay() };
}

/** True from 09:15 to 15:45 IST, Monday to Friday (holidays simply return no new candles). */
export function inMarketWindow(ms: number): boolean {
  const { minutes, weekday } = istClock(ms);
  return weekday >= 1 && weekday <= 5 && minutes >= FIRST_CANDLE_MIN && minutes <= 15 * 60 + 45;
}

/** Turns a Yahoo v8 chart response into finished, session-only 5-minute candles. */
export function parseYahooChart(symbol: string, json: any, nowMs: number): Candle[] {
  const result = json?.chart?.result?.[0];
  if (!result) throw new Error(json?.chart?.error?.description ?? "empty Yahoo response");
  const stamps: number[] = result.timestamp ?? [];
  const q = result.indicators?.quote?.[0] ?? {};
  const out: Candle[] = [];
  for (let i = 0; i < stamps.length; i++) {
    const t = stamps[i];
    if (t % 300 !== 0) continue;                      // only aligned 5-minute bars
    if ((t + 300) * 1000 > nowMs) continue;           // candle still forming
    const { minutes, weekday } = istClock(t * 1000);
    if (weekday < 1 || weekday > 5 || minutes < FIRST_CANDLE_MIN || minutes > LAST_CANDLE_MIN) continue;
    const o = q.open?.[i], h = q.high?.[i], l = q.low?.[i], c = q.close?.[i], v = q.volume?.[i];
    if ([o, h, l, c].some((x) => typeof x !== "number" || !Number.isFinite(x) || x <= 0)) continue;
    out.push({
      symbol,
      ts: new Date(t * 1000).toISOString(),
      open: o,
      high: Math.max(h, o, c),
      low: Math.min(l, o, c),
      close: c,
      volume: typeof v === "number" && v > 0 ? Math.round(v) : 0,
      source: "live",
    });
  }
  return out;
}

async function fetchYahoo(ticker: string): Promise<any> {
  const path = `/v8/finance/chart/${encodeURIComponent(ticker)}?interval=5m&range=1d&includePrePost=false`;
  let lastErr = "";
  for (const host of ["https://query1.finance.yahoo.com", "https://query2.finance.yahoo.com"]) {
    try {
      const res = await fetch(host + path, {
        headers: { "User-Agent": "Mozilla/5.0 (compatible; PredictedBot/1.0)", "Accept": "application/json" },
      });
      if (res.ok) return await res.json();
      lastErr = `${res.status} ${await res.text().then((t) => t.slice(0, 120))}`;
    } catch (e) {
      lastErr = String(e);
    }
  }
  throw new Error(`Yahoo ${ticker}: ${lastErr}`);
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });

  const supabase = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
  const body = await req.json().catch(() => ({}));
  const now = Date.now();

  if (!inMarketWindow(now) && !body?.force) {
    return json({ status: "skipped", reason: "outside market hours (09:15-15:45 IST, Mon-Fri)" });
  }

  // Throttle: one real ingest per minute, however many people press Refresh.
  const { data: recent } = await supabase.from("worker_runs").select("started_at")
    .eq("job", "ingest").in("status", ["ok", "running"])
    .gte("started_at", new Date(now - THROTTLE_SECONDS * 1000).toISOString()).limit(1);
  if (recent && recent.length > 0) return json({ status: "skipped", reason: "ran less than a minute ago" });

  const { data: run } = await supabase.from("worker_runs").insert({ job: "ingest", status: "running" })
    .select("id").single();

  const details: Record<string, unknown> = {};
  let total = 0;
  let errors = 0;
  try {
    const { data: symbols, error } = await supabase.from("symbols").select("symbol, yahoo_ticker").eq("active", true);
    if (error) throw error;
    const results = await Promise.allSettled(symbols!.map(async (s) => {
      const candles = parseYahooChart(s.symbol, await fetchYahoo(s.yahoo_ticker), now);
      if (candles.length) {
        const rows = candles.map((c) => ({ ...c, updated_at: new Date().toISOString() }));
        const { error: upErr } = await supabase.from("candles").upsert(rows, { onConflict: "symbol,ts" });
        if (upErr) throw upErr;
      }
      return { symbol: s.symbol, n: candles.length, last: candles.at(-1)?.ts ?? null };
    }));
    results.forEach((r, i) => {
      const sym = symbols![i].symbol;
      if (r.status === "fulfilled") { details[sym] = { n: r.value.n, last: r.value.last }; total += r.value.n; }
      else { details[sym] = { error: String(r.reason).slice(0, 200) }; errors++; }
    });
    const { data: resolved } = await supabase.rpc("resolve_predictions");
    details.resolved = resolved ?? 0;
    const status = errors === symbols!.length ? "error" : "ok";
    await supabase.from("worker_runs").update({
      status, finished_at: new Date().toISOString(), details,
      message: `${total} candles upserted, ${errors} symbol errors, ${resolved ?? 0} predictions resolved`,
    }).eq("id", run!.id);
    return json({ status, total, errors, details });
  } catch (e) {
    await supabase.from("worker_runs").update({
      status: "error", finished_at: new Date().toISOString(), message: String(e).slice(0, 1000), details,
    }).eq("id", run!.id);
    return json({ status: "error", message: String(e) }, 500);
  }
});
```

### worker-api/index.ts

```typescript
// worker-api: the ONLY door the GitHub worker uses to read and write Predicted's database.
// Lovable Cloud does not expose database keys, so the worker calls this function over HTTPS
// with the shared secret WORKER_TOKEN in the `x-worker-token` header.
// verify_jwt = false (see supabase/config.toml); the token check below is the authentication.
import { createClient } from "npm:@supabase/supabase-js@2.45.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-worker-token",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const MODEL_KEYS = ["naive", "lightgbm", "lstm", "ensemble", "chronos2", "kronos"];
const SYMBOL_RE = /^[A-Z0-9]{2,20}$/;

/** Constant-time string comparison. */
export function safeEqual(a: string, b: string): boolean {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

const num = (x: unknown) => typeof x === "number" && Number.isFinite(x);
const isoOk = (x: unknown) => typeof x === "string" && !Number.isNaN(Date.parse(x));

export function validateCandle(r: any): string | null {
  if (!r || !SYMBOL_RE.test(r.symbol)) return "bad symbol";
  if (!isoOk(r.ts)) return "bad ts";
  if (![r.open, r.high, r.low, r.close].every((x) => num(x) && x > 0)) return "bad price";
  if (r.high < r.low) return "high < low";
  if (!(num(r.volume) && r.volume >= 0)) return "bad volume";
  if (!["csv", "live", "backfill"].includes(r.source)) return "bad source";
  return null;
}

export function validatePrediction(r: any): string | null {
  if (!r || !SYMBOL_RE.test(r.symbol)) return "bad symbol";
  if (!MODEL_KEYS.includes(r.model_key)) return "bad model_key";
  if (!["30m", "eod"].includes(r.horizon)) return "bad horizon";
  if (!isoOk(r.made_at) || !isoOk(r.target_ts)) return "bad timestamps";
  if (Date.parse(r.target_ts) <= Date.parse(r.made_at)) return "target before made_at";
  if (![r.base_price, r.pred_return, r.pred_price].every(num)) return "bad numbers";
  if (!["up", "down"].includes(r.pred_direction)) return "bad direction";
  if (r.prob_up != null && !(num(r.prob_up) && r.prob_up >= 0 && r.prob_up <= 1)) return "bad prob_up";
  return null;
}

const PRED_FIELDS = ["symbol", "model_key", "horizon", "made_at", "target_ts", "base_price", "pred_return",
  "pred_price", "pred_direction", "prob_up", "band_low", "band_high", "model_version_id"];

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });

  if (!safeEqual(req.headers.get("x-worker-token") ?? "", Deno.env.get("WORKER_TOKEN") ?? "")) {
    return json({ error: "unauthorized" }, 401);
  }
  const supabase = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
  let body: any;
  try { body = await req.json(); } catch { return json({ error: "invalid JSON" }, 400); }

  try {
    switch (body.action) {
      case "get_candles": {
        // Pages through all rows (the API returns at most 1000 per request).
        const symbols: string[] = (body.symbols ?? []).filter((s: string) => SYMBOL_RE.test(s));
        if (!symbols.length || !isoOk(body.since)) return json({ error: "symbols and since required" }, 400);
        const rows: unknown[] = [];
        for (let from = 0; ; from += 1000) {
          const { data, error } = await supabase.from("candles")
            .select("symbol, ts, open, high, low, close, volume")
            .in("symbol", symbols).gte("ts", body.since)
            .order("symbol").order("ts").range(from, from + 999);
          if (error) throw error;
          rows.push(...data);
          if (data.length < 1000 || rows.length >= 200_000) break;
        }
        return json({ rows });
      }

      case "upsert_candles": {
        // mode "fill": insert missing candles only (CSV seed, backfill)
        // mode "overwrite": replace existing candles (not used by the worker today)
        const rows: any[] = body.rows ?? [];
        if (rows.length > 2000) return json({ error: "max 2000 rows per call" }, 400);
        const bad = rows.map(validateCandle).findIndex((e) => e !== null);
        if (bad >= 0) return json({ error: `row ${bad}: ${validateCandle(rows[bad])}` }, 400);
        const clean = rows.map((r) => ({
          symbol: r.symbol, ts: r.ts, open: r.open, high: r.high, low: r.low, close: r.close,
          volume: Math.round(r.volume), source: r.source, updated_at: new Date().toISOString(),
        }));
        const { error, count } = await supabase.from("candles")
          .upsert(clean, { onConflict: "symbol,ts", ignoreDuplicates: body.mode !== "overwrite", count: "exact" });
        if (error) throw error;
        return json({ count: count ?? clean.length });
      }

      case "upsert_predictions": {
        const rows: any[] = body.rows ?? [];
        if (rows.length > 1000) return json({ error: "max 1000 rows per call" }, 400);
        const bad = rows.map(validatePrediction).findIndex((e) => e !== null);
        if (bad >= 0) return json({ error: `row ${bad}: ${validatePrediction(rows[bad])}` }, 400);
        const clean = rows.map((r) => Object.fromEntries(PRED_FIELDS.map((k) => [k, r[k] ?? null])));
        const { error } = await supabase.from("predictions")
          .upsert(clean, { onConflict: "symbol,model_key,horizon,made_at" });
        if (error) throw error;
        return json({ count: clean.length });
      }

      case "create_model_version": {
        const r = body.row ?? {};
        if (!MODEL_KEYS.includes(r.model_key) || typeof r.version !== "string") return json({ error: "bad row" }, 400);
        const { data, error } = await supabase.from("model_versions").upsert({
          model_key: r.model_key, version: r.version, train_start: r.train_start ?? null,
          train_end: r.train_end ?? null, n_train: r.n_train ?? null, params: r.params ?? {},
          backtest: r.backtest ?? {}, trained_at: new Date().toISOString(),
        }, { onConflict: "model_key,version" }).select("id").single();
        if (error) throw error;
        return json({ id: data.id });
      }

      case "log_run": {
        const r = body.row ?? {};
        if (body.id) {
          const { error } = await supabase.from("worker_runs").update({
            status: r.status, message: r.message ?? null, details: r.details ?? {},
            finished_at: new Date().toISOString(),
          }).eq("id", body.id);
          if (error) throw error;
          return json({ id: body.id });
        }
        const { data, error } = await supabase.from("worker_runs")
          .insert({ job: r.job, status: r.status ?? "running", message: r.message ?? null, details: r.details ?? {} })
          .select("id").single();
        if (error) throw error;
        return json({ id: data.id });
      }

      case "resolve": {
        const { data, error } = await supabase.rpc("resolve_predictions");
        if (error) throw error;
        return json({ resolved: data ?? 0 });
      }

      case "status": {
        const { data, error } = await supabase.from("data_health").select("*").order("sort_order");
        if (error) throw error;
        return json({ data_health: data });
      }

      default:
        return json({ error: `unknown action ${body.action}` }, 400);
    }
  } catch (e: any) {
    console.error("worker-api error", body?.action, e);
    return json({ error: e?.message ?? String(e) }, 500);
  }
});
```

### import-candles/index.ts

```typescript
// import-candles: receives candles parsed from an uploaded CSV on the Data page and inserts
// the ones not already stored (never overwrites). Protected by the ADMIN_PASSCODE secret.
// verify_jwt = false (see supabase/config.toml); the passcode check is the authentication.
import { createClient } from "npm:@supabase/supabase-js@2.45.4";

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "authorization, x-client-info, apikey, content-type, x-admin-passcode",
  "Access-Control-Allow-Methods": "POST, OPTIONS",
};
const SYMBOL_RE = /^[A-Z0-9]{2,20}$/;

export function safeEqual(a: string, b: string): boolean {
  if (!a || !b || a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/** Keeps only finished NSE-session 5-minute candles with sane prices. Returns [rows, rejected]. */
export function cleanRows(rows: any[]): [any[], number] {
  const out: any[] = [];
  let rejected = 0;
  for (const r of rows) {
    const t = Date.parse(r?.ts);
    const p = [r?.open, r?.high, r?.low, r?.close].map(Number);
    const ok = SYMBOL_RE.test(r?.symbol ?? "") && !Number.isNaN(t) && p.every((x) => Number.isFinite(x) && x > 0);
    if (!ok) { rejected++; continue; }
    const ist = new Date(t + 330 * 60_000);
    const minutes = ist.getUTCHours() * 60 + ist.getUTCMinutes();
    const day = ist.getUTCDay();
    if (day < 1 || day > 5 || minutes < 555 || minutes > 925 || minutes % 5 !== 0 || t + 300_000 > Date.now()) {
      rejected++;
      continue;
    }
    const [o, h, l, c] = p;
    out.push({
      symbol: r.symbol, ts: new Date(t).toISOString(), open: o, high: Math.max(o, h, l, c),
      low: Math.min(o, h, l, c), close: c, volume: Math.max(0, Math.round(Number(r.volume) || 0)),
      source: "csv",
    });
  }
  return [out, rejected];
}

Deno.serve(async (req) => {
  if (req.method === "OPTIONS") return new Response(null, { headers: corsHeaders });
  const json = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), { status, headers: { ...corsHeaders, "Content-Type": "application/json" } });

  if (!safeEqual(req.headers.get("x-admin-passcode") ?? "", Deno.env.get("ADMIN_PASSCODE") ?? "")) {
    return json({ error: "wrong passcode" }, 401);
  }
  const supabase = createClient(Deno.env.get("SUPABASE_URL")!, Deno.env.get("SUPABASE_SERVICE_ROLE_KEY")!);
  const body = await req.json().catch(() => null);
  const rows: any[] = body?.rows ?? [];
  if (!Array.isArray(rows) || rows.length === 0 || rows.length > 2000) {
    return json({ error: "send 1 to 2000 rows per call" }, 400);
  }
  const { data: known } = await supabase.from("symbols").select("symbol");
  const allowed = new Set((known ?? []).map((s) => s.symbol));
  const [clean, rejected] = cleanRows(rows);
  const valid = clean.filter((r) => allowed.has(r.symbol));
  const { error, count } = await supabase.from("candles")
    .upsert(valid, { onConflict: "symbol,ts", ignoreDuplicates: true, count: "exact" });
  if (error) return json({ error: error.message }, 500);
  if (body?.last_batch) {
    await supabase.from("worker_runs").insert({
      job: "import", status: "ok", finished_at: new Date().toISOString(),
      message: `CSV import finished for ${body?.symbol ?? "?"}`, details: { file: body?.file_name ?? null },
    });
  }
  return json({ received: rows.length, inserted: count ?? valid.length, skipped: rows.length - valid.length - rejected, rejected });
});
```

## 5. Models

Six models predict the same two things for every stock at every finished candle: the log return to the close 30 minutes later, and to the 15:25 candle's close, each with a probability of up and an 80% range. The code is in the `predicted` Python package in the zip; Lovable never runs it.

| Key | Model | What it learns from | Training | Direction from | 80% range from |
| --- | --- | --- | --- | --- | --- |
| `naive` | Naive baseline: "price won't change" | Nothing (up/down base rate only) | Nightly | Base rate of up moves | Its backtest errors |
| `lightgbm` | LightGBM gradient-boosted trees: Huber-loss regressor + binary classifier per horizon | 37 features, all 4 stocks pooled, stock as a category | Nightly, early stopping on the last 5 days | Classifier | Its backtest errors |
| `lstm` | PyTorch LSTM (48 units), one network for both horizons (return + direction each), 3 seeds averaged | Last 24 candles (2 hours) of the same features + stock one-hot | Nightly, early stopping | Its direction outputs | Its backtest errors |
| `ensemble` | Average of LightGBM and LSTM | Its members | Via members | Average probability | Its backtest errors |
| `chronos2` | [Amazon Chronos-2](https://huggingface.co/amazon/chronos-2), 120M-parameter pretrained forecaster | Last 512 candles of the gap-removed price | None (zero-shot) | Its quantile forecast | Its own 10th and 90th percentiles |
| `kronos` | [Kronos-small](https://github.com/shiyu-coder/Kronos), pretrained on candlesticks from 45+ exchanges (MIT licence, \~40k GitHub stars, pinned commit `67b630e`) | Last 400 OHLCV candles | None (zero-shot) | Sign of its predicted move | Naive's backtest errors |

**Cleaning and outliers**

| Problem in Yahoo data | Handling |
| --- | --- |
| Missing candles (e.g. 15:15, 15:20) | Rebuild the 09:15 to 15:25 grid each day; a missing candle becomes flat at the previous close and is flagged |
| Zero volume on 09:15 and 15:15 candles | Replaced with the neighbouring candle's volume, flagged |
| Overnight gaps (the largest jumps) | Kept as a separate "gap today" feature; removed from the price series used for indicators |
| Spike candles (move over 5 robust standard deviations, median/MAD) | Flagged and clipped for indicators |
| Extreme feature values | Clipped at the 0.5% and 99.5% levels learned from training rows only |
| Outliers in the target | Huber loss for LightGBM and LSTM returns |
| A candle still forming | Never stored, never used |

**Features (37).** Returns over 1, 3, 6, 12 and 24 candles; volatility (12 and 36 candles) and their ratio; RSI(14); MACD histogram; Bollinger z-score; distance from 48-candle EMA; overnight gap; move since today's open; position in today's range; today's range; distance from VWAP; volume z-score; volume so far vs yesterday; candle body and wicks; yesterday's move and range; minute of day, candles left, time-of-day sine and cosine; weekday; spike flag; Nifty 50 30-minute move and move since open; Nifty Bank 30-minute move; India VIX level and 1-hour change; stock's 30-minute move relative to Nifty; stock code. Every feature uses only information available when the candle closes.

**Targets.** 30m: `ln(close 6 candles later / close now)`, same day only, so predictions stop after 14:55. EOD: `ln(15:25 close / close now)`, so predictions stop after 15:20. Predicted price = current price × e^(predicted return).

**Evaluation, two ways**

1. **Backtest (nightly):** walk-forward over the last 8 blocks of 5 trading days. Train on everything before each block, test on the block, never shuffle. Chronos-2 is scored on every 3rd test candle and Kronos on every 12th (30m only) to keep the run under about 30 minutes.
2. **Live (forward test):** every prediction is stored before its outcome and scored when the target candle closes. This is the number to trust.

**Metrics shown**

| Metric | Meaning | Good sign |
| --- | --- | --- |
| Direction accuracy | Share of up/down calls that were right | Above Naive's |
| Balanced accuracy | Accuracy on up moves and on down moves, averaged; immune to "always say down" | Above 50% |
| p vs naive | Chance the accuracy edge over Naive is luck (binomial test; optimistic, as neighbouring predictions overlap) | Below 0.05 |
| AUC | Quality of the up-probability | Above 0.5 |
| MAPE | Average price error as % of price | Below Naive's |
| Skill vs naive | `1 − Σ model % error / Σ naive % error` | Above 0 |
| IC | Rank correlation between predicted and actual moves | Above 0.05, in most blocks |
| Backtest return | Simple strategy after 0.05% costs: trade when the predicted move beats costs (backtest only) | Positive in most blocks |

**First backtest on your four files** (6 blocks of 5 days, 3 Sep to 16 Oct 2026, run while building this; Chronos-2 and Kronos were not included because their weights could not be downloaded in the build environment)

| Model | 30m direction | 30m balanced | 30m skill vs naive | 30m IC | EOD direction | EOD balanced | EOD skill vs naive | EOD IC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Naive | 55.6% | 50.0% | 0.00% |  | 55.2% | 50.0% | 0.00% |  |
| LightGBM | 56.0% | 51.2% | +0.32% | 0.01 | 52.4% | 48.7% | −2.11% | −0.08 |
| LSTM | 55.7% | 50.1% | +0.51% | 0.03 | 55.7% | 53.2% | +0.03% | 0.09 |
| Ensemble | 55.8% | 50.4% | +0.57% | 0.03 | 56.6% | 53.1% | +0.20% | 0.05 |

Reading: at 30 minutes, no model is distinguishable from "no change". At end of day the Ensemble and LSTM show a faint edge (balanced accuracy about 53%), but it changes sign between test blocks. The dashboard must show this as "weak or no signal" until live results confirm it.

**"No reliable signal" rule (used by the UI).** A model is marked **Edge** for a horizon only if, in its latest backtest, skill vs naive > 0, balanced accuracy > 52% and p vs naive < 0.05, and, once it has 200+ live predictions, its live skill vs naive > 0 too. Otherwise it is **No edge**. When no model has Edge, the forecast card says "No reliable signal: models do not beat 'no change' yet".

## 6. UI requirements

Four pages in a light theme: **Overview** (today's forecasts), **Models** (who is actually right), **History** (every prediction and its outcome) and **Data** (health, jobs, CSV import). Built with React, Tailwind, shadcn/ui, Recharts and TanStack Query, reading tables and views through the Lovable Cloud client.

### Global

| Item | Specification |
| --- | --- |
| Top bar | Wordmark "Predicted" left; nav Overview, Models, History, Data; right: market pill, "Data as of HH:MM IST", Refresh button |
| Market pill | Green dot "Market open" (Mon to Fri, 09:15 to 15:30 IST) or grey "Market closed"; live IST clock |
| Refresh button | Calls `ingest-live` with `{}`, then refetches all queries; shows a spinner; if the reply is `skipped`, shows its reason in a toast |
| Auto refresh | Refetch every 60 s while the market is open; off when closed |
| Stale banner | Amber, under the top bar, when the market is open and the newest prediction is over 15 min old: "Predictions are delayed: last run at 10:20 IST. Scheduled runs can start a few minutes late." |
| Empty state | Before the first predictions: "Waiting for the first prediction run. Runs start at 09:22 IST on market days." |
| Time | Always IST (`Asia/Kolkata`), 24-hour "10:45"; dates "Mon 19 Oct" |
| Numbers | Prices "₹216.40"; moves with sign "−0.04%" (from log return: `(exp(r) − 1) × 100`); probabilities "43%"; tabular figures |
| Up / down | Never colour alone: ▲ Up in green, ▼ Down in red, plus the word |
| Footer | "Research tool, not investment advice. Data: Yahoo Finance, may be delayed." |
| Responsive | At phone width: stock cards scroll sideways, model tables become one card per model, charts full width |

**Light theme tokens**

| Token | Value | Use |
| --- | --- | --- |
| Background | `#F8FAFC` | Page |
| Surface | `#FFFFFF`, border `#E2E8F0`, radius 12 px, shadow `0 1px 2px rgba(15,23,42,.06)` | Cards, tables |
| Text | `#0F172A`; muted `#64748B` | Body; labels |
| Accent | `#4F46E5` | Active nav, buttons, focus rings |
| Up / Down | `#059669` / `#DC2626` | Direction, moves |
| Warning | `#D97706` | Stale banner, "No edge" badge outline |
| Font | Inter (Google Fonts), 14 px base, numbers `font-variant-numeric: tabular-nums` | All |
| Model colours (fixed everywhere) | Naive `#94A3B8` (dashed lines), LightGBM `#2563EB`, LSTM `#7C3AED`, Ensemble `#0891B2`, Chronos-2 `#EA580C`, Kronos `#DB2777` | Charts, dots in tables |

### Overview (`/`)

1. **Stock strip:** four cards (PC Jeweller, Jio Financial, Suzlon, JSW Energy). Each: symbol, name, last close, change since today's open (%), a sparkline of today's closes, and two consensus chips: "30m: 4▼ 1▲" and "EOD: 3▼ 2▲" (Naive excluded). Click selects the stock; the choice lives in the URL (`?s=JIOFIN`).
2. **Stock header:** full name, last close, "as of 10:45 IST", today's high and low.
3. **Price chart** (Recharts `ComposedChart`, full width, 320 px tall): x = time today, 09:15 to 15:30 IST; y = price, auto-scaled with 0.5% padding.
   - Solid line: actual 5-minute closes.
   - Dotted line in the chosen model's colour: what that model predicted, 30 minutes earlier, for each candle today (30m predictions, plotted at `target_ts`). This shows how well it has tracked the price today.
   - The open forecasts: a point at the 30m target time and one at 15:25, each with an error bar for the 80% range.
   - Model picker above the chart (default: the model with the best live skill for 30m, falling back to Ensemble); toggle "Today / Last 2 days".
4. **Two forecast cards** side by side (stacked on phones): **"Next 30 minutes · target 11:15 IST"** and **"End of day · 15:25 candle close"**. Each card:
   - Signal line on top: green "Edge: LSTM, Ensemble" or amber "No reliable signal: models do not beat 'no change' yet" (rule in section 5).
   - Consensus: "4 of 5 models say ▼ Down".
   - Table, one row per model from `latest_predictions` for this stock and horizon (only rows whose `made_at` is today): colour dot + model name; direction; predicted price; move %; P(up) as a small bar with the %; 80% range "₹215.53 to ₹217.01"; live direction accuracy, last 30 days, with n (from `model_scoreboard`, this stock); Edge or No edge badge.
   - Naive row pinned at the bottom, muted, labelled "Baseline".
   - Card footer: "Made from the 10:45 candle at 10:52 IST" (`made_at`, `created_at`).
   - After 14:55 the 30-minute card shows "No 30-minute forecasts after 14:55 (target would pass the close)"; after 15:20 the end-of-day card shows today's final scorecard instead.
5. **Today's scorecard** (small table below): per model, predictions scored today for this stock, direction accuracy, average error %; both horizons.

### Models (`/models`)

1. Controls: tabs **Live (last 30 days)** | **Backtest (latest nightly)**; horizon toggle **30 min** | **End of day**; stock filter (All, or one stock).
2. **Leaderboard table**, sorted by skill vs naive: Model, Predictions (n), Direction accuracy, Balanced accuracy, MAPE, Skill vs naive, IC, Verdict (Edge / No edge). Backtest tab adds p vs naive, AUC, backtest return, trades. Live data from `model_scoreboard` (`symbol = 'ALL'` or the stock); backtest data from `latest_model_versions.backtest[horizon][stock or 'ALL']`. Naive pinned as the reference row. Tooltips on every column header explain the metric in one line (texts from section 5).
3. **Chart: direction accuracy by day** (line per model, from `model_daily_scores` for the chosen horizon, 50% reference line, y 30% to 70%). Live tab only.
4. **Chart: IC per backtest block** (grouped bars per model, from `backtest.fold_ic[horizon]`, zero line). Backtest tab only. Caption: "A real edge shows up in most blocks, not one."
5. **Model cards** (grid of 6): plain-language description (from section 5), version, trained at, training range, rows, key settings from `params`; LightGBM card lists its top 10 features from `params.top_features` as a small horizontal bar list.
6. Banner when no live data yet: "Live scores appear after the first scored predictions (30 minutes after the first run)."

### History (`/history`)

1. Filters: stock, model, horizon, status (open, resolved, expired), date range (default: today). Filters live in the URL.
2. Summary line for the filter: predictions, scored, direction accuracy, average error %.
3. Table, 50 rows a page, newest first: Made at, Stock, Model, Horizon, Target, Price then, Predicted, 80% range, Actual, Error ₹, Error %, Direction (✓ / ✗ / –), Status pill.
4. "Export CSV" downloads the filtered rows (up to 10,000).

### Data (`/data`)

1. **Health table** from `data_health`: symbol, name, type, candles, trading days, first candle, last candle, candles today, status dot (green if the last candle is under 10 minutes old during market hours, grey when closed, amber otherwise).
2. **Job cards:** last ingest, last prediction run, last training run, from `worker_runs`, each with status, time and message; amber if the last prediction run is older than 15 minutes during market hours.
3. **Job log:** last 50 `worker_runs`, filter by job; columns: job, status pill, started, duration, message; click a row to see `details` as formatted JSON.
4. **Import CSV files** (admin):
   - Passcode field (kept only in memory, never stored) and a drop zone for one or more `.csv` files.
   - The stock is read from the file name (must contain PCJEWELLER, JIOFIN, SUZLON or JSWENERGY); otherwise a dropdown asks for it.
   - Parse with Papa Parse; expected columns `datetime, open, high, low, close, volume` (the Colab downloader format, e.g. `2026-07-24 09:15:00+05:30`). Convert `datetime` to ISO UTC.
   - Send to `import-candles` in batches of 1000 rows with header `x-admin-passcode`; body `{rows, symbol, file_name, last_batch}`; rows `{symbol, ts, open, high, low, close, volume}`.
   - Per-file progress bar and result: "4,338 rows · 4,338 new · 0 already stored · 0 rejected". Wrong passcode shows "Wrong passcode" and stops.
5. **Refresh data now** button (same as the top bar) and a short explainer of the schedules (section 2 timing list).

## 7. Lovable build prompts

Paste the knowledge text once, then send the seven prompts in order, checking each result before the next. Attach the files named in each prompt from the zip's `lovable/` and `docs/` folders (Lovable accepts up to 10 files per message, including Markdown and code files, per [Lovable docs](https://docs.lovable.dev/features/projects/chat)).

### Step 0: project knowledge

Create a new Lovable project named **Predicted**. Open **Knowledge (chat actions menu → Project → Knowledge)** and paste the text below (about 4,800 characters; also in `docs/LOVABLE_KNOWLEDGE.md`).

```text
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
naive "Naive (baseline)" #94A3B8 dashed; lightgbm "LightGBM" #2563EB; lstm "LSTM" #7C3AED; ensemble "Ensemble" #0891B2; chronos2 "Chronos-2" #EA580C; kronos "Kronos" #DB2777. Show models that have data; never invent rows.

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

LOOK (light theme)
Background #F8FAFC; cards #FFFFFF, border #E2E8F0, radius 12px, subtle shadow; text #0F172A, muted #64748B; accent #4F46E5; warning #D97706. Font Inter, 14px base, tabular numbers. Clean, calm, data-first; no gradients, no dark mode, no stock photos. shadcn/ui components, Recharts for charts, TanStack Query for data, Papa Parse for CSV.

PAGES
/ Overview (stock strip, price chart with predicted-vs-actual overlay, "Next 30 minutes" and "End of day" forecast cards, today's scorecard) · /models (live vs backtest leaderboard, accuracy-by-day chart, IC-per-block chart, model cards) · /history (filterable prediction ledger, CSV export) · /data (data health, job cards and log, CSV import with passcode, Refresh). Full spec: attached PREDICTED_SPEC.md, section 6.
```

### Prompt 1: database and shell

Attach: `docs/PREDICTED_SPEC.md`, `lovable/migration.sql`

```text
We're building "Predicted" (spec attached: PREDICTED_SPEC.md; rules in Project Knowledge).
This step only:
1. Enable Lovable Cloud.
2. Run the attached migration.sql EXACTLY as written, as one migration. Don't rename, add or drop anything.
3. Build the app shell only: top bar (wordmark "Predicted"; nav Overview, Models, History, Data; market-open pill with a live IST clock), empty pages for the four routes, and the footer disclaimer. Light theme tokens and Inter font from Knowledge.
When done, confirm: tables symbols, candles, predictions, model_versions, worker_runs; views latest_predictions, model_scoreboard, model_daily_scores, data_health, latest_model_versions; function resolve_predictions; 7 rows in symbols.
```

### Prompt 2: backend functions, secrets, schedule

Attach: `lovable/functions/ingest-live/index.ts`, `lovable/functions/worker-api/index.ts`, `lovable/functions/import-candles/index.ts`, `lovable/config.toml`, `lovable/cron.sql`

```text
Create three edge functions using the attached code EXACTLY (file name = function name): ingest-live, worker-api, import-candles.
- Add the attached config.toml settings to supabase/config.toml (verify_jwt = false for these three).
- Ask me to add two secrets: WORKER_TOKEN and ADMIN_PASSCODE.
- Make sure the pg_cron and pg_net extensions are enabled, then run cron.sql, replacing <INGEST_LIVE_URL> with the real URL of the ingest-live function.
- Finally tell me the full URL of the worker-api function; I need it for GitHub.
Don't build any UI in this step.
```

### Prompt 3: Data page and CSV import

```text
Build the /data page exactly as in spec section 6 "Data": data health table from the data_health view; job cards and job log from worker_runs (click a row to see details JSON); "Import CSV files" panel (passcode field kept in memory only, multi-file drop zone, stock detected from the file name, Papa Parse, convert datetime like "2026-07-24 09:15:00+05:30" to ISO UTC, send batches of 1000 rows to the import-candles function with header x-admin-passcode and body {rows, symbol, file_name, last_batch}, per-file progress and result counts); "Refresh data now" button calling ingest-live with {} and showing the reply's reason in a toast if skipped.
Also wire the top-bar Refresh button and "Data as of HH:MM IST" (newest candle ts of the four stocks).
```

After this prompt: open the Data page, enter your passcode and drop the four `*_5min.csv` files. Expect about 4,300 to 4,400 rows per stock, all new.

### Prompt 4: Overview page

```text
Build the Overview page (/) exactly as in spec section 6 "Overview":
1. Stock strip (4 cards, consensus chips excluding Naive, selection in ?s=).
2. Stock header.
3. Price chart (Recharts ComposedChart): today's 5-min closes; dotted line of the chosen model's 30m predictions plotted at target_ts; open forecasts at the 30m target and at 15:25 with 80% error bars; model picker (default = best live 30m skill_vs_naive_pct from model_scoreboard for this stock, else ensemble); Today / Last 2 days toggle.
4. "Next 30 minutes" and "End of day" forecast cards from latest_predictions (made_at today only), with signal line (Edge rule in Knowledge), consensus, per-model rows, Naive pinned last as Baseline, live accuracy from model_scoreboard, footer with made_at and created_at, and the after-14:55 / after-15:20 messages.
5. Today's scorecard.
Query only what's needed: candles for the selected stock since the previous trading day's 09:15; predictions for the selected stock from today. Empty state and stale banner as in Knowledge. Refetch every 60 s while the market is open.
```

### Prompt 5: Models page

```text
Build /models exactly as in spec section 6 "Models": Live / Backtest tabs, 30 min / End of day toggle, stock filter; leaderboard (live from model_scoreboard; backtest from latest_model_versions.backtest[horizon][stock or 'ALL']), Naive pinned as reference, verdict badges using the Edge rule, header tooltips; direction-accuracy-by-day line chart from model_daily_scores with a 50% line (Live tab); IC-per-block grouped bars from backtest.fold_ic (Backtest tab); six model cards with descriptions from spec section 5, version info and LightGBM's top features from params.top_features.
Accuracies are fractions (0.556 -> 55.6%); *_pct fields are already percentages.
```

### Prompt 6: History page

```text
Build /history exactly as in spec section 6 "History": URL-synced filters (stock, model, horizon, status, date range defaulting to today in IST), summary line, paginated table of predictions (50 per page, newest first) with Error ₹ = abs_error, Error % = abs_error / actual_price * 100, direction ✓/✗/–, status pill; Export CSV of the filtered rows (max 10,000).
```

### Prompt 7: polish and checks

```text
Polish and check the whole app:
- Phone width (375 px): stock strip scrolls sideways, forecast tables become one card per model, charts full width, no horizontal page scroll.
- Every number formatted per Knowledge (₹, signed %, IST times); up/down always with arrow and word.
- Loading skeletons; empty states; error toasts with the server message.
- Confirm no browser code writes to any table (only functions do).
- Model colours identical in every chart and table.
- Lighthouse accessibility 90+; visible focus rings.
List anything in spec section 6 that is not implemented yet.
```

## 8. GitHub setup

Upload the whole `predicted-worker` folder to a new **public** GitHub repository, add two secrets, and run the setup workflow once; after that everything runs on its own. Public is recommended because Actions minutes are free and unlimited for public repositories on standard runners, while a free private repository gets 2,000 minutes a month ([GitHub billing docs](https://docs.github.com/en/billing/concepts/product-billing/github-actions)), which 5-minute runs would use up in about a week. Your secrets stay private either way; only the code and run logs are visible.

**What is in the zip**

| Path | What it is | Where it goes |
| --- | --- | --- |
| `predicted/` | Python worker: data, features, 6 models, backtest, jobs | GitHub |
| `.github/workflows/` | `predict.yml`, `nightly.yml`, `setup.yml`, `keepalive.yml` | GitHub (hidden folder: keep it) |
| `requirements.txt`, `README.md`, `.gitignore` | Packages, instructions | GitHub |
| `tests/` | Replay, SQL, function and contract tests | GitHub (optional) |
| `data/seed/README.md` | Note: keep the CSVs out of a public repo; import them in the app | GitHub |
| `lovable/` | Migration, cron SQL, function settings, 3 edge functions | Attach to Lovable prompts 1 and 2 (also kept in GitHub for reference) |
| `docs/` | Lovable knowledge text and this spec in Markdown | Lovable (Knowledge and Prompt 1) |

**Manual steps**

1. **Account.** Sign up at github.com with `akshay_chitlangia@qbadvisory.com`, or add it under Settings → Emails on an existing account and verify it. As it is a work address, add a personal email as a backup too, so you keep access if that address changes.
2. **Repository.** New repository → name `predicted-worker` → Public → leave "Add a README" unticked → Create.
3. **Upload.** Either drag the *contents* of the unzipped folder onto the repository page ("uploading an existing file"); on a Mac press Cmd+Shift+. in Finder first so the hidden `.github` folder is visible and gets uploaded. Or use git:

```bash
git config --global user.name "Akshay Chitlangia"
git config --global user.email "akshay_chitlangia@qbadvisory.com"
cd predicted-worker
git init -b main
git add .
git commit -m "Predicted worker v1"
git remote add origin https://github.com/<your-username>/predicted-worker.git
git push -u origin main
```

4. **Secrets.** Repository → Settings → Secrets and variables → Actions → New repository secret:

| Secret | Value |
| --- | --- |
| `PREDICTED_API_URL` | The `worker-api` URL Lovable gave you in Prompt 2 |
| `PREDICTED_WORKER_TOKEN` | Exactly the value you saved in Lovable as `WORKER_TOKEN` |
| `PREDICTED_ANON_KEY` | Optional. Only if the setup run fails with a 401 "missing authorization" error: your Lovable project's publishable (anon) key |

Optional **Variables** (same page, Variables tab): `ENABLE_CHRONOS`, `ENABLE_KRONOS`, `ENABLE_LSTM` = `0` to switch a model off, for example if runs get slow.

5. **Import data first.** In Predicted's Data page, import the four CSV files (after Prompt 3).
6. **First run.** Repository → Actions → enable workflows if asked → **First-time setup** → Run workflow (leave "Also load data/seed" unticked). It checks the connection, pulls 59 days of Nifty 50, Nifty Bank and India VIX, backtests and trains. Expect 20 to 40 minutes on the first run while the Chronos-2 and Kronos weights download.
7. **Check.** Predicted's Data page shows a green `train` job; the Models page shows the Backtest tab filled in.
8. **Next market day.** Nothing to do: `predict.yml` runs every 5 minutes from 09:22 IST and the nightly run retrains at 16:10 IST. Watch the first few runs under Actions → Predict.

**What runs on its own afterwards**

| When (IST) | Where | What |
| --- | --- | --- |
| Every 5 min, 09:16 to 15:41 | Lovable (pg\_cron) | `ingest-live` stores new candles and scores due predictions |
| Every 5 min, 09:22 to 15:27 | GitHub (`predict.yml`) | Predictions from all models, both horizons |
| 16:10, Mon to Fri | GitHub (`nightly.yml`) | Fill gaps, backtest, retrain, publish model versions |
| 02:00 daily | Lovable (pg\_cron) | Delete job-log rows older than 30 days |
| 1st of the month | GitHub (`keepalive.yml`) | Tiny commit so GitHub doesn't pause schedules after 60 quiet days |

## 9. Testing, acceptance and limitations

Everything that could be tested without your Lovable project was tested on your four CSV files; what remains is a live check during one market day.

**Tests already run while building**

| Test | What it proved | Result |
| --- | --- | --- |
| Local seed → train → predict | 17,442 candles loaded; 6 backtest blocks; models saved; 32 predictions (4 stocks × 4 models × 2 horizons) with correct target times (10:45 → 11:15 and 15:25 IST) | Pass |
| Replay of 16 Oct 2026, 14 moments | 432 predictions; every one scored against the right candle; no 30m prediction after 14:55 | Pass, 0 mismatches |
| Migration on Postgres 17 (PGlite) | Tables, policies, views; scoring incl. a missing target candle and expiry; re-running scores nothing twice; anonymous users can read but not write | Pass |
| Edge functions (Deno 2) | Type-check; Yahoo parsing drops forming, misaligned, out-of-session and empty candles; market window; input validation; CSV cleaning | Pass, 5/5 |
| Contract: real `worker-api` server + real Python worker | Seed, train, predict over HTTP; wrong token rejected (401); re-import inserts 0 rows | Pass |
| Chronos-2 and Kronos wrappers | With small random-weight models of the real architectures: shapes, horizons, 15:25 and 14:55 cut-offs | Pass (forecast quality not testable without the real weights) |

**Acceptance checklist (first market day)**

- [ ] Data page: all 7 symbols show candles; the four stocks have about 60 trading days.
- [ ] At 09:21 IST, `ingest` jobs appear every 5 minutes with 7 symbols and no errors.
- [ ] By 09:35 IST, Overview shows forecasts for all four stocks from at least Naive, LightGBM, LSTM and Ensemble.
- [ ] Chronos-2 and Kronos rows appear too (check the Predict run log if not).
- [ ] 30 minutes after the first run, History shows the first `resolved` predictions with actual prices matching the chart.
- [ ] After 14:55 the 30-minute card shows its "no more forecasts" message; after 15:20 the end-of-day card shows the day's scorecard.
- [ ] At about 16:40 IST, a new model version appears on the Models page.
- [ ] Refresh twice within a minute: the second press says "ran less than a minute ago".
- [ ] On a phone, Overview is readable with no sideways page scroll.

**Known limitations and risks**

| Risk | Effect | Mitigation |
| --- | --- | --- |
| Yahoo's chart API is unofficial; it may rate-limit cloud servers or change format | Live candles stop arriving from `ingest-live` | The worker tops up today's candles from Yahoo (yfinance) on every run and fills gaps nightly; the Data page shows staleness. Moving to a broker API later only changes `ingest-live` and the worker's `fetch_yahoo` |
| Yahoo NSE data can be delayed | Predictions are made from slightly old prices | Each prediction shows the candle it used; the stale banner shows delays |
| GitHub scheduled runs can start late or be dropped at busy times ([GitHub docs](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows)) | Some 5-minute slots get no prediction | Runs are idempotent and always use the latest candle; schedules avoid the top of the hour |
| Chronos-2 and Kronos untested with real weights here | They could error on the first run | Each is optional and wrapped: a failure is logged and the other models still publish. Switch off with `ENABLE_CHRONOS=0` / `ENABLE_KRONOS=0` |
| Only 60 days of history | Weak, unstable signals; backtest uses 6 blocks until more data arrives | Live ledger grows daily; the nightly backtest automatically uses up to 8 blocks and the full history (up to 250 days) |
| Seed CSVs were saved split- and dividend-adjusted; live data is not | Small price mismatch only if a corporate action happens | Re-import or backfill after a split or bonus |
| NSE holidays and special sessions (e.g. Muhurat trading) | No candles that day; a special session may be partly captured | Handled as "no new candle"; nothing breaks |
| Public repository | Code and run logs are public | Secrets are masked; keep the CSV files out of the repo |
