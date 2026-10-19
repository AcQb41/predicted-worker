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
