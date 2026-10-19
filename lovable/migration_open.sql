-- Predicted: "Tomorrow's open" add-on. Run as ONE migration AFTER migration.sql.
-- Adds: horizon 'open', model 'linear', job types 'train_open' / 'predict_open',
-- scoring against the 09:15 candle's OPEN, a stage-aware scoreboard, and a merged
-- latest_model_versions view (intraday + opening-price backtests side by side).
-- Safe to run more than once.

-- ============================================================ 1. allowed values
-- Drop the existing check constraints on these columns, whatever they are named.
do $$
declare r record;
begin
  for r in
    select c.conrelid::regclass as tbl, c.conname
    from pg_constraint c
    where c.contype = 'c'
      and c.conrelid in ('public.predictions'::regclass, 'public.model_versions'::regclass, 'public.worker_runs'::regclass)
      and (pg_get_constraintdef(c.oid) ilike '%horizon%'
           or pg_get_constraintdef(c.oid) ilike '%model_key%'
           or pg_get_constraintdef(c.oid) ilike '%job %'
           or pg_get_constraintdef(c.oid) ilike '%(job%')
  loop
    execute format('alter table %s drop constraint %I', r.tbl, r.conname);
  end loop;
end $$;

alter table public.predictions add constraint predictions_horizon_check
  check (horizon in ('30m', 'eod', 'open'));
alter table public.predictions add constraint predictions_model_key_check
  check (model_key in ('naive', 'lightgbm', 'lstm', 'ensemble', 'chronos2', 'kronos', 'linear'));
alter table public.model_versions add constraint model_versions_model_key_check
  check (model_key in ('naive', 'lightgbm', 'lstm', 'ensemble', 'chronos2', 'kronos', 'linear'));
alter table public.worker_runs add constraint worker_runs_job_check
  check (job in ('ingest', 'seed', 'import', 'backfill', 'train', 'predict', 'train_open', 'predict_open'));

-- ============================================================ 2. scoring
-- 30m / eod: unchanged (close of the target candle).
-- open: the OPEN of the first 09:15 IST candle at or after target_ts (a holiday simply
-- moves it to the next session), once that candle is 6 minutes old. Expires after 6 days.
create or replace function public.resolve_predictions()
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
  n1 integer;
  n2 integer;
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
      and p.horizon in ('30m', 'eod')
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
  get diagnostics n1 = row_count;

  with hit as (
    select p.id, c.open as px
    from public.predictions p
    join lateral (
      select c.open
      from public.candles c
      where c.symbol = p.symbol
        and c.ts >= p.target_ts
        and c.ts < p.target_ts + interval '6 days'
        and to_char(c.ts at time zone 'Asia/Kolkata', 'HH24:MI') = '09:15'
        and c.ts <= now() - interval '6 minutes'
      order by c.ts
      limit 1
    ) c on true
    where p.status = 'open'
      and p.horizon = 'open'
  )
  update public.predictions p
     set actual_price      = hit.px,
         actual_return     = ln(hit.px / p.base_price),
         abs_error         = abs(p.pred_price - hit.px),
         direction_correct = case when hit.px = p.base_price then null
                                  else (p.pred_direction = 'up') = (hit.px > p.base_price) end,
         status            = 'resolved',
         resolved_at       = now()
    from hit
   where p.id = hit.id;
  get diagnostics n2 = row_count;

  update public.predictions
     set status = 'expired', resolved_at = now()
   where status = 'open'
     and ((horizon in ('30m', 'eod') and now() > target_ts + interval '1 day')
       or (horizon = 'open' and now() > target_ts + interval '6 days'));

  return n1 + n2;
end;
$$;
revoke all on function public.resolve_predictions() from public, anon, authenticated;
grant execute on function public.resolve_predictions() to service_role;

-- ============================================================ 3. views
-- Live scores for tomorrow's open, split by stage:
--   'evening'  = made after the close (made_at 15:30 IST)
--   'pre_open' = made before the open (made_at 08:45 IST)
-- symbol = 'ALL' row per model and stage. Same columns as model_scoreboard plus `stage`.
create or replace view public.open_scoreboard
with (security_invoker = on) as
with r as (
  select p.symbol, p.model_key, p.made_at, p.pred_return, p.actual_return,
         p.direction_correct, p.abs_error,
         case when extract(hour from p.made_at at time zone 'Asia/Kolkata') >= 12
              then 'evening' else 'pre_open' end         as stage,
         p.abs_error / p.actual_price                    as rel_err,
         n.abs_error / n.actual_price                    as naive_rel_err
  from public.predictions p
  left join public.predictions n
    on n.symbol = p.symbol and n.horizon = 'open' and n.made_at = p.made_at
   and n.model_key = 'naive' and n.status = 'resolved'
  where p.horizon = 'open'
    and p.status = 'resolved'
    and p.made_at >= now() - interval '90 days'
)
select coalesce(symbol, 'ALL')                                         as symbol,
       model_key,
       stage,
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
group by grouping sets ((symbol, model_key, stage), (model_key, stage));

-- Latest model version per model, merging the newest INTRADAY version (30m / eod) with the
-- newest OPENING-PRICE version (params.scope = 'open'), so the dashboard reads both:
--   backtest -> {"30m": ..., "eod": ..., "open": ..., "open_evening": ...,
--                "fold_ic": {"30m": ..., "eod": ..., "open": ..., "open_evening": ...}, "resid_q": <intraday>}
--   params   -> intraday params, plus "open": <opening-price params>
-- Same columns as before. A model with only one kind (e.g. 'linear', 'lstm') shows that one.
drop view if exists public.latest_model_versions;
create view public.latest_model_versions
with (security_invoker = on) as
with intra as (
  select distinct on (model_key) *
  from public.model_versions
  where coalesce(params->>'scope', 'intraday') <> 'open'
  order by model_key, trained_at desc
), opn as (
  select distinct on (model_key) *
  from public.model_versions
  where params->>'scope' = 'open'
  order by model_key, trained_at desc
)
select coalesce(i.id, o.id)                   as id,
       coalesce(i.model_key, o.model_key)     as model_key,
       coalesce(i.version, o.version)         as version,
       coalesce(i.trained_at, o.trained_at)   as trained_at,
       coalesce(i.train_start, o.train_start) as train_start,
       coalesce(i.train_end, o.train_end)     as train_end,
       coalesce(i.n_train, o.n_train)         as n_train,
       case when o.id is null then i.params
            when i.id is null then jsonb_build_object('open', o.params)
            else i.params || jsonb_build_object('open', o.params) end as params,
       coalesce(i.backtest, '{}'::jsonb)
         || coalesce(o.backtest - 'resid_q' - 'fold_ic', '{}'::jsonb)
         || jsonb_build_object('fold_ic', coalesce(i.backtest->'fold_ic', '{}'::jsonb)
                                          || coalesce(o.backtest->'fold_ic', '{}'::jsonb)) as backtest
from intra i
full outer join opn o on o.model_key = i.model_key;
