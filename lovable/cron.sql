-- Predicted: scheduled jobs (pg_cron + pg_net). Replace <INGEST_LIVE_URL> with the
-- ingest-live function URL (Cloud -> Edge functions -> ingest-live -> Copy URL).

-- Pull Yahoo candles 1 minute after every 5-minute candle closes, Mon-Fri.
-- Window 03:01-10:56 UTC (08:31-16:26 IST); the function itself skips outside 09:15-15:45 IST.
select cron.schedule(
  'predicted-ingest-live',
  '1-56/5 3-10 * * 1-5',
  $$
  select net.http_post(
    url     := '<INGEST_LIVE_URL>',
    headers := '{"Content-Type": "application/json"}'::jsonb,
    body    := '{}'::jsonb
  );
  $$
);

-- Keep the job log small: delete runs older than 30 days, nightly at 02:00 IST.
select cron.schedule(
  'predicted-cleanup-runs',
  '30 20 * * *',
  $$ delete from public.worker_runs where started_at < now() - interval '30 days'; $$
);
