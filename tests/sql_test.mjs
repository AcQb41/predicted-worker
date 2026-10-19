// Runs lovable/migration.sql on PGlite (real Postgres compiled to WASM) and checks
// resolution and the dashboard views with a small synthetic scenario.
// Run: npm i @electric-sql/pglite && MIGRATION=lovable/migration.sql node tests/sql_test.mjs
import { PGlite } from "@electric-sql/pglite";
import fs from "node:fs";
import assert from "node:assert/strict";

const db = new PGlite();
await db.exec(`create role anon; create role authenticated; create role service_role;`);
await db.exec(fs.readFileSync(process.env.MIGRATION, "utf8"));
console.log("migration applied");

// Candles: JIOFIN 10:00..10:40 IST yesterday, close rising 100 -> 100.8
await db.exec(`
  insert into candles (symbol, ts, open, high, low, close, volume, source)
  select 'JIOFIN', ((current_date - 1) + time '10:00' + (i * interval '5 minutes')) at time zone 'Asia/Kolkata',
         100 + i*0.1, 100.2 + i*0.1, 99.9 + i*0.1, 100 + i*0.1, 1000, 'csv'
  from generate_series(0, 8) i;
  -- JSWENERGY: target candle missing (10:30), 10:25 present -> fallback within 10 minutes
  insert into candles (symbol, ts, open, high, low, close, volume, source)
  select 'JSWENERGY', ((current_date - 1) + time '10:00' + (i * interval '5 minutes')) at time zone 'Asia/Kolkata',
         500, 501, 499, 500 - i, 1000, 'live'
  from generate_series(0, 5) i;`);

const made = `((current_date - 1) + time '10:00') at time zone 'Asia/Kolkata'`;
const tgt = `((current_date - 1) + time '10:30') at time zone 'Asia/Kolkata'`;
await db.exec(`
  insert into model_versions (model_key, version) values ('lightgbm','v1'), ('naive','v1');
  insert into predictions (symbol, model_key, horizon, made_at, target_ts, base_price, pred_return, pred_price, pred_direction, prob_up, model_version_id)
  values
   ('JIOFIN','lightgbm','30m', ${made}, ${tgt}, 100, 0.004, 100.4, 'up', 0.6, 1),
   ('JIOFIN','naive','30m', ${made}, ${tgt}, 100, 0, 100, 'down', 0.45, 2),
   ('JSWENERGY','lightgbm','30m', ${made}, ${tgt}, 500, 0.001, 500.5, 'up', 0.55, 1),
   ('JSWENERGY','naive','30m', ${made}, ${tgt}, 500, 0, 500, 'down', 0.45, 2),
   -- open prediction whose target is in the future: must stay open
   ('JIOFIN','lightgbm','eod', ${made}, now() + interval '1 hour', 100, 0.01, 101, 'up', 0.7, 1),
   -- target 2 days ago with no candles: must expire
   ('SUZLON','lightgbm','30m', now() - interval '3 days', now() - interval '2 days', 40, 0, 40, 'up', 0.5, 1);`);

const n = (await db.query(`select resolve_predictions() as n`)).rows[0].n;
assert.equal(n, 4, "4 predictions resolved");
const rows = (await db.query(`select symbol, model_key, horizon, status, actual_price, abs_error, direction_correct from predictions order by id`)).rows;
console.table(rows);
assert.equal(rows[0].actual_price, 100.6);          // JIOFIN 10:30 close = 100 + 6*0.1
assert.equal(rows[0].direction_correct, true);
assert.equal(rows[1].direction_correct, false);     // naive said down
assert.equal(rows[2].actual_price, 495);            // JSW 10:25 candle (10:30 missing)
assert.equal(rows[2].direction_correct, false);
assert.equal(rows[4].status, "open");
assert.equal(rows[5].status, "expired");

// Re-running resolves nothing new
assert.equal((await db.query(`select resolve_predictions() as n`)).rows[0].n, 0);

const sb = (await db.query(`select symbol, model_key, horizon, n, dir_acc, balanced_acc, round(mape_pct::numeric,4) mape, round(skill_vs_naive_pct::numeric,2) skill from model_scoreboard order by symbol, model_key`)).rows;
console.table(sb);
const all = sb.find(r => r.symbol === "ALL" && r.model_key === "lightgbm");
assert.equal(Number(all.n), 2);
assert.equal(Number(all.dir_acc), 0.5);
console.table((await db.query(`select * from latest_predictions`)).rows.map(r => ({ s: r.symbol, m: r.model_key, h: r.horizon, st: r.status })));
console.table((await db.query(`select symbol, candles, trading_days from data_health order by sort_order`)).rows);
console.table((await db.query(`select * from model_daily_scores`)).rows);
console.table((await db.query(`select model_key, version from latest_model_versions`)).rows);

// RLS: anon can read, cannot write
await db.exec(`grant usage on schema public to anon; grant select, insert on all tables in schema public to anon;`);
await db.exec(`set role anon;`);
assert.ok((await db.query(`select count(*) from predictions`)).rows[0].count >= 0);
await assert.rejects(db.exec(`insert into worker_runs (job, status) values ('ingest','ok')`), /row-level security/);
await db.exec(`reset role;`);
console.log("SQL TESTS OK");
