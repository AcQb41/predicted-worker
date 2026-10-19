// Applies lovable/migration.sql then lovable/migration_open.sql on PGlite (real Postgres in WASM)
// and checks the "Tomorrow's open" scoring, constraints and views.
// Run: npm i @electric-sql/pglite && node tests/sql_open_test.mjs
import { PGlite } from "@electric-sql/pglite";
import fs from "node:fs";
import assert from "node:assert/strict";

const db = new PGlite();
await db.exec(`create role anon; create role authenticated; create role service_role;`);
await db.exec(fs.readFileSync("lovable/migration.sql", "utf8"));
await db.exec(fs.readFileSync("lovable/migration_open.sql", "utf8"));
await db.exec(fs.readFileSync("lovable/migration_open.sql", "utf8"));   // idempotent
console.log("migrations applied (open add-on twice)");

const ist = (d, t) => `((current_date - ${d}) + time '${t}') at time zone 'Asia/Kolkata'`;
// Evening forecast made 3 days ago (15:30) for "2 days ago" 09:15; that day is a holiday (no candles),
// so it must be scored on the NEXT session's 09:15 open (yesterday).
await db.exec(`
  insert into candles (symbol, ts, open, high, low, close, volume, source) values
   ('JIOFIN', ${ist(1, "09:15")}, 102, 103, 101, 102.5, 1000, 'live'),
   ('JIOFIN', ${ist(1, "09:20")}, 102.5, 104, 102, 103.9, 1000, 'live'),
   ('SUZLON', ${ist(0, "09:15")}, 41, 41, 40, 40.5, 1000, 'live');
  insert into model_versions (model_key, version, params, backtest) values
   ('lightgbm', 'intra', '{"num_leaves":15}', '{"30m":{"ALL":{"n":10}},"eod":{"ALL":{"n":5}},"resid_q":{"30m":[-0.01,0.01]},"fold_ic":{"30m":{"1":0.02}}}'),
   ('lightgbm', 'open', '{"scope":"open","num_leaves":7}', '{"open":{"ALL":{"n":250}},"open_evening":{"ALL":{"n":250}},"resid_q":{"open":[-0.02,0.02]},"fold_ic":{"open":{"1":0.3}}}'),
   ('linear', 'open', '{"scope":"open"}', '{"open":{"ALL":{"n":250}}}'),
   ('lstm', 'intra', '{}', '{"30m":{"ALL":{"n":9}}}');
  insert into predictions (symbol, model_key, horizon, made_at, target_ts, base_price, pred_return, pred_price, pred_direction, prob_up)
  values
   ('JIOFIN','linear','open', ${ist(3, "15:30")}, ${ist(2, "09:15")}, 100, 0.01, 101, 'up', 0.6),
   ('JIOFIN','naive','open',  ${ist(3, "15:30")}, ${ist(2, "09:15")}, 100, 0, 100, 'down', 0.45),
   ('JIOFIN','linear','open', ${ist(1, "08:45")}, ${ist(1, "09:15")}, 100, 0.03, 103.05, 'up', 0.8),
   ('JIOFIN','naive','open',  ${ist(1, "08:45")}, ${ist(1, "09:15")}, 100, 0, 100, 'down', 0.45),
   ('SUZLON','lightgbm','open', ${ist(1, "15:30")}, ${ist(0, "09:15")}, 40, -0.01, 39.6, 'down', 0.4),
   ('PCJEWELLER','kronos','open', now() - interval '8 days', now() - interval '7 days', 20, 0, 20, 'up', null);`);

const nowMin = (await db.query(`select extract(epoch from now() - (${ist(0, "09:15")})) / 60 as m`)).rows[0].m;
const n = (await db.query(`select resolve_predictions() as n`)).rows[0].n;
const rows = (await db.query(`select symbol, model_key, to_char(made_at at time zone 'Asia/Kolkata','DD HH24:MI') made, status, actual_price, round(actual_return::numeric,5) ar, direction_correct from predictions order by id`)).rows;
console.table(rows);
assert.equal(rows[0].actual_price, 102, "holiday: scored on the next session's 09:15 OPEN (not close 102.5)");
assert.equal(rows[0].direction_correct, true);
assert.equal(rows[1].direction_correct, false);
assert.equal(rows[2].actual_price, 102);
assert.equal(rows[5].status, "expired");
// SUZLON: today's 09:15 candle only counts once it is 6+ minutes old
assert.equal(rows[4].status, nowMin >= 6 ? "resolved" : "open");
assert.equal(n, nowMin >= 6 ? 5 : 4);

const sb = (await db.query(`select symbol, model_key, stage, n, dir_acc, round(skill_vs_naive_pct::numeric,2) skill from open_scoreboard order by symbol, model_key, stage`)).rows;
console.table(sb);
const lin = sb.filter(r => r.symbol === "ALL" && r.model_key === "linear");
assert.deepEqual(lin.map(r => r.stage).sort(), ["evening", "pre_open"]);
assert.ok(Number(lin.find(r => r.stage === "pre_open").skill) > 0, "linear beat naive on the pre-open run");

const lv = (await db.query(`select model_key, version, params, backtest from latest_model_versions order by model_key`)).rows;
console.log(JSON.stringify(lv, null, 1).slice(0, 1500));
const lg = lv.find(r => r.model_key === "lightgbm");
assert.ok(lg.backtest["30m"] && lg.backtest["eod"] && lg.backtest["open"] && lg.backtest["open_evening"]);
assert.deepEqual(Object.keys(lg.backtest.fold_ic).sort(), ["30m", "open"]);
assert.deepEqual(lg.backtest.resid_q, { "30m": [-0.01, 0.01] });
assert.equal(lg.params.num_leaves, 15);
assert.equal(lg.params.open.num_leaves, 7);
assert.ok(lv.find(r => r.model_key === "linear").backtest.open);
assert.ok(lv.find(r => r.model_key === "lstm").backtest["30m"]);

// constraints
await assert.rejects(db.exec(`insert into predictions (symbol, model_key, horizon, made_at, target_ts, base_price, pred_return, pred_price, pred_direction) values ('JIOFIN','linear','1d', now(), now()+interval '1 day', 1, 0, 1, 'up')`));
await assert.rejects(db.exec(`insert into worker_runs (job, status) values ('bogus','ok')`));
await assert.rejects(db.exec(`insert into worker_runs (job, status) values ('train_open','bogus')`), "status check kept");
await db.exec(`insert into worker_runs (job, status) values ('train_open','ok'), ('predict_open','running')`);
// old horizons still resolve as before
await db.exec(`insert into candles (symbol, ts, open, high, low, close, volume, source) values ('JSWENERGY', ${ist(1, "10:30")}, 500, 501, 499, 500.5, 1, 'live');
  insert into predictions (symbol, model_key, horizon, made_at, target_ts, base_price, pred_return, pred_price, pred_direction) values ('JSWENERGY','lightgbm','30m', ${ist(1, "10:00")}, ${ist(1, "10:30")}, 500, 0.001, 500.5, 'up')`);
assert.equal((await db.query(`select resolve_predictions() as n`)).rows[0].n, 1);
assert.equal((await db.query(`select actual_price from predictions where symbol='JSWENERGY'`)).rows[0].actual_price, 500.5);
console.log("SQL OPEN TEST OK");
