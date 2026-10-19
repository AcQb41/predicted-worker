// Unit tests for the edge functions' pure logic. Run: deno test -A tests/functions_test.ts
import { ok as assert, deepStrictEqual as assertEquals } from "node:assert";

// Stop the modules from starting servers when imported.
(Deno as any).serve = () => ({});
const ingest = await import("../lovable/functions/ingest-live/index.ts");
const api = await import("../lovable/functions/worker-api/index.ts");
const imp = await import("../lovable/functions/import-candles/index.ts");

// 2026-10-16 (Friday) 09:15 IST = 03:45 UTC
const t0 = Date.UTC(2026, 9, 16, 3, 45) / 1000;

function yahoo(stamps: number[], closes: (number | null)[]) {
  return {
    chart: {
      result: [{
        timestamp: stamps,
        indicators: { quote: [{ open: closes, high: closes.map((c) => c && c + 1), low: closes.map((c) => c && c - 1), close: closes, volume: closes.map(() => 1000) }] },
      }],
      error: null,
    },
  };
}

Deno.test("parseYahooChart keeps finished, aligned, in-session candles only", () => {
  const stamps = [t0 - 300, t0, t0 + 300, t0 + 333, t0 + 600, t0 + 900];
  const closes = [100, 101, null, 102, 103, 104];
  const now = (t0 + 900) * 1000 + 60_000; // 1 min into the 09:30 candle -> 09:30 still forming
  const out = ingest.parseYahooChart("JIOFIN", yahoo(stamps, closes), now);
  assertEquals(out.map((c) => c.close), [101, 103]);      // pre-open, null, misaligned and forming dropped
  assertEquals(out[0].ts, "2026-10-16T03:45:00.000Z");
  assertEquals(out[0].source, "live");
});

Deno.test("parseYahooChart drops the post-15:25 candles and throws on errors", () => {
  const last = t0 + 74 * 300; // 15:25 IST
  const out = ingest.parseYahooChart("SUZLON", yahoo([last, last + 300], [40, 41]), (last + 3600) * 1000);
  assertEquals(out.length, 1);
  let threw = false;
  try { ingest.parseYahooChart("X", { chart: { result: null, error: { description: "No data found" } } }, 0); } catch { threw = true; }
  assert(threw);
});

Deno.test("market window is 09:15-15:45 IST on weekdays", () => {
  assert(ingest.inMarketWindow(t0 * 1000));
  assert(!ingest.inMarketWindow((t0 - 60) * 1000));
  assert(ingest.inMarketWindow((t0 + 390 * 60) * 1000));        // 15:45
  assert(!ingest.inMarketWindow((t0 + 391 * 60) * 1000));
  assert(!ingest.inMarketWindow((t0 + 86400) * 1000));           // Saturday
});

Deno.test("worker-api validation", () => {
  const c = { symbol: "JIOFIN", ts: "2026-10-16T03:45:00Z", open: 1, high: 2, low: 1, close: 1.5, volume: 10, source: "backfill" };
  assertEquals(api.validateCandle(c), null);
  assertEquals(api.validateCandle({ ...c, high: 0.5 }), "high < low");
  assertEquals(api.validateCandle({ ...c, symbol: "jio; drop" }), "bad symbol");
  const p = { symbol: "JIOFIN", model_key: "lstm", horizon: "30m", made_at: "2026-10-16T05:15:00Z", target_ts: "2026-10-16T05:45:00Z",
    base_price: 216.4, pred_return: -0.0004, pred_price: 216.3, pred_direction: "down", prob_up: 0.43 };
  assertEquals(api.validatePrediction(p), null);
  assertEquals(api.validatePrediction({ ...p, model_key: "gpt" }), "bad model_key");
  assertEquals(api.validatePrediction({ ...p, model_key: "linear", horizon: "open" }), null);
  assertEquals(api.validatePrediction({ ...p, horizon: "1d" }), "bad horizon");
  assertEquals(api.validatePrediction({ ...p, prob_up: 1.2 }), "bad prob_up");
  assertEquals(api.validatePrediction({ ...p, target_ts: p.made_at }), "target before made_at");
  assert(api.safeEqual("abc123", "abc123"));
  assert(!api.safeEqual("abc123", "abc124"));
  assert(!api.safeEqual("", ""));
});

Deno.test("import-candles cleanRows", () => {
  const rows = [
    { symbol: "JIOFIN", ts: "2026-10-16 09:15:00+05:30", open: "216", high: "217", low: "215", close: "216.5", volume: "0" },
    { symbol: "JIOFIN", ts: "2026-10-16 15:30:00+05:30", open: 1, high: 1, low: 1, close: 1, volume: 1 },   // after session
    { symbol: "JIOFIN", ts: "not a date", open: 1, high: 1, low: 1, close: 1, volume: 1 },
    { symbol: "JIOFIN", ts: "2026-10-16 09:20:00+05:30", open: 216, high: 215, low: 217, close: 216, volume: 5 }, // swapped hi/lo
  ];
  const [out, rejected] = imp.cleanRows(rows);
  assertEquals(rejected, 2);
  assertEquals(out[0].ts, "2026-10-16T03:45:00.000Z");
  assertEquals(out[1].high, 217);
  assertEquals(out[1].low, 215);
});
