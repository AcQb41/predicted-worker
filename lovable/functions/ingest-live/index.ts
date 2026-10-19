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
