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
const MODEL_KEYS = ["naive", "lightgbm", "lstm", "ensemble", "chronos2", "kronos", "linear"];
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
  if (!["30m", "eod", "open"].includes(r.horizon)) return "bad horizon";
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
