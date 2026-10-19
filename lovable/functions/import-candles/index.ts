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
