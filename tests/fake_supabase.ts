// In-memory stand-in for supabase-js, just enough for worker-api's queries.
// Used by tests/contract_test.sh through a Deno import map.
type Row = Record<string, any>;
const tables: Record<string, Row[]> = { candles: [], predictions: [], model_versions: [], worker_runs: [], symbols: [] };
const seq: Record<string, number> = {};
const keyCols: Record<string, string[]> = {
  candles: ["symbol", "ts"],
  predictions: ["symbol", "model_key", "horizon", "made_at"],
  model_versions: ["model_key", "version"],
};
const norm = (v: any, col: string) =>
  ["ts", "made_at", "target_ts", "started_at"].includes(col) && typeof v === "string" ? new Date(v).toISOString() : v;

class Q {
  op = "select"; rows: Row[] = []; vals: Row = {}; opts: any = {}; filters: ((r: Row) => boolean)[] = [];
  orders: string[] = []; from = 0; to = Infinity; one = false; returning = false;
  constructor(public table: string) {}
  select(_c?: string) { if (this.op !== "select") this.returning = true; return this; }
  insert(rows: Row | Row[]) { this.op = "insert"; this.rows = [rows].flat(); return this; }
  upsert(rows: Row | Row[], opts: any = {}) { this.op = "upsert"; this.rows = [rows].flat(); this.opts = opts; return this; }
  update(vals: Row) { this.op = "update"; this.vals = vals; return this; }
  eq(c: string, v: any) { this.filters.push((r) => r[c] === norm(v, c)); return this; }
  in(c: string, v: any[]) { this.filters.push((r) => v.includes(r[c])); return this; }
  gte(c: string, v: any) { this.filters.push((r) => r[c] >= norm(v, c)); return this; }
  order(c: string) { this.orders.push(c); return this; }
  range(a: number, b: number) { this.from = a; this.to = b; return this; }
  limit(n: number) { this.to = n - 1; return this; }
  single() { this.one = true; return this; }
  then(res: (v: any) => any, rej?: (e: any) => any) { try { return Promise.resolve(this.run()).then(res, rej); } catch (e) { return rej?.(e); } }
  run() {
    const t = (tables[this.table] ??= []);
    if (this.table === "data_health") {
      const by: Record<string, number> = {};
      for (const c of tables.candles) by[c.symbol] = (by[c.symbol] ?? 0) + 1;
      return { data: Object.entries(by).map(([symbol, candles]) => ({ symbol, candles })), error: null };
    }
    if (this.op === "select") {
      let out = t.filter((r) => this.filters.every((f) => f(r)));
      out.sort((a, b) => { for (const c of this.orders) { if (a[c] < b[c]) return -1; if (a[c] > b[c]) return 1; } return 0; });
      out = out.slice(this.from, this.to + 1);
      return { data: this.one ? out[0] : out, error: null };
    }
    if (this.op === "update") {
      t.filter((r) => this.filters.every((f) => f(r))).forEach((r) => Object.assign(r, this.vals));
      return { data: null, error: null };
    }
    const keys = keyCols[this.table];
    let affected = 0; const written: Row[] = [];
    for (const raw of this.rows) {
      const r: Row = Object.fromEntries(Object.entries(raw).map(([k, v]) => [k, norm(v, k)]));
      const hit = keys && this.op === "upsert" ? t.find((x) => keys.every((k) => x[k] === r[k])) : undefined;
      if (hit) { if (!this.opts.ignoreDuplicates) { Object.assign(hit, r); affected++; written.push(hit); } }
      else { r.id ??= (seq[this.table] = (seq[this.table] ?? 0) + 1); t.push(r); affected++; written.push(r); }
    }
    return { data: this.returning ? (this.one ? written[0] : written) : null, error: null, count: affected };
  }
}

export function createClient(_url: string, _key: string) {
  return {
    from: (t: string) => new Q(t),
    rpc: (_fn: string) => Promise.resolve({ data: 0, error: null }),
    _tables: tables,
  };
}
