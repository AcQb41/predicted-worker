#!/usr/bin/env bash
# End-to-end contract test: the REAL worker-api edge function (served by Deno, with an
# in-memory database stand-in) called by the REAL Python worker (seed -> train -> predict).
set -euo pipefail
cd "$(dirname "$0")/.."
TMP=$(mktemp -d)
cat > "$TMP/import_map.json" <<JSON
{ "imports": { "npm:@supabase/supabase-js@2.45.4": "$(pwd)/tests/fake_supabase.ts" } }
JSON
export WORKER_TOKEN=test-token SUPABASE_URL=http://unused SUPABASE_SERVICE_ROLE_KEY=unused
deno run -A --import-map "$TMP/import_map.json" lovable/functions/worker-api/index.ts > "$TMP/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; rm -rf "$TMP"' EXIT
for i in $(seq 1 30); do curl -s localhost:8000 >/dev/null 2>&1 && break; sleep 1; done

# wrong token must be rejected
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST localhost:8000 -H "x-worker-token: nope" -d '{"action":"status"}')
[ "$code" = "401" ] || { echo "expected 401, got $code"; exit 1; }

export PREDICTED_API_URL=http://localhost:8000 PREDICTED_WORKER_TOKEN=test-token
export PREDICTED_MODEL_DIR="$TMP/models" ENABLE_CHRONOS=0 ENABLE_KRONOS=0 PREDICTED_NOW=2026-10-19T10:51:00+05:30
unset PREDICTED_LOCAL_STORE || true
python3 -m predicted seed "${SEED_DIR:-data/seed}/*.csv" | tail -3
python3 -m predicted train --skip-backfill 2>&1 | grep -E "fold|trained" | tail -3
python3 -m predicted predict --no-refresh 2>&1 | grep -E "predictions as of"
python3 -m predicted status | head -8
# re-seeding must insert nothing new (fill mode)
python3 - <<'PY'
import os
from predicted.store import ApiStore
from predicted.marketdata import load_seed_csv
s = ApiStore()
n = s.upsert_candles(load_seed_csv(os.path.join(os.environ.get("SEED_DIR","data/seed"),"JIOFIN_5min.csv")), source="csv", mode="fill")
assert n == 0, f"fill mode re-inserted {n} rows"
print("re-seed inserted 0 rows: OK")
PY
grep -i error "$TMP/server.log" && exit 1 || true
echo "CONTRACT TEST OK"
