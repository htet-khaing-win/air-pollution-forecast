#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# e2e_test.sh — Local end-to-end verification (Updated)
# ──────────────────────────────────────────────────────────────────────────────

set -e
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'

pass() { echo -e "${GREEN}✅ $1${NC}"; }
fail() { echo -e "${RED}❌ $1${NC}"; exit 1; }
info() { echo -e "${YELLOW}→   $1${NC}"; }

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   PM2.5 Forecast — End-to-End Container Verification"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── 1. Check .env exists ───────────────────────────────────────────────────────
info "Checking .env file..."
if [ ! -f .env ]; then
  fail ".env not found. Run: cp .env.example .env  then fill in your API keys."
fi
pass ".env file present"

# ── 2. Start containers if not running ────────────────────────────────────────
info "Checking containers..."
# Silence the obsolete version attribute warning for cleaner output
export DOCKER_DEFAULT_PLATFORM=linux/amd64
RUNNING=$(docker compose ps --status running --services 2>/dev/null | wc -l)
if [ "$RUNNING" -lt 3 ]; then
  info "Starting containers..."
  docker compose up -d --build
  info "Waiting 45s for services to initialise..."
  sleep 45
else
  pass "Containers already running ($RUNNING services)"
fi

# ── 3. Check all services are healthy ─────────────────────────────────────────
info "Checking service health..."

check_service() {
  local SERVICE=$1
  # Improved JSON parsing to handle different docker compose output formats
  local STATUS
  STATUS=$(docker compose ps "$SERVICE" --format json 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    if isinstance(data, list): data = data[0]
    print(data.get('Health', data.get('Status', 'unknown')).lower())
except:
    print('unknown')
" 2>/dev/null || echo "unknown")

  if [[ "$STATUS" == *"healthy"* ]] || [[ "$STATUS" == *"running"* ]]; then
    pass "$SERVICE → $STATUS"
  else
    echo -e "${YELLOW}⚠   $SERVICE → $STATUS (may still be starting)${NC}"
  fi
}

check_service postgres
check_service scheduler
check_service api

# ── 4. Verify MLflow is reachable from host ────────────────────────────────────
info "Verifying MLflow UI (host → scheduler:5000)..."
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:5000 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ] || [ "$HTTP_CODE" = "302" ]; then
  pass "MLflow UI reachable at http://localhost:5000 (HTTP $HTTP_CODE)"
else
  fail "MLflow not reachable at http://localhost:5000 (HTTP $HTTP_CODE)"
fi

# ── 5. Verify MLflow is reachable from inside the API container ───────────────
info "Verifying MLflow reachable from API container (api → scheduler:5000)..."
# Bypassing environment proxies and setting a User-Agent to avoid 403 Forbidden
CONTAINER_CHECK=$(docker compose exec -T api python3 -c "
import urllib.request, sys
try:
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    req = urllib.request.Request('http://scheduler:5000/', headers={'User-Agent': 'ContainerCheck/1.0'})
    with opener.open(req, timeout=5) as resp:
        print('ok')
except Exception as e:
    print(f'DEBUG_ERR: {e}')
" 2>&1 || echo "EXEC_FAILED")

if [[ "$CONTAINER_CHECK" == *"ok"* ]]; then
  pass "API container can reach MLflow at http://scheduler:5000"
else
  echo -e "${RED}❌ Connection Failed${NC}"
  echo "    Detail: $CONTAINER_CHECK"
  echo "    Hint: If you see 403, ensure no_proxy=scheduler is set in your docker-compose.yaml"
  exit 1
fi

# ── 6. Check FastAPI health endpoint ──────────────────────────────────────────
info "Checking FastAPI /health endpoint..."
HEALTH_RESPONSE=$(curl -sf http://localhost:8000/health 2>/dev/null || echo '{"error":"unreachable"}')
echo "    Response: $HEALTH_RESPONSE"

# Use python3 to safely extract fields even if the response is slightly different
STATUS=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('status','unknown'))" 2>/dev/null || echo "parse_error")
MODEL_LOADED=$(echo "$HEALTH_RESPONSE" | python3 -c "import sys,json; print(str(json.load(sys.stdin).get('model_loaded','False')).lower())" 2>/dev/null || echo "false")

if [ "$STATUS" = "ok" ] && [ "$MODEL_LOADED" = "true" ]; then
  pass "FastAPI /health → status=ok, model_loaded=true"
elif [ "$STATUS" = "degraded" ] || [ "$MODEL_LOADED" = "false" ]; then
  echo -e "${YELLOW}⚠   FastAPI /health → status=degraded (model not loaded yet)${NC}"
  echo "    Ensure a model is registered in MLflow with the '@champion' tag."
else
  fail "FastAPI /health returned unexpected status: $STATUS"
fi

# ── 7. Test GET /predict ───────────────────────────────────────────────────────
info "Testing GET /predict..."
PREDICT_RESPONSE=$(curl -s http://localhost:8000/predict 2>/dev/null || echo '{"error":"unreachable"}')
echo "    Response: $PREDICT_RESPONSE"

PM25=$(echo "$PREDICT_RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin).get('pm25_predicted','MISSING'))" 2>/dev/null || echo "MISSING")

if [ "$PM25" != "MISSING" ] && [ "$PM25" != "null" ]; then
  pass "GET /predict → pm25_predicted=$PM25 µg/m³"
else
  echo -e "${YELLOW}⚠   GET /predict did not return a prediction (check API logs)${NC}"
fi

# ── 8. Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   Verification complete."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"