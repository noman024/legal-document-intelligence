#!/usr/bin/env bash
# Smoke-test all HTTP endpoints. Requires: venv, Ollama running, model pulled, PYTHONPATH.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# shellcheck source=/dev/null
source .venv/bin/activate
export PYTHONPATH=.
export LEGAL_OLLAMA_BASE_URL="${LEGAL_OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
export LEGAL_OLLAMA_MODEL="${LEGAL_OLLAMA_MODEL:-llama3.2}"

HOST="${1:-127.0.0.1:8000}"
BASE="http://${HOST}"
PDF="${ROOT}/examples/sample_memo.pdf"

echo "== GET /health =="
curl -sf "$BASE/health" | python -m json.tool

echo "== GET /health/ready =="
curl -sf "$BASE/health/ready" | python -m json.tool

echo "== POST /ingest (multipart) =="
curl -sf -X POST -F "file=@${PDF}" "$BASE/ingest" | python -m json.tool

echo "== POST /draft =="
DRAFT_JSON=$(curl -sf -X POST "$BASE/draft" -H "Content-Type: application/json" \
  -d '{"task":"internal_memo","query":"Summarize risks and key dates."}')
echo "$DRAFT_JSON" | python -m json.tool
DRAFT_ID=$(echo "$DRAFT_JSON" | python -c "import sys,json; print(json.load(sys.stdin)['draft_id'])")

echo "== POST /feedback =="
# Build operator revision JSON from draft response (append note or tweak heading casing).
export DRAFT_JSON_FOR_SMOKE="$DRAFT_JSON"
FEEDBACK_JSON="$(python -c "
import json, os
d = json.loads(os.environ['DRAFT_JSON_FOR_SMOKE'])
text = d['text'].rstrip()
note = (
    \"\\n\\n— Operator note: group hard deadlines under a single Key dates bullet list when possible; \"
    \"keep inline citations like [1] tied to the memo.\"
)
edited = text + note if note not in text else text.replace('**Key Terms', '**Key terms', 1)
print(json.dumps({'draft_id': d['draft_id'], 'edited_text': edited}, ensure_ascii=False))
")"
unset DRAFT_JSON_FOR_SMOKE
curl -sf -X POST "$BASE/feedback" -H "Content-Type: application/json" -d "$FEEDBACK_JSON" | python -m json.tool

echo "== POST /draft (second call) =="
curl -sf -X POST "$BASE/draft" -H "Content-Type: application/json" \
  -d '{"task":"internal_memo","query":"Summarize risks and key dates."}' | python -m json.tool | head -n 50

echo "OK: smoke_api finished"
