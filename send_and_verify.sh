#!/usr/bin/env bash
set -euo pipefail

# ===== CHANGE THESE =====
URL="${1:-http://localhost:8000}"
REQUEST_FOLDER="html_request/request_005_20260316_221014"   # <-- change this to switch requests
# MODEL_PATH="${HF_MODELS}/Qwen/Qwen3-VL-30B-A3B-Instruct"   # <-- change to match your running model
MODEL_PATH="${HF_MODELS}/Qwen/Qwen3-VL-8B-Instruct"   # <-- change to match your running model

TRACE_DIR="qwen3vl-log/tree_sparse_traces"                  # <-- where SGLang saves runtime traces
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_DIR="$SCRIPT_DIR/$REQUEST_FOLDER"
TRACE_DIR="$SCRIPT_DIR/$TRACE_DIR"

# Find the request.json
REQ_JSON=$(find "$REQ_DIR" -name "request.json" | head -1)
if [[ -z "$REQ_JSON" ]]; then
  echo "Error: No request.json found in $REQ_DIR" >&2
  exit 1
fi

echo "================================================================"
echo "  SEND AND VERIFY: End-to-End Tree Sparse Attention Test"
echo "================================================================"
echo "  Server:    $URL"
echo "  Request:   $REQ_JSON"
echo "  Model:     $MODEL_PATH"
echo "  Trace dir: $TRACE_DIR"
echo "================================================================"
echo

# ── Step 1: Record existing trace files ──
echo "==> [Step 1] Recording existing trace files..."
BEFORE_TRACES=$(mktemp)
ls "$TRACE_DIR"/*.json 2>/dev/null | sort > "$BEFORE_TRACES" || true
BEFORE_COUNT=$(wc -l < "$BEFORE_TRACES")
echo "    Found $BEFORE_COUNT existing trace(s)"
echo

# ── Step 2: Send request to SGLang ──
echo "==> [Step 2] Sending request to $URL/v1/chat/completions ..."
RESPONSE_FILE=$(mktemp)
RESPONSE_CODE=$(curl -s -o "$RESPONSE_FILE" -w "%{http_code}" \
  "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQ_JSON")

if [[ "$RESPONSE_CODE" != "200" ]]; then
  echo "    ERROR: Server returned HTTP $RESPONSE_CODE"
  cat "$RESPONSE_FILE"
  rm -f "$BEFORE_TRACES" "$RESPONSE_FILE"
  exit 1
fi

# Extract and show the response summary
echo "    Response received (HTTP $RESPONSE_CODE)"
python3 -c "
import json, sys
with open('$RESPONSE_FILE') as f:
    r = json.load(f)
usage = r.get('usage', {})
choices = r.get('choices', [])
content = choices[0]['message']['content'][:200] if choices else '(empty)'
print(f'    Model: {r.get(\"model\", \"?\")}'  )
print(f'    Prompt tokens:     {usage.get(\"prompt_tokens\", \"?\")}'  )
print(f'    Completion tokens: {usage.get(\"completion_tokens\", \"?\")}'  )
print(f'    Response preview:  {content}...'  )
" 2>/dev/null || echo "    (could not parse response)"
echo

# ── Step 3: Find the new trace file ──
echo "==> [Step 3] Looking for new trace file..."
# Wait a moment for the trace to be finalized
sleep 2

AFTER_TRACES=$(mktemp)
ls "$TRACE_DIR"/*.json 2>/dev/null | sort > "$AFTER_TRACES" || true

# Find new files (in AFTER but not in BEFORE)
NEW_TRACES=$(comm -13 "$BEFORE_TRACES" "$AFTER_TRACES")
rm -f "$BEFORE_TRACES" "$AFTER_TRACES"

if [[ -z "$NEW_TRACES" ]]; then
  # Retry after a longer wait
  echo "    No new trace yet, waiting 5s..."
  sleep 5
  NEW_TRACES=$(ls -t "$TRACE_DIR"/*.json 2>/dev/null | head -1)
fi

if [[ -z "$NEW_TRACES" ]]; then
  echo "    WARNING: No new trace file found in $TRACE_DIR"
  echo "    Running verification without trace comparison..."
  TRACE_FLAG=""
else
  # Use the latest new trace
  TRACE_JSON=$(echo "$NEW_TRACES" | tail -1)
  echo "    Found new trace: $(basename "$TRACE_JSON")"
  TRACE_FLAG="--trace-json $TRACE_JSON"
fi
echo

# ── Step 4: Run verification ──
echo "==> [Step 4] Running tree parser verification..."
echo "================================================================"

python3 "$SCRIPT_DIR/verify_tree_parser.py" \
  --request-json "$REQ_JSON" \
  --model-path "$MODEL_PATH" \
  $TRACE_FLAG

echo
echo "================================================================"
echo "==> Done! Verification complete."
echo "================================================================"

# Cleanup
rm -f "$RESPONSE_FILE"
