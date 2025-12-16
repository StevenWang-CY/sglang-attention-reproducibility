#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:8000}"

# Create a minimal request JSON in a temp file
REQ="$(mktemp /tmp/sglang_req.XXXXXX.json)"
cat > "$REQ" <<'JSON'
{
  "model": "Qwen3-VL-30B-A3B-Instruct",
  "temperature": 0.0,
  "max_tokens": 32,
  "messages": [
    {
      "role": "system",
      "content": [
        { "type": "text", "text": "You are a helpful assistant." }
      ]
    },
    {
      "role": "user",
      "content": [
        { "type": "text", "text": "Say hello." }
      ]
    }
  ]
}
JSON

echo "==> POST $URL/v1/chat/completions"
echo "==> Request JSON: $REQ"
echo

# Send request
curl -s "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQ" | python -m json.tool

echo
echo "==> Done. (Temp JSON kept at: $REQ)"

