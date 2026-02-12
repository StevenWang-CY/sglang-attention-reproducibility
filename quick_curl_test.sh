#!/usr/bin/env bash
set -euo pipefail

URL="${1:-http://localhost:8000}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="${2:-$SCRIPT_DIR/html_test_request.json}"

if [[ ! -f "$REQ" ]]; then
  echo "Error: JSON file not found: $REQ" >&2
  exit 1
fi

echo "==> POST $URL/v1/chat/completions"
echo "==> Request JSON: $REQ"
echo

# Send request
curl -s "$URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d @"$REQ" | python -m json.tool

echo
echo "==> Done."

