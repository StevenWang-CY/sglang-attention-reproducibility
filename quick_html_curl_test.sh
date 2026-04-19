#!/usr/bin/env bash
set -euo pipefail

# ===== CHANGE THESE =====
URL="${1:-http://localhost:8000}"
REQUEST_FOLDER="html_request/request_005_20260316_221014"   # <-- change this to switch requests
# =========================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ_DIR="${2:-$SCRIPT_DIR/$REQUEST_FOLDER}"

if [[ ! -d "$REQ_DIR" ]]; then
  echo "Error: Request directory not found: $REQ_DIR" >&2
  exit 1
fi

# Find all request.json files, sorted by folder name
REQ_FILES=()
while IFS= read -r f; do
  REQ_FILES+=("$f")
done < <(find "$REQ_DIR" -name "request.json" | sort)

if [[ ${#REQ_FILES[@]} -eq 0 ]]; then
  echo "Error: No request.json files found in $REQ_DIR" >&2
  exit 1
fi

echo "==> Found ${#REQ_FILES[@]} request(s) in $REQ_DIR"
echo "==> Target: $URL/v1/chat/completions"
echo

for REQ in "${REQ_FILES[@]}"; do
  REQ_NAME="$(basename "$(dirname "$REQ")")"
  echo "================================================================"
  echo "==> [$REQ_NAME] Sending request..."
  echo "==> File: $REQ"
  echo "================================================================"

  curl -s "$URL/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d @"$REQ" | python3 -m json.tool 2>/dev/null || true

  echo
  echo "==> [$REQ_NAME] Done."
  echo
done

echo "==> All ${#REQ_FILES[@]} requests completed."
