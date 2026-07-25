#!/usr/bin/env bash
# Verify fragmentation and prefix contiguity at the report-16 measurement-arm configuration.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/sglang_0b2e/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
DRIVER="${DRIVER:-$SCRIPT_DIR/drive_fragmentation_experiment.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/report-16-verification}"
REPEAT_TAG="${1:-rep1}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-$HOME/sglang_git/python}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$DRIVER" ]] || { echo "Driver not found: $DRIVER" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SGLANG_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

LOG_FILE="$OUTPUT_DIR/verification_${REPEAT_TAG}.log"
INDEX_FILE="$OUTPUT_DIR/verification_${REPEAT_TAG}_kv_indices.json"
SGLANG_DUMP_KVIDX="$INDEX_FILE" \
  timeout --signal=TERM --kill-after=20s 700 \
  "$PYTHON_BIN" "$DRIVER" --model-path "$MODEL_PATH" --backend triton \
  --page-size 1 --frag --eager --mem-fraction-static 0.85 \
  --churn-waves 3 --churn-bs 16 --churn-gen 2048 \
  --batch 32 --prefix-words 3630 --gen 12 --rounds 1 \
  > "$LOG_FILE" 2>&1
echo "exit=$? $(date '+%F %T')" > "$OUTPUT_DIR/verification_${REPEAT_TAG}.done"
