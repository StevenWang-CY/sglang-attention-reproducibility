#!/usr/bin/env bash
# Report 16 A1: shared-prefix steady decode across XQA, FlashInfer, and Triton.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/sglang_0b2e/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
DRIVER="${DRIVER:-$SCRIPT_DIR/drive_fragmentation_experiment.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/report-16-shared-prefix-engine/a1logs}"
PROGRESS_FILE="${PROGRESS_FILE:-$OUTPUT_DIR/progress.txt}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-260}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
VENV_BIN="${VENV_BIN:-$(dirname "$PYTHON_BIN")}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-$HOME/sglang_git/python}"
REPEAT_TAG="${1:-r1}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -f "$DRIVER" ]] || { echo "Driver not found: $DRIVER" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
export CUDA_HOME
export PATH="$VENV_BIN:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SGLANG_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

run_cell() {
  local tag=$1 backend=$2 page_size=$3 batch_size=$4
  local output_file="$OUTPUT_DIR/fr_${tag}.log"
  [[ -s "$output_file" ]] && { echo "skip $tag"; return; }

  timeout --signal=TERM --kill-after=20s "$TIMEOUT_SECONDS" \
    "$PYTHON_BIN" "$DRIVER" --model-path "$MODEL_PATH" --backend "$backend" \
    --page-size "$page_size" --batch "$batch_size" --prefix-words 3630 \
    --gen 256 --rounds 3 > "$output_file" 2>&1
  local return_code=$?
  echo "$tag exit=$return_code $(date '+%F %T')" >> "$PROGRESS_FILE"

  for _ in $(seq 1 20); do
    local used
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    [[ "${used:-9999}" -lt 500 ]] && break
    sleep 3
  done
}

for batch_size in 8 16 32; do
  for page_size in 16 32 64; do
    run_cell "trtllm_ps${page_size}_bs${batch_size}_${REPEAT_TAG}" \
      trtllm_mha "$page_size" "$batch_size"
  done
  for page_size in 1 16 128; do
    run_cell "FI_ps${page_size}_bs${batch_size}_${REPEAT_TAG}" \
      flashinfer "$page_size" "$batch_size"
  done
  for page_size in 1 16 128; do
    run_cell "T_ps${page_size}_bs${batch_size}_${REPEAT_TAG}" \
      triton "$page_size" "$batch_size"
  done
done

echo "complete $(date '+%F %T')" > "$OUTPUT_DIR/${REPEAT_TAG}.done"
