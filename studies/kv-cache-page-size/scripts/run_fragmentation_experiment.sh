#!/usr/bin/env bash
# Report 15 constructed fragmentation arms. Sequential, one exclusive GPU.
set -uo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/sglang_0b2e/bin/python}"
MODEL_PATH="${MODEL_PATH:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
DRIVER="${DRIVER:-$SCRIPT_DIR/drive_fragmentation_experiment.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/report-15-constructed-fragmentation}"
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

run_cell() {
  local tag=$1
  shift
  local output_file="$OUTPUT_DIR/fr_${tag}.log"
  [[ -s "$output_file" ]] && { echo "skip $tag"; return; }
  timeout --signal=TERM --kill-after=20s 900 \
    "$PYTHON_BIN" "$DRIVER" --model-path "$MODEL_PATH" "$@" \
    > "$output_file" 2>&1
  echo "$tag exit=$? $(date '+%F %T')" >> "$OUTPUT_DIR/progress.txt"
}

# Graph measurement arms. The pool and churn settings are defined by the driver defaults.
run_cell T_ps1_frag    --backend triton     --page-size 1   --frag
run_cell T_ps128_frag  --backend triton     --page-size 128 --frag
run_cell T_ps1_fresh   --backend triton     --page-size 1
run_cell T_ps128_fresh --backend triton     --page-size 128
run_cell FI_ps1_frag   --backend flashinfer --page-size 1   --frag
run_cell FI_ps128_frag --backend flashinfer --page-size 128 --frag

# Eager verification arms with retained index-contiguity dumps.
SGLANG_DUMP_KVIDX="$OUTPUT_DIR/fr_kvidx_frag.json" \
  run_cell T_ps1_fragver --backend triton --page-size 1 --frag --eager \
    --mem-fraction-static 0.40 --churn-waves 2 --churn-gen 1024 --gen 12 --rounds 1
SGLANG_DUMP_KVIDX="$OUTPUT_DIR/fr_kvidx_fresh.json" \
  run_cell T_ps1_freshver --backend triton --page-size 1 --eager \
    --mem-fraction-static 0.40 --gen 12 --rounds 1

echo "complete $(date '+%F %T')" > "$OUTPUT_DIR/run.done"
