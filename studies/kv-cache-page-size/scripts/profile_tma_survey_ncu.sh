#!/usr/bin/env bash
# Profile the report-6 stack-wide TMA surveys with Nsight Compute.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(eval echo "~$TARGET_USER")"

PYTHON_BIN="${PYTHON_BIN:-$TARGET_HOME/venvs/bench_sglang/bin/python}"
NCU_BIN="${NCU_BIN:-$(find /opt/nvidia/nsight-compute -type f -name ncu 2>/dev/null | sort -V | tail -1)}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/xqa_profile/tma-survey}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
METRICS="${METRICS:-gpu__time_duration.sum,l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$NCU_BIN" ]] || { echo "Nsight Compute not executable: $NCU_BIN" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
export CUDA_HOME
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

profile_survey() {
  local name=$1
  local survey="$SCRIPT_DIR/$name.py"
  local output_base="$OUTPUT_DIR/$name"
  [[ -f "$survey" ]] || { echo "Survey not found: $survey" >&2; exit 2; }
  [[ -s "${output_base}.csv" ]] && { echo "skip $name"; return; }

  "$NCU_BIN" --target-processes all --profile-from-start off --replay-mode kernel \
    --metrics "$METRICS" -o "$output_base" --force-overwrite \
    "$PYTHON_BIN" "$survey" > "${output_base}.stdout" 2>&1
  "$NCU_BIN" --import "${output_base}.ncu-rep" --page raw --csv \
    > "${output_base}.csv" 2>> "${output_base}.stdout"
}

profile_survey tma_survey
profile_survey tma_survey2

if [[ $(id -u) -eq 0 ]]; then
  chown -R "$TARGET_USER" "$OUTPUT_DIR" 2>/dev/null || true
fi
echo "TMA survey profiling complete"
