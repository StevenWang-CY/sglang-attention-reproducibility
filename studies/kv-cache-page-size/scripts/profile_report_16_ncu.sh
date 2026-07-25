#!/usr/bin/env bash
# Report 16 C1: profile XQA and FlashInfer decode in shared/distinct regimes.
# Run as root when performance-counter permissions require it.
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(eval echo "~$TARGET_USER")"
PYTHON_BIN="${PYTHON_BIN:-$TARGET_HOME/venvs/sglang_0b2e/bin/python}"
NCU_BIN="${NCU_BIN:-$(find /opt/nvidia/nsight-compute -type f -name ncu 2>/dev/null | sort -V | tail -1)}"
BENCHMARK="${BENCHMARK:-$SCRIPT_DIR/bench_xqa.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/report-16-ncu-primary}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
VENV_BIN="${VENV_BIN:-$(dirname "$PYTHON_BIN")}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-$TARGET_HOME/sglang_git/python}"

[[ -x "$PYTHON_BIN" ]] || { echo "Python not executable: $PYTHON_BIN" >&2; exit 2; }
[[ -x "$NCU_BIN" ]] || { echo "Nsight Compute not executable: $NCU_BIN" >&2; exit 2; }
[[ -f "$BENCHMARK" ]] || { echo "Benchmark not found: $BENCHMARK" >&2; exit 2; }

mkdir -p "$OUTPUT_DIR"
export CUDA_HOME
export PATH="$VENV_BIN:$CUDA_HOME/bin:$PATH"
export PYTHONPATH="$SGLANG_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,l1tex__t_sector_hit_rate.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_inst_executed_op_global_ld.sum,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum"

profile_cell() {
  local backend=$1 page_size=$2 batch_size=$3 sequence_length=$4 kv_mode=$5 page_order=$6
  local tag="${backend}_${kv_mode}_${page_order}_ps${page_size}_bs${batch_size}_kv${sequence_length}"
  local output_base="$OUTPUT_DIR/$tag"

  [[ -s "${output_base}.csv" ]] && { echo "skip $tag"; return; }
  echo "=== ncu $tag ==="
  "$NCU_BIN" --target-processes all --profile-from-start off --replay-mode kernel \
    --cache-control all --metrics "$METRICS" -o "$output_base" --force-overwrite \
    "$PYTHON_BIN" "$BENCHMARK" --single --backend-single "$backend" \
    --page-size "$page_size" --batch-size "$batch_size" --seq-len "$sequence_length" \
    --kv-mode "$kv_mode" --page-order "$page_order" --warmup 5 \
    >> "${output_base}.stdout" 2>&1
  "$NCU_BIN" --import "${output_base}.ncu-rep" --page raw --csv \
    > "${output_base}.csv" 2>/dev/null
}

# Shared/off-wall regime: XQA page sensitivity and order robustness at B32/L8192.
profile_cell xqa 16 32 8192 shared contig
profile_cell xqa 32 32 8192 shared contig
profile_cell xqa 64 32 8192 shared contig
profile_cell xqa 128 32 8192 shared contig
profile_cell xqa 16 32 8192 shared shuffle
profile_cell xqa 128 32 8192 shared shuffle
profile_cell flashinfer 16 32 8192 shared contig

# Distinct/on-wall regime.
profile_cell xqa 16 8 4096 distinct contig
profile_cell xqa 128 8 4096 distinct contig
profile_cell xqa 16 8 4096 distinct shuffle

if [[ $(id -u) -eq 0 ]]; then
  chown -R "$TARGET_USER" "$OUTPUT_DIR" 2>/dev/null || true
fi
echo "Report 16 C1 profiling complete"
