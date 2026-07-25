#!/usr/bin/env bash
# Report 16 deep-dive — P3/P4: ncu the Triton + FlashInfer decode kernels (engine-faithful, via
# bench_pagecost.py) across page LAYOUT {contig=ps1-fresh, block128=ps128, scatter=ps1-frag} x
# reuse {shared,distinct}. Tests H3: their kernel is page-INVARIANT (contig≈block128) -> the ps1
# e2e edge is the engine allocator (report 15), not the attention kernel. Root (perf counters).
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(eval echo "~$TARGET_USER")"
PY=${PY:-$TARGET_HOME/venvs/sglang_0b2e/bin/python}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
BENCH=${BENCH:-$SCRIPT_DIR/bench_pagecost.py}
OUT=${OUT:-$STUDY_ROOT/data/raw/report-16-mechanism-sweep}; mkdir -p "$OUT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="${SGLANG_PYTHONPATH:-$TARGET_HOME/sglang_git/python}${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
M="gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__sass_inst_executed_op_global_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active"
cell(){ # backend reuse pattern bs sl pagesize
  local be=$1 ru=$2 pat=$3 bs=$4 sl=$5 pz=$6
  local tag="${be}_${ru}_${pat}_ps${pz}_bs${bs}_kv${sl}"; local o="$OUT/$tag"
  [ -s "${o}.csv" ] && { echo "skip $tag"; return; }
  "$NCU" --target-processes all --profile-from-start off --replay-mode kernel --cache-control all \
    --metrics "$M" -o "$o" --force-overwrite \
    "$PY" "$BENCH" --single --backend "$be" --reuse "$ru" --pattern "$pat" \
        --page-size "$pz" --batch-size "$bs" --seq-len "$sl" --warmup 5 >> "${o}.stdout" 2>&1
  "$NCU" --import "${o}.ncu-rep" --page raw --csv > "${o}.csv" 2>/dev/null
  echo "done $tag"
}
# cell args: backend reuse pattern BS SL PAGESIZE. Triton page constexpr {1,128}; patterns
# contig(=ps1-fresh) / block128(=ps128) / scatter(=ps1-frag). bs32, L8192.
for ru in shared distinct; do
  for pat in contig block128 scatter; do
    cell triton     $ru $pat 32 8192 1      # ps1 constexpr, layout=pattern
    cell triton     $ru $pat 32 8192 128    # ps128 constexpr
    cell flashinfer $ru $pat 32 8192 1      # FI engine-faithful (plan->1)
  done
done
chown -R "$TARGET_USER" "$OUT" 2>/dev/null || true
echo R16_TRITONFI_DONE
