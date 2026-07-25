#!/usr/bin/env bash
# Report 16 deep-dive — P1 XQA transition sweep (page x kv-mode x seq-len) + full metric set incl.
# warp-stall reasons + occupancy (the off-wall latency-bound signature). Root (perf counters).
#   echo <pw> | sudo -S -E bash profile_report_16_mechanism.sh
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER="${SUDO_USER:-$(id -un)}"
TARGET_HOME="$(eval echo "~$TARGET_USER")"
PY=${PY:-$TARGET_HOME/venvs/sglang_0b2e/bin/python}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
BENCH=${BENCH:-$SCRIPT_DIR/bench_xqa.py}
OUT=${OUT:-$STUDY_ROOT/data/raw/report-16-mechanism-sweep}; mkdir -p "$OUT"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export PYTHONPATH="${SGLANG_PYTHONPATH:-$TARGET_HOME/sglang_git/python}${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
# scalar metric set (all verified to resolve on this ncu 2025.3.1 / sm120)
M="gpu__time_duration.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__sass_inst_executed_op_global_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active"
cell(){ # km ps bs sl
  local km=$1 ps=$2 bs=$3 sl=$4
  local tag="xqa_${km}_ps${ps}_bs${bs}_kv${sl}"; local o="$OUT/$tag"
  [ -s "${o}.csv" ] && { echo "skip $tag"; return; }
  "$NCU" --target-processes all --profile-from-start off --replay-mode kernel --cache-control all \
    --metrics "$M" -o "$o" --force-overwrite \
    "$PY" "$BENCH" --single --backend-single xqa --page-size "$ps" --batch-size "$bs" \
        --seq-len "$sl" --kv-mode "$km" --page-order contig --warmup 5 >> "${o}.stdout" 2>&1
  "$NCU" --import "${o}.ncu-rep" --page raw --csv > "${o}.csv" 2>/dev/null
  echo "done $tag"
}
deepcell(){ # km ps bs sl — full section set (stall reasons, occupancy, memory chart)
  local km=$1 ps=$2 bs=$3 sl=$4
  local tag="deep_xqa_${km}_ps${ps}_bs${bs}_kv${sl}"; local o="$OUT/$tag"
  [ -s "${o}.csv" ] && { echo "skip $tag"; return; }
  "$NCU" --target-processes all --profile-from-start off --replay-mode kernel --cache-control all \
    --set full -o "$o" --force-overwrite \
    "$PY" "$BENCH" --single --backend-single xqa --page-size "$ps" --batch-size "$bs" \
        --seq-len "$sl" --kv-mode "$km" --page-order contig --warmup 5 >> "${o}.stdout" 2>&1
  "$NCU" --import "${o}.ncu-rep" --page raw --csv > "${o}.csv" 2>/dev/null
  echo "deep done $tag"
}
MODE=${1:-verify}
if [ "$MODE" = "verify" ]; then
  cell shared 16 32 8192          # single cell to confirm metric names resolve
elif [ "$MODE" = "deep" ]; then
  # P2 pivotal cells: shared vs distinct x ps16 vs ps64, bs32/kv8192 (stall-reason proof)
  deepcell shared 16 32 8192; deepcell shared 64 32 8192
  deepcell distinct 16 32 8192; deepcell distinct 64 32 8192
else
  for sl in 2048 4096 8192 16384; do
    for km in shared distinct; do
      for ps in 16 32 64 128; do cell $km $ps 32 $sl; done
    done
  done
  cell shared 16 8 8192; cell shared 64 8 8192   # bs8 batch-effect spot check
  cell distinct 16 8 8192; cell distinct 64 8 8192
fi
chown -R "$TARGET_USER" "$OUT" 2>/dev/null || true
echo R16MECH_${MODE}_DONE
