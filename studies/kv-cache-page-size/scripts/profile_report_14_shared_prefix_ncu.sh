#!/usr/bin/env bash
# Report 14 -- ncu profile of the Triton decode kernel's synthetic scattered-kv-index
# control in the L2-reuse (shared-prefix) regime, versus FlashInfer
# vs distinct (on-wall).  Crosses {backend triton|flashinfer} x {reuse shared|distinct} x
# {pattern contig|block128|scatter} on bench_pagecost.py --single.
#
# The decisive counter is l1tex ..._sectors_per_request..._op_ld.ratio (sectors/request).
# This bounds synthetic layout sensitivity; it does not explain the retracted report-3 +25% metric.
# scatter-vs-block128 == the engine's real ps1-vs-ps128 (page_size lives only in the allocator;
# the experiment build's Triton decode kernel is page-size-agnostic).
#
# Env: BACKENDS "triton flashinfer"  REUSES "shared distinct"  PATTERNS "contig block128 scatter"
#      CELLS "8 6144|16 6144|32 4096"  PAGES "1"  QH/KH heads  CACHE all|none  WARMUP n
# Run (root for ncu; password piped, NEVER written to disk):
#   ssh phastform 'echo PW | sudo -S -E BACKENDS="triton flashinfer" REUSES="shared distinct" \
#      CELLS="8 6144|16 6144|32 4096" bash <checkout>/studies/kv-cache-page-size/scripts/profile_report_14_shared_prefix_ncu.sh'
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/report-14-shared-prefix-page-kernel-rtx5060ti}
SUDO_LOG=${SUDO_LOG:-$OUTDIR/provenance.md}
BENCH=${BENCH:-$SCRIPT_DIR/bench_pagecost.py}
WARMUP=${WARMUP:-20}          # >= JIT compile + autotune before the profiled launch
CACHE=${CACHE:-all}           # cold = faithful; intra-launch reuse is preserved (report 12)
BACKENDS=${BACKENDS:-"triton flashinfer"}
REUSES=${REUSES:-"shared distinct"}
PATTERNS=${PATTERNS:-"contig block128 scatter"}
PAGES=${PAGES:-1}
QH=${QH:-16}
KH=${KH:-8}
CELLS_RAW=${CELLS:-"8 6144|16 6144|32 4096"}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor"

umask 022
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "=== profile_report_14_shared_prefix_ncu.sh start (BACKENDS='$BACKENDS' REUSES='$REUSES' PATTERNS='$PATTERNS' CACHE=$CACHE QH=$QH KH=$KH ncu=$NCU user=$TARGET_USER) ==="

idle_gate(){
  for i in $(seq 1 60); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge 12000 ]; then return 0; fi
    echo "  idle_gate wait ($i): procs=$procs free=${free}MiB"; sleep 15
  done
  echo "  idle_gate TIMEOUT"; return 1
}

run_ncu(){ # $1 tag, rest = target cmd
  local tag=$1; shift
  local out="$OUTDIR/$tag"
  [ -s "${out}.csv" ] && { echo "skip $tag (exists)"; return 0; }
  idle_gate || exit 1
  log "ncu $tag"
  "$NCU" --target-processes all --profile-from-start off \
         --replay-mode kernel --cache-control "$CACHE" --metrics "$METRICS" \
         -o "$out" --force-overwrite "$@" >> "${out}.stdout" 2>&1
  "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>>"${out}.stdout"
  if [ -s "${out}.csv" ]; then echo "wrote ${out}.csv"; else echo "FAILED $tag (see ${out}.stdout)"; fi
}

IFS='|' read -r -a CELLS <<< "$CELLS_RAW"
for be in $BACKENDS; do
 for reuse in $REUSES; do
  for cell in "${CELLS[@]}"; do
    set -- $cell; B=$1; L=$2
    for ps in $PAGES; do
     for pat in $PATTERNS; do
      run_ncu "pc_${be}_${reuse}_ps${ps}_${pat}_h${QH}k${KH}_bs${B}_kv${L}" \
        "$PY" "$BENCH" --single --backend "$be" --reuse "$reuse" --page-size "$ps" \
              --pattern "$pat" --num-q-heads "$QH" --num-kv-heads "$KH" \
              --batch-size "$B" --seq-len "$L" --warmup "$WARMUP"
     done
    done
  done
 done
done

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR"
log "=== profile_report_14_shared_prefix_ncu.sh done (h${QH}k${KH}); chown -R $TARGET_USER $OUTDIR ==="
