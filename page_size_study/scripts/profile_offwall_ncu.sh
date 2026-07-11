#!/usr/bin/env bash
# Report 13 — ncu profile of DISTINCT (true-batch / individual-request) decode at SMALL WORK
# (small batch x short context), to CONSTRUCT and map the regime where the decode-attention
# kernel is NOT DRAM-bandwidth-bound (DRAM% < 50 vs ~95 at standard sizes), and to attribute
# page-size / fragmentation costs inside it.
#
# Adds to the report-12 metric set: launch__grid_size, launch__block_size,
# launch__waves_per_multiprocessor (the latency-/occupancy-bound mechanism evidence) and
# l1tex sectors/request (the report-10/11 coalescing-vs-latency attribution instrument).
# Metric set smoke-validated on ncu 2025.3.1 / sm120 (2026-07-11, B2/L1024 cell).
#
# TOOL=xqa      -> bench_xqa.py --single   (native page_size axis; contiguous kv_indices)
# TOOL=pagecost -> bench_pagecost.py --single (sglang-faithful ps1 token index; PATTERNS =
#                  contig|block128|scatter = worst-case ps1 fragmentation)
# Env: CELLS "B L|B L|..."  PAGES "1 128"  MODES "distinct"  CACHE all|none  QH/KH heads
#      PATTERNS "contig scatter" (pagecost only)
# Run: ssh phastform 'echo PW | sudo -S -E CACHE=all TOOL=xqa QH=16 KH=8 \
#        CELLS="2 1024|1 512" PAGES="1 128" bash ~/sglang_log/profile_offwall_ncu.sh'
# Never writes the sudo password; logs every privileged action to SUDO_CHANGES.md.
set -u
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=$TARGET_HOME/venvs/bench_sglang/bin/python
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
OUTDIR=${OUTDIR:-$TARGET_HOME/sglang_log/offline_batch_results/offwall_profile}
SUDO_LOG=${SUDO_LOG:-$TARGET_HOME/sglang_log/offline_batch_results/offwall_profile/SUDO_CHANGES.md}
WARMUP=8
CACHE=${CACHE:-all}
MODES=${MODES:-distinct}
PAGES=${PAGES:-128}
TOOL=${TOOL:-xqa}
QH=${QH:-16}
KH=${KH:-8}
TC=${TC:-0}          # 1 -> --tensor-cores (sglang's kernel for GQA group>=4); tag gets 'tc'
PATTERNS=${PATTERNS:-contig scatter}
CELLS_RAW=${CELLS:-"2 1024|1 512"}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export HOME=$TARGET_HOME

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
log "=== profile_offwall_ncu.sh start (TOOL=$TOOL CACHE=$CACHE QH=$QH KH=$KH MODES='$MODES' PAGES='$PAGES' ncu=$NCU user=$TARGET_USER) ==="

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
TCFLAG=(); TCTAG=""
if [ "$TC" = "1" ]; then TCFLAG=(--tensor-cores); TCTAG="tc"; fi

for cell in "${CELLS[@]}"; do
  set -- $cell; B=$1; L=$2
  if [ "$TOOL" = "xqa" ]; then
    BENCH=$TARGET_HOME/sglang_log/bench_xqa.py
    for mode in $MODES; do for ps in $PAGES; do
      run_ncu "flashinfer${TCTAG}_${mode}_${CACHE}_h${QH}k${KH}_ps${ps}_bs${B}_kv${L}" \
        "$PY" "$BENCH" --single --backend-single flashinfer --kv-mode "$mode" \
              --num-q-heads "$QH" --num-kv-heads "$KH" "${TCFLAG[@]}" \
              --page-size "$ps" --batch-size "$B" --seq-len "$L" --warmup "$WARMUP"
    done; done
  else
    BENCH=$TARGET_HOME/sglang_log/bench_pagecost.py
    for pat in $PATTERNS; do
      run_ncu "pagecost${TCTAG}_${pat}_${CACHE}_h${QH}k${KH}_bs${B}_kv${L}" \
        "$PY" "$BENCH" --single --pattern "$pat" \
              --num-q-heads "$QH" --num-kv-heads "$KH" "${TCFLAG[@]}" \
              --batch-size "$B" --seq-len "$L" --warmup "$WARMUP"
    done
  fi
done

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR"
log "=== profile_offwall_ncu.sh done (TOOL=$TOOL CACHE=$CACHE h${QH}k${KH}); chown -R $TARGET_USER $OUTDIR ==="
