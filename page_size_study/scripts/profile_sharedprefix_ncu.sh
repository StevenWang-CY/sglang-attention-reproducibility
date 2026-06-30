#!/usr/bin/env bash
# =============================================================================
# profile_sharedprefix_ncu.sh — ncu hardware-counter profile of the decode-
#   attention kernel, SHARED vs DISTINCT KV, to explain WHY page_size=1 loses
#   with a shared prefix but not in a true batch (Report 7).
#
# Controlled variable: --kv-mode {shared,distinct} in bench_xqa.py.
#   shared   = ONE physical KV copy re-read by all B seqs (shared prefix) -> reuse
#   distinct = B independent KV copies (true batch) -> no reuse, DRAM-streamed
# Within each: page_size {1,128}. Cells include bs8x39k (the FlashInfer penalty
#   regime) — reachable in BOTH modes here because the microbench has no model
#   weights (distinct bs8x39k ~1.3GB fits; the engine's true-batch never could).
#
# WHY ROOT: ncu needs GPU perf-counter access (RmProfilingAdminOnly=1) -> run via
#   sudo.  Audit: every privileged action is appended to $SUDO_LOG; outputs are
#   chowned back to the invoking user. Isolation: bench_xqa.py --single wraps ONE
#   launch in cudaProfilerStart/Stop; --profile-from-start off captures just it.
#
# Usage (on phastform):  sudo -E bash profile_sharedprefix_ncu.sh
# =============================================================================
set -u
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
BENCH=${BENCH:-$TARGET_HOME/sglang_log/bench_xqa.py}
OUTDIR=${OUTDIR:-$TARGET_HOME/sglang_log/offline_batch_results/sharedprefix_profile}
SUDO_LOG=${SUDO_LOG:-$TARGET_HOME/sglang_log/page_size_study/report_7_sharedprefix_mechanism/SUDO_CHANGES.md}
WARMUP=${WARMUP:-5}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export HOME=$TARGET_HOME

BACKEND=${BACKEND:-flashinfer}                       # flashinfer = the page-level wrapper (clean)
PAGES=(${PAGES:-1 128})
CELLS=(${CELLS:-"8 6144" "8 39168" "16 3072"})       # "batch seqlen"
MODES=(${MODES:-shared distinct})

# report-6 metric set (DRAM-bound vs cache-bound discriminators)
METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
l1tex__t_sector_hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_inst_executed_op_global_ld.sum"

umask 022; mkdir -p "$OUTDIR" "$(dirname "$SUDO_LOG")"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "================ profile_sharedprefix_ncu.sh START (report 7) ================"
log "host=$(hostname) uid=$(id -u) (root via sudo for ncu perf counters)"
log "PY=$PY NCU=$NCU BENCH=$BENCH OUTDIR=$OUTDIR BACKEND=$BACKEND"
"$NCU" --version 2>&1 | head -1 | while read l; do log "ncu: $l"; done

idle_gate(){ for i in $(seq 1 60); do
  local n f; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c .||true)
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "${n:-0}" -eq 0 ] && [ "${f:-0}" -ge 8000 ] && return 0
  log "idle_gate wait($i) procs=$n free=${f}MiB"; sleep 15; done; log "idle_gate TIMEOUT"; }

cell(){ local be=$1 mode=$2 ps=$3 bs=$4 sl=$5
  local tag="${be}_${mode}_ps${ps}_bs${bs}_kv${sl}"; local out="$OUTDIR/$tag"
  [ -s "${out}.csv" ] && { log "skip $tag (exists)"; return; }
  idle_gate; log "ncu $tag"
  "$NCU" --target-processes all --profile-from-start off \
         --replay-mode kernel --cache-control all --metrics "$METRICS" \
         -o "$out" --force-overwrite \
         "$PY" "$BENCH" --single --backend-single "$be" --kv-mode "$mode" \
              --page-size "$ps" --batch-size "$bs" --seq-len "$sl" --warmup "$WARMUP" \
         >> "${out}.stdout" 2>&1
  [ $? -ne 0 ] && log "  WARN $tag rc!=0 (see ${tag}.stdout)"
  "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>/dev/null
}

for c in "${CELLS[@]}"; do set -- $c; B=$1; L=$2
  for mode in "${MODES[@]}"; do for ps in "${PAGES[@]}"; do cell "$BACKEND" "$mode" "$ps" "$B" "$L"; done; done
done
chown -R "$TARGET_USER":"$TARGET_USER" "$OUTDIR" 2>/dev/null || true
log "================ profile_sharedprefix_ncu.sh DONE; chowned outputs to $TARGET_USER ================"
