#!/usr/bin/env bash
# =============================================================================
# profile_smallkv_ncu.sh — ncu hardware-counter profile of the FlashInfer decode
#   kernel in DISTINCT (true-batch) KV mode, swept across KV FOOTPRINT
#   (L2-resident -> DRAM-bound), to test the premise "shrink KV -> escape the
#   DRAM-bandwidth bottleneck" (Report 8). Page {1,128} -> is the page pattern
#   exposed once off the ceiling?
#
# Two cache regimes (both informative):
#   CACHE=all  (default) — flush caches each replay = COLD = faithful to real
#              steady-state decode (28 layers thrash the 32 MB L2, so each layer
#              re-streams its KV from DRAM every step; small KV just streams less).
#   CACHE=none — warm/resident = what cache-residency WOULD give (the regime the
#              warm 100-iter latency microbench actually measures).
#
# WHY ROOT: ncu needs GPU perf-counter access (RmProfilingAdminOnly=1). Every
#   privileged action is appended to $SUDO_LOG; outputs are chowned back. Isolation:
#   bench_xqa.py --single wraps ONE launch in cudaProfilerStart/Stop; -from-start off.
# Usage (phastform):  sudo -E CACHE=all  bash profile_smallkv_ncu.sh
#                     sudo -E CACHE=none bash profile_smallkv_ncu.sh
# =============================================================================
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
BENCH=${BENCH:-$SCRIPT_DIR/bench_xqa.py}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/smallkv_profile}
SUDO_LOG=${SUDO_LOG:-$STUDY_ROOT/reports/08-small-kv-true-batch/provenance.md}
WARMUP=${WARMUP:-8}
CACHE=${CACHE:-all}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

BACKEND=${BACKEND:-flashinfer}
PAGES=(${PAGES:-1 128})
# B8 footprint sweep: per-layer KV = B*L*4096 B; L2 = 32 MB <=> B*L ~ 8192 tok
CELLS=(${CELLS:-"8 256" "8 512" "8 1024" "8 2048" "8 6144" "8 16384"})   # 8MB..512MB
MODES=(${MODES:-distinct})

METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,l1tex__throughput.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_inst_executed_op_global_ld.sum"

umask 022; mkdir -p "$OUTDIR" "$(dirname "$SUDO_LOG")"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "================ profile_smallkv_ncu.sh START (report 8, CACHE=$CACHE) ================"
log "host=$(hostname) uid=$(id -u) (root via sudo for ncu perf counters)"
log "PY=$PY NCU=$NCU BENCH=$BENCH OUTDIR=$OUTDIR BACKEND=$BACKEND CACHE=$CACHE"
"$NCU" --version 2>&1 | head -1 | while read l; do log "ncu: $l"; done

idle_gate(){ for i in $(seq 1 60); do
  local n f; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c .||true)
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "${n:-0}" -eq 0 ] && [ "${f:-0}" -ge 8000 ] && return 0
  log "idle_gate wait($i) procs=$n free=${f}MiB"; sleep 15; done; log "idle_gate TIMEOUT"; }

cell(){ local be=$1 mode=$2 ps=$3 bs=$4 sl=$5
  local tag="${be}_${mode}_${CACHE}_ps${ps}_bs${bs}_kv${sl}"; local out="$OUTDIR/$tag"
  [ -s "${out}.csv" ] && { log "skip $tag (exists)"; return; }
  idle_gate; log "ncu $tag"
  "$NCU" --target-processes all --profile-from-start off \
         --replay-mode kernel --cache-control "$CACHE" --metrics "$METRICS" \
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
log "================ profile_smallkv_ncu.sh DONE (CACHE=$CACHE); chowned to $TARGET_USER ================"
