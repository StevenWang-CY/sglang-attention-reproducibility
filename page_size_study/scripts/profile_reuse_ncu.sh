#!/usr/bin/env bash
# Report 12 — ncu profile of the FlashInfer decode kernel to show L2-hit is governed by
# the intra-kernel REUSE factor (shared = B-way reuse) not footprint. Adds L2 hit/miss
# SECTOR COUNT metrics so the hit-RATE can be decomposed (kill the "rate-metric quirk").
#
# Env-driven: CELLS="B L ..." (space-separated pairs), MODES="distinct shared", PAGES="128",
#             CACHE=all|none (all=cold=faithful; none=warm=residency artifact).
# Run: ssh phastform 'echo PW | sudo -S -E CACHE=all MODES="distinct shared" \
#        CELLS="1 1024|4 1024|8 1024" PAGES=128 bash ~/sglang_log/profile_reuse_ncu.sh'
#   (CELLS pairs separated by | to survive sudo arg passing; we split on | below.)
# Never writes the sudo password; logs every privileged action to SUDO_CHANGES.md.
set -u
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=$TARGET_HOME/venvs/bench_sglang/bin/python
NCU=/usr/local/cuda-12.8/bin/ncu
BENCH=$TARGET_HOME/sglang_log/bench_xqa.py
OUTDIR=$TARGET_HOME/sglang_log/offline_batch_results/reuse_profile
SUDO_LOG=$OUTDIR/SUDO_CHANGES.md
WARMUP=8
CACHE=${CACHE:-all}
MODES=${MODES:-distinct shared}
PAGES=${PAGES:-128}
# CELLS as "B L" pairs separated by "|" (robust under sudo); default = a tiny smoke set.
CELLS_RAW=${CELLS:-"8 1024|8 8192"}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export HOME=$TARGET_HOME

# report-8 metric set + L2 hit/miss SECTOR COUNTS (verified present on ncu 2025.3.1 sm120)
METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
lts__t_sectors_lookup_hit.sum,lts__t_sectors_lookup_miss.sum,\
lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum,lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active"

umask 022
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "=== profile_reuse_ncu.sh start (CACHE=$CACHE, MODES='$MODES', PAGES='$PAGES', ncu=$NCU, user=$TARGET_USER) ==="

idle_gate(){
  for i in $(seq 1 60); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge 15500 ]; then return 0; fi
    echo "  idle_gate wait ($i): procs=$procs free=${free}MiB"; sleep 15
  done
  echo "  idle_gate TIMEOUT"; return 1
}

IFS='|' read -r -a CELLS <<< "$CELLS_RAW"
BE=flashinfer

for cell in "${CELLS[@]}"; do
  set -- $cell; B=$1; L=$2
  for mode in $MODES; do
    for ps in $PAGES; do
      tag="${BE}_${mode}_${CACHE}_ps${ps}_bs${B}_kv${L}"
      out="$OUTDIR/$tag"
      [ -s "${out}.csv" ] && { echo "skip $tag (exists)"; continue; }
      idle_gate || exit 1
      log "ncu $tag : --cache-control $CACHE  $mode B$B L$L ps$ps"
      "$NCU" --target-processes all --profile-from-start off \
             --replay-mode kernel --cache-control "$CACHE" --metrics "$METRICS" \
             -o "$out" --force-overwrite \
             "$PY" "$BENCH" --single --backend-single "$BE" --kv-mode "$mode" \
                   --page-size "$ps" --batch-size "$B" --seq-len "$L" --warmup "$WARMUP" \
             >> "${out}.stdout" 2>&1
      "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>>"${out}.stdout"
      if [ -s "${out}.csv" ]; then echo "wrote ${out}.csv"; else echo "FAILED $tag (see ${out}.stdout)"; fi
    done
  done
done

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR"
log "=== profile_reuse_ncu.sh done; chown -R $TARGET_USER $OUTDIR ==="
