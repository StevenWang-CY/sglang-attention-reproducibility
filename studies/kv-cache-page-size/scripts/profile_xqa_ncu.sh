#!/usr/bin/env bash
# =============================================================================
# profile_xqa_ncu.sh  —  ncu hardware-counter profile of decode-attention
#   kernels (XQA / FlashInfer) vs page_size x batch x seq_len.  Report 6.
#
# WHY ROOT: ncu needs GPU perf-counter access; CIS boxes set
#   RmProfilingAdminOnly=1, so ncu must run as root.  Run the whole script
#   with sudo:
#       sudo -E bash profile_xqa_ncu.sh                 # inherits PY/NCU/... env
#   or override paths explicitly:
#       sudo PY=/home/wangcy07/venvs/bench_sglang/bin/python \
#            NCU=/opt/nvidia/nsight-compute/2025.1.1/ncu \
#            bash profile_xqa_ncu.sh
#
# AUDIT: every privileged action (and the only persistent footprint —
#   the .ncu-rep/.csv outputs and the flashinfer JIT cache) is appended to
#   $SUDO_LOG so the PI's "record what your sudo changed" instruction is met.
#   Outputs are chowned back to the invoking user at the end.
#
# ISOLATION: bench_xqa.py --single wraps ONE attention launch in
#   cudaProfilerStart/Stop; we pass --profile-from-start off so ncu captures
#   exactly that kernel regardless of its name.
# =============================================================================
set -u

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")

PY=${PY:-$(ls "$TARGET_HOME"/venvs/xqa_bench/bin/python "$TARGET_HOME"/venvs/bench_sglang/bin/python 2>/dev/null | head -1)}
# auto-pick the NEWEST Nsight Compute (older ones fail "LibraryNotLoaded" on new drivers)
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
BENCH=${BENCH:-$SCRIPT_DIR/bench_xqa.py}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/xqa_profile}
SUDO_LOG=${SUDO_LOG:-$STUDY_ROOT/reports/06-xqa-tma-profile/provenance.md}
WARMUP=${WARMUP:-5}
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda-12.8}
# venv bin (ninja for flashinfer JIT) + cuda bin (nvcc) on PATH
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
# Keep JIT/cache output out of root's cache without changing HOME.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

# ---- experiment matrix (override via env) -----------------------------------
XQA_PAGES=(${XQA_PAGES:-16 32 64 128})            # XQA supports only >=16
FI_PAGES=(${FI_PAGES:-1 16 32 64 128})            # FlashInfer also does ps1
BATCHES=(${BATCHES:-1 8 32 64})
SEQS=(${SEQS:-1024 4096 16384})
FULLSET_CELLS=(${FULLSET_CELLS:-"xqa 64 8 4096" "flashinfer 64 8 4096" "xqa 64 8 16384" "flashinfer 64 8 16384"})

# ---- curated metrics: TMA-vs-LSU contrast + memory + occupancy --------------
# Verified available on RTX 5060 Ti (sm120) via ncu 2025.3.1 --query-metrics.
# The TMA load-bytes counter is the direct "does this kernel use TMA" signal.
METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
l1tex__t_sector_hit_rate.pct,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__sass_inst_executed_op_global_ld.sum,\
l1tex__m_xbar2l1tex_read_bytes_mem_global_op_tma_ld.sum,\
l1tex__m_xbar2l1tex_read_sectors_mem_global_op_tma_ld.sum,\
l1tex__m_l1tex2xbar_req_cycles_active_op_tma.sum,\
l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ldgsts.sum"

umask 022   # root-created outputs world-readable so the user can poll/rsync before chown
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG" ; }

log "================ profile_xqa_ncu.sh START ================"
log "host=$(hostname) uid=$(id -u) (running as root via sudo for ncu perf counters)"
log "TARGET_USER=$TARGET_USER  PY=$PY  NCU=$NCU  OUTDIR=$OUTDIR"
"$NCU" --version 2>&1 | head -1 | while read l; do log "ncu: $l"; done
nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader | while read l; do log "gpu: $l"; done

# ---- 0) discover TMA / async-copy metric names on this ncu+GPU --------------
"$NCU" --query-metrics 2>/dev/null | grep -iE 'op_tma|tma_ld|bulk|cp_async|ldgsts|tensor_memory' \
   > "$OUTDIR/tma_metric_names.txt" 2>/dev/null || true
log "dumped TMA/async metric names -> tma_metric_names.txt (metric set is explicit; see METRICS)"

idle_gate(){   # wait until GPU is free of foreign procs and has memory headroom
  for i in $(seq 1 60); do
    local nproc free
    nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "${nproc:-0}" -eq 0 ] && [ "${free:-0}" -ge 8000 ] && return 0
    log "idle_gate wait ($i): procs=$nproc free=${free}MiB"; sleep 15
  done
  log "idle_gate TIMEOUT — proceeding anyway"; return 0
}

run_cell(){   # backend page batch seq
  local be=$1 ps=$2 bs=$3 sl=$4
  local tag="${be}_ps${ps}_bs${bs}_kv${sl}"
  local out="$OUTDIR/$tag"
  [ -s "${out}.csv" ] && { log "skip $tag (exists)"; return; }
  log "ncu cell $tag"
  "$NCU" --target-processes all --profile-from-start off \
         --replay-mode kernel --cache-control all \
         --metrics "$METRICS" \
         -o "$out" --force-overwrite \
         "$PY" "$BENCH" --single --backend-single "$be" \
              --page-size "$ps" --batch-size "$bs" --seq-len "$sl" --warmup "$WARMUP" \
         >> "${out}.stdout" 2>&1
  local rc=$?
  [ $rc -ne 0 ] && log "  WARN $tag exited rc=$rc (see ${tag}.stdout)"
  # export a parseable wide CSV from the report (the -o report holds the data;
  # --log-file would only capture ncu's banner). --import needs no GPU/root.
  "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>/dev/null
}

run_fullset(){  # backend page batch seq  -> complete --set full report (.ncu-rep)
  local be=$1 ps=$2 bs=$3 sl=$4
  local tag="full_${be}_ps${ps}_bs${bs}_kv${sl}"
  local out="$OUTDIR/$tag"
  [ -s "${out}.ncu-rep" ] && { log "skip $tag (exists)"; return; }
  log "ncu FULLSET $tag"
  "$NCU" --target-processes all --profile-from-start off --set full \
         -o "$out" --force-overwrite \
         "$PY" "$BENCH" --single --backend-single "$be" \
              --page-size "$ps" --batch-size "$bs" --seq-len "$sl" --warmup "$WARMUP" \
         >> "${out}.stdout" 2>&1 || log "  WARN $tag rc=$?"
}

# ---- 1) curated-metric sweep -----------------------------------------------
idle_gate
for sl in "${SEQS[@]}"; do
  for bs in "${BATCHES[@]}"; do
    for ps in "${XQA_PAGES[@]}"; do run_cell xqa        "$ps" "$bs" "$sl"; done
    for ps in "${FI_PAGES[@]}";  do run_cell flashinfer "$ps" "$bs" "$sl"; done
  done
done

# ---- 2) full-set reports on representative cells ----------------------------
for cell in "${FULLSET_CELLS[@]}"; do run_fullset $cell; done

# ---- 3) hand outputs back to the user + record footprint -------------------
chown -R "$TARGET_USER":"$TARGET_USER" "$OUTDIR" 2>/dev/null && log "chowned $OUTDIR -> $TARGET_USER"
[ -d "$TARGET_HOME/.cache/flashinfer" ] && chown -R "$TARGET_USER":"$TARGET_USER" "$TARGET_HOME/.cache/flashinfer" 2>/dev/null \
   && log "chowned flashinfer JIT cache -> $TARGET_USER"
log "PERSISTENT FOOTPRINT: wrote *.ncu-rep/*.csv under $OUTDIR; flashinfer JIT cache under $TARGET_HOME/.cache/flashinfer (if recompiled). No sudoers/module/driver changes made by this script."
log "================ profile_xqa_ncu.sh DONE ================"
