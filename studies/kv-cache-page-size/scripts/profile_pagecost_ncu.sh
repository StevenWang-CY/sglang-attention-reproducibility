#!/usr/bin/env bash
# ncu profile of the FlashInfer decode KERNEL with CONTIGUOUS vs SCATTERED kv_indices,
# at a deeply DRAM-bound distinct footprint, to confirm WHY scatter is gather-robust
# (kernel stays DRAM-bound; each token's 4 KB KV read is internally coalesced).
# Run the checked-out script with sudo; override PY/NCU/OUTDIR as needed.
# Never writes the sudo password anywhere; logs each privileged action to provenance.md.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NCU=${NCU:-/usr/local/cuda-12.8/bin/ncu}
BENCH=${BENCH:-$SCRIPT_DIR/bench_pagecost.py}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/pagecost_profile}
SUDO_LOG=$OUTDIR/provenance.md
WARMUP=8
CACHE=${CACHE:-all}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

METRICS="gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__t_sectors.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

umask 022
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "=== profile_pagecost_ncu.sh start (CACHE=$CACHE, ncu=$NCU, user=$TARGET_USER) ==="

idle_gate(){
  for i in $(seq 1 40); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge 12000 ]; then return 0; fi
    echo "  idle_gate wait ($i): procs=$procs free=${free}MiB"; sleep 15
  done
  echo "  idle_gate TIMEOUT"; return 1
}

CELLS=("8 32768" "8 131072")
PATS=(contig scatter)

for cell in "${CELLS[@]}"; do
  set -- $cell; B=$1; L=$2
  for PAT in "${PATS[@]}"; do
    tag="flashinfer_${PAT}_${CACHE}_bs${B}_kv${L}"
    out="$OUTDIR/$tag"
    [ -f "${out}.csv" ] && { echo "skip $tag (exists)"; continue; }
    idle_gate || exit 1
    log "ncu $tag : $NCU --cache-control $CACHE on $BENCH --single --pattern $PAT -b $B -l $L"
    "$NCU" --target-processes all --profile-from-start off \
           --replay-mode kernel --cache-control "$CACHE" --metrics "$METRICS" \
           -o "$out" --force-overwrite \
           "$PY" "$BENCH" --single --pattern "$PAT" --batch-size "$B" --seq-len "$L" --warmup "$WARMUP" \
           >> "${out}.stdout" 2>&1
    "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>>"${out}.stdout"
    echo "wrote ${out}.csv"
  done
done

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR"
log "=== profile_pagecost_ncu.sh done; chown -R $TARGET_USER $OUTDIR ==="
