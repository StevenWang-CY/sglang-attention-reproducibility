#!/usr/bin/env bash
# ncu profile of the FlashInfer MLA decode kernel, contig vs scatter kv_indices, to pin
# the mechanism of MLA's ~3-4% scatter cost (DRAM-bound? sectors/request? L2/TLB?).
# Run the checked-out script with sudo; override PY/NCU/OUTDIR as needed.
# Never writes the sudo password; logs each privileged action to provenance.md.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NCU=${NCU:-/usr/local/cuda-12.8/bin/ncu}
BENCH=${BENCH:-$SCRIPT_DIR/bench_mla_pagecost.py}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/pagecost_profile}
SUDO_LOG=$OUTDIR/provenance.md
WARMUP=8
CACHE=${CACHE:-all}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARGET_HOME/.cache}"

METRICS="gpu__time_duration.sum,dram__bytes_op_read.sum,dram__throughput.avg.pct_of_peak_sustained_elapsed,lts__t_sector_hit_rate.pct,lts__t_sectors.sum,lts__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sector_hit_rate.pct,l1tex__t_sectors.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,sm__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active"

umask 022
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "=== profile_mla_pagecost_ncu.sh start (CACHE=$CACHE, ncu=$NCU, user=$TARGET_USER) ==="

idle_gate(){
  for i in $(seq 1 40); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge 12000 ]; then return 0; fi
    echo "  idle_gate wait ($i): procs=$procs free=${free}MiB"; sleep 15
  done
  echo "  idle_gate TIMEOUT"; return 1
}

# cells: "B L H"
CELLS=("8 131072 16" "8 131072 128" "1 524288 16")
PATS=(contig scatter)

for cell in "${CELLS[@]}"; do
  set -- $cell; B=$1; L=$2; H=$3
  for PAT in "${PATS[@]}"; do
    tag="mla_h${H}_${PAT}_${CACHE}_bs${B}_kv${L}"
    out="$OUTDIR/$tag"
    [ -f "${out}.csv" ] && { echo "skip $tag (exists)"; continue; }
    idle_gate || exit 1
    log "ncu $tag : --cache-control $CACHE on $BENCH --single h=$H $PAT B$B L$L"
    "$NCU" --target-processes all --profile-from-start off \
           --replay-mode kernel --cache-control "$CACHE" --metrics "$METRICS" \
           -o "$out" --force-overwrite \
           "$PY" "$BENCH" --single --pattern "$PAT" --batch-size "$B" --seq-len "$L" \
                 --num-heads-single "$H" --warmup "$WARMUP" \
           >> "${out}.stdout" 2>&1
    "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>>"${out}.stdout"
    echo "wrote ${out}.csv"
  done
done

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR"
log "=== profile_mla_pagecost_ncu.sh done; chown -R $TARGET_USER $OUTDIR ==="
