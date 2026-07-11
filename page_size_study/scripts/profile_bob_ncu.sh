#!/usr/bin/env bash
# Report 13 — ncu INSIDE the engine: profile the FlashInfer decode-attention kernel
# (BatchDecodeWithPagedKVCacheKernel) during real `sglang.bench_one_batch` decode steps,
# to show the constructed off-wall cell (Qwen2.5-3B GQA-2, B2/L1024) has DRAM% < 50
# *in the literal bench_one_batch*, vs Qwen3-VL-2B (GQA-8) at ~80% and a DRAM-side
# reference cell at ~95%.
#
# Method: eager mode (--disable-cuda-graph) so every decode step launches the kernel
# visibly; -k regex filters decode-attention launches only (prefill/merge/GEMM excluded);
# with --output-len 16 the warmup run does 15 decode steps (bench_one_batch warmup uses
# min(32, output_len)) and the measured run 15 more, n_layers launches per step, so
# --launch-skip 20*n_layers lands inside the MEASURED phase; --launch-count 2*n_layers
# profiles two full decode steps. --cache-control all (cold) per study convention;
# per-launch counters are medianed by the analyzer (multi-row CSV).
#
# WHY ROOT: RmProfilingAdminOnly=1. HOME pinned to the user; outputs + caches chowned back.
# Run: ssh phastform 'echo PW | sudo -S -E bash ~/sglang_log/profile_bob_ncu.sh'
set -u
TARGET_USER=${SUDO_USER:-$(id -un)}
TARGET_HOME=$(eval echo "~$TARGET_USER")
PY=$TARGET_HOME/venvs/bench_sglang/bin/python
NCU=${NCU:-$(ls -d /opt/nvidia/nsight-compute/*/ncu 2>/dev/null | sort -V | tail -1)}
OUTDIR=$TARGET_HOME/sglang_log/offline_batch_results/offwall_profile
SUDO_LOG=$OUTDIR/SUDO_CHANGES.md
OL=16
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export HOME=$TARGET_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Two decode paths (sglang should_use_tensor_core): GQA group<4 -> the CUDA-core decode
# kernel (BatchDecodeWithPagedKVCacheKernel, e.g. Qwen3-VL-2B 16q/8kv); group>=4 -> the
# tensor-core prefill-template kernel (BatchPrefillWithPagedKVCacheKernel, e.g.
# Qwen2.5-3B 16q/2kv). Match both; prefill-phase launches also match, and the skip math
# below still lands inside measured decode for both models (see report §4).
KNAME="regex:BatchDecodeWithPagedKVCacheKernel|BatchPrefillWithPagedKVCacheKernel"
METRICS="gpu__time_duration.sum,\
dram__bytes_op_read.sum,dram__bytes_op_write.sum,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,lts__t_sectors.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sector_hit_rate.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
launch__grid_size,launch__block_size,launch__waves_per_multiprocessor"

umask 022
mkdir -p "$OUTDIR"
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$SUDO_LOG"; }
log "=== profile_bob_ncu.sh start (ncu=$NCU user=$TARGET_USER OL=$OL) ==="

idle_gate(){
  for i in $(seq 1 60); do
    procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [ "${procs:-1}" -eq 0 ] && [ "${free:-0}" -ge 12000 ]; then return 0; fi
    echo "  idle_gate wait ($i): procs=$procs free=${free}MiB"; sleep 15
  done
  echo "  idle_gate TIMEOUT"; return 1
}

# args: mdl path wgb kvkb nlayers B L ps
cell(){ local mdl=$1 path=$2 wgb=$3 kvkb=$4 nl=$5 B=$6 L=$7 ps=$8
  # floor 0.55: eager + no graph leaves VRAM free, and ps1's per-token pool metadata
  # needs headroom beyond the raw KV bytes (mf=0.41 OOM'd the 3B pool at ps1)
  local mf=$(awk -v B=$B -v L=$L -v OL=$OL -v w=$wgb -v kv=$kvkb 'BEGIN{need=B*(L+OL); gb=need*kv*1024/1e9*1.08; mf=(gb+w)/15.47; if(mf<0.55)mf=0.55; if(mf>0.72)mf=0.72; printf "%.2f", mf}')
  local skip=$((20*nl)); local count=$((2*nl))
  local tag="bob_${mdl}_all_ps${ps}_bs${B}_kv${L}"; local out="$OUTDIR/$tag"
  [ -s "${out}.csv" ] && { echo "skip $tag (exists)"; return 0; }
  idle_gate || exit 1
  log "ncu-engine $tag (mf=$mf skip=$skip count=$count layers=$nl)"
  timeout 1200 "$NCU" --target-processes all -k "$KNAME" --launch-skip "$skip" --launch-count "$count" \
         --replay-mode kernel --cache-control all --metrics "$METRICS" \
         -o "$out" --force-overwrite \
         "$PY" -m sglang.bench_one_batch --model-path "$path" --trust-remote-code \
               --batch-size "$B" --input-len "$L" --output-len $OL --page-size "$ps" \
               --attention-backend flashinfer --disable-cuda-graph \
               --mem-fraction-static "$mf" --run-name "$tag" \
               --result-filename "$OUTDIR/$tag.jsonl" >> "${out}.stdout" 2>&1
  "$NCU" --import "${out}.ncu-rep" --page raw --csv > "${out}.csv" 2>>"${out}.stdout"
  if [ -s "${out}.csv" ]; then echo "wrote ${out}.csv"; else echo "FAILED $tag (see ${out}.stdout)"; fi
}

M3=$TARGET_HOME/hf_models/Qwen2.5-3B-Instruct
M2=$TARGET_HOME/hf_models/Qwen3-VL-2B-Instruct

# flagship off-wall cell, both pages, both models + DRAM-side in-engine reference
cell q25_3b  "$M3" 6.2 36  36 2 1024 128
cell q25_3b  "$M3" 6.2 36  36 2 1024 1
cell q3vl_2b "$M2" 4.1 112 28 2 1024 128
cell q3vl_2b "$M2" 4.1 112 28 2 1024 1
cell q25_3b  "$M3" 6.2 36  36 8 4096 128

chown -R "$TARGET_USER:$TARGET_USER" "$OUTDIR" "$TARGET_HOME/.cache" 2>/dev/null || true
log "=== profile_bob_ncu.sh done; chown -R $TARGET_USER $OUTDIR ==="
