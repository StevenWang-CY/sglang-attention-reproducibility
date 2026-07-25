#!/usr/bin/env bash
# Report 13 — latency tier (sudo-free, run as user on phastform, detached).
# CUDA-event kernel latency (median of 100, 20 warmup) for:
#   (1) FlashInfer native page ladder {1..128} on the GQA-8 (16q/8kv) small-work grid
#   (2) same ladder on the GQA-2 arm (16q/2kv = Qwen2.5-3B shape; the constructed
#       bs2/L1k off-wall case)
#   (3) XQA ladder {16..128} (XQA rejects page<16 — report 6) on the GQA-8 grid corner
#   (4) bench_pagecost gather: contig/block128/scatter (worst-case ps1 fragmentation)
# 3 rounds, fresh process per sweep, skip-guarded => resumable/paired-in-time per round.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
PY="${PYTHON_BIN:-$HOME/venvs/bench_sglang/bin/python}"
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
OUT="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/offwall_profile}"
mkdir -p "$OUT"

wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c .||true);
  [ "${f:-0}" -ge 12000 ] && [ "${p:-1}" -eq 0 ] && return; echo "wait_idle free=$f procs=$p"; sleep 20; done; }

sweep(){ local out=$1; shift
  [ -s "$out" ] && { echo "skip $(basename $out) (exists)"; return; }
  wait_idle; echo "$(date '+%F %T') === $(basename $out) ==="
  timeout 3000 $PY "$@" --output "$out" >> "${out%.json}.log" 2>&1 || echo "  WARN rc=$? (see log)"
}

echo "$(date '+%F %T') OFFWALL_LAT_START"
for r in 1 2 3; do
  echo "===== ROUND $r ====="
  # (1) GQA-8 FlashInfer ladder — grid spans off-wall corner -> DRAM-bound anchor
  sweep "$OUT/lat_offwall_h16k8_r${r}.json" "$SCRIPT_DIR/bench_xqa.py" --latency \
    --backends flashinfer --kv-mode distinct --num-q-heads 16 --num-kv-heads 8 \
    --page-sizes 1 2 4 8 16 32 64 128 --batch-sizes 1 2 4 8 \
    --seq-lens 128 256 512 1024 2048 4096 6144
  # (2) GQA-2 arm (Qwen2.5-3B shape)
  sweep "$OUT/lat_offwall_h16k2_r${r}.json" "$SCRIPT_DIR/bench_xqa.py" --latency \
    --backends flashinfer --kv-mode distinct --num-q-heads 16 --num-kv-heads 2 \
    --page-sizes 1 2 4 8 16 32 64 128 --batch-sizes 1 2 4 8 \
    --seq-lens 512 1024 2048 4096 8192
  # (3) XQA ladder (page>=16 only), GQA-8
  sweep "$OUT/lat_offwall_xqa_r${r}.json" "$SCRIPT_DIR/bench_xqa.py" --latency \
    --backends xqa --kv-mode distinct --num-q-heads 16 --num-kv-heads 8 \
    --page-sizes 16 32 64 128 --batch-sizes 1 2 4 8 --seq-lens 256 1024 4096
  # (4) worst-case ps1 fragmentation (token-index contig/block128/scatter)
  sweep "$OUT/pagecost_offwall_h16k8_r${r}.json" "$SCRIPT_DIR/bench_pagecost.py" --mode gather \
    --num-q-heads 16 --num-kv-heads 8 --patterns contig block128 scatter \
    --cells "1 512" "2 1024" "4 512" "8 4096"
  sweep "$OUT/pagecost_offwall_h16k2_r${r}.json" "$SCRIPT_DIR/bench_pagecost.py" --mode gather \
    --num-q-heads 16 --num-kv-heads 2 --patterns contig block128 scatter \
    --cells "2 1024" "2 4096" "8 1024" "8 8192"
done
echo "$(date '+%F %T') OFFWALL_LAT_DONE"
