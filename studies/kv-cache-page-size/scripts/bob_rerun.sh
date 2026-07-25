#!/bin/bash
# Re-run ONLY the long-ctx cells that OOM'd in bob_sweep.sh (mf was too high → no
# activation headroom). Lower mem-fraction: pool still covers ~49k KV need but
# frees ~5.7GB for FlashInfer workspace + the long single-seq prefill.
# Decode TPOT is pool-size-independent, so these pair correctly with the existing
# cells within each backend. Skip-guard ([ -s out ]) leaves already-done cells.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti; mkdir -p $OUT
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local tag=$1 B=$2 L=$3 mf=$4 be=$5 ps=$6; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (B=$B L=$L mf=$mf $be ps$ps) ==="
  timeout 600 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len 64 --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  grep -iE "out of memory|Decode\.  median" $OUT/$tag.log | tail -1; }
# config: "B L mf"  (lower mf than the original sweep)
CONFIGS=("16 3072 0.64" "8 6144 0.64" "2 24576 0.62")
echo "$(date) BOB_RERUN_START"
for r in 1 2; do
  for c in "${CONFIGS[@]}"; do set -- $c; B=$1; L=$2; mf=$3
    for be in triton flashinfer; do
      for ps in 1 8 32 128; do run "bob_r${r}_b${B}_l${L}_${be}_ps${ps}" $B $L $mf $be $ps; done
    done
  done
done
echo "$(date) BOB_RERUN_DONE"
