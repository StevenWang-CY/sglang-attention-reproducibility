#!/usr/bin/env bash
# Report 9 verification + mechanism: confirm the >5% small-block (blk16) penalty in vLLM true batch
# is REAL (not order/thermal noise) and characterize it.
#  (1) REVERSED block order {128,64,32,16} on the break cells -> rules out "first-run-slow" bias
#      (tag vfy_); each block_size is a separate process anyway.
#  (2) GRAPH OFF (enforce_eager) on the break cells -> is the penalty graph-related? (tag eag_)
#  (3) finer total-KV ladder at fixed high batch -> where does the penalty turn on? (tag vfy_)
# repeat 6 for tight CIs. FlashInfer, no prefix sharing, distinct random tokens.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${VLLM_ENV_FILE:-$SCRIPT_DIR/vllm_env.sh}"
PY="${PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
MEASURE="${MEASURE_SCRIPT:-$SCRIPT_DIR/measure_vllm_smallkv.py}"
OUT="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/vllm_smallkv_5060ti}"
MODEL="${MODEL_PATH:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
mkdir -p "$OUT"
MT=128; REP=6; UTIL=0.90
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d " ");
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 14000 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local pfx=$1 B=$2 L=$3 blk=$4 eager=${5:-}; local mml=$((L+160))
  local tag=${pfx}_b${B}_l${L}_blk${blk}; [ -n "$eager" ] && tag=${tag}_eager
  local out="$OUT/$tag.json"; [ -s "$out" ] && grep -q tpot_median_ms "$out" && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag ==="
  timeout 400 "$PY" "$MEASURE" --model-path "$MODEL" --block-size "$blk" --batch-size "$B" \
    --input-len "$L" --max-tokens "$MT" --repeat "$REP" --gpu-mem-util "$UTIL" --max-model-len "$mml" \
    --attention-backend FLASHINFER $eager --output "$out" > "$OUT/$tag.log" 2>&1
  grep -E "TPOT=|VLLM_LLM_INIT_FAILED|ValueError|out of memory" "$OUT/$tag.log" | tail -1; }
echo "$(date) VLLM_VERIFY_START"
# (1) reversed-order confirm on the break cells + the reversed-sign control
for c in "128 512" "64 1024" "32 2048" "256 128"; do set -- $c; B=$1; L=$2
  for blk in 128 64 32 16; do run vfy $B $L $blk; done
done
# (2) graph OFF on the two strongest break cells
for c in "128 512" "64 1024"; do set -- $c; B=$1; L=$2
  for blk in 16 32 64 128; do run eag $B $L $blk --enforce-eager; done
done
# (3) total-KV turn-on ladder at fixed batch 128 (L grows -> total KV grows): 16k..82k tok
for L in 128 256 384 512 640; do for blk in 16 128; do run vfy 128 $L $blk; done; done
echo "$(date) VLLM_VERIFY_DONE"
