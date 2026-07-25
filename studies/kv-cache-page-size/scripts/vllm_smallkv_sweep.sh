#!/usr/bin/env bash
# Report 9 — vLLM TRUE-batch (NO prefix sharing) decode TPOT vs block_size, FlashInfer backend,
# CUDA graph ON, on sm120. Hunt for a >=5% smallest-block (page_size) latency disadvantage across
# batch sizes and block sizes. vLLM-FlashInfer rejects block_size <16 (probe: 1,8 -> "block_size
# not supported"); supported = {16,32,64,128}, so the smallest expressible page is 16.
# Distinct random-token prompts + enable_prefix_caching=False => genuine true batch.
# Tags vllm_b{B}_l{L}_blk{blk}.json in studies/kv-cache-page-size/data/raw/vllm_smallkv_5060ti/.
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "${VLLM_ENV_FILE:-$SCRIPT_DIR/vllm_env.sh}"
PY="${PYTHON_BIN:-$HOME/venvs/vllm/bin/python}"
MEASURE="${MEASURE_SCRIPT:-$SCRIPT_DIR/measure_vllm_smallkv.py}"
OUT="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/vllm_smallkv_5060ti}"
MODEL="${MODEL_PATH:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
mkdir -p "$OUT"
MT=128; REP=4; UTIL=0.90
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d " ");
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 14000 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local B=$1 L=$2 blk=$3; local mml=$((L+160))
  local tag=vllm_b${B}_l${L}_blk${blk}; local out="$OUT/$tag.json"
  [ -s "$out" ] && grep -q tpot_median_ms "$out" && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag ==="
  timeout 400 "$PY" "$MEASURE" --model-path "$MODEL" --block-size "$blk" --batch-size "$B" \
    --input-len "$L" --max-tokens "$MT" --repeat "$REP" --gpu-mem-util "$UTIL" --max-model-len "$mml" \
    --attention-backend FLASHINFER --output "$out" > "$OUT/$tag.log" 2>&1
  grep -E "TPOT=|VLLM_LLM_INIT_FAILED|ValueError|out of memory" "$OUT/$tag.log" | tail -1; }
BLK=(16 32 64 128)
# feasible cells: B*(L+128) <= ~90k tokens (fits 16GB KV pool alongside 4.1GB weights)
CELLS=("1 1024" "1 2048" "1 4096" "1 8192" \
       "8 1024" "8 2048" "8 4096" \
       "32 512" "32 1024" "32 2048" \
       "64 256" "64 512" "64 1024" \
       "128 128" "128 256" "128 512" \
       "256 128")
echo "$(date) VLLM_SWEEP_START"
for c in "${CELLS[@]}"; do set -- $c; B=$1; L=$2
  for blk in "${BLK[@]}"; do run $B $L $blk; done
done
echo "$(date) VLLM_SWEEP_DONE"
