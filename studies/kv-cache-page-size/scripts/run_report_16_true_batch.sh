#!/usr/bin/env bash
# Report 16 Arm A2 — TRUE-batch bench_one_batch, trtllm_mha vs flashinfer vs triton.
# Same sglang_0b2e build (commit 0eded9e208) as reports 15/16 so only the backend differs.
# trtllm_mha: pages {16,32,64}; FI/Triton: {1,16,128}. One round by default, resumable.
# Cells: on-wall (Qwen3-VL-2B, report-5) + off-wall (Qwen2.5-3B GQA-2, report-13 flagship).
# TPOT = JSONL median_decode_latency * 1000 (already the correct per-step metric; first step discarded).
set -uo pipefail
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$HOME/venvs/sglang_0b2e/bin/python}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}"
SGLANG_PYTHONPATH="${SGLANG_PYTHONPATH:-$HOME/sglang_git/python}"
MODEL_Q25_3B="${MODEL_Q25_3B:-$HOME/hf_models/Qwen2.5-3B-Instruct}"
MODEL_Q3VL_2B="${MODEL_Q3VL_2B:-$HOME/hf_models/Qwen3-VL-2B-Instruct}"
OUTPUT_DIR="${OUTPUT_DIR:-$STUDY_ROOT/data/raw/report-16-true-batch-engine}"
export PATH="$(dirname "$PYTHON_BIN"):$CUDA_HOME/bin:$PATH"
export CUDA_HOME
export PYTHONPATH="$SGLANG_PYTHONPATH${PYTHONPATH:+:$PYTHONPATH}"
export SGLANG_DISABLE_CUDNN_CHECK=1 SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$OUTPUT_DIR"
OUTPUT_LENGTH=64
wait_idle(){ while true; do p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c .||true);
  [ "${p:-1}" -eq 0 ] && return; sleep 15; done; }
# args: mdl path wgb kvkb B L be ps
run(){ local mdl=$1 path=$2 wgb=$3 kvkb=$4 B=$5 L=$6 be=$7 ps=$8
  local mf=$(awk -v B=$B -v L=$L -v OL=$OUTPUT_LENGTH -v w=$wgb -v kv=$kvkb 'BEGIN{need=B*(L+OL); gb=need*kv*1024/1e9*1.08; mf=(gb+w)/15.47; if(mf<0.55)mf=0.55; if(mf>0.72)mf=0.72; printf "%.2f", mf}')
  local tag=bt_${mdl}_r${RND}_b${B}_l${L}_${be}_ps${ps}; local out=$OUTPUT_DIR/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  # trtllm_mha: split backends (prefill trtllm is SM100-only) -> prefill=flashinfer, decode=trtllm_mha
  local backend_args=(--attention-backend "$be")
  [ "$be" = "trtllm_mha" ] && backend_args=(--decode-attention-backend trtllm_mha --prefill-attention-backend flashinfer)
  wait_idle; echo "$(date '+%F %T') === $tag (mf=$mf) ==="
  timeout 500 "$PYTHON_BIN" -m sglang.bench_one_batch --model-path "$path" --trust-remote-code \
    --batch-size "$B" --input-len "$L" --output-len "$OUTPUT_LENGTH" --page-size "$ps" "${backend_args[@]}" \
    --cuda-graph-bs "$B" --mem-fraction-static "$mf" --run-name "$tag" --result-filename "$out" > "$OUTPUT_DIR/$tag.log" 2>&1
  echo "  rc=$? $(grep -iE 'out of memory|no attribute|not support|invalid|Decode\.  median|Traceback' "$OUTPUT_DIR/$tag.log" | tail -1)"; }

TR=(16 32 64); FT=(1 16 128)   # trtllm engine pages {16,32,64} (128 coerced); FI/Triton full
echo "$(date '+%F %T') BT_START"
for RND in ${ROUNDS:-1}; do
  echo "===== ROUND $RND ====="
  # off-wall flagship (Qwen2.5-3B GQA-2, B2/L1024 — report-13 exposed cell)
  for ps in "${TR[@]}"; do run q25_3b "$MODEL_Q25_3B" 6.2 36 2 1024 trtllm_mha "$ps"; done
  for ps in "${FT[@]}"; do run q25_3b "$MODEL_Q25_3B" 6.2 36 2 1024 flashinfer "$ps"; done
  for ps in "${FT[@]}"; do run q25_3b "$MODEL_Q25_3B" 6.2 36 2 1024 triton     "$ps"; done
  # on-wall (Qwen3-VL-2B GQA-8, B8/L4096 — report-5 headline cell)
  for ps in "${TR[@]}"; do run q3vl_2b "$MODEL_Q3VL_2B" 4.1 112 8 4096 trtllm_mha "$ps"; done
  for ps in "${FT[@]}"; do run q3vl_2b "$MODEL_Q3VL_2B" 4.1 112 8 4096 flashinfer "$ps"; done
  for ps in "${FT[@]}"; do run q3vl_2b "$MODEL_Q3VL_2B" 4.1 112 8 4096 triton     "$ps"; done
done
echo BTDONE > "$OUTPUT_DIR/bt.done"
echo "$(date '+%F %T') BT_DONE"
