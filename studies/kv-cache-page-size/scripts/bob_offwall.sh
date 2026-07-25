#!/bin/bash
# Report 13 engine tier — TRUE-batch bench_one_batch at the OFF-WALL cells (small batch x
# short ctx), CUDA graph ON, independent seqs. Two models:
#   Qwen2.5-3B  (GQA-2, 36 KB/token)  — its decode-attn kernel at B2/L1024 is OFF the DRAM
#                                       wall (the constructed case; microbench + engine ncu)
#   Qwen3-VL-2B (GQA-8, 112 KB/token) — same cell sits AT ~80% DRAM (contrast)
# Pages: FlashInfer fine grid on the flagship cell, coarse elsewhere; Triton control.
# 3 rounds paired (pages back-to-back per round), idle-gated, skip-guarded => resumable.
# NB engine TPOT at these cells is expected weight-DRAM-bound (report 8: ~4-6 GB weights
# re-streamed per step dwarf the KV) -> page effects wash out in TPOT; the kernel-level
# page story lives in the microbench + engine-ncu tiers. We still measure it rigorously.
# Tag: bobo_{mdl}_r{r}_b{B}_l{L}_{be}_ps{ps}.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
# venv bin MUST be on PATH: sglang's runtime JIT needs `ninja` (phastform gotcha —
# without it every run dies post-graph-capture with FileNotFoundError: 'ninja')
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$HOME/venvs/bench_sglang/bin:/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_offwall_5060ti; mkdir -p $OUT
OL=64
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|grep -c .||true);
  [ "${f:-0}" -ge 15000 ] && [ "${p:-1}" -eq 0 ] && return; sleep 20; done; }
# args: mdl path wgb kvkb B L be ps
run(){ local mdl=$1 path=$2 wgb=$3 kvkb=$4 B=$5 L=$6 be=$7 ps=$8
  # floor 0.55: the 3B model's pool math rejects mf<=0.41 at ps1 (per-token metadata
  # headroom) — see report 13 §4; ceiling 0.72 keeps graph-capture headroom
  local mf=$(awk -v B=$B -v L=$L -v OL=$OL -v w=$wgb -v kv=$kvkb 'BEGIN{need=B*(L+OL); gb=need*kv*1024/1e9*1.08; mf=(gb+w)/15.47; if(mf<0.55)mf=0.55; if(mf>0.72)mf=0.72; printf "%.2f", mf}')
  local tag=bobo_${mdl}_r${RND}_b${B}_l${L}_${be}_ps${ps}; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date '+%F %T') === $tag (mf=$mf) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $path --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c "Capture cuda graph bs \[$B\]" $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log | tail -1)"; }

M3=$HOME/hf_models/Qwen2.5-3B-Instruct
M2=$HOME/hf_models/Qwen3-VL-2B-Instruct
FINE=(1 2 4 8 16 32 128)
COARSE=(1 8 32 128)

echo "$(date '+%F %T') BOBO_START"
for RND in 1 2 3; do
  echo "===== ROUND $RND ====="
  # Qwen2.5-3B — flagship off-wall cell (fine page grid) + off-wall B1 + DRAM-side reference
  for ps in "${FINE[@]}";   do run q25_3b $M3 6.2 36 2 1024 flashinfer $ps; done
  for ps in "${COARSE[@]}"; do run q25_3b $M3 6.2 36 2 1024 triton     $ps; done
  for ps in "${COARSE[@]}"; do run q25_3b $M3 6.2 36 1 512  flashinfer $ps; done
  for ps in "${COARSE[@]}"; do run q25_3b $M3 6.2 36 1 512  triton     $ps; done
  for ps in "${COARSE[@]}"; do run q25_3b $M3 6.2 36 8 4096 flashinfer $ps; done
  for ps in "${COARSE[@]}"; do run q25_3b $M3 6.2 36 8 4096 triton     $ps; done
  # Qwen3-VL-2B — same flagship cell (GQA-8 contrast; sits at ~80% DRAM in the kernel)
  for ps in "${COARSE[@]}"; do run q3vl_2b $M2 4.1 112 2 1024 flashinfer $ps; done
  for ps in "${COARSE[@]}"; do run q3vl_2b $M2 4.1 112 2 1024 triton     $ps; done
done
echo "$(date '+%F %T') BOBO_DONE"
