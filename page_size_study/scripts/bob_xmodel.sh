#!/bin/bash
# Cross-model adversarial break hunt (Phase F): does a TRUE-batch FlashInfer+graph-ON scenario exist where
# ps1 is NOT fastest, on models MORE vulnerable than Qwen3-VL-2B?
#  - Qwen2.5-3B  : GQA-2, ~36 KB/token KV (4x less per-layer than 2B GQA-8) -> index overhead 4x more visible;
#                  tiny KV also lets batch go MUCH higher (512+) where overhead could finally dominate.
#  - Qwen3-VL-4B : 36 layers, GQA-8 (144 KB/token).
# FlashInfer primary + Triton control. Fine-ish page grid. 3 rounds (paired), idle-gated, skip-guard.
# Per-model mem-fraction auto-sized (pool ~1.15x B*(L+OL)*KVKB + weights). Tag bobm_{mdl}_r{r}_b{B}_l{L}_{be}_ps{ps}.
set -u
H=$HOME/sglang_log; cd $H
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python
OUT=$H/offline_batch_results/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
# args: mdl path wgb kvkb B L be ps
run(){ local mdl=$1 path=$2 wgb=$3 kvkb=$4 B=$5 L=$6 be=$7 ps=$8
  # pool just above the KV need (margin 1.08) so high-batch graph CAPTURE keeps ample headroom; clamp [0.40,0.72]
  local mf=$(awk -v B=$B -v L=$L -v OL=$OL -v w=$wgb -v kv=$kvkb 'BEGIN{need=B*(L+OL); gb=need*kv*1024/1e9*1.08; mf=(gb+w)/15.47; if(mf<0.40)mf=0.40; if(mf>0.72)mf=0.72; printf "%.2f", mf}')
  local tag=bobm_${mdl}_r${RND}_b${B}_l${L}_${be}_ps${ps}; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $path --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c "Capture cuda graph bs \[$B\]" $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log | tail -1)"; }

FINE=(1 2 4 8 16 32 128)
echo "$(date) BOBM_START"
for RND in 1 2 3; do
  # Qwen2.5-3B (GQA-2) : push batch HIGH (small KV fits) -- the prime break candidate
  M=$HOME/hf_models/Qwen2.5-3B-Instruct
  for c in "128 128" "256 128" "512 128" "256 64" "512 64" "256 256"; do set -- $c; B=$1; L=$2
    for ps in "${FINE[@]}"; do run q25_3b $M 6.2 36 $B $L flashinfer $ps; done
    for ps in 1 8 32 128; do run q25_3b $M 6.2 36 $B $L triton $ps; done
  done
done
for RND in 1 2; do
  # Qwen3-VL-4B (36 layers, GQA-8; 8GB weights + 144KB/tok -> only lower batch fits with capture headroom)
  M=$HOME/hf_models/Qwen3-VL-4B-Instruct
  for c in "64 128" "96 128" "128 128" "96 256"; do set -- $c; B=$1; L=$2
    for ps in "${FINE[@]}"; do run q3vl_4b $M 8.0 144 $B $L flashinfer $ps; done
    for ps in 1 8 32 128; do run q3vl_4b $M 8.0 144 $B $L triton $ps; done
  done
done
echo "$(date) BOBM_DONE"
