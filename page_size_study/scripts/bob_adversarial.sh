#!/bin/bash
# Adversarial hunt: a TRUE-batch (bench_one_batch) scenario where FlashInfer ps1 is NOT the fastest,
# CUDA graph ON. Attacks the corners report-5 never tested: HIGH batch, L2-resident small KV, long
# single-seq. Qwen3-VL-2B. Baseline pages {1,8,32,128}; FlashInfer primary + Triton control.
# Per-cell mem-fraction auto-sized (pool ~1.1x the B*(L+OL) need; weights ~4.1GB). Idle-gated, skip-guard,
# round-robin (pages back-to-back within a round). Tag prefix bobx_.
set -u
H=$HOME/sglang_log; cd $H
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/offline_batch_results/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
# mem-fraction so pool >= 1.1 * B*(L+OL) tokens (114688 B/tok, ~4.1GB weights, 15.47GB usable); clamp [0.45,0.82]
mf_for(){ awk -v B=$1 -v L=$2 -v OL=$OL 'BEGIN{need=B*(L+OL); gb=need*1.1*114688/1e9; mf=(gb+4.1)/15.47;
  if(mf<0.45)mf=0.45; if(mf>0.82)mf=0.82; printf "%.2f", mf}'; }
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local B=$1 L=$2 be=$3 ps=$4; local mf=$(mf_for $B $L); local tag=bobx_b${B}_l${L}_${be}_ps${ps}
  local out=$OUT/$tag.jsonl; [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (B=$B L=$L mf=$mf $be ps$ps) ==="
  timeout 600 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  local cap=$(grep -c "Capture cuda graph bs \[$B\]" $OUT/$tag.log)
  echo "  graphcap=$cap $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log | tail -1)"; }

# (B L) configs per phase  -- B*L kept <= ~49k so it fits 16GB
PHASE_B=("48 256" "64 256" "96 256" "128 256" "192 256" "64 512" "96 512" "48 1024")     # high batch x short ctx
PHASE_C=("8 256" "16 256" "32 256" "16 512" "16 768" "32 512" "8 1024")                  # L2-resident small KV (2k..16k)
PHASE_D=("1 32768" "1 49152" "2 32768")                                                  # long single/few seq
CONFIGS=("${PHASE_B[@]}" "${PHASE_C[@]}" "${PHASE_D[@]}")
echo "$(date) BOBX_START (${#CONFIGS[@]} configs x4 pages x2 rounds, flashinfer+triton)"
for r in 1 2; do
  for c in "${CONFIGS[@]}"; do set -- $c; B=$1; L=$2
    for be in flashinfer triton; do
      for ps in 1 8 32 128; do run $B $L $be $ps; done
    done
  done
done
echo "$(date) BOBX_DONE"
