#!/bin/bash
# Re-run Qwen3-VL-4B cells that SKIPPED in bob_xmodel.sh (mf clamp 0.72 too low for 8GB weights + 144KB/tok
# -> pool < B*(L+OL), "max batch size limit"). Higher clamp (0.84). FlashInfer + Triton, fine grid, 3 rounds.
# Tag bobm_q3vl_4b_r{r}_b{B}_l{L}_{be}_ps{ps} (same as bob_xmodel so the analyzer/skip-guard line up).
set -u
H=$HOME/sglang_log; cd $H
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; M=$HOME/hf_models/Qwen3-VL-4B-Instruct
OUT=$H/offline_batch_results/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
# drop empty 4B stubs so skip-guard re-runs them
find $OUT -name 'bobm_q3vl_4b_*.jsonl' -empty -delete
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local r=$1 B=$2 L=$3 be=$4 ps=$5
  local mf=$(awk -v B=$B -v L=$L -v OL=$OL 'BEGIN{need=B*(L+OL); gb=need*144*1024/1e9*1.08; mf=(gb+8.0)/15.47; if(mf<0.45)mf=0.45; if(mf>0.84)mf=0.84; printf "%.2f", mf}')
  local tag=bobm_q3vl_4b_r${r}_b${B}_l${L}_${be}_ps${ps}; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $M --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c "Capture cuda graph bs \[$B\]" $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log | tail -1)"; }
FINE=(1 2 4 8 16 32 128)
echo "$(date) BOB4B_START"
for r in 1 2 3; do for c in "64 128" "96 128" "128 128" "96 256"; do set -- $c; B=$1; L=$2
  for ps in "${FINE[@]}"; do run $r $B $L flashinfer $ps; done
  for ps in 1 8 32 128; do run $r $B $L triton $ps; done
done; done
echo "$(date) BOB4B_DONE"
