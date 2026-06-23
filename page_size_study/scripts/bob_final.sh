#!/bin/bash
# Fill the last two gaps, correctly sized:
#  (1) Qwen3-VL-4B at LOW batch (VL weights ~8.4GB + ~3GB overhead floor -> only small KV fits on 16GB).
#  (2) Phase G adversarial graph knobs, properly sized: decode-at-grown-context (output-len 192) and
#      mismatched --cuda-graph-max-bs (capture a big graph, run small) -> does ps1 over-walk?
# FlashInfer primary + Triton control. Tags: bobm_q3vl_4b_* (4B) and bobg_* (knobs).
set -u
H=$HOME/sglang_log; cd $H
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python
OUT=$H/offline_batch_results/bench_one_batch_5060ti; mkdir -p $OUT
find $OUT -name 'bobg_*.jsonl' -empty -delete; find $OUT -name 'bobm_q3vl_4b_*.jsonl' -empty -delete
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
# generic: path B L OL ps be mf extra tag
go(){ local path=$1 B=$2 L=$3 ol=$4 ps=$5 be=$6 mf=$7 extra=$8 tag=$9; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf ol=$ol $extra) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $path --trust-remote-code \
    --batch-size $B --input-len $L --output-len $ol --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf $extra --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c 'Capture cuda graph' $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log|tail -1)"; }
M4=$HOME/hf_models/Qwen3-VL-4B-Instruct
echo "$(date) BOBFINAL_START"
# (1) 4B low-batch (mf high; small KV) -- fine grid
for r in 1 2; do for c in "8 256" "16 256" "8 512" "16 512"; do set -- $c; B=$1; L=$2
  for ps in 1 2 4 8 16 32; do go $M4 $B $L 64 $ps flashinfer 0.85 "" bobm_q3vl_4b_r${r}_b${B}_l${L}_flashinfer_ps${ps}; done
  for ps in 1 8 32; do go $M4 $B $L 64 $ps triton 0.85 "" bobm_q3vl_4b_r${r}_b${B}_l${L}_triton_ps${ps}; done
done; done
# (2) Phase G on Qwen3-VL-2B (fits): decode-at-grown-ctx (OL192) + mismatched graph-max-bs
M2=$HOME/hf_models/Qwen3-VL-2B-Instruct
for r in 1 2; do
  for ps in 1 8 16 32; do go $M2 128 128 192 $ps flashinfer 0.58 "" bobg_ol192_r${r}_b128_l128_flashinfer_ps${ps}; done   # decode grows 128->320
  for ps in 1 8 16 32; do go $M2 64 1024 64 $ps flashinfer 0.62 "--cuda-graph-max-bs 256" bobg_gmax256_r${r}_b64_l1024_flashinfer_ps${ps}; done  # capture big graph, run small
done
echo "$(date) BOBFINAL_DONE"
