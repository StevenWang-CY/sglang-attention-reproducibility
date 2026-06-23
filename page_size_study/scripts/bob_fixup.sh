#!/bin/bash
# Fix-up for the 2 remaining gaps after bob_rerun.sh:
#  (1) B2/L24576: pool at mf0.62 held 49125 tok but batch needs 49280 -> "skipping
#      due to max batch size limit" (empty jsonl). mf0.63 -> pool ~50.5k (fits) and
#      headroom ~5.7GB (clears the long 24k prefill). Both backends, all ps, 2 rounds.
#  (2) B16/L3072 flashinfer ps128: flaky 'TreeCacheNamespace has no attribute evict'
#      crash (ps1/8/32 OK). Re-run ps128 (skip-guard leaves the good cells).
set -u
H=$HOME/sglang_log; cd $H
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/offline_batch_results/bench_one_batch_5060ti; mkdir -p $OUT
# drop the empty B2/L24576 stubs so the skip-guard re-runs them
find $OUT -name 'bob_*b2_l24576*.jsonl' -empty -delete
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local tag=$1 B=$2 L=$3 mf=$4 be=$5 ps=$6; local out=$OUT/$tag.jsonl
  [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (B=$B L=$L mf=$mf $be ps$ps) ==="
  timeout 600 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len 64 --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  grep -iE "out of memory|max batch size limit|has no attribute|Decode\.  median" $OUT/$tag.log | tail -1; }
echo "$(date) BOB_FIXUP_START"
for r in 1 2; do
  # (1) B2/L24576 both backends, all pages, mf 0.63
  for be in triton flashinfer; do
    for ps in 1 8 32 128; do run "bob_r${r}_b2_l24576_${be}_ps${ps}" 2 24576 0.63 $be $ps; done
  done
  # (2) B16/L3072 flashinfer ps128 only (skip-guard leaves ps1/8/32), mf 0.64
  run "bob_r${r}_b16_l3072_flashinfer_ps128" 16 3072 0.64 flashinfer 128
done
echo "$(date) BOB_FIXUP_DONE"
