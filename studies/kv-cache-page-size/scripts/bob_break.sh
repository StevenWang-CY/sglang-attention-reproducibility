#!/bin/bash
# Map the candidate BREAK: FlashInfer + CUDA graph ON + TRUE batch, high batch x short ctx, where ps1 is
# NOT the fastest page (smoke: B256/L128 ps1 40.4ms > ps16 38.1ms, +6%). Fine page grid to locate the
# minimum and characterise batch- and ctx-dependence. Qwen3-VL-2B. 3 rounds, pages back-to-back (paired),
# idle-gated, skip-guard, per-cell mem-fraction. Tag bobx_r{r}_b{B}_l{L}_{be}_ps{ps}.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
mf_for(){ awk -v B=$1 -v L=$2 -v OL=$OL 'BEGIN{need=B*(L+OL); gb=need*1.1*114688/1e9; mf=(gb+4.1)/15.47;
  if(mf<0.45)mf=0.45; if(mf>0.82)mf=0.82; printf "%.2f", mf}'; }
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local r=$1 B=$2 L=$3 be=$4 ps=$5; local mf=$(mf_for $B $L); local tag=bobx_r${r}_b${B}_l${L}_${be}_ps${ps}
  local out=$OUT/$tag.jsonl; [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c "Capture cuda graph bs \[$B\]" $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|Decode\.  median' $OUT/$tag.log | tail -1)"; }
FINE=(1 2 4 8 16 32 64 128)
# batch ladder @ L128 ; ctx ladder @ B256  (B256/L128 in both)
LADDER=("64 128" "128 128" "192 128" "256 128" "320 128" "256 64" "256 256" "256 512")
echo "$(date) BOBX_BREAK_START"
for r in 1 2 3; do
  for c in "${LADDER[@]}"; do set -- $c; B=$1; L=$2
    for ps in "${FINE[@]}"; do run $r $B $L flashinfer $ps; done       # FlashInfer fine grid (primary)
  done
done
# Triton control on the same ladder, baseline pages, 2 rounds
for r in 1 2; do
  for c in "${LADDER[@]}"; do set -- $c; B=$1; L=$2
    for ps in 1 8 32 128; do run $r $B $L triton $ps; done
  done
done
echo "$(date) BOBX_BREAK_DONE"
