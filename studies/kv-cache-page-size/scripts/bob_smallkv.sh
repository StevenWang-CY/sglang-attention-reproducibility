#!/bin/bash
# Report 8 Tier A — TRUE-batch bench_one_batch with SMALL KV (does ps1 lose once decode is no
# longer DRAM-bound *because the KV is small*?). Qwen3-VL-2B, FlashInfer + Triton, CUDA graph ON,
# independent seqs (no shared prefix). Footprint ladder L1 -> L2 -> DRAM-bound.
# NB (load-bearing): shrinking KV at the ENGINE level does NOT move attention into a cache regime;
# it makes the MODEL WEIGHTS (~4 GB/step) the dominant DRAM traffic -> TPOT goes flat for a reason
# unrelated to page_size. We still measure it rigorously (paired, multi-round); the decisive
# off-ceiling test is the weightless kernel microbench (Tier B). Tags bobk_r{r}_b{B}_l{L}_{be}_ps{ps}.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_smallkv_5060ti; mkdir -p $OUT
OL=64
mf_for(){ awk -v B=$1 -v L=$2 -v OL=$OL 'BEGIN{need=B*(L+OL); gb=need*1.1*114688/1e9; mf=(gb+4.1)/15.47;
  if(mf<0.45)mf=0.45; if(mf>0.82)mf=0.82; printf "%.2f", mf}'; }
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15000 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local r=$1 B=$2 L=$3 be=$4 ps=$5
  local mf=$(mf_for $B $L); local tag=bobk_r${r}_b${B}_l${L}_${be}_ps${ps}
  local out=$OUT/$tag.jsonl; [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  $(grep -ciE 'out of memory|has no attribute|max batch size' $OUT/$tag.log|sed 's/^0$//;s/^[1-9].*/ERR/') $(grep 'Decode.  median' $OUT/$tag.log|tail -1)"; }

# footprint ladder: B*L*4096 B per layer; L2=32MB <=> B*L ~ 8192 tok
CELLS=("1 256" "1 1024" "1 4096" "1 16384" "1 32768" \
       "1 128" "8 128" "32 128" "64 128" "128 128" \
       "8 256" "16 256" "32 256" "8 512" "16 512" "32 512")
HEADLINE=("8 256" "32 256" "1 4096" "128 128")
FI_COARSE=(1 8 32 128); FI_FINE=(2 4 16); TRI=(1 8 32 128)
is_headline(){ local key="$1 $2"; for c in "${HEADLINE[@]}"; do [ "$c" = "$key" ] && return 0; done; return 1; }

echo "$(date) BOBK_START"
for r in 1 2 3 4 5 6; do
  echo "===== ROUND $r ====="
  for c in "${CELLS[@]}"; do set -- $c; B=$1; L=$2
    if [ "$r" -le 3 ] || is_headline $B $L; then
      for ps in "${FI_COARSE[@]}"; do run $r $B $L flashinfer $ps; done
      for ps in "${TRI[@]}";        do run $r $B $L triton     $ps; done
    fi
  done
  for c in "${HEADLINE[@]}"; do set -- $c; B=$1; L=$2     # fine FI grid on headline cells, all rounds
    for ps in "${FI_FINE[@]}"; do run $r $B $L flashinfer $ps; done
  done
done
echo "$(date) BOBK_DONE"
