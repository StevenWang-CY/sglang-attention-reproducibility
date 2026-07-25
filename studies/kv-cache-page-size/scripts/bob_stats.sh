#!/bin/bash
# AUDIT: high-rep PAIRED statistical test of the report-5 "ps1 fastest in a true batch" claim.
# For each decisive cell, run pages BACK-TO-BACK (thermally matched) for 12 rounds -> gives
#   (a) cross-launch NOISE FLOOR (std of median_decode_latency per page across 12 launches), and
#   (b) PAIRED ps1-vs-best diff with a 12-sample CI -> definitively excludes (or finds) a >=5% break.
# FlashInfer + CUDA graph ON, true batch. Tag bobs_{mdl}_b{B}_l{L}_r{r}_ps{ps}.
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python
Q2B=$HOME/hf_models/Qwen3-VL-2B-Instruct; Q25=$HOME/hf_models/Qwen2.5-3B-Instruct
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
# args: mdltag path wgb kvB B L pages...
runcell(){ local mtag=$1 path=$2 wgb=$3 kvb=$4 B=$5 L=$6; shift 6; local pages=("$@")
  local mf=$(awk -v B=$B -v L=$L -v OL=$OL -v w=$wgb -v kv=$kvb 'BEGIN{need=B*(L+OL); gb=need*kv*1.1/1e9; mf=(gb+w)/15.47; if(mf<0.45)mf=0.45; if(mf>0.80)mf=0.80; printf "%.2f", mf}')
  for ps in "${pages[@]}"; do
    local tag=bobs_${mtag}_b${B}_l${L}_r${RND}_ps${ps}; local out=$OUT/$tag.jsonl
    [ -s "$out" ] && { echo "skip $tag"; continue; }
    wait_idle
    timeout 500 $PY -m sglang.bench_one_batch --model-path $path --trust-remote-code \
      --batch-size $B --input-len $L --output-len $OL --page-size $ps --attention-backend flashinfer \
      --cuda-graph-bs $B --mem-fraction-static $mf --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
    v=$(grep "Decode.  median latency" $OUT/$tag.log|tail -1|grep -oE "0\.[0-9]+")
    echo "r$RND $tag mf=$mf -> ${v}s $(grep -ciE 'out of memory|has no attribute|max batch' $OUT/$tag.log|sed 's/^0$//;s/^[1-9].*/ERR/')"
  done; }
echo "$(date) BOBS_START"
for RND in $(seq 1 12); do
  echo "=== ROUND $RND ==="
  runcell q2b  $Q2B 4.1 114688 8   6144  1 8 128     # 49k long-ctx (report-4 mirror, ps128 works)
  runcell q2b  $Q2B 4.1 114688 16  2048  1 8 128     # 32k main-sweep regime
  runcell q2b  $Q2B 4.1 114688 256 128   1 8 32      # high batch (apparent-candidate region; ps128 evict-bug)
  runcell q25  $Q25 6.2 36864  512 64    1 8 32      # GQA-2 extreme batch (most vulnerable)
done
echo "$(date) BOBS_DONE"
