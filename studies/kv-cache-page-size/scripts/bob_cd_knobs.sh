#!/bin/bash
# Remaining adversarial corners on Qwen3-VL-2B (FlashInfer primary + Triton control), CUDA graph ON, true batch:
#   Phase C  L2-resident small KV  (B*L in {2k..16k}) -> weight-bound vs index-overhead crossover
#   Phase D  long single/few seq   (B1 very long)     -> longest per-request ps1 index walk
#   Phase G  adversarial graph knobs                  -> over-long output (decode at grown ctx) + big cuda-graph-max-bs
# Fine-ish page grid; 2 rounds; idle-gated; skip-guard; per-cell mem-fraction. Tags bobx_r{r}_b{B}_l{L}_{be}_ps{ps}
# (+ bobg_* for the knob variants).
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
H="$(cd -- "$STUDY_ROOT/../.." && pwd)"; cd "$H"
export CUDA_HOME=/usr/local/cuda-12.8; export PATH=/usr/local/cuda-12.8/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=$HOME/venvs/bench_sglang/bin/python; MODEL=$HOME/hf_models/Qwen3-VL-2B-Instruct
OUT=$H/studies/kv-cache-page-size/data/raw/bench_one_batch_5060ti; mkdir -p $OUT
OL=64
mf_for(){ awk -v B=$1 -v L=$2 -v OL=$3 'BEGIN{need=B*(L+OL); gb=need*1.1*114688/1e9; mf=(gb+4.1)/15.47;
  if(mf<0.45)mf=0.45; if(mf>0.82)mf=0.82; printf "%.2f", mf}'; }
wait_idle(){ while true; do f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits|head -1|tr -d ' ');
  p=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader|wc -l);
  [ "${f:-0}" -ge 15500 ] && [ "$p" -eq 0 ] && return; sleep 20; done; }
run(){ local r=$1 B=$2 L=$3 be=$4 ps=$5 ol=${6:-$OL} extra=${7:-} pfx=${8:-bobx}
  local mf=$(mf_for $B $L $ol); local tag=${pfx}_r${r}_b${B}_l${L}_${be}_ps${ps}
  [ -n "$extra" ] && tag=${pfx}_r${r}_b${B}_l${L}_${be}_ps${ps}_$(echo $extra|tr -d ' -=' )
  local out=$OUT/$tag.jsonl; [ -s "$out" ] && { echo "skip $tag"; return; }
  wait_idle; echo "$(date) === $tag (mf=$mf ol=$ol $extra) ==="
  timeout 500 $PY -m sglang.bench_one_batch --model-path $MODEL --trust-remote-code \
    --batch-size $B --input-len $L --output-len $ol --page-size $ps --attention-backend $be \
    --cuda-graph-bs $B --mem-fraction-static $mf $extra --run-name $tag --result-filename $out > $OUT/$tag.log 2>&1
  echo "  cap=$(grep -c "Capture cuda graph" $OUT/$tag.log) $(grep -iE 'out of memory|max batch size limit|has no attribute|Decode\.  median' $OUT/$tag.log | tail -1)"; }
FINE=(1 2 4 8 16 32 128)
echo "$(date) BOBCD_START"
# Phase C: L2-resident small KV (2k..16k)
PHASE_C=("8 256" "16 256" "32 256" "8 512" "16 512" "8 1024" "16 768" "32 512")
for r in 1 2; do for c in "${PHASE_C[@]}"; do set -- $c; B=$1; L=$2
  for ps in "${FINE[@]}"; do run $r $B $L flashinfer $ps; done
  for ps in 1 8 32 128; do run $r $B $L triton $ps; done
done; done
# Phase D: long single/few sequence
PHASE_D=("1 32768" "1 49152" "1 65536" "2 32768")
for r in 1 2; do for c in "${PHASE_D[@]}"; do set -- $c; B=$1; L=$2
  for ps in "${FINE[@]}"; do run $r $B $L flashinfer $ps; done
  for ps in 1 8 32 128; do run $r $B $L triton $ps; done
done; done
# Phase G: adversarial graph knobs on a representative high-batch cell (over-long decode; mismatched graph max-bs)
for r in 1 2; do
  for ps in 1 8 16 32; do run $r 256 128 flashinfer $ps 256 "" bobg; done                       # output-len 256 (decode at grown ctx)
  for ps in 1 8 16 32; do run $r 64 2048 flashinfer $ps 64 "--cuda-graph-max-bs 256" bobg; done  # capture big graph, run small
done
echo "$(date) BOBCD_DONE"
