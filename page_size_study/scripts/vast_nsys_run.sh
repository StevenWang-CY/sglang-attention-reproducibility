#!/bin/bash
# Report-13 nsys experiments on the rented 5060 Ti (exclusive GPU).
# A: microbench floor decomposition (cuda-core + tensor-core wrappers)
# B: engine per-step launch structure (B2/L1024 + the unresolved B8/L4096)
set -ex
cd /workspace
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | sort -V | tail -1)
OUT=/workspace/nsys_out; mkdir -p $OUT; cd $OUT
M=/workspace/Qwen2.5-3B-Instruct

$NSYS profile -t cuda --force-overwrite true -o mb_cudacore python /workspace/bench_xqa.py --latency \
  --backends flashinfer --kv-mode distinct --num-q-heads 16 --num-kv-heads 2 \
  --page-sizes 128 --batch-sizes 2 --seq-lens 1024 --output $OUT/mb_cudacore.json > mb_cudacore.log 2>&1
$NSYS profile -t cuda --force-overwrite true -o mb_tc python /workspace/bench_xqa.py --latency \
  --backends flashinfer --kv-mode distinct --num-q-heads 16 --num-kv-heads 2 --tensor-cores \
  --page-sizes 128 --batch-sizes 2 --seq-lens 1024 --output $OUT/mb_tc.json > mb_tc.log 2>&1
$NSYS profile -t cuda --force-overwrite true -o eng_b2l1024 python -m sglang.bench_one_batch \
  --model-path $M --trust-remote-code --batch-size 2 --input-len 1024 --output-len 16 --page-size 128 \
  --attention-backend flashinfer --disable-cuda-graph --mem-fraction-static 0.55 \
  --run-name nsys_b2 --result-filename $OUT/nsys_b2.jsonl > eng_b2l1024.log 2>&1
$NSYS profile -t cuda --force-overwrite true -o eng_b8l4096 python -m sglang.bench_one_batch \
  --model-path $M --trust-remote-code --batch-size 8 --input-len 4096 --output-len 16 --page-size 128 \
  --attention-backend flashinfer --disable-cuda-graph --mem-fraction-static 0.55 \
  --run-name nsys_b8 --result-filename $OUT/nsys_b8.jsonl > eng_b8l4096.log 2>&1

for r in mb_cudacore mb_tc eng_b2l1024 eng_b8l4096; do
  $NSYS export --type sqlite --force-overwrite true $r.nsys-rep
done
$NSYS stats --report cuda_api_trace --format csv --output mb_cudacore_api mb_cudacore.nsys-rep || true
grep -hE "Decode\.  median" *.log || true
echo VAST_EXPS_DONE
