#!/usr/bin/env bash
# Report 14 -- nsys per-kernel + per-CUDA-API decomposition of a shared-prefix decode step,
# to LOCALIZE report-3's ps1-vs-ps128 gap. Step 0 shows the decode kv_indices are CONTIGUOUS
# at both page sizes, so the attention kernel does ~identical work -> the gap (if real) must be
# in another kernel (KV append / index build / sampling) or host-side. This traces the real
# sgl.Engine (radix ON, Triton, eager) at ps1 and ps128 over a few decode steps and dumps
# per-kernel GPU time + CUDA API time, so a diff pinpoints the cost.
#
# nsys traces the scheduler SUBPROCESS too (whole process tree).
# Env: PS "1 128"  BS 32  GEN 24  PROMPT prompt10k.json  MODEL <path>
set -u
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
STUDY_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
TARGET_HOME=$(eval echo "~$(id -un)")
PY=${PY:-$TARGET_HOME/venvs/bench_sglang/bin/python}
NSYS=${NSYS:-$(command -v nsys || echo /usr/local/bin/nsys)}
OUTDIR=${OUTDIR:-$STUDY_ROOT/data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/nsys}
MODEL=${MODEL:-$TARGET_HOME/hf_models/Qwen3-VL-2B-Instruct}
PROMPT=${PROMPT:-$STUDY_ROOT/data/raw/report-14-shared-prefix-page-kernel-rtx5060ti/prompt10k.json}
HARNESS=${HARNESS:-$SCRIPT_DIR/measure_batch_latency_offline.py}
BS=${BS:-32}
GEN=${GEN:-24}
PS=${PS:-"1 128"}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$(dirname "$PY"):$CUDA_HOME/bin:$PATH"
export SGLANG_DISABLE_CUDNN_CHECK=1
mkdir -p "$OUTDIR"

for ps in $PS; do
  rep="$OUTDIR/nsys_sharedprefix_ps${ps}_bs${BS}"
  echo "=== nsys ps$ps bs$BS ==="
  # short run: one batch, GEN decode steps, eager (all kernels visible), radix ON (shared)
  "$NSYS" profile -t cuda,nvtx --force-overwrite true -o "$rep" \
     "$PY" "$HARNESS" "$PROMPT" \
        --model-path "$MODEL" --attention-backend triton \
        --batch-sizes "$BS" --max-tokens "$GEN" --repeat 1 --page-size "$ps" \
        --mem-fraction-static 0.85 --context-length 20480 \
        --output "$OUTDIR/tpot_ps${ps}.json" > "${rep}.stdout" 2>&1
  # per-kernel GPU time and per-CUDA-API time (CSV)
  "$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-overwrite true \
     --output "$rep" "${rep}.nsys-rep"  >/dev/null 2>&1
  "$NSYS" stats --report cuda_api_sum --format csv --force-overwrite true \
     --output "${rep}_api" "${rep}.nsys-rep"  >/dev/null 2>&1
  echo "  wrote ${rep}_cuda_gpu_kern_sum.csv"
done
echo "DONE nsys decompose -> $OUTDIR"
