#!/bin/bash
#
# Simple script to confirm Tensor Core activation at batch=32
# Uses ncu to capture kernel names for QKV projection
#

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=10  # Very short run

mkdir -p "$OUTPUT_DIR"

export TREE_SPARSE_TIMING=0  # Disable to reduce noise

echo "========================================================================"
echo "Confirming Tensor Core Activation at Batch=32"
echo "========================================================================"
echo ""
echo "This script captures CUDA kernel names at batch 16 vs batch 32"
echo "to confirm the GEMM kernel transition that causes QKV speedup."
echo ""
echo "Expected findings:"
echo "  Batch 16: Generic GEMM kernels"
echo "  Batch 32: Tensor Core optimized kernels (names contain 'tensorop')"
echo ""
echo "========================================================================"
echo ""

profile_kernels() {
    local BATCH=$1
    echo ""
    echo "----------------------------------------"
    echo "Profiling Batch Size: $BATCH"
    echo "----------------------------------------"

    # Use ncu with minimal metrics to just capture kernel names
    # --kernel-name regex:.*gemm.*: Only profile GEMM kernels
    # --metrics sm__inst_executed_pipe_tensor.avg: Capture Tensor Core usage
    # --page raw: Get raw output

    ncu \
        --kernel-name regex:.*gemm.* \
        --metrics sm__inst_executed_pipe_tensor.avg \
        --csv \
        --log-file "${OUTPUT_DIR}/ncu_batch${BATCH}.csv" \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/results_batch${BATCH}.json" \
            2>&1 | grep -i "kernel\|tensorop\|gemm" | head -20

    echo ""
    echo "Log saved to: ${OUTPUT_DIR}/ncu_batch${BATCH}.csv"
}

# Profile batch 16 and 32
for BATCH in 16 32; do
    profile_kernels $BATCH
    sleep 1
done

echo ""
echo "========================================================================"
echo "Analysis Complete!"
echo "========================================================================"
echo ""
echo "Check the CSV files for kernel names:"
ls -lh "$OUTPUT_DIR"/ncu_batch*.csv
echo ""
echo "Look for these patterns in kernel names:"
echo "  - Batch 16: ampere_sgemm_*, s884gemm_*, etc. (generic GEMM)"
echo "  - Batch 32: *tensorop*, *tn_tensorop*, cutlass_* (Tensor Core GEMM)"
echo ""
echo "Tensor Core pipeline utilization metric:"
echo "  - sm__inst_executed_pipe_tensor.avg > 0 means Tensor Cores are active"
echo "========================================================================"
