#!/bin/bash
#
# Detailed profiling of QKV kernels using ncu (NVIDIA Nsight Compute)
# This captures Tensor Core utilization metrics
#

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=20  # Very short for detailed profiling

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Detailed QKV Kernel Profiling with ncu"
echo "========================================================================"
echo ""

export TREE_SPARSE_TIMING=1
export TREE_SPARSE_TIMING_INTERVAL=5

profile_with_ncu() {
    local BATCH_SIZE=$1
    local PROFILE_FILE="${OUTPUT_DIR}/ncu_batch${BATCH_SIZE}"

    echo ""
    echo "Profiling batch size $BATCH_SIZE with ncu..."
    echo "(This will be slow - captures detailed metrics)"
    echo ""

    # Use ncu to capture Tensor Core utilization
    # --set full: Capture all available metrics
    # --target-processes all: Profile all processes
    # --kernel-name-base mangled: Show full kernel names

    ncu \
        --set full \
        --target-processes all \
        --kernel-name-base mangled \
        --export "${PROFILE_FILE}" \
        --force-overwrite \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH_SIZE \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/results_ncu_batch${BATCH_SIZE}.json" \
            2>&1 | tee "${OUTPUT_DIR}/ncu_batch${BATCH_SIZE}.log"

    echo ""
    echo "Profile saved to: ${PROFILE_FILE}.ncu-rep"
}

# Profile batch 16 (before transition) and batch 32 (after transition)
for BATCH in 16 32; do
    profile_with_ncu $BATCH
done

echo ""
echo "========================================================================"
echo "To view detailed metrics, use:"
echo "  ncu-ui $OUTPUT_DIR/ncu_batch16.ncu-rep"
echo "  ncu-ui $OUTPUT_DIR/ncu_batch32.ncu-rep"
echo ""
echo "Look for these metrics in the UI:"
echo "  - sm__sass_thread_inst_executed_op_*_pred_on.sum: Tensor Core ops"
echo "  - sm__inst_executed_pipe_tensor.avg: Tensor Core pipeline utilization"
echo "========================================================================"
