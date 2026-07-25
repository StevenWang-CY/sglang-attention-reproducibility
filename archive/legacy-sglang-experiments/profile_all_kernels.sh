#!/bin/bash
#
# Profile ALL kernels at batch 16 vs 32 (no filtering)
# Then analyze the results with a Python script
#

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=10

mkdir -p "$OUTPUT_DIR"

export TREE_SPARSE_TIMING=0

echo "========================================================================"
echo "Profiling ALL Kernels (No Filter)"
echo "========================================================================"
echo ""
echo "This will profile every kernel launched during decode"
echo "WARNING: This will be VERY SLOW (~5-10 minutes per batch)"
echo "========================================================================"
echo ""

profile_all() {
    local BATCH=$1
    echo ""
    echo "========================================================================"
    echo "Profiling Batch Size: $BATCH"
    echo "========================================================================"
    echo "Start time: $(date)"
    echo ""

    # Profile ALL kernels with key metrics
    # No --kernel-name filter = profile everything
    # Minimal metrics to reduce overhead

    ncu \
        --metrics gpu__time_duration.sum,sm__inst_executed_pipe_tensor.avg \
        --csv \
        --log-file "${OUTPUT_DIR}/all_kernels_batch${BATCH}.csv" \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/results_all_batch${BATCH}.json" \
            2>&1 | tee "${OUTPUT_DIR}/all_kernels_batch${BATCH}_console.log"

    echo ""
    echo "End time: $(date)"
    echo "Profile saved to: ${OUTPUT_DIR}/all_kernels_batch${BATCH}.csv"
}

# Profile batch 16 and 32
for BATCH in 16 32; do
    profile_all $BATCH
    echo ""
    sleep 2
done

echo ""
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo ""
echo "Results:"
ls -lh "$OUTPUT_DIR"/all_kernels_batch*.csv
echo ""
echo "Now analyze with:"
echo "  python analyze_all_kernel_profiles.py"
echo ""
echo "========================================================================"
