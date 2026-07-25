#!/bin/bash
#
# Profile kernels with BASIC metrics only (no performance counter permission needed)
# This uses only public metrics that don't require GPU performance counter access
#

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=10

mkdir -p "$OUTPUT_DIR"

export TREE_SPARSE_TIMING=0

echo "========================================================================"
echo "Profiling Kernels (Basic Metrics - No Permissions Required)"
echo "========================================================================"
echo ""
echo "Using only kernel launch info and basic timing"
echo "No GPU performance counters needed"
echo "========================================================================"
echo ""

profile_basic() {
    local BATCH=$1
    echo ""
    echo "========================================================================"
    echo "Profiling Batch Size: $BATCH"
    echo "========================================================================"
    echo "Start time: $(date)"
    echo ""

    # Use --query-metrics to see available metrics without profiling
    # Then profile with NO metrics (just kernel names and basic timing)

    ncu \
        --csv \
        --log-file "${OUTPUT_DIR}/basic_kernels_batch${BATCH}.csv" \
        --target-processes all \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/results_basic_batch${BATCH}.json" \
            2>&1 | tee "${OUTPUT_DIR}/basic_kernels_batch${BATCH}_console.log"

    echo ""
    echo "End time: $(date)"
    echo "Profile saved to: ${OUTPUT_DIR}/basic_kernels_batch${BATCH}.csv"
}

# Profile batch 16 and 32
for BATCH in 16 32; do
    profile_basic $BATCH
    echo ""
    sleep 2
done

echo ""
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo ""
echo "Results:"
ls -lh "$OUTPUT_DIR"/basic_kernels_batch*.csv
echo ""
echo "This gives us kernel names and launch counts, which is enough to"
echo "confirm that different kernels are used at batch 16 vs 32!"
echo ""
echo "========================================================================"
