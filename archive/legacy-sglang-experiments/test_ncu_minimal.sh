#!/bin/bash
#
# Test what ncu can capture without requiring GPU performance counter permissions
#

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"

export TREE_SPARSE_TIMING=0

echo "Testing ncu with different permission levels..."
echo ""

# Test 1: No metrics at all (just kernel trace)
echo "Test 1: No metrics - just kernel trace"
ncu \
    --kernel-name regex:nvjet.* \
    --print-summary per-kernel \
    --csv \
    --log-file "${OUTPUT_DIR}/test_no_metrics.csv" \
    python measure_batch_latency_offline.py "$REQUEST_FILE" \
        --model-path "$MODEL_PATH" \
        --batch-sizes 16 \
        --max-tokens 5 \
        --repeat 1 \
        --output "${OUTPUT_DIR}/test_result.json" \
        2>&1 | head -50

echo ""
echo "Check if it worked:"
if [ -f "${OUTPUT_DIR}/test_no_metrics.csv" ]; then
    echo "File created, size: $(wc -l < ${OUTPUT_DIR}/test_no_metrics.csv) lines"
    echo "First 20 lines:"
    head -20 "${OUTPUT_DIR}/test_no_metrics.csv"
else
    echo "File not created - permission issue"
fi
