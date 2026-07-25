#!/bin/bash
#
# Profile with nsys (NVIDIA Nsight Systems) - works without GPU perf counter permissions
#

cd /vast/projects/liuv/pennnetworks/jiaheng/sglang_log

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=10

mkdir -p "$OUTPUT_DIR"

export TREE_SPARSE_TIMING=0

echo "========================================================================"
echo "Profiling with nsys (No GPU Performance Counter Permissions Needed)"
echo "========================================================================"
echo ""

profile_with_nsys() {
    local BATCH=$1

    echo ""
    echo "========================================================================"
    echo "Profiling Batch Size: $BATCH"
    echo "========================================================================"
    echo ""

    # nsys profile with CUDA trace
    # This captures kernel names and timings without needing GPU perf counters
    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --output="${OUTPUT_DIR}/nsys_batch${BATCH}" \
        --force-overwrite=true \
        --export=sqlite \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/nsys_results_batch${BATCH}.json"

    echo ""
    echo "Profile saved to: ${OUTPUT_DIR}/nsys_batch${BATCH}.nsys-rep"
    echo ""

    # Export to SQLite for analysis
    echo "Exporting kernel statistics..."

    # Use nsys stats to extract kernel info
    nsys stats \
        --report cuda_gpu_kern_sum \
        --format csv \
        --output "${OUTPUT_DIR}/nsys_kernels_batch${BATCH}" \
        "${OUTPUT_DIR}/nsys_batch${BATCH}.nsys-rep"

    if [ -f "${OUTPUT_DIR}/nsys_kernels_batch${BATCH}.csv" ]; then
        echo "Kernel summary saved to: ${OUTPUT_DIR}/nsys_kernels_batch${BATCH}.csv"
        echo ""
        echo "Top kernels by time:"
        head -20 "${OUTPUT_DIR}/nsys_kernels_batch${BATCH}.csv"
    fi
}

# Profile both batch sizes
for BATCH in 16 32; do
    profile_with_nsys $BATCH
    echo ""
    sleep 2
done

echo ""
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo ""
echo "To analyze the results:"
echo "  1. View kernel summaries: cat ${OUTPUT_DIR}/nsys_kernels_batch*.csv"
echo "  2. Compare batch 16 vs 32 kernel names"
echo "  3. Open in GUI: nsys-ui ${OUTPUT_DIR}/nsys_batch16.nsys-rep"
echo ""
echo "========================================================================"
