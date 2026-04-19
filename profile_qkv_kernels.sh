#!/bin/bash
#
# Profile QKV projection kernels at different batch sizes to confirm
# the Tensor Core activation hypothesis at batch=32
#
# This script uses nsys (NVIDIA Nsight Systems) to capture kernel launches
# and identify which GEMM kernels are used for QKV projection at different batch sizes.

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="profile_results"
MAX_TOKENS=50  # Short run for profiling

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "QKV Projection Kernel Profiling"
echo "========================================================================"
echo "This script profiles batch sizes 16, 32, and 64 to identify"
echo "the GEMM kernel change at batch=32 that causes QKV speedup"
echo ""
echo "Profiling method: nsys (NVIDIA Nsight Systems)"
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"
echo ""

# Enable detailed decode timing
export TREE_SPARSE_TIMING=1
export TREE_SPARSE_TIMING_INTERVAL=5

# Function to profile a single batch size
profile_batch_size() {
    local BATCH_SIZE=$1
    local PROFILE_FILE="${OUTPUT_DIR}/profile_batch${BATCH_SIZE}"

    echo ""
    echo "========================================================================"
    echo "Profiling Batch Size: $BATCH_SIZE"
    echo "========================================================================"

    # Run with nsys profiling
    # --trace=cuda,nvtx: Trace CUDA API and NVTX markers
    # --cuda-memory-usage=true: Track memory
    # --force-overwrite=true: Overwrite existing files
    # --sample=none: Disable CPU sampling to reduce overhead

    nsys profile \
        --trace=cuda,nvtx \
        --cuda-memory-usage=true \
        --force-overwrite=true \
        --sample=none \
        --output="${PROFILE_FILE}" \
        python measure_batch_latency_offline.py "$REQUEST_FILE" \
            --model-path "$MODEL_PATH" \
            --batch-sizes $BATCH_SIZE \
            --max-tokens $MAX_TOKENS \
            --repeat 1 \
            --output "${OUTPUT_DIR}/results_batch${BATCH_SIZE}.json" \
            --log-file "${OUTPUT_DIR}/log_batch${BATCH_SIZE}.log"

    echo ""
    echo "Profile saved to: ${PROFILE_FILE}.nsys-rep"
    echo ""

    # Extract GEMM kernel information from the profile
    echo "Analyzing GEMM kernels..."
    nsys stats --report cuda_gpu_kern_sum "${PROFILE_FILE}.nsys-rep" \
        --format csv \
        --output "${PROFILE_FILE}_kernel_summary" \
        2>/dev/null || true

    # Filter for GEMM-related kernels
    if [ -f "${PROFILE_FILE}_kernel_summary_cuda_gpu_kern_sum.csv" ]; then
        echo "Top GEMM kernels for batch size $BATCH_SIZE:"
        grep -i "gemm\|cutlass\|tensorop" "${PROFILE_FILE}_kernel_summary_cuda_gpu_kern_sum.csv" | head -20
        echo ""
    fi
}

# Profile the critical batch sizes around the transition
for BATCH in 16 32 64; do
    profile_batch_size $BATCH
    sleep 2
done

echo ""
echo "========================================================================"
echo "Profiling Complete!"
echo "========================================================================"
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"/*.nsys-rep 2>/dev/null
echo ""
echo "To view profiles interactively, use:"
echo "  nsys-ui $OUTPUT_DIR/profile_batch16.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/profile_batch32.nsys-rep"
echo "  nsys-ui $OUTPUT_DIR/profile_batch64.nsys-rep"
echo ""
echo "To extract kernel statistics:"
echo "  nsys stats --report cuda_gpu_kern_sum <profile>.nsys-rep"
echo ""
echo "Look for these kernel patterns:"
echo "  - Batch 16: generic GEMM kernels (sgemm, s884gemm, etc.)"
echo "  - Batch 32+: Tensor Core kernels (s*_tn_tensorop*, cutlass, etc.)"
echo ""
echo "========================================================================"
