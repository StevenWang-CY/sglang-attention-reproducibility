#!/bin/bash
# Capture nsys profile for BS=32 and BS=64 to inspect CUDA stream assignments.
# Generates .nsys-rep files that can be opened in Nsight Systems GUI.

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
OUTPUT_DIR="nsys_profiles"
mkdir -p "$OUTPUT_DIR"

# Disable the custom profiler so it doesn't interfere
export TREE_SPARSE_TIMING=0

# Short run: 1 repeat, only 32 decode tokens (enough to see the pattern)
COMMON_ARGS="$REQUEST_FILE \
    --model-path $MODEL_PATH \
    --max-tokens 32 \
    --repeat 1 \
    --ignore-eos"

for BS in 32 64; do
    echo "========================================"
    echo "Profiling BS=$BS"
    echo "========================================"

    PROFILE_NAME="${OUTPUT_DIR}/bs${BS}"

    nsys profile \
        --output "$PROFILE_NAME" \
        --force-overwrite true \
        --trace cuda,nvtx \
        -- python -u measure_batch_latency_offline.py $COMMON_ARGS \
            --batch-sizes $BS \
            --output "${OUTPUT_DIR}/results_bs${BS}.json"

    echo ""
    echo "Profile saved: ${PROFILE_NAME}.nsys-rep"
    echo ""
done

echo "========================================"
echo "Done! To view profiles:"
echo "  nsys stats ${OUTPUT_DIR}/bs32.nsys-rep"
echo "  nsys stats ${OUTPUT_DIR}/bs64.nsys-rep"
echo ""
echo "Or open in Nsight Systems GUI for timeline view."
echo "========================================"
