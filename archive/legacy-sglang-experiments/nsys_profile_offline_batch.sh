#!/bin/bash
# Profile offline batch experiments with nsys to capture kernel names

# =============================================================================
# Configuration - Edit these to control what to profile
# =============================================================================

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
REPEAT=1
TOKEN_SIZE=256

# Batch sizes to profile (space-separated)
# BATCH_SIZES=(16 32)
BATCH_SIZES=(64)
# BATCH_SIZES=(1 2 4 8 16 32 64)
# BATCH_SIZES=(8 16)

OUTPUT_DIR="nsys_offline_profiles"

# =============================================================================
# Start profiling
# =============================================================================

mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "Profiling Offline Batch Experiments with nsys"
echo "========================================================================"
echo "Model: $MODEL_PATH"
echo "Token size: $TOKEN_SIZE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Repeat: $REPEAT"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Profile each batch size
TOTAL=${#BATCH_SIZES[@]}
for i in "${!BATCH_SIZES[@]}"; do
    BATCH_SIZE=${BATCH_SIZES[$i]}
    IDX=$((i + 1))

    echo "========================================================================"
    echo "[$IDX/$TOTAL] Profiling batch size = $BATCH_SIZE"
    echo "========================================================================"

    SGLANG_LOG="$OUTPUT_DIR/sglang_output_batch_${BATCH_SIZE}.log"

    nsys profile \
      --output="$OUTPUT_DIR/batch_${BATCH_SIZE}_tokens_${TOKEN_SIZE}" \
      --force-overwrite=true \
      --trace=cuda,nvtx,osrt \
      --stats=true \
      python measure_batch_latency_offline.py "$REQUEST_FILE" \
        --model-path "$MODEL_PATH" \
        --batch-sizes $BATCH_SIZE \
        --max-tokens $TOKEN_SIZE \
        --repeat $REPEAT \
        --output "$OUTPUT_DIR/results_batch_${BATCH_SIZE}.json" \
      2>&1 | tee "$SGLANG_LOG"

    echo ""
    echo "✓ Batch $BATCH_SIZE profiling complete"
    echo "  SGLang log: $SGLANG_LOG"
    echo ""
done

# Extract kernel summaries to text files
echo "========================================================================"
echo "Extracting kernel summaries and NVTX markers"
echo "========================================================================"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    NSYS_REP="$OUTPUT_DIR/batch_${BATCH_SIZE}_tokens_${TOKEN_SIZE}.nsys-rep"
    KERNEL_TXT="$OUTPUT_DIR/kernels_batch_${BATCH_SIZE}.txt"
    NVTX_TXT="$OUTPUT_DIR/nvtx_batch_${BATCH_SIZE}.txt"

    if [ -f "$NSYS_REP" ]; then
        echo "Extracting kernels for batch $BATCH_SIZE..."
        nsys stats --report cuda_gpu_kern_sum "$NSYS_REP" > "$KERNEL_TXT"

        echo "Extracting NVTX markers for batch $BATCH_SIZE..."
        nsys stats --report nvtx_sum "$NSYS_REP" > "$NVTX_TXT"
    else
        echo "⚠ Warning: $NSYS_REP not found, skipping"
    fi
done

echo ""
echo "✓ Kernel summaries and NVTX markers extracted"
echo ""

# Compare kernels (if we have at least 2 batch sizes)
if [ ${#BATCH_SIZES[@]} -ge 2 ]; then
    echo "========================================================================"
    echo "Comparing kernels between batch sizes"
    echo "========================================================================"

    BATCH_1=${BATCH_SIZES[0]}
    BATCH_2=${BATCH_SIZES[1]}

    # Extract just kernel names
    grep -v "^Time\|^---\|^$\|^CUDA\|Executing" "$OUTPUT_DIR/kernels_batch_${BATCH_1}.txt" 2>/dev/null | \
      awk '{$1=$2=$3=$4=$5=$6=$7=""; print $0}' | sed 's/^[ \t]*//' | sort | uniq > /tmp/kernels_b${BATCH_1}.txt

    grep -v "^Time\|^---\|^$\|^CUDA\|Executing" "$OUTPUT_DIR/kernels_batch_${BATCH_2}.txt" 2>/dev/null | \
      awk '{$1=$2=$3=$4=$5=$6=$7=""; print $0}' | sed 's/^[ \t]*//' | sort | uniq > /tmp/kernels_b${BATCH_2}.txt

    echo "Kernels in BOTH batch $BATCH_1 and $BATCH_2 (same kernel used):"
    comm -12 /tmp/kernels_b${BATCH_1}.txt /tmp/kernels_b${BATCH_2}.txt | head -20
    echo ""
    echo "(Showing first 20, see full list in kernel summary files)"
    echo ""

    echo "Kernels ONLY in batch $BATCH_1:"
    comm -23 /tmp/kernels_b${BATCH_1}.txt /tmp/kernels_b${BATCH_2}.txt | head -10
    echo ""

    echo "Kernels ONLY in batch $BATCH_2:"
    comm -13 /tmp/kernels_b${BATCH_1}.txt /tmp/kernels_b${BATCH_2}.txt | head -10
    echo ""
fi

# Summary
echo "========================================================================"
echo "Summary"
echo "========================================================================"
echo ""
echo "Generated files in $OUTPUT_DIR/:"
ls -lh "$OUTPUT_DIR/" | tail -n +2
echo ""
echo "Kernel summary files:"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    KERNEL_TXT="$OUTPUT_DIR/kernels_batch_${BATCH_SIZE}.txt"
    if [ -f "$KERNEL_TXT" ]; then
        echo "  - $KERNEL_TXT"
    fi
done
echo ""
echo "NVTX marker files:"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    NVTX_TXT="$OUTPUT_DIR/nvtx_batch_${BATCH_SIZE}.txt"
    if [ -f "$NVTX_TXT" ]; then
        echo "  - $NVTX_TXT"
    fi
done
echo ""
echo "SGLang output logs:"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    SGLANG_LOG="$OUTPUT_DIR/sglang_output_batch_${BATCH_SIZE}.log"
    if [ -f "$SGLANG_LOG" ]; then
        echo "  - $SGLANG_LOG"
    fi
done
echo ""
echo "To view full kernel list for a specific batch:"
echo "  cat $OUTPUT_DIR/kernels_batch_<N>.txt"
echo ""
echo "To view NVTX markers (shows which layer each kernel belongs to):"
echo "  cat $OUTPUT_DIR/nvtx_batch_<N>.txt"
echo ""
echo "To compare two batches with compare script:"
echo "  bash compare_nsys_kernels.sh \\"
echo "    $OUTPUT_DIR/batch_<N1>_tokens_${TOKEN_SIZE}.nsys-rep \\"
echo "    $OUTPUT_DIR/batch_<N2>_tokens_${TOKEN_SIZE}.nsys-rep"
echo ""
echo "To view in GUI (with NVTX markers for layer identification):"
for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "  nsys-ui $OUTPUT_DIR/batch_${BATCH_SIZE}_tokens_${TOKEN_SIZE}.nsys-rep &"
done
echo "========================================================================"
