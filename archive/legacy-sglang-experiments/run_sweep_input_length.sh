#!/bin/bash

# Sweep input sequence lengths to measure how TPOT scales with context length.
# Generates dummy prompts at various token counts and benchmarks each.
#
# Always run from the script's own directory
cd "$(dirname "$0")"

# Fix FlashInfer JIT linking: add libcuda.so stub to library path (needed for MoE models)
export LIBRARY_PATH="${CONDA_PREFIX}/lib/stubs:${LIBRARY_PATH:-}"
#
# Usage:
#   ./run_sweep_input_length.sh              # default: tpot mode (cleanest measurement)
#   ./run_sweep_input_length.sh tpot         # no profiling, cleanest TPOT
#   ./run_sweep_input_length.sh profile      # with profiling (sync all)

# MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"

# MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-30B-A3B-Instruct"

MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-32B-Instruct"


REPEAT=3
MAX_DECODE_TOKENS=128

# Input lengths to sweep (in tokens)
# INPUT_LENGTHS=(10000 20000 30000 40000 50000 60000 70000)
INPUT_LENGTHS=(10000 50000 100000 150000 200000)
# INPUT_LENGTHS=(10000)
# INPUT_LENGTHS=(100000 200000 400000 600000 800000 1000000)

# Batch sizes to test at each input length
# BATCH_SIZES=(1 4 16 64)
BATCH_SIZES=(1)
# Parse mode
MODE="${1:-tpot}"

case "$MODE" in
    profile)
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=1
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/sweep_input_length_profile"
        MODE_DESC="Profiling (sync all)"
        ;;
    tpot)
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/sweep_input_length_tpot"
        MODE_DESC="TPOT only (no profiling)"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [tpot|profile]"
        exit 1
        ;;
esac

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_NAME=$(basename "$MODEL_PATH")
RESULTS_DIR="${OUTPUT_DIR}/sweep_${MODEL_NAME}_${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

# Save run metadata
cat > "$RESULTS_DIR/run_info.json" <<METAEOF
{
  "timestamp": "$TIMESTAMP",
  "mode": "$MODE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "input_lengths": [$(IFS=,; echo "${INPUT_LENGTHS[*]}")],
  "batch_sizes": [$(IFS=,; echo "${BATCH_SIZES[*]}")],
  "max_decode_tokens": $MAX_DECODE_TOKENS,
  "repeat": $REPEAT,
  "hostname": "$(hostname)",
  "gpu": "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
}
METAEOF

# Shared dummy requests — generated once per model series (same tokenizer)
# Extract series name: e.g. "Qwen3-VL" from "Qwen3-VL-8B-Instruct"
MODEL_SERIES=$(echo "$MODEL_NAME" | sed 's/-[0-9]*[BbMm].*$//')
DUMMY_DIR="dummy_requests/${MODEL_SERIES}"

echo "========================================================================"
echo "Input Length Sweep — SGLang Offline Benchmark"
echo "========================================================================"
echo "Date: $(date)"
echo "Mode: $MODE_DESC"
echo "Model: $MODEL_PATH"
echo "Input lengths: ${INPUT_LENGTHS[@]}"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Decode tokens: $MAX_DECODE_TOKENS"
echo "Repetitions: $REPEAT"
echo "Results: $RESULTS_DIR"
echo "========================================================================"
echo ""

# Step 1: Generate only missing dummy requests
MISSING_LENGTHS=()
for LEN in "${INPUT_LENGTHS[@]}"; do
    [ ! -f "${DUMMY_DIR}/dummy_${LEN}tok.json" ] && MISSING_LENGTHS+=("$LEN")
done

if [ ${#MISSING_LENGTHS[@]} -gt 0 ]; then
    echo "Generating dummy requests for: ${MISSING_LENGTHS[*]}"
    python generate_dummy_requests.py \
        --lengths ${MISSING_LENGTHS[@]} \
        --model-path "$MODEL_PATH" \
        --output-dir "$DUMMY_DIR"
else
    echo "Reusing existing dummy requests from ${DUMMY_DIR}/"
fi
echo ""

START_TIME=$(date +%s)

# Step 2: Run benchmark for each input length
BATCH_ARGS=$(printf " %s" "${BATCH_SIZES[@]}")
ALL_RESULTS_FILE="${RESULTS_DIR}/all_results.json"

# Initialize the combined results file
echo "{}" > "$ALL_RESULTS_FILE"

for INPUT_LEN in "${INPUT_LENGTHS[@]}"; do
    REQUEST_FILE="${DUMMY_DIR}/dummy_${INPUT_LEN}tok.json"

    if [ ! -f "$REQUEST_FILE" ]; then
        echo "WARNING: $REQUEST_FILE not found, skipping"
        continue
    fi

    echo "========================================================================"
    echo "Input length: ${INPUT_LEN} tokens"
    echo "========================================================================"

    OUTPUT_FILE="${RESULTS_DIR}/results_input${INPUT_LEN}.json"
    LOG_FILE="${RESULTS_DIR}/runtime_input${INPUT_LEN}.log"

    python -u measure_batch_latency_offline.py "$REQUEST_FILE" \
        --model-path "$MODEL_PATH" \
        --batch-sizes $BATCH_ARGS \
        --max-tokens $MAX_DECODE_TOKENS \
        --repeat $REPEAT \
        --output "$OUTPUT_FILE" \
        --ignore-eos 2>&1 | tee "$LOG_FILE"

    echo ""
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Step 3: Combine results and plot
echo ""
echo "Combining results and generating plots..."
python plot_input_length_sweep.py "$RESULTS_DIR"

echo ""
echo "========================================================================"
echo "SWEEP COMPLETED!"
echo "========================================================================"
echo "Mode: $MODE_DESC"
echo "Total time: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
echo ""
echo "Results: $RESULTS_DIR"
echo ""
echo "Generated files:"
for f in "$RESULTS_DIR/"*.png "$RESULTS_DIR/"*.json; do
    [ -f "$f" ] && echo "  $(basename "$f") ($(du -h "$f" | cut -f1))"
done
echo "========================================================================"
