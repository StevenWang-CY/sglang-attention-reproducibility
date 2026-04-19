#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# This provides direct control over batching without HTTP server overhead
#
# Usage:
#   ./run_batch_experiments_offline.sh              # default: profiling with sync
#   ./run_batch_experiments_offline.sh profile       # accurate per-op timing (sync all)  -> offline_batch_results/profile_sync/
#   ./run_batch_experiments_offline.sh profile-raw   # raw CUDA events (no sync, fast)    -> offline_batch_results/profile_raw/
#   ./run_batch_experiments_offline.sh tpot          # no profiling, cleanest TPOT         -> offline_batch_results/tpot/

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=3

# Parse mode from first argument (default: profile)
MODE="${1:-profile}"

case "$MODE" in
    profile)
        # Accurate per-op timing: sync before every op to prevent PDL overlap
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=1
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/profile_sync"
        MODE_DESC="Profiling (sync all — accurate per-op timing)"
        ;;
    profile-raw)
        # Raw CUDA events: fast but per-op times may be inaccurate on Blackwell
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=0
        unset TREE_SPARSE_TIMING_SYNC_OPS
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/profile_raw"
        MODE_DESC="Profiling (raw CUDA events — no sync, fast)"
        ;;
    tpot)
        # No profiling: cleanest TPOT measurement
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/tpot"
        MODE_DESC="TPOT only (no profiling overhead)"
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [profile|profile-raw|tpot]"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of batch sizes to test
BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
# Array of token lengths to test
TOKEN_SIZES=(256)

echo "========================================================================"
echo "SGLang Offline Batch Latency Experiments"
echo "========================================================================"
echo "Date: $(date)"
echo "Mode: $MODE_DESC"
echo "Model: $MODEL_PATH"
echo "Request file: $REQUEST_FILE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Token sizes: ${TOKEN_SIZES[@]}"
echo "Repetitions: $REPEAT per batch size"
echo ""
echo "Output directory: $OUTPUT_DIR"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

# Outer loop: iterate over token sizes
for TOKEN_SIZE in "${TOKEN_SIZES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Starting experiments for TOKEN_SIZE: $TOKEN_SIZE"
    echo "======================================================================"

    TOKEN_START_TIME=$(date +%s)

    # Prepare output files
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_FILE="${OUTPUT_DIR}/results_batch_${BATCH_SIZES[0]}-${BATCH_SIZES[-1]}_tokens_${TOKEN_SIZE}_${TIMESTAMP}.json"
    LOG_FILE="${OUTPUT_DIR}/runtime_offline_tokens_${TOKEN_SIZE}_${TIMESTAMP}.log"

    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo ""

    # Build batch sizes argument
    BATCH_ARGS=$(printf " %s" "${BATCH_SIZES[@]}")

    # Run the offline experiment
    CMD="python measure_batch_latency_offline.py \"$REQUEST_FILE\" \
        --model-path \"$MODEL_PATH\" \
        --batch-sizes $BATCH_ARGS \
        --max-tokens $TOKEN_SIZE \
        --repeat $REPEAT \
        --output \"$OUTPUT_FILE\""

    # Add min-tokens if specified
    if [ "$MIN_TOKENS" -gt 0 ]; then
        CMD="$CMD --min-tokens $MIN_TOKENS"
    fi

    # Add ignore-eos if enabled
    if [ -n "$IGNORE_EOS" ]; then
        CMD="$CMD $IGNORE_EOS"
    fi

    echo "Running command:"
    echo "$CMD"
    echo ""

    # Execute and capture all output (including decode timing from subprocess)
    eval $CMD 2>&1 | tee "$LOG_FILE"

    TOKEN_END_TIME=$(date +%s)
    TOKEN_DURATION=$((TOKEN_END_TIME - TOKEN_START_TIME))

    echo ""
    echo "======================================================================"
    echo "Completed TOKEN_SIZE=$TOKEN_SIZE"
    echo "Time: ${TOKEN_DURATION} seconds"
    echo "======================================================================"
    echo ""

    # Generate TPOT plot
    if [ -f "$OUTPUT_FILE" ]; then
        echo "[1/3] Generating TPOT plot..."
        PLOT_FILE="${OUTPUT_FILE%.json}_tpot.png"
        python plot_offline_tpot.py "$OUTPUT_FILE" --output "$PLOT_FILE"
        echo ""
    else
        echo "WARNING: Results file not found, skipping plot generation"
    fi

    # Parse decode timing breakdown (skip in tpot mode — no profiling data)
    if [ "$MODE" != "tpot" ] && [ -f "$LOG_FILE" ]; then
        # Create a results directory for this token size
        RESULTS_DIR="${OUTPUT_DIR}/results_token_${TOKEN_SIZE}_${TIMESTAMP}"
        mkdir -p "$RESULTS_DIR"

        # Copy log file to results dir
        cp "$LOG_FILE" "$RESULTS_DIR/"

        echo "[2/3] Parsing timing data from logs..."
        python parse_qwen3vl_logs.py "$RESULTS_DIR"

        # Generate filtered plots with only the experimental batch sizes
        if [ -f "$RESULTS_DIR/timing_data.json" ]; then
            echo "[3/3] Generating latency breakdown plots..."
            BATCH_LIST=$(IFS=,; echo "${BATCH_SIZES[*]}")
            python plot_latency_breakdown_filtered.py "$RESULTS_DIR/timing_data.json" "$BATCH_LIST"
        else
            echo "  No timing_data.json found (profiler may not have produced DECODE STEP TIMING logs)"
            echo "  Check that attention_backend=flashinfer is set (DecodeStepTimer lives in flashinfer backend)"
        fi

        # Copy TPOT plot into results dir for convenience
        if [ -f "$PLOT_FILE" ]; then
            cp "$PLOT_FILE" "$RESULTS_DIR/"
        fi

        echo ""
        echo "========================================================================"
        echo "All results for TOKEN_SIZE=$TOKEN_SIZE"
        echo "========================================================================"
        echo ""
        echo "Results directory: $RESULTS_DIR"
        echo ""
        echo "Generated files:"
        for f in "$RESULTS_DIR/"*.png "$RESULTS_DIR/"*.json; do
            [ -f "$f" ] && echo "  $(basename "$f") ($(du -h "$f" | cut -f1))"
        done
        echo ""
    fi

    # Small delay before next token size
    sleep 2
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Mode: $MODE_DESC"
echo "Total time: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lt $OUTPUT_DIR/"
echo ""
echo "========================================================================"
