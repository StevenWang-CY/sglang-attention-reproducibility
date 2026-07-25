#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# Tree-sparse attention version — for comparing against full FlashInfer baseline

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=5
OUTPUT_DIR="offline_batch_results_tree_sparse"

# Enable detailed decode step timing profiler
export TREE_SPARSE_TIMING=1
export TREE_SPARSE_TIMING_INTERVAL=10  # Report every N decode steps

# Tree sparse attention config (must match server config)
TREE_SPARSE_TOP_K=8
TREE_SPARSE_MIN_SEQ_LEN=512
TREE_SPARSE_MIN_CHUNK_SIZE=16
TREE_SPARSE_MAX_CHUNK_SIZE=256
TREE_SPARSE_RECENT_TOKENS=128

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of batch sizes to test
# BATCH_SIZES=(1 2 4 8 16 32 64)
BATCH_SIZES=(1)
# Array of token lengths to test
TOKEN_SIZES=(256)

echo "========================================================================"
echo "SGLang Offline Batch Latency Experiments (Tree-Sparse Attention)"
echo "========================================================================"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo "Attention backend: tree_sparse"
echo "Request file: $REQUEST_FILE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Token sizes: ${TOKEN_SIZES[@]}"
echo "Repetitions: $REPEAT per batch size"
echo ""
echo "Tree-Sparse Config:"
echo "  top_k: $TREE_SPARSE_TOP_K"
echo "  min_seq_len: $TREE_SPARSE_MIN_SEQ_LEN"
echo "  min_chunk_size: $TREE_SPARSE_MIN_CHUNK_SIZE"
echo "  max_chunk_size: $TREE_SPARSE_MAX_CHUNK_SIZE"
echo "  recent_tokens: $TREE_SPARSE_RECENT_TOKENS"
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
    LOG_FILE="${OUTPUT_DIR}/runtime_offline_tree_sparse_tokens_${TOKEN_SIZE}_${TIMESTAMP}.log"

    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo ""

    # Build batch sizes argument
    BATCH_ARGS=$(printf " %s" "${BATCH_SIZES[@]}")

    # Run the offline experiment with tree-sparse attention
    CMD="python measure_batch_latency_offline.py \"$REQUEST_FILE\" \
        --model-path \"$MODEL_PATH\" \
        --batch-sizes $BATCH_ARGS \
        --max-tokens $TOKEN_SIZE \
        --repeat $REPEAT \
        --output \"$OUTPUT_FILE\" \
        --attention-backend tree_sparse \
        --tree-sparse-top-k $TREE_SPARSE_TOP_K \
        --tree-sparse-min-seq-len $TREE_SPARSE_MIN_SEQ_LEN \
        --tree-sparse-min-chunk-size $TREE_SPARSE_MIN_CHUNK_SIZE \
        --tree-sparse-max-chunk-size $TREE_SPARSE_MAX_CHUNK_SIZE \
        --tree-sparse-recent-tokens $TREE_SPARSE_RECENT_TOKENS"

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

    # Parse decode timing breakdown from the log file
    if [ -f "$LOG_FILE" ]; then
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
        fi

        # Copy TPOT plot into results dir for convenience
        if [ -f "$PLOT_FILE" ]; then
            cp "$PLOT_FILE" "$RESULTS_DIR/"
        fi

        # Extract TPOT summary to a simple text file for quick reference
        if [ -f "$OUTPUT_FILE" ]; then
            TPOT_FILE="$RESULTS_DIR/tpot.txt"
            python -c "
import json, sys
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
with open('$TPOT_FILE', 'w') as out:
    out.write('TPOT Summary (Time Per Output Token)\n')
    out.write('=' * 50 + '\n')
    out.write(f'Model: $MODEL_NAME\n')
    out.write(f'Attention: tree_sparse (top_k=$TREE_SPARSE_TOP_K)\n')
    out.write(f'Max tokens: $TOKEN_SIZE\n')
    out.write(f'Timestamp: $TIMESTAMP\n')
    out.write('=' * 50 + '\n\n')
    for bs_key, result in sorted(data.items(), key=lambda x: int(x[0])):
        bs = result['batch_size']
        tpot = result['tpot_mean_ms']
        std = result['tpot_std_ms']
        ttft = result['ttft_mean_ms']
        tps = result['throughput_tokens_per_sec']
        out.write(f'BS={bs}:  TPOT={tpot:.2f}ms (±{std:.2f})  TTFT={ttft:.1f}ms  Throughput={tps:.1f} tok/s\n')
    out.write('\n')
print(open('$TPOT_FILE').read())
"
            echo "  TPOT summary: $TPOT_FILE"
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
echo "Total time: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
echo ""
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "To view results:"
echo "  ls -lt $OUTPUT_DIR/"
echo ""
echo "To compare with FlashInfer baseline:"
echo "  python compare_attention_backends.py offline_batch_results/ $OUTPUT_DIR/"
echo ""
echo "========================================================================"
