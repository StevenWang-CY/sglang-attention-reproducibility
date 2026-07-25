#!/bin/bash

# Script to run batch latency experiments with different concurrency levels (batch sizes)
# This script runs the client-side experiments AND captures the SGLang server logs
#
# KV Cache Management:
#   - FLUSH_CACHE_BETWEEN_TOKENS: Clear cache between token size experiments (recommended)
#   - FLUSH_CACHE_BETWEEN_BATCHES: Clear cache between batch size experiments
#   - RECORD_CACHE_STATS: Record cache usage statistics throughout experiments
#
# Cache statistics are recorded via:
#   - Server API endpoints (/server_info for cache stats, /flush_cache to clear cache)
#   - Server log parsing (extracts #cached-token, #running-req, etc.)

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL="Qwen3-VL-8B-Instruct"
MIN_TOKENS=0  # Set to 0 to disable minimum length, or set to desired minimum (e.g., 100)
IGNORE_EOS=true  # Set to true to force generation to max_tokens (recommended for consistent TPOT)
REPEAT=5
CLIENT_LOG_DIR="batch_experiment_logs"
SERVER_LOG_DIR="qwen3vl-log"
SERVER_URL="http://localhost:8000"  # SGLang server URL

# KV Cache control options
FLUSH_CACHE_BETWEEN_BATCHES=false  # Set to true to flush cache between batch sizes
FLUSH_CACHE_BETWEEN_TOKENS=true    # Set to true to flush cache between token sizes (recommended)
RECORD_CACHE_STATS=true            # Set to true to record cache usage statistics

# Create log directory if it doesn't exist
mkdir -p "$CLIENT_LOG_DIR"

# ======================================================================
# KV Cache Management Functions
# ======================================================================

# Function to flush KV cache
flush_kv_cache() {
    echo "  [Cache] Flushing KV cache..."
    RESPONSE=$(curl -s -X POST "${SERVER_URL}/flush_cache" -H "Content-Type: application/json" 2>&1)
    if [ $? -eq 0 ]; then
        echo "  [Cache] ✓ KV cache flushed successfully"
        return 0
    else
        echo "  [Cache] ⚠ Warning: Failed to flush cache (server may not support this endpoint)"
        return 1
    fi
}

# Function to get and record KV cache statistics
get_cache_stats() {
    local output_file="$1"
    local label="$2"

    echo "  [Cache] Recording cache statistics ($label)..."

    # Get server info which includes internal states and cache statistics
    STATS=$(curl -s "${SERVER_URL}/server_info" 2>&1)

    if [ $? -eq 0 ] && [ -n "$STATS" ]; then
        echo "========================================" >> "$output_file"
        echo "Cache Statistics - $label" >> "$output_file"
        echo "Time: $(date)" >> "$output_file"
        echo "========================================" >> "$output_file"
        echo "$STATS" | python3 -m json.tool >> "$output_file" 2>/dev/null || echo "$STATS" >> "$output_file"
        echo "" >> "$output_file"

        # Extract key cache metrics from internal_states
        KVCACHE_GB=$(echo "$STATS" | grep -o '"kvcache":[^,}]*' | head -1 | cut -d':' -f2 || echo "N/A")
        TOKEN_CAPACITY=$(echo "$STATS" | grep -o '"token_capacity":[^,}]*' | head -1 | cut -d':' -f2 || echo "N/A")
        THROUGHPUT=$(echo "$STATS" | grep -o '"last_gen_throughput":[^,}]*' | head -1 | cut -d':' -f2 || echo "N/A")

        echo "  [Cache] ✓ Stats recorded: KV cache=${KVCACHE_GB}GB, Capacity=${TOKEN_CAPACITY} tokens, Throughput=${THROUGHPUT} tok/s"
        return 0
    else
        echo "  [Cache] ⚠ Warning: Failed to get cache stats (server may not support this endpoint)"
        return 1
    fi
}

# Function to record cache stats from server logs
extract_cache_stats_from_log() {
    local log_file="$1"
    local output_file="$2"
    local label="$3"

    if [ ! -f "$log_file" ]; then
        return 1
    fi

    echo "  [Cache] Extracting cache stats from logs ($label)..."

    # Extract the most recent cache-related information from logs
    CACHE_INFO=$(tail -100 "$log_file" | grep -E "#cached-token|#running-req|#queue-req|tree_cache" | tail -5)

    if [ -n "$CACHE_INFO" ]; then
        echo "========================================" >> "$output_file"
        echo "Cache Info from Logs - $label" >> "$output_file"
        echo "Time: $(date)" >> "$output_file"
        echo "========================================" >> "$output_file"
        echo "$CACHE_INFO" >> "$output_file"
        echo "" >> "$output_file"
        echo "  [Cache] ✓ Log-based stats extracted"
        return 0
    fi

    return 1
}

# Array of batch sizes to test
# BATCH_SIZES=(1 2 4 8 16 32 64)
BATCH_SIZES=(64 32 16 8 4 2 1)
# BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
# BATCH_SIZES=(128)

# Array of token lengths to test
# TOKEN_SIZES=(512)
TOKEN_SIZES=(256)
# TOKEN_SIZES=(256 512 1024 2048)
echo "========================================================================"
echo "Batch Size Latency Experiments (Multi-Token-Length)"
echo "========================================================================"
echo "Date: $(date)"
echo "Request file: $REQUEST_FILE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Token sizes: ${TOKEN_SIZES[@]}"
echo "Repetitions: $REPEAT per batch size"
echo ""
echo "KV Cache Management:"
echo "  Flush between token sizes: $FLUSH_CACHE_BETWEEN_TOKENS"
echo "  Flush between batch sizes: $FLUSH_CACHE_BETWEEN_BATCHES"
echo "  Record cache statistics: $RECORD_CACHE_STATS"
echo ""
echo "IMPORTANT: Make sure SGLang server is running at $SERVER_URL"
echo "  Check with: ps aux | grep sglang"
echo "  Check API: curl -s ${SERVER_URL}/server_info | python3 -m json.tool"
echo "========================================================================"
echo ""

START_TIME=$(date +%s)

# Outer loop: iterate over token sizes
for TOKEN_SIZE in "${TOKEN_SIZES[@]}"; do
    echo ""
    echo "======================================================================"
    echo "Starting experiments for TOKEN_SIZE: $TOKEN_SIZE"
    echo "======================================================================"

    # Find the most recent server log at the start of each token size iteration
    CURRENT_SERVER_LOG=$(ls -t ${SERVER_LOG_DIR}/runtime_*.log 2>/dev/null | head -1)
    if [ -z "$CURRENT_SERVER_LOG" ]; then
        echo "WARNING: No SGLang server log found in ${SERVER_LOG_DIR}/"
        echo "Make sure the server is running and logging to ${SERVER_LOG_DIR}/"
        echo ""
    fi

    # Prepare cache statistics file for this token size
    CACHE_STATS_FILE="${SERVER_LOG_DIR}/cache_stats_token_${TOKEN_SIZE}_$(date +%Y%m%d_%H%M%S).log"

    # Flush cache at the start of each token size if enabled
    if [ "$FLUSH_CACHE_BETWEEN_TOKENS" = "true" ]; then
        echo ""
        echo "Flushing KV cache before TOKEN_SIZE=$TOKEN_SIZE experiments..."
        flush_kv_cache
        sleep 2
    fi

    # Record initial cache state
    if [ "$RECORD_CACHE_STATS" = "true" ]; then
        echo ""
        get_cache_stats "$CACHE_STATS_FILE" "Before TOKEN_SIZE=$TOKEN_SIZE"
        if [ -n "$CURRENT_SERVER_LOG" ] && [ -f "$CURRENT_SERVER_LOG" ]; then
            extract_cache_stats_from_log "$CURRENT_SERVER_LOG" "$CACHE_STATS_FILE" "Initial State"
        fi
    fi

    # Record the current log size (offset) before experiments start
    LOG_START_OFFSET=0
    if [ -n "$CURRENT_SERVER_LOG" ] && [ -f "$CURRENT_SERVER_LOG" ]; then
        LOG_START_OFFSET=$(wc -c < "$CURRENT_SERVER_LOG")
        echo "Recording log offset: $LOG_START_OFFSET bytes"
    fi

    TOKEN_START_TIME=$(date +%s)

    # Inner loop: iterate over batch sizes
    for batch_size in "${BATCH_SIZES[@]}"; do
        echo ""
        echo "Running: batch_size=$batch_size, token_size=$TOKEN_SIZE"
        echo "Time: $(date)"
        echo "--------------------------------"

        CLIENT_LOG="${CLIENT_LOG_DIR}/batch_${batch_size}_tokens_${TOKEN_SIZE}_$(date +%Y%m%d_%H%M%S).log"

        # Build command with optional parameters
        CMD="python measure_batch_latency_async.py \"$REQUEST_FILE\" \
            --concurrency \"$batch_size\" \
            --repeat \"$REPEAT\" \
            --model \"$MODEL\" \
            --max-tokens \"$TOKEN_SIZE\""

        # Add min-tokens if specified (> 0)
        if [ "$MIN_TOKENS" -gt 0 ]; then
            CMD="$CMD --min-tokens \"$MIN_TOKENS\""
        fi

        # Add ignore-eos if enabled
        if [ "$IGNORE_EOS" = "true" ]; then
            CMD="$CMD --ignore-eos"
        fi

        # Run the experiment (client-side)
        eval $CMD 2>&1 | tee "$CLIENT_LOG"

        echo ""
        echo "✓ Completed batch_size=$batch_size, token_size=$TOKEN_SIZE"
        echo "  Client log: $CLIENT_LOG"

        # Record cache stats after this batch size
        if [ "$RECORD_CACHE_STATS" = "true" ]; then
            get_cache_stats "$CACHE_STATS_FILE" "After batch_size=$batch_size"
            if [ -n "$CURRENT_SERVER_LOG" ] && [ -f "$CURRENT_SERVER_LOG" ]; then
                extract_cache_stats_from_log "$CURRENT_SERVER_LOG" "$CACHE_STATS_FILE" "batch_size=$batch_size completed"
            fi
        fi

        # Flush cache between batch sizes if enabled
        if [ "$FLUSH_CACHE_BETWEEN_BATCHES" = "true" ]; then
            echo ""
            flush_kv_cache
            sleep 2
        fi

        echo "--------------------------------"

        # Small delay between experiments
        sleep 3
    done

    TOKEN_END_TIME=$(date +%s)
    TOKEN_DURATION=$((TOKEN_END_TIME - TOKEN_START_TIME))

    # Record final cache state for this token size
    if [ "$RECORD_CACHE_STATS" = "true" ]; then
        echo ""
        echo "Recording final cache statistics..."
        get_cache_stats "$CACHE_STATS_FILE" "After all batches for TOKEN_SIZE=$TOKEN_SIZE"
        if [ -n "$CURRENT_SERVER_LOG" ] && [ -f "$CURRENT_SERVER_LOG" ]; then
            extract_cache_stats_from_log "$CURRENT_SERVER_LOG" "$CACHE_STATS_FILE" "Final State"
        fi
    fi

    echo ""
    echo "======================================================================"
    echo "Completed all batch sizes for TOKEN_SIZE=$TOKEN_SIZE"
    echo "Time for this token size: ${TOKEN_DURATION} seconds"
    echo "======================================================================"
    echo ""
    echo "Creating results directory and generating plots..."
    echo ""

    # Create results folder with batch size range, token size, and timestamp
    BATCH_RANGE=$(IFS=_; echo "${BATCH_SIZES[*]}")
    RESULTS_DIR="${SERVER_LOG_DIR}/results_batch_${BATCH_RANGE}_token_${TOKEN_SIZE}_$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$RESULTS_DIR"

    echo "Results directory: $RESULTS_DIR"

    # Extract only the relevant portion of the server log (from offset to current end)
    if [ -n "$CURRENT_SERVER_LOG" ] && [ -f "$CURRENT_SERVER_LOG" ]; then
        echo "Extracting relevant log portion (from offset $LOG_START_OFFSET)..."
        SERVER_LOG_NAME="runtime_${TOKEN_SIZE}tokens_$(date +%Y%m%d_%H%M%S).log"
        EXTRACTED_LOG="$RESULTS_DIR/$SERVER_LOG_NAME"

        # Extract from offset to end of file
        LOG_END_OFFSET=$(wc -c < "$CURRENT_SERVER_LOG")
        LOG_SIZE=$((LOG_END_OFFSET - LOG_START_OFFSET))

        if [ "$LOG_SIZE" -gt 0 ]; then
            tail -c +$((LOG_START_OFFSET + 1)) "$CURRENT_SERVER_LOG" > "$EXTRACTED_LOG"
            echo "  ✓ Extracted: $SERVER_LOG_NAME ($(du -h "$EXTRACTED_LOG" | cut -f1))"
            echo "  Original log: $(basename "$CURRENT_SERVER_LOG") ($(du -h "$CURRENT_SERVER_LOG" | cut -f1))"
        else
            echo "  ⚠ No new log data found (log size: $LOG_SIZE bytes)"
            # Copy full log as fallback
            cp "$CURRENT_SERVER_LOG" "$EXTRACTED_LOG"
            echo "  ✓ Copied full log: $SERVER_LOG_NAME"
        fi
    else
        echo "WARNING: Server log not found, skipping extraction"
    fi

    # Copy cache statistics file to results directory
    if [ "$RECORD_CACHE_STATS" = "true" ] && [ -f "$CACHE_STATS_FILE" ]; then
        cp "$CACHE_STATS_FILE" "$RESULTS_DIR/"
        echo "  ✓ Copied cache statistics: $(basename "$CACHE_STATS_FILE")"
    fi

    echo ""
    echo "========================================================================"
    echo "Running analysis on server logs (TOKEN_SIZE=$TOKEN_SIZE)..."
    echo "========================================================================"

    # Parse the server logs
    echo ""
    echo "[1/2] Parsing timing data from logs..."
    python parse_qwen3vl_logs.py "$RESULTS_DIR"

    # Check if parsing succeeded
    if [ ! -f "$RESULTS_DIR/timing_data.json" ]; then
        echo "ERROR: Parsing failed for TOKEN_SIZE=$TOKEN_SIZE, timing_data.json not found"
        continue  # Skip to next token size
    fi

    # Generate filtered plots with only the experimental batch sizes
    echo ""
    echo "[2/2] Generating latency breakdown plots (filtered to experimental batch sizes)..."
    BATCH_LIST=$(IFS=,; echo "${BATCH_SIZES[*]}")
    python plot_latency_breakdown_filtered.py "$RESULTS_DIR/timing_data.json" "$BATCH_LIST"

    echo ""
    echo "========================================================================"
    echo "Analysis complete for TOKEN_SIZE=$TOKEN_SIZE!"
    echo "========================================================================"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo ""

    # Verify and list generated files
    echo "Generated files:"
    if [ -f "$RESULTS_DIR/timing_data.json" ]; then
        echo "  ✓ timing_data.json ($(du -h "$RESULTS_DIR/timing_data.json" | cut -f1))"
    else
        echo "  ✗ timing_data.json (NOT FOUND)"
    fi

    if [ -f "$RESULTS_DIR/latency_breakdown_simple_filtered.png" ]; then
        echo "  ✓ latency_breakdown_simple_filtered.png ($(du -h "$RESULTS_DIR/latency_breakdown_simple_filtered.png" | cut -f1))"
    else
        echo "  ✗ latency_breakdown_simple_filtered.png (NOT FOUND)"
    fi

    if [ -f "$RESULTS_DIR/latency_breakdown_detailed_filtered.png" ]; then
        echo "  ✓ latency_breakdown_detailed_filtered.png ($(du -h "$RESULTS_DIR/latency_breakdown_detailed_filtered.png" | cut -f1))"
    else
        echo "  ✗ latency_breakdown_detailed_filtered.png (NOT FOUND)"
    fi

    if [ -f "$RESULTS_DIR/$SERVER_LOG_NAME" ]; then
        echo "  ✓ $SERVER_LOG_NAME ($(du -h "$RESULTS_DIR/$SERVER_LOG_NAME" | cut -f1))"
    fi

    if [ "$RECORD_CACHE_STATS" = "true" ] && [ -f "$RESULTS_DIR/$(basename "$CACHE_STATS_FILE")" ]; then
        echo "  ✓ $(basename "$CACHE_STATS_FILE") ($(du -h "$RESULTS_DIR/$(basename "$CACHE_STATS_FILE")" | cut -f1))"
    fi

    echo ""
    echo "All results for TOKEN_SIZE=$TOKEN_SIZE are in: $RESULTS_DIR"
    echo ""

    # Small delay before next token size
    sleep 5
done

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "========================================================================"
echo "ALL EXPERIMENTS COMPLETED!"
echo "========================================================================"
echo "Total time: ${DURATION} seconds ($(($DURATION / 60)) minutes)"
echo ""
echo "Tested configurations:"
echo "  Batch sizes: ${BATCH_SIZES[@]}"
echo "  Token sizes: ${TOKEN_SIZES[@]}"
echo "  Total combinations: $((${#BATCH_SIZES[@]} * ${#TOKEN_SIZES[@]}))"
echo ""
echo "KV Cache Management:"
echo "  Flush between token sizes: $FLUSH_CACHE_BETWEEN_TOKENS"
echo "  Flush between batch sizes: $FLUSH_CACHE_BETWEEN_BATCHES"
echo "  Record cache statistics: $RECORD_CACHE_STATS"
echo ""
echo "Results directories created in: $SERVER_LOG_DIR"
echo ""
echo "To view all results:"
echo "  ls -lt $SERVER_LOG_DIR/results_batch_*/"
echo ""
echo "To view plots for a specific token size (example for 256 tokens):"
echo "  display $SERVER_LOG_DIR/results_batch_*_token_256_*/latency_breakdown_simple_filtered.png &"
echo "  display $SERVER_LOG_DIR/results_batch_*_token_256_*/latency_breakdown_detailed_filtered.png &"
echo ""
echo "========================================================================"
