#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# DeepSeek V3.2 AWQ Quantized — MoE model with NSA (Neural Sparse Attention)

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="${HF_MODELS}/QuantTrio/DeepSeek-V3.2-AWQ"
MODEL_NAME="DeepSeek-V3.2-AWQ"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=3
OUTPUT_DIR="offline_batch_results_deepseek_v32_quant"

# DeepSeek V3.2 AWQ engine configuration
DTYPE="bfloat16"
KV_CACHE_DTYPE="bfloat16"
CONTEXT_LENGTH=10240
CHUNKED_PREFILL_SIZE=512
MEM_FRACTION_STATIC=0.7
MOE_RUNNER_BACKEND="auto"
NSA_PREFILL_BACKEND="flashmla_sparse"
NSA_DECODE_BACKEND="flashmla_sparse"
WATCHDOG_TIMEOUT=600
QUANTIZATION="awq_marlin"

# Auto-detect number of visible GPUs
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    TP_SIZE=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, using ${TP_SIZE} GPU(s)"
else
    TP_SIZE=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "No CUDA_VISIBLE_DEVICES set, detected ${TP_SIZE} total GPU(s)"
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of batch sizes to test
# BATCH_SIZES=(1 2 4 8 16 32 64)

BATCH_SIZES=(1)
# Array of token lengths to test
TOKEN_SIZES=(256)

echo "========================================================================"
echo "SGLang Offline Batch Latency Experiments (DeepSeek V3.2 AWQ)"
echo "========================================================================"
echo "Date: $(date)"
echo "Model: $MODEL_PATH"
echo "Quantization: $QUANTIZATION"
echo "Attention backend: nsa"
echo "Request file: $REQUEST_FILE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Token sizes: ${TOKEN_SIZES[@]}"
echo "Repetitions: $REPEAT per batch size"
echo "TP size: $TP_SIZE"
echo ""
echo "DeepSeek V3.2 AWQ Config:"
echo "  dtype: $DTYPE"
echo "  kv_cache_dtype: $KV_CACHE_DTYPE"
echo "  quantization: $QUANTIZATION"
echo "  context_length: $CONTEXT_LENGTH"
echo "  chunked_prefill_size: $CHUNKED_PREFILL_SIZE"
echo "  moe_runner_backend: $MOE_RUNNER_BACKEND"
echo "  nsa_prefill_backend: $NSA_PREFILL_BACKEND"
echo "  nsa_decode_backend: $NSA_DECODE_BACKEND"
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
    LOG_FILE="${OUTPUT_DIR}/runtime_offline_deepseek_v32_quant_tokens_${TOKEN_SIZE}_${TIMESTAMP}.log"

    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo ""

    # Build batch sizes argument
    BATCH_ARGS=$(printf " %s" "${BATCH_SIZES[@]}")

    # Run the offline experiment with DeepSeek V3.2 AWQ config
    CMD="python measure_batch_latency_offline.py \"$REQUEST_FILE\" \
        --model-path \"$MODEL_PATH\" \
        --batch-sizes $BATCH_ARGS \
        --max-tokens $TOKEN_SIZE \
        --repeat $REPEAT \
        --tp-size $TP_SIZE \
        --output \"$OUTPUT_FILE\" \
        --attention-backend nsa \
        --dtype $DTYPE \
        --kv-cache-dtype $KV_CACHE_DTYPE \
        --context-length $CONTEXT_LENGTH \
        --chunked-prefill-size $CHUNKED_PREFILL_SIZE \
        --mem-fraction-static $MEM_FRACTION_STATIC \
        --moe-runner-backend $MOE_RUNNER_BACKEND \
        --nsa-prefill-backend $NSA_PREFILL_BACKEND \
        --nsa-decode-backend $NSA_DECODE_BACKEND \
        --watchdog-timeout $WATCHDOG_TIMEOUT \
        --quantization $QUANTIZATION \
        --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 64}'"

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

    # Execute and capture all output
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
        echo "[1/2] Generating TPOT plot..."
        PLOT_FILE="${OUTPUT_FILE%.json}_tpot.png"
        python plot_offline_tpot.py "$OUTPUT_FILE" --output "$PLOT_FILE" 2>/dev/null && \
            echo "  Plot saved: $PLOT_FILE" || \
            echo "  WARNING: plot_offline_tpot.py not found or failed, skipping"
        echo ""
    else
        echo "WARNING: Results file not found, skipping plot generation"
    fi

    # Save results summary
    if [ -f "$OUTPUT_FILE" ]; then
        RESULTS_DIR="${OUTPUT_DIR}/results_token_${TOKEN_SIZE}_${TIMESTAMP}"
        mkdir -p "$RESULTS_DIR"

        # Copy log and results files
        cp "$LOG_FILE" "$RESULTS_DIR/"
        cp "$OUTPUT_FILE" "$RESULTS_DIR/"

        # Extract TPOT summary
        echo "[2/2] Generating TPOT summary..."
        TPOT_FILE="$RESULTS_DIR/tpot.txt"
        python -c "
import json
with open('$OUTPUT_FILE') as f:
    data = json.load(f)
with open('$TPOT_FILE', 'w') as out:
    out.write('TPOT Summary (Time Per Output Token)\n')
    out.write('=' * 60 + '\n')
    out.write('Model: $MODEL_NAME\n')
    out.write('Attention: NSA (Neural Sparse Attention)\n')
    out.write('Quantization: AWQ (awq_marlin)\n')
    out.write('TP size: $TP_SIZE\n')
    out.write('Max tokens: $TOKEN_SIZE\n')
    out.write('Timestamp: $TIMESTAMP\n')
    out.write('=' * 60 + '\n\n')
    for bs_key, result in sorted(data.items(), key=lambda x: int(x[0])):
        bs = result['batch_size']
        tpot = result['tpot_mean_ms']
        std = result['tpot_std_ms']
        ttft = result['ttft_mean_ms']
        tps = result['throughput_tokens_per_sec']
        out.write(f'BS={bs:>4}:  TPOT={tpot:.2f}ms (+-{std:.2f})  TTFT={ttft:.1f}ms  Throughput={tps:.1f} tok/s\n')
    out.write('\n')
print(open('$TPOT_FILE').read())
"
        echo "  TPOT summary: $TPOT_FILE"

        # Copy TPOT plot into results dir
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
        for f in "$RESULTS_DIR/"*.png "$RESULTS_DIR/"*.json "$RESULTS_DIR/"*.txt; do
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
echo "========================================================================"
