#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# DeepSeek V3.2 (FP8) — MoE model with NSA (Neural Sparse Attention)
#
# Usage:
#   ./run_batch_experiments_offline_deepseek_v32.sh              # default: tpot mode
#   ./run_batch_experiments_offline_deepseek_v32.sh tpot          # no profiling, cleanest TPOT         -> offline_batch_results_deepseek_v32/tpot/
#   ./run_batch_experiments_offline_deepseek_v32.sh tpot-no-share         # no profiling, no prefix sharing              -> offline_batch_results_deepseek_v32/tpot_no_share/
#   ./run_batch_experiments_offline_deepseek_v32.sh tpot-no-share-page16  # no profiling, no prefix sharing, page_size=16 -> offline_batch_results_deepseek_v32/tpot_no_share_page16/
#   ./run_batch_experiments_offline_deepseek_v32.sh nsys                # nsys GPU profiling (SM%, DRAM BW, with sharing)    -> offline_batch_results_deepseek_v32/nsys/
#   ./run_batch_experiments_offline_deepseek_v32.sh nsys-no-share  # nsys GPU profiling (no KV sharing)                 -> offline_batch_results_deepseek_v32/nsys_no_share/
#   ./run_batch_experiments_offline_deepseek_v32.sh ncu                 # ncu kernel profiling: exact DRAM bytes/step (with KV sharing)   -> offline_batch_results_deepseek_v32/ncu/
#   ./run_batch_experiments_offline_deepseek_v32.sh ncu-no-share        # ncu kernel profiling: exact DRAM bytes/step (no KV sharing)     -> offline_batch_results_deepseek_v32/ncu_no_share/

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="${HF_MODELS}/deepseek-ai/DeepSeek-V3.2"
MODEL_NAME="DeepSeek-V3.2"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=3

# DeepSeek V3.2 engine configuration
# The model has native FP8 quantization (e4m3, block_size 128x128) in config.json,
# so SGLang will auto-detect it — no need to specify --quantization explicitly.
DTYPE="bfloat16"
KV_CACHE_DTYPE="bfloat16"
CONTEXT_LENGTH=10240
CHUNKED_PREFILL_SIZE=512
MEM_FRACTION_STATIC=0.7
MOE_RUNNER_BACKEND="triton"
NSA_PREFILL_BACKEND="flashmla_sparse"
NSA_DECODE_BACKEND="flashmla_sparse"
WATCHDOG_TIMEOUT=600

# Parse mode from first argument (default: tpot)
MODE="${1:-tpot}"

case "$MODE" in
    tpot)
        # No prefix sharing disabled: cleanest TPOT measurement
        OUTPUT_DIR="offline_batch_results_deepseek_v32/tpot"
        MODE_DESC="TPOT only (no profiling overhead)"
        EXTRA_ARGS=""
        ;;
    tpot-no-share)
        # No profiling, no prefix sharing (simulate different prompts)
        OUTPUT_DIR="offline_batch_results_deepseek_v32/tpot_no_share"
        MODE_DESC="TPOT only (no prefix sharing — simulates different prompts)"
        EXTRA_ARGS="--disable-radix-cache"
        ;;
    tpot-no-share-page16)
        # No profiling, no prefix sharing, page_size=16 (matches quest benchmark config)
        OUTPUT_DIR="offline_batch_results_deepseek_v32/tpot_no_share_page16"
        MODE_DESC="TPOT only (no prefix sharing, page_size=16)"
        EXTRA_ARGS="--disable-radix-cache --page-size 16"
        ;;
    nsys)
        # Nsight Systems profiling: accurate SM% and DRAM BW from hardware counters
        OUTPUT_DIR="offline_batch_results_deepseek_v32/nsys"
        MODE_DESC="Nsight Systems GPU profiling (SM%, DRAM BW, with KV sharing)"
        EXTRA_ARGS=""
        REPEAT=1
        ;;
    nsys-no-share)
        # Nsight Systems profiling, no prefix sharing
        OUTPUT_DIR="offline_batch_results_deepseek_v32/nsys_no_share"
        MODE_DESC="Nsight Systems GPU profiling (SM%, DRAM BW, no KV sharing)"
        EXTRA_ARGS="--disable-radix-cache --disable-layerwise-nvtx-marker"
        REPEAT=1
        ;;
    ncu)
        # Nsight Compute profiling: exact DRAM bytes read/written per attention kernel
        OUTPUT_DIR="offline_batch_results_deepseek_v32/ncu"
        MODE_DESC="NCU kernel profiling (exact DRAM bytes, with KV sharing)"
        EXTRA_ARGS=""
        REPEAT=1
        ;;
    ncu-no-share)
        # Nsight Compute profiling, no prefix sharing
        OUTPUT_DIR="offline_batch_results_deepseek_v32/ncu_no_share"
        MODE_DESC="NCU kernel profiling (exact DRAM bytes, no KV sharing)"
        EXTRA_ARGS="--disable-radix-cache"
        REPEAT=1
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [tpot|tpot-no-share|tpot-no-share-page16|nsys|nsys-no-share|ncu|ncu-no-share]"
        exit 1
        ;;
esac

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

# Array of batch sizes to test (can be overridden per mode above)
if [ ${#BATCH_SIZES[@]} -eq 0 ]; then
    # BATCH_SIZES=(1 2 4 8 16 32 64)
    BATCH_SIZES=(1)
fi
# Array of token lengths to test (can be overridden per mode above)
if [ ${#TOKEN_SIZES[@]} -eq 0 ]; then
    TOKEN_SIZES=(256)
fi

echo "========================================================================"
echo "SGLang Offline Batch Latency Experiments (DeepSeek V3.2 FP8)"
echo "========================================================================"
echo "Date: $(date)"
echo "Mode: $MODE_DESC"
echo "Model: $MODEL_PATH"
echo "Attention backend: nsa"
echo "Request file: $REQUEST_FILE"
echo "Batch sizes: ${BATCH_SIZES[@]}"
echo "Token sizes: ${TOKEN_SIZES[@]}"
echo "Repetitions: $REPEAT per batch size"
echo "TP size: $TP_SIZE"
echo ""
echo "DeepSeek V3.2 Config:"
echo "  dtype: $DTYPE"
echo "  kv_cache_dtype: $KV_CACHE_DTYPE"
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
    LOG_FILE="${OUTPUT_DIR}/runtime_offline_deepseek_v32_tokens_${TOKEN_SIZE}_${TIMESTAMP}.log"

    # Create results directory for this token size
    RESULTS_DIR="${OUTPUT_DIR}/results_token_${TOKEN_SIZE}_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR"

    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo "Results dir: $RESULTS_DIR"
    echo ""

    # Build batch sizes argument
    BATCH_ARGS=$(printf " %s" "${BATCH_SIZES[@]}")

    # Run the offline experiment with DeepSeek V3.2 config
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
        --model-loader-extra-config '{\"enable_multithread_load\": true, \"num_threads\": 64}'"

    # Add min-tokens if specified
    if [ "$MIN_TOKENS" -gt 0 ]; then
        CMD="$CMD --min-tokens $MIN_TOKENS"
    fi

    # Add ignore-eos if enabled
    if [ -n "$IGNORE_EOS" ]; then
        CMD="$CMD $IGNORE_EOS"
    fi

    # Add mode-specific extra args (e.g. --disable-radix-cache)
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Running command:"
    echo "$CMD"
    echo ""

    # Execute — dispatch based on profiling mode
    if [ "$MODE" == "ncu" ] || [ "$MODE" == "ncu-no-share" ]; then
        NCU_OUT="${RESULTS_DIR}/ncu_report"
        NCU_CMD="ncu \
            --target-processes all \
            --replay-mode kernel \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_read_lookup_hit_in_l1tex_pipe_lsu.sum \
            --kernel-name-base function \
            --kernel-name 'regex:(batch_decode|paged_attn|flashinfer|decode_attention|flash_attn|flashmla)' \
            --csv \
            --log-file ${NCU_OUT}.csv \
            -o ${NCU_OUT} \
            --force-overwrite \
            $CMD"
        echo "ncu command:"
        echo "$NCU_CMD"
        echo ""
        eval $NCU_CMD 2>&1 | tee "$LOG_FILE"

        echo ""
        echo "======================================================================"
        echo "NCU report: ${NCU_OUT}.ncu-rep"
        echo "To view interactively: ncu-ui ${NCU_OUT}.ncu-rep"
        echo ""
        echo "DRAM bytes summary (attention kernels):"
        if [ -f "${NCU_OUT}.csv" ]; then
            head -5 "${NCU_OUT}.csv" | cut -c1-120
            echo "..."
            echo "(Full CSV: ${NCU_OUT}.csv)"
            echo ""
            NCU_CSV="${NCU_OUT}.csv" python3 - <<'PYEOF'
import csv, sys, os
csv_path = os.environ.get('NCU_CSV', '')
if not os.path.exists(csv_path):
    print(f"CSV not found at {csv_path!r}, skipping summary")
    sys.exit(0)

with open(csv_path) as f:
    all_lines = f.readlines()

data_lines = [l for l in all_lines if not l.startswith('==') and l.strip()]

total_read = 0.0
total_write = 0.0
num_invocations = 0
try:
    reader = csv.DictReader(data_lines)
    for row in reader:
        for k, v in row.items():
            clean = v.strip().replace(',', '')
            try:
                val = float(clean) if clean else 0.0
            except ValueError:
                continue
            kl = k.lower()
            if 'dram__bytes_read' in kl:
                total_read += val
                num_invocations += 1
            elif 'dram__bytes_write' in kl:
                total_write += val
    print(f"Attention kernel invocations profiled: {num_invocations}")
    print(f"Total DRAM read  (attention kernels): {total_read/1e9:.3f} GB")
    print(f"Total DRAM write (attention kernels): {total_write/1e9:.3f} GB")
except Exception as e:
    print(f"Could not parse CSV: {e}")
PYEOF
        else
            echo "WARNING: NCU CSV not found at ${NCU_OUT}.csv"
        fi
        echo "======================================================================"

    elif [ "$MODE" == "nsys" ] || [ "$MODE" == "nsys-no-share" ]; then
        NSYS_OUT="${RESULTS_DIR}/nsys_report"
        GPU_PHYS="${CUDA_VISIBLE_DEVICES:-0}"
        GPU_PHYS="${GPU_PHYS%%,*}"
        NSYS_CMD="nsys profile \
            --output=${NSYS_OUT} \
            --force-overwrite=true \
            --trace=nvtx \
            --gpu-metrics-devices=${GPU_PHYS} \
            --stats=false \
            --resolve-symbols=false \
            $CMD"
        echo "nsys command:"
        echo "$NSYS_CMD"
        echo ""
        eval $NSYS_CMD 2>&1 | tee "$LOG_FILE"

        NSYS_REP="${NSYS_OUT}.nsys-rep"
        SQLITE="${NSYS_OUT}.sqlite"
        echo ""
        echo "======================================================================"
        echo "Extracting per-BS GPU metrics from nsys report (via NVTX ranges)..."
        echo "======================================================================"
        if [ ! -f "$SQLITE" ] && [ -f "$NSYS_REP" ]; then
            echo "Exporting nsys-rep to SQLite (this may take a minute)..."
            nsys export --type=sqlite --output="$SQLITE" --force-overwrite=true "$NSYS_REP" 2>&1
        fi
        if [ -f "$SQLITE" ]; then
            python3 - "$SQLITE" <<'PYEOF'
import sqlite3, sys

sqlite_path = sys.argv[1]
con = sqlite3.connect(sqlite_path)
peak = 7672  # B200 peak HBM BW in GB/s

try:
    nvtx_rows = con.execute("""
        SELECT text, start, end
        FROM NVTX_EVENTS
        WHERE text LIKE 'bs_%'
        ORDER BY start
    """).fetchall()
except Exception as e:
    print(f"Could not read NVTX_EVENTS: {e}")
    sys.exit(1)

if not nvtx_rows:
    print("No bs_* NVTX ranges found — did the Python script run with NVTX markers?")
    sys.exit(0)

print(f"\n{'BS':>4}  {'SMs Avg':>8} {'SMs Max':>8}  {'Tensor Avg':>10} {'Tensor Max':>10}  {'DRAM Rd Avg':>11} {'DRAM Rd Max':>11}  {'DRAM Wr Avg':>11}")
print("-" * 100)

for text, t0, t1 in nvtx_rows:
    bs = text
    rows = con.execute("""
        SELECT metricId, AVG(value), MAX(value)
        FROM GPU_METRICS
        WHERE metricId IN (3, 5, 9, 10)
          AND timestamp BETWEEN ? AND ?
        GROUP BY metricId ORDER BY metricId
    """, (t0, t1)).fetchall()

    metrics = {mid: (avg, mx) for mid, avg, mx in rows}
    sm_avg,  sm_mx  = metrics.get(3,  (0, 0))
    ten_avg, ten_mx = metrics.get(5,  (0, 0))
    rd_avg,  rd_mx  = metrics.get(9,  (0, 0))
    wr_avg,  wr_mx  = metrics.get(10, (0, 0))

    print(f"{bs:>4}  "
          f"{sm_avg:7.1f}% {sm_mx:7.1f}%  "
          f"{ten_avg:9.1f}% {ten_mx:9.1f}%  "
          f"{rd_avg:8.1f}%={rd_avg/100*peak:5.0f}GB/s "
          f"{rd_mx:8.1f}%={rd_mx/100*peak:5.0f}GB/s  "
          f"{wr_avg:8.1f}%={wr_avg/100*peak:4.0f}GB/s")

print()
PYEOF
        else
            echo "WARNING: nsys SQLite not found at $SQLITE"
        fi
    else
        # tpot / tpot-no-share / tpot-no-share-page16
        eval $CMD 2>&1 | tee "$LOG_FILE"
    fi

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

    # Copy log and results files
    cp "$LOG_FILE" "$RESULTS_DIR/"
    [ -f "$OUTPUT_FILE" ] && cp "$OUTPUT_FILE" "$RESULTS_DIR/"
    [ -f "$PLOT_FILE" ] && cp "$PLOT_FILE" "$RESULTS_DIR/"

    # Extract TPOT summary
    if [ -f "$OUTPUT_FILE" ]; then
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
    out.write('Mode: $MODE_DESC\n')
    out.write('Attention: NSA (Neural Sparse Attention)\n')
    out.write('Quantization: FP8 (native e4m3)\n')
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
    fi

    echo ""
    echo "========================================================================"
    echo "All results for TOKEN_SIZE=$TOKEN_SIZE"
    echo "========================================================================"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo ""
    echo "Generated files:"
    for f in "$RESULTS_DIR/"*.png "$RESULTS_DIR/"*.json "$RESULTS_DIR/"*.txt \
             "$RESULTS_DIR/"*.log "$RESULTS_DIR/"*.csv "$RESULTS_DIR/"*.ncu-rep "$RESULTS_DIR/"*.nsys-rep; do
        [ -f "$f" ] && echo "  $(basename "$f") ($(du -h "$f" | cut -f1))"
    done
    echo ""

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
