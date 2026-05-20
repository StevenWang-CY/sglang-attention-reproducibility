#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# This provides direct control over batching without HTTP server overhead
#
# Usage:
#   ./run_batch_experiments_offline.sh              # default: tpot mode
#   ./run_batch_experiments_offline.sh profile       # accurate per-op timing (sync all)  -> offline_batch_results/profile_sync/
#   ./run_batch_experiments_offline.sh profile-raw   # raw CUDA events (no sync, fast)    -> offline_batch_results/profile_raw/
#   ./run_batch_experiments_offline.sh tpot          # no profiling, cleanest TPOT         -> offline_batch_results/tpot/
#   ./run_batch_experiments_offline.sh tpot-no-share    # no profiling, no prefix sharing  -> offline_batch_results/tpot_no_share/
#   ./run_batch_experiments_offline.sh profile-no-share # profiling, no prefix sharing    -> offline_batch_results/profile_sync_no_share/
#   ./run_batch_experiments_offline.sh nsys                # nsys GPU profiling (SM%, DRAM BW, with sharing)    -> offline_batch_results/nsys/
#   ./run_batch_experiments_offline.sh nsys-tpot-no-share  # nsys GPU profiling (no KV sharing)                 -> offline_batch_results/nsys_no_share/
#   ./run_batch_experiments_offline.sh ncu                 # ncu kernel profiling: exact DRAM bytes/step (with KV sharing)   -> offline_batch_results/ncu/
#   ./run_batch_experiments_offline.sh ncu-no-share        # ncu kernel profiling: exact DRAM bytes/step (no KV sharing)     -> offline_batch_results/ncu_no_share/

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=3

# Parse mode from first argument (default: profile)
MODE="${1:-tpot}"

case "$MODE" in
    profile)
        # Accurate per-op timing: sync before every op to prevent PDL overlap
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=1
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/profile_sync"
        MODE_DESC="Profiling (sync all — accurate per-op timing)"
        EXTRA_ARGS=""
        ;;
    profile-raw)
        # Raw CUDA events: fast but per-op times may be inaccurate on Blackwell
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=0
        unset TREE_SPARSE_TIMING_SYNC_OPS
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/profile_raw"
        MODE_DESC="Profiling (raw CUDA events — no sync, fast)"
        EXTRA_ARGS=""
        ;;
    tpot)
        # No profiling: cleanest TPOT measurement
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/tpot"
        MODE_DESC="TPOT only (no profiling overhead)"
        EXTRA_ARGS=""
        ;;
    tpot-no-share)
        # No profiling, no prefix sharing (simulate different prompts)
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/tpot_no_share"
        MODE_DESC="TPOT only (no prefix sharing — simulates different prompts)"
        EXTRA_ARGS="--disable-radix-cache"
        ;;
    profile-no-share)
        # Profiling with sync, no prefix sharing
        export TREE_SPARSE_TIMING=1
        export TREE_SPARSE_TIMING_SYNC_ALL=1
        export TREE_SPARSE_TIMING_INTERVAL=10
        OUTPUT_DIR="offline_batch_results/profile_sync_no_share"
        MODE_DESC="Profiling (sync all, no prefix sharing)"
        EXTRA_ARGS="--disable-radix-cache"
        ;;
    nsys)
        # Nsight Systems profiling: accurate SM% and DRAM BW from hardware counters
        # Runs all batch sizes (defined below), with KV sharing
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/nsys"
        MODE_DESC="Nsight Systems GPU profiling (SM%, DRAM BW, with KV sharing)"
        EXTRA_ARGS=""
        REPEAT=1
        ;;
    nsys-tpot-no-share)
        # Nsight Systems profiling, no prefix sharing (matches tpot-no-share scenario)
        # Runs all batch sizes (defined below)
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/nsys_no_share"
        MODE_DESC="Nsight Systems GPU profiling (SM%, DRAM BW, no KV sharing)"
        EXTRA_ARGS="--disable-radix-cache"
        REPEAT=1
        ;;
    ncu)
        # Nsight Compute profiling: exact DRAM bytes read/written per attention kernel
        # Reports dram__bytes_read.sum / dram__bytes_write.sum for FlashInfer decode kernels.
        # MUCH more accurate than nvidia-smi memory% — gives actual byte counts.
        # NOTE: ncu replays each kernel multiple times → TPOT timings are not meaningful.
        #       Use this mode only for bandwidth/byte-count analysis, not TPOT.
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/ncu"
        MODE_DESC="NCU kernel profiling (exact DRAM bytes, with KV sharing)"
        EXTRA_ARGS=""
        REPEAT=1
        ;;
    ncu-no-share)
        # Nsight Compute profiling, no prefix sharing (real per-request KV loading)
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/ncu_no_share"
        MODE_DESC="NCU kernel profiling (exact DRAM bytes, no KV sharing)"
        EXTRA_ARGS="--disable-radix-cache"
        REPEAT=1
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [profile|profile-raw|tpot|tpot-no-share|profile-no-share|nsys|nsys-tpot-no-share|ncu|ncu-no-share]"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of batch sizes to test
# BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
BATCH_SIZES=(1 2 4 8 16 32 64 )
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

    # Create results directory for this token size (needed by nsys output path too)
    RESULTS_DIR="${OUTPUT_DIR}/results_token_${TOKEN_SIZE}_${TIMESTAMP}"
    mkdir -p "$RESULTS_DIR"

    echo "Output file: $OUTPUT_FILE"
    echo "Log file: $LOG_FILE"
    echo "Results dir: $RESULTS_DIR"
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

    # Add mode-specific extra args (e.g. --disable-radix-cache)
    if [ -n "$EXTRA_ARGS" ]; then
        CMD="$CMD $EXTRA_ARGS"
    fi

    echo "Running command:"
    echo "$CMD"
    echo ""

    # Execute — dispatch based on profiling mode
    if [ "$MODE" == "ncu" ] || [ "$MODE" == "ncu-no-share" ]; then
        # ── NCU mode: exact DRAM bytes per kernel ────────────────────────────
        # Metrics:
        #   dram__bytes_read.sum        — total HBM bytes read by each kernel
        #   dram__bytes_write.sum       — total HBM bytes written by each kernel
        #   sm__throughput.avg.pct_of_peak_sustained_elapsed — SM utilization %
        # --replay-mode kernel replays each kernel independently (faster than
        # application replay, accurate enough for memory metrics).
        # --kernel-name filters to only attention/decode kernels to keep runtime
        # manageable (skip all weight-loading, norm, etc. kernels).
        NCU_OUT="${RESULTS_DIR}/ncu_report"
        NCU_CMD="ncu \
            --target-processes all \
            --replay-mode kernel \
            --metrics dram__bytes_read.sum,dram__bytes_write.sum,sm__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_bytes_read_lookup_hit_in_l1tex_pipe_lsu.sum \
            --kernel-name-base function \
            --kernel-name 'regex:(batch_decode|paged_attn|flashinfer|decode_attention|flash_attn)' \
            --csv \
            --log-file ${NCU_OUT}.csv \
            -o ${NCU_OUT} \
            --force-overwrite \
            $CMD"
        echo "ncu command:"
        echo "$NCU_CMD"
        echo ""
        eval $NCU_CMD 2>&1 | tee "$LOG_FILE"

        # Post-process: sum DRAM bytes across all attention kernel invocations
        echo ""
        echo "======================================================================"
        echo "NCU report: ${NCU_OUT}.ncu-rep"
        echo "To view interactively: ncu-ui ${NCU_OUT}.ncu-rep"
        echo ""
        echo "DRAM bytes summary (attention kernels):"
        if [ -f "${NCU_OUT}.csv" ]; then
            # Print header + data rows (dram__bytes_read.sum column)
            head -5 "${NCU_OUT}.csv" | cut -c1-120
            echo "..."
            echo "(Full CSV: ${NCU_OUT}.csv)"
            echo ""
            # Sum dram__bytes_read.sum column (pass CSV path via env var)
            NCU_CSV="${NCU_OUT}.csv" python3 - <<'PYEOF'
import csv, sys, os
csv_path = os.environ.get('NCU_CSV', '')
if not os.path.exists(csv_path):
    print(f"CSV not found at {csv_path!r}, skipping summary")
    sys.exit(0)

with open(csv_path) as f:
    all_lines = f.readlines()

# ncu CSV has metadata lines starting with "==" and a double-header row;
# find the actual data lines (after the last header)
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

    # Execute — nsys mode: single process (model loads once), NVTX range per BS
    # Post-process slices GPU_METRICS by NVTX window to get per-BS SM%/DRAM BW
    elif [ "$MODE" == "nsys" ] || [ "$MODE" == "nsys-tpot-no-share" ]; then
        NSYS_OUT="${RESULTS_DIR}/nsys_report"
        NSYS_CMD="nsys profile \
            --output=${NSYS_OUT} \
            --force-overwrite=true \
            --trace=cuda,nvtx \
            --gpu-metrics-devices=0 \
            --stats=true \
            $CMD"
        echo "nsys command:"
        echo "$NSYS_CMD"
        echo ""
        eval $NSYS_CMD 2>&1 | tee "$LOG_FILE"

        # Post-process: extract per-BS GPU metrics using NVTX range timestamps
        NSYS_REP="${NSYS_OUT}.nsys-rep"
        SQLITE="${NSYS_OUT}.sqlite"
        echo ""
        echo "======================================================================"
        echo "Extracting per-BS GPU metrics from nsys report (via NVTX ranges)..."
        echo "======================================================================"
        if [ -f "$SQLITE" ]; then
            python3 - "$SQLITE" <<'PYEOF'
import sqlite3, sys

sqlite_path = sys.argv[1]
con = sqlite3.connect(sqlite_path)
peak = 7672  # B200 peak HBM BW in GB/s

# Find NVTX ranges named "bs_N"
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

names = {3: 'SMs Active   ', 5: 'Tensor Active', 9: 'DRAM Read BW ', 10: 'DRAM Write BW'}

print(f"\n{'BS':>4}  {'SMs Avg':>8} {'SMs Max':>8}  {'Tensor Avg':>10} {'Tensor Max':>10}  {'DRAM Rd Avg':>11} {'DRAM Rd Max':>11}  {'DRAM Wr Avg':>11}")
print("-" * 100)

for text, t0, t1 in nvtx_rows:
    bs = text  # e.g. "bs_8"
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
        echo "[1/3] Generating TPOT plot..."
        PLOT_FILE="${OUTPUT_FILE%.json}_tpot.png"
        python plot_offline_tpot.py "$OUTPUT_FILE" --output "$PLOT_FILE"
        echo ""
    else
        echo "WARNING: Results file not found, skipping plot generation"
    fi

    # Copy log and results files
    cp "$LOG_FILE" "$RESULTS_DIR/"
    [ -f "$OUTPUT_FILE" ] && cp "$OUTPUT_FILE" "$RESULTS_DIR/"
    [ -f "$PLOT_FILE" ] && cp "$PLOT_FILE" "$RESULTS_DIR/"

    # Generate GPU stats plot from results JSON
    if [ -f "$OUTPUT_FILE" ]; then
        echo "Generating GPU stats plot..."
        python plot_gpu_stats.py "$OUTPUT_FILE" --output "$RESULTS_DIR/"
    fi

    # Parse decode timing breakdown (profile modes only)
    if [ "$MODE" == "profile" ] || [ "$MODE" == "profile-raw" ] || [ "$MODE" == "profile-no-share" ]; then
        echo "Parsing timing data from logs..."
        python parse_qwen3vl_logs.py "$RESULTS_DIR"

        # Generate filtered plots with only the experimental batch sizes
        if [ -f "$RESULTS_DIR/timing_data.json" ]; then
            echo "Generating latency breakdown plots..."
            BATCH_LIST=$(IFS=,; echo "${BATCH_SIZES[*]}")
            python plot_latency_breakdown_filtered.py "$RESULTS_DIR/timing_data.json" "$BATCH_LIST"
        fi
    fi

    echo ""
    echo "========================================================================"
    echo "All results for TOKEN_SIZE=$TOKEN_SIZE"
    echo "========================================================================"
    echo ""
    echo "Results directory: $RESULTS_DIR"
    echo ""
    echo "Generated files:"
    for f in "$RESULTS_DIR/"*.png "$RESULTS_DIR/"*.json "$RESULTS_DIR/"*.log \
             "$RESULTS_DIR/"*.csv "$RESULTS_DIR/"*.ncu-rep "$RESULTS_DIR/"*.nsys-rep; do
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
