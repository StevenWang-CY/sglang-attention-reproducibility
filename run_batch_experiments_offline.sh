#!/bin/bash

# Script to run batch latency experiments using sglang.Engine (offline mode)
# This provides direct control over batching without HTTP server overhead
#
# Usage:
#   ./run_batch_experiments_offline.sh              # default: tpot mode
#   ./run_batch_experiments_offline.sh profile       # accurate per-op timing (sync all)  -> offline_batch_results/profile_sync/
#   ./run_batch_experiments_offline.sh profile-raw   # raw CUDA events (no sync, fast)    -> offline_batch_results/profile_raw/
#   ./run_batch_experiments_offline.sh tpot          # no profiling, cleanest TPOT         -> offline_batch_results/tpot/
#   ./run_batch_experiments_offline.sh tpot-no-share         # no profiling, no prefix sharing              -> offline_batch_results/tpot_no_share/
#   ./run_batch_experiments_offline.sh tpot-no-share-page16  # no profiling, no prefix sharing, page_size=16 -> offline_batch_results/tpot_no_share_page16/
#   ./run_batch_experiments_offline.sh profile-no-share # profiling, no prefix sharing    -> offline_batch_results/profile_sync_no_share/
#   ./run_batch_experiments_offline.sh nsys                # nsys GPU profiling (SM%, DRAM BW, with sharing)    -> offline_batch_results/nsys/
#   ./run_batch_experiments_offline.sh nsys-tpot-no-share  # nsys GPU profiling (no KV sharing)                 -> offline_batch_results/nsys_no_share/
#   ./run_batch_experiments_offline.sh ncu                 # ncu kernel profiling: exact DRAM bytes/step (with KV sharing)   -> offline_batch_results/ncu/
#   ./run_batch_experiments_offline.sh ncu-no-share        # ncu kernel profiling: exact DRAM bytes/step (no KV sharing)     -> offline_batch_results/ncu_no_share/
#   ./run_batch_experiments_offline.sh page-size-sweep          # sweep page_size=1..128, no prefix sharing (default token=4096)  -> offline_batch_results/page_size_sweep/
#   ./run_batch_experiments_offline.sh nsys-page-size-sweep     # nsys kernel timeline per page size (bs=1, 256 tokens default)   -> offline_batch_results/nsys_page_size_sweep/
#   ./run_batch_experiments_offline.sh page-size-sweep 256 4096 8192  # same but with multiple token sizes
#   ./run_batch_experiments_offline.sh page-size-sweep-bs            # sweep page_size across batch sizes (attention-bound regime — finds where page_size=1 is NOT lowest)  -> offline_batch_results/page_size_sweep_bs/
#   ./run_batch_experiments_offline.sh page-size-sweep-bs 8184       # same, longest decode (longest KV) only
# export SGLANG_DISABLE_CUDNN_CHECK=1

# Ensure sglang conda env is used (prepend to PATH so nsys inherits the right python)
SGLANG_ENV="/vast/projects/liuv/pennnetworks/jiaheng/miniconda3/envs/sglang"
export PATH="${SGLANG_ENV}/bin:${PATH}"
# Ensure libcudart is findable by the JIT kernel linker (tvm_ffi/ninja uses -lcudart)
export LIBRARY_PATH="${SGLANG_ENV}/lib:${SGLANG_ENV}/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"

REQUEST_FILE="html_request/request_005_20260316_221014/request.json"
MODEL_PATH="/vast/projects/liuv/pennnetworks/hf_models/Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME="Qwen3-VL-8B-Instruct"
MIN_TOKENS=0
IGNORE_EOS="--ignore-eos"
REPEAT=3

# Parse mode from first argument (default: profile)
MODE="${1:-tpot}"
# Parse optional token sizes from remaining positional args: e.g. page-size-sweep 256 4096 8192
if [ $# -ge 2 ]; then
    TOKEN_SIZES=("${@:2}")
fi

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
    tpot-no-share-page16)
        # No profiling, no prefix sharing, page_size=16 (matches quest benchmark config)
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/tpot_no_share_page16"
        MODE_DESC="TPOT only (no prefix sharing, page_size=16)"
        EXTRA_ARGS="--disable-radix-cache --page-size 16"
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
        # --disable-layerwise-nvtx-marker: prevents millions of per-layer NVTX events
        # that make QdstrmImporter take ~2 hours. We only need the bs_N markers.
        EXTRA_ARGS="--disable-radix-cache --disable-layerwise-nvtx-marker"
        REPEAT=1
        BATCH_SIZES=(1 2 4 8 16 32 64)
        TOKEN_SIZES=(256)
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
        BATCH_SIZES=(1 2 4 8 16 32 64)
        TOKEN_SIZES=(256)
        ;;
    nsys-page-size-sweep)
        # nsys profiling across page sizes — one engine boot per page size, captures
        # kernel timeline so you can compare attention kernel time between page sizes.
        # Use nsys-ui to open the .nsys-rep files side by side.
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/nsys_page_size_sweep"
        MODE_DESC="nsys page size sweep — kernel timeline per page size"
        EXTRA_ARGS="--disable-radix-cache --disable-layerwise-nvtx-marker"
        PAGE_SIZES_SWEEP=(1 2 4 8 16 32 64 128)
        REPEAT=1
        BATCH_SIZES=(2 4 32)
        if [ ${#TOKEN_SIZES[@]} -eq 0 ]; then
            TOKEN_SIZES=(256 512 1024 2048 4096 8184) # 512 1024 2048 4096 8184
        fi
        ;;
    page-size-sweep)
        # Sweep page_size=1,2,4,8,16,32,64,128 — no prefix sharing, no profiling
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/page_size_sweep"
        MODE_DESC="Page size sweep (1,2,4,8,16,32,64,128) — no prefix sharing"
        EXTRA_ARGS="--disable-radix-cache"
        PAGE_SIZES_SWEEP=(1 2 4 8 16 32 64 128)
        # 4096 output tokens keeps bs=64 within the KV pool (~9.7k prompt + 4k output < 14.3k/req budget)
        BATCH_SIZES=(1) # 2 4 8 16 32 64
        # TOKEN_SIZES may already be set from CLI args (positional args after mode)
        if [ ${#TOKEN_SIZES[@]} -eq 0 ]; then
            TOKEN_SIZES=(256 512 1024 2048 4096 8184)
        fi
        ;;
    page-size-sweep-bs)
        # Attention-bound variant of page-size-sweep: sweep page_size across MULTIPLE
        # batch sizes at a long decode length, so the measurement runs in the regime
        # where KV-cache reads — not weight loading — dominate TPOT.
        #
        # WHY: at bs=1 / short context the decode step is weight-bound (~16 GB of
        # Qwen3-VL-8B weights streamed from HBM every step), and the few-KB-per-token
        # attention read is negligible, so page_size barely moves TPOT and page_size=1
        # ties/wins inside run-to-run noise. As batch size and context grow, per-step
        # KV-cache reads scale (bytes ≈ batch × kv_len × 144 KB) and eventually dwarf
        # the fixed weight read; then the attention kernel dominates and larger pages
        # win (more coalesced HBM reads, less page-table indirection). That is the
        # scenario in which page_size=1 stops being the lowest-latency choice.
        #
        # The repo's isolated kernel microbenchmark (benchmark_page_size_attention.py /
        # page_size_attn_bench.json) already shows this: page_size=1 is the lowest in
        # only 1 of 12 (batch, kv_len) cells; a larger page (typically 8–64) wins the
        # other 11. This mode reproduces it end-to-end in the SGLang engine.
        export TREE_SPARSE_TIMING=0
        unset TREE_SPARSE_TIMING_SYNC_ALL
        unset TREE_SPARSE_TIMING_SYNC_OPS
        OUTPUT_DIR="offline_batch_results/page_size_sweep_bs"
        MODE_DESC="Page size sweep across batch sizes (attention-bound — finds where page_size=1 is NOT lowest)"
        EXTRA_ARGS="--disable-radix-cache"
        PAGE_SIZES_SWEEP=(1 2 4 8 16 32 64 128)
        # Batch sizes are chosen to push attention's share of the step up while still
        # fitting the ~9.7k-token prompt in the KV pool on a single 80 GB GPU.
        # IMPORTANT: check "concurrency_ok"/max_running_req_observed in each
        # results.json — if a batch did not actually run concurrently (KV pool limit)
        # its TPOT is not comparable across page sizes. Drop the largest batch (or
        # raise --mem-fraction-static) if you see a mismatch warning.
        BATCH_SIZES=(1 4 8 16)
        # Longest decode by default => longest KV => most attention-bound.
        if [ ${#TOKEN_SIZES[@]} -eq 0 ]; then
            TOKEN_SIZES=(8184)
        fi
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 [profile|profile-raw|tpot|tpot-no-share|tpot-no-share-page16|profile-no-share|nsys|nsys-tpot-no-share|ncu|ncu-no-share|page-size-sweep|nsys-page-size-sweep]"
        exit 1
        ;;
esac

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Array of batch sizes to test (can be overridden per mode above)
if [ ${#BATCH_SIZES[@]} -eq 0 ]; then
    # BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
    BATCH_SIZES=(1 2 4 8 16 32 64)
    # BATCH_SIZES=(64)
fi
# Array of token lengths to test (can be overridden per mode above)
if [ ${#TOKEN_SIZES[@]} -eq 0 ]; then
    TOKEN_SIZES=(256)
    # TOKEN_SIZES=(64000)
fi

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

    # Default results directory (modes that override RESULTS_DIR do their own mkdir)
    RESULTS_DIR="${OUTPUT_DIR}/results_token_${TOKEN_SIZE}_${TIMESTAMP}"
    if [[ "$MODE" != "page-size-sweep" && "$MODE" != "page-size-sweep-bs" && "$MODE" != "nsys-page-size-sweep" ]]; then
        mkdir -p "$RESULTS_DIR"
    fi

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
        # Determine physical GPU ID for nsys: CUDA_VISIBLE_DEVICES may remap logical 0 to a physical GPU
        GPU_PHYS="${CUDA_VISIBLE_DEVICES:-0}"
        # If multiple GPUs in CUDA_VISIBLE_DEVICES, use only the first
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

        # Post-process: extract per-BS GPU metrics using NVTX range timestamps
        NSYS_REP="${NSYS_OUT}.nsys-rep"
        SQLITE="${NSYS_OUT}.sqlite"
        echo ""
        echo "======================================================================"
        echo "Extracting per-BS GPU metrics from nsys report (via NVTX ranges)..."
        echo "======================================================================"
        # Auto-export nsys-rep to sqlite if not already present
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
    elif [ "$MODE" == "nsys-page-size-sweep" ]; then
        # ── nsys page size sweep: one engine boot + nsys profile per page size ─
        RESULTS_DIR="${OUTPUT_DIR}/token_${TOKEN_SIZE}_${TIMESTAMP}"
        mkdir -p "$RESULTS_DIR"
        echo "Results dir: $RESULTS_DIR"
        GPU_PHYS="${CUDA_VISIBLE_DEVICES:-0}"
        GPU_PHYS="${GPU_PHYS%%,*}"
        for PAGE_SIZE in "${PAGE_SIZES_SWEEP[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                PS_OUT="${RESULTS_DIR}/page_size_${PAGE_SIZE}/bs_${BATCH_SIZE}"
                PS_JSON="${PS_OUT}/results.json"
                PS_LOG="${PS_OUT}/runtime.log"
                NSYS_OUT="${PS_OUT}/nsys_report"
                mkdir -p "$PS_OUT"
                PS_CMD="python measure_batch_latency_offline.py \"$REQUEST_FILE\" \
                    --model-path \"$MODEL_PATH\" \
                    --batch-sizes $BATCH_SIZE \
                    --max-tokens $TOKEN_SIZE \
                    --repeat $REPEAT \
                    --page-size $PAGE_SIZE \
                    --enable-cuda-graph \
                    --output \"$PS_JSON\""
                if [ "$MIN_TOKENS" -gt 0 ]; then
                    PS_CMD="$PS_CMD --min-tokens $MIN_TOKENS"
                fi
                if [ -n "$IGNORE_EOS" ]; then
                    PS_CMD="$PS_CMD $IGNORE_EOS"
                fi
                PS_CMD="$PS_CMD $EXTRA_ARGS"
                NSYS_CMD="nsys profile \
                    --output=${NSYS_OUT} \
                    --force-overwrite=true \
                    --trace=cuda,nvtx \
                    --cuda-graph-trace=node \
                    --gpu-metrics-devices=cuda-visible \
                    --stats=false \
                    --resolve-symbols=false \
                    $PS_CMD"
                echo ""
                echo "--- page_size=$PAGE_SIZE bs=$BATCH_SIZE ---"
                echo "$NSYS_CMD"
                eval $NSYS_CMD 2>&1 | tee "$PS_LOG"
                if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                    echo "!!! page_size=$PAGE_SIZE bs=$BATCH_SIZE nsys run failed" >&2
                    exit 1
                fi
                echo "  nsys report: ${NSYS_OUT}.nsys-rep"
            done
        done

        echo ""
        echo "======================================================================"
        echo "nsys reports saved (open with nsys-ui for kernel timeline comparison):"
        for PAGE_SIZE in "${PAGE_SIZES_SWEEP[@]}"; do
            for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
                REP="${RESULTS_DIR}/page_size_${PAGE_SIZE}/bs_${BATCH_SIZE}/nsys_report.nsys-rep"
                [ -f "$REP" ] && echo "  page_size=$PAGE_SIZE bs=$BATCH_SIZE: $REP"
            done
        done
        echo "======================================================================"

        # ── Plot figures for this token size ──────────────────────────────────
        ANALYZE_SCRIPT="$(dirname "$0")/offline_batch_results/nsys_page_size_sweep/analyze_nsys.py"
        if [ -f "$ANALYZE_SCRIPT" ]; then
            echo "Plotting figures for $RESULTS_DIR ..."
            python "$ANALYZE_SCRIPT" "$RESULTS_DIR" && \
                echo "Plots saved to $RESULTS_DIR" || \
                echo "WARNING: plotting failed (non-fatal)"
        fi

    elif [ "$MODE" == "page-size-sweep" ] || [ "$MODE" == "page-size-sweep-bs" ]; then
        # ── Page size sweep: one engine boot per page size ────────────────────
        # Override RESULTS_DIR to reflect cuda-graph is enabled
        RESULTS_DIR="${OUTPUT_DIR}/cuda_graph_token_${TOKEN_SIZE}_${TIMESTAMP}"
        mkdir -p "$RESULTS_DIR"
        echo "Results dir (overridden): $RESULTS_DIR"
        for PAGE_SIZE in "${PAGE_SIZES_SWEEP[@]}"; do
            PS_OUT="${RESULTS_DIR}/page_size_${PAGE_SIZE}"
            PS_JSON="${PS_OUT}/results.json"
            PS_LOG="${PS_OUT}/runtime.log"
            mkdir -p "$PS_OUT"
            PS_CMD="python measure_batch_latency_offline.py \"$REQUEST_FILE\" \
                --model-path \"$MODEL_PATH\" \
                --batch-sizes $BATCH_ARGS \
                --max-tokens $TOKEN_SIZE \
                --repeat $REPEAT \
                --page-size $PAGE_SIZE \
                --enable-cuda-graph \
                --output \"$PS_JSON\""
            if [ "$MIN_TOKENS" -gt 0 ]; then
                PS_CMD="$PS_CMD --min-tokens $MIN_TOKENS"
            fi
            if [ -n "$IGNORE_EOS" ]; then
                PS_CMD="$PS_CMD $IGNORE_EOS"
            fi
            PS_CMD="$PS_CMD $EXTRA_ARGS"
            echo ""
            echo "--- page_size=$PAGE_SIZE ---"
            echo "$PS_CMD"
            eval $PS_CMD 2>&1 | tee "$PS_LOG"
            if [ "${PIPESTATUS[0]}" -ne 0 ]; then
                echo "!!! page_size=$PAGE_SIZE run failed" >&2
                exit 1
            fi
        done

        # Print comparison table across all page sizes
        echo ""
        echo "======================================================================"
        echo "Page size sweep summary (median TPOT ms)"
        echo "======================================================================"
        python3 - "$RESULTS_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

results_dir = Path(sys.argv[1])
page_dirs = sorted(results_dir.glob("page_size_*"),
                   key=lambda p: int(p.name.split("_")[-1]))

batch_sizes = []
all_data = {}
for pd in page_dirs:
    ps = int(pd.name.split("_")[-1])
    jf = pd / "results.json"
    if not jf.exists():
        continue
    with open(jf) as f:
        data = json.load(f)
    all_data[ps] = data
    if not batch_sizes:
        batch_sizes = sorted(data.keys(), key=int)

if not all_data:
    print("No results found.")
    sys.exit(0)

# Print table
header = f"{'page_size':>10}" + "".join(f"  bs={bs:>4}" for bs in batch_sizes)
print(header)
print("-" * len(header))
for ps, data in sorted(all_data.items()):
    row = f"{ps:>10}"
    for bs in batch_sizes:
        r = data.get(str(bs), data.get(bs, {}))
        tpot = r.get("tpot_median_ms", r.get("tpot_mean_ms", float("nan")))
        row += f"  {tpot:>7.2f}"
    print(row)

# ── Which page_size is fastest for each batch size? (answers the question) ────
print()
print("Lowest-latency page_size per batch size:")
ps_list = sorted(all_data.keys())
any_non_one = False
for bs in batch_sizes:
    col = []
    for ps in ps_list:
        r = all_data[ps].get(str(bs), all_data[ps].get(bs, {}))
        t = r.get("tpot_median_ms", r.get("tpot_mean_ms"))
        ok = all_data[ps].get(str(bs), all_data[ps].get(bs, {})).get("concurrency_ok")
        if t is not None and t > 0:
            col.append((ps, t, ok))
    if not col:
        continue
    best_ps, best_t, _ = min(col, key=lambda x: x[1])
    ps1_t = next((t for ps, t, _ in col if ps == 1), None)
    bad_conc = [ps for ps, _, ok in col if ok is False]
    if best_ps != 1:
        any_non_one = True
        verdict = f"page_size=1 is NOT lowest — loses to page_size={best_ps}"
    else:
        verdict = "page_size=1 IS lowest"
    extra = ""
    if ps1_t is not None and best_t > 0:
        extra = f"  [ps1={ps1_t:.3f} ms vs best={best_t:.3f} ms, ps1 is +{(ps1_t/best_t-1)*100:.2f}%]"
    warn = f"  (WARNING: page_size {bad_conc} did NOT run at full concurrency — TPOT not comparable)" if bad_conc else ""
    print(f"  bs={bs:>4}: best page_size={best_ps:>4} ({best_t:.3f} ms)  ->  {verdict}{extra}{warn}")
print()
print("=> Found a scenario where page_size=1 is NOT the lowest latency."
      if any_non_one else
      "=> page_size=1 was lowest for every batch size here — push batch size / context"
      "\n   higher (more attention-bound) to surface the crossover.")
print()

# Plot: one curve per page_size, x=batch_size, y=median TPOT
fig, ax = plt.subplots(figsize=(9, 5))
for ps, data in sorted(all_data.items()):
    xs, ys = [], []
    for bs in batch_sizes:
        r = data.get(str(bs), data.get(bs, {}))
        tpot = r.get("tpot_median_ms", r.get("tpot_mean_ms", None))
        if tpot is not None:
            xs.append(int(bs))
            ys.append(tpot)
    ax.plot(xs, ys, marker="o", label=f"page_size={ps}")

ax.set_xlabel("Batch size")
ax.set_ylabel("Median TPOT (ms)")
ax.set_title("TPOT vs Batch Size — page size sweep")
ax.legend(title="page_size", bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = results_dir / "page_size_sweep_tpot.png"
fig.savefig(plot_path, dpi=150)
print(f"\nPlot saved: {plot_path}")
PYEOF
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
