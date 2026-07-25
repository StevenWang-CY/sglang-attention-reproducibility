#!/bin/bash

# Optimized configuration for batch size 64
# Based on: 64 requests × ~9660 tokens each = ~618K tokens

# Add the sglang_log directory to Python path so custom handlers can be imported
pkill -f sglang.launch_server
export PYTHONPATH="~/sglang_log:$PYTHONPATH"

# Set CUDA library paths for FlashInfer compilation
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Skip FlashInfer JIT compilation if it fails
export FLASHINFER_ALLOW_JIT_FAILURE=1

# Enable decode step timing profiler
export TREE_SPARSE_TIMING=1
export TREE_SPARSE_TIMING_INTERVAL=10

# Create log directory
LOG_DIR="qwen3vl-log"
mkdir -p ${LOG_DIR}

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/runtime_qwen3vl_batch64_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SGLang (Optimized for Batch Size 64)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="

# Configuration variables
MODEL_PATH="${HF_MODELS}/Qwen/Qwen3-VL-8B-Instruct"
MODEL_NAME=$(basename ${MODEL_PATH})

# Auto-detect number of visible GPUs
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, using ${NUM_GPUS} GPU(s)"
else
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No CUDA_VISIBLE_DEVICES set, detected ${NUM_GPUS} total GPU(s)"
fi

TP_SIZE=${NUM_GPUS}
PORT=8000
HOST="0.0.0.0"
CONTEXT_LENGTH=40960

# OPTIMIZED SETTINGS FOR BATCH SIZE 64
# max-prefill-tokens: 800K (handles 64 × 9660 tokens + headroom)
# chunked-prefill-size: 12K (balance latency and throughput)
MAX_PREFILL_TOKENS=800000
CHUNKED_PREFILL_SIZE=12288

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Batch 64 optimizations:"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - max-prefill-tokens: ${MAX_PREFILL_TOKENS}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - chunked-prefill-size: ${CHUNKED_PREFILL_SIZE}"

# Start SGLang server
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name ${MODEL_NAME} \
  --host ${HOST} \
  --port ${PORT} \
  --context-length ${CONTEXT_LENGTH} \
  --tp ${TP_SIZE} \
  --trust-remote-code \
  --log-level debug \
  --log-requests \
  --attention-backend flashinfer \
  --disable-cuda-graph \
  --log-requests-level 3 \
  --enable-multimodal \
  --max-prefill-tokens ${MAX_PREFILL_TOKENS} \
  --chunked-prefill-size ${CHUNKED_PREFILL_SIZE} \
  --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 64}' \
  --watchdog-timeout 600 > ${LOG_FILE} 2>&1 &

SERVER_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SGLang server started with PID: ${SERVER_PID}"

# Wait for server to be ready
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to be ready..."
MAX_WAIT=1200
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Server process died! Check log: ${LOG_FILE}"
    exit 1
  fi

  if grep -q "Uvicorn running on" ${LOG_FILE} 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready!"
    break
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server not ready yet, waiting... (${WAIT_TIME}s)"
  sleep ${SLEEP_INTERVAL}
  WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Timeout waiting for server"
  kill ${SERVER_PID}
  exit 1
fi

echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready and running."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server PID: ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Configuration:"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Max prefill tokens: ${MAX_PREFILL_TOKENS}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Chunked prefill size: ${CHUNKED_PREFILL_SIZE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Context length: ${CONTEXT_LENGTH}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')]   - Tensor parallel: ${TP_SIZE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] To stop: kill ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="

# Keep server running
wait ${SERVER_PID}
