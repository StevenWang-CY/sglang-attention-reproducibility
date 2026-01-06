#!/bin/bash

# Add the sglang_log directory to Python path so custom handlers can be imported
pkill -f sglang.launch_server
export PYTHONPATH="~/sglang_log:$PYTHONPATH"

# Set CUDA library paths for FlashInfer compilation
# Add common CUDA library locations
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Skip FlashInfer JIT compilation if it fails (use pre-compiled kernels)
export FLASHINFER_ALLOW_JIT_FAILURE=1
export CUDA_LAUNCH_BLOCKING=1

# Optional: Enable token ID debug prints (uncomment to enable)
# export SGLANG_TOKEN_ID_DEBUG=1

# Optional: Save attention patterns for debugging (comment out if not needed)
# export SGLANG_SAVE_ATTN=1
# export SGLANG_SAVE_ATTN_LAYERS=0,1,9,16,23
# export SGLANG_SAVE_ATTN_HEADS=0,1

# Create log directory if it doesn't exist
LOG_DIR="qwen3vl-log"
mkdir -p ${LOG_DIR}

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/runtime_qwen3vl_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SGLang with Qwen3-VL-30B-A3B-Instruct"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="

# Configuration variables
MODEL_PATH="${HF_MODELS}/Qwen/Qwen3-VL-30B-A3B-Instruct"

# Auto-detect number of visible GPUs from CUDA_VISIBLE_DEVICES
if [ -n "${CUDA_VISIBLE_DEVICES}" ]; then
    # Count comma-separated GPU IDs in CUDA_VISIBLE_DEVICES
    NUM_GPUS=$(echo "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}, using ${NUM_GPUS} GPU(s)"
else
    # Fallback: count all available GPUs
    NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No CUDA_VISIBLE_DEVICES set, detected ${NUM_GPUS} total GPU(s)"
fi

TP_SIZE=${NUM_GPUS}  # Tensor parallel size automatically set to number of visible GPUs
PORT=8000
HOST="0.0.0.0"
CONTEXT_LENGTH=40960  # Qwen3-VL supports large context

# Start SGLang server with logging
# Note: --disable-radix-cache can be added to avoid cache-related token recording issues (TODO)
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --host ${HOST} \
  --port ${PORT} \
  --context-length ${CONTEXT_LENGTH} \
  --tp ${TP_SIZE} \
  --trust-remote-code \
  --log-level debug \
  --log-requests \
  --log-requests-level 3 \
  --enable-multimodal \
  --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 64}' \
  --watchdog-timeout 600 > ${LOG_FILE} 2>&1 &

# Save the server PID
SERVER_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] SGLang server started with PID: ${SERVER_PID}"

# Wait for server to be ready by monitoring log file for ready message
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for server to be ready..."
MAX_WAIT=1200  # Maximum wait time in seconds (20 minutes)
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
  # Check if server process is still running
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Server process died! Check log file: ${LOG_FILE}"
    exit 1
  fi

  # Check log file for the definitive ready message
  if grep -q "The server is fired up and ready to roll!" ${LOG_FILE} 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready! (warmup completed)"
    break
  fi

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server not ready yet, waiting... (${WAIT_TIME}s elapsed)"
  sleep ${SLEEP_INTERVAL}
  WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: Server did not become ready within ${MAX_WAIT} seconds"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Killing server process..."
  kill ${SERVER_PID}
  exit 1
fi

# Execute the Python request script
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing send_sglang_request.py"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
python send_sglang_request.py

# Execute the Python request script again
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing send_sglang_request.py again"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
python send_sglang_request.py


# Keep the server running
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server is ready and running."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server PID: ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] To stop the server, run: kill ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="


# Wait for the server process (keeps script running)
wait ${SERVER_PID}
# kill ${SERVER_PID}