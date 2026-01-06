#!/bin/bash

# Add the sglang_log directory to Python path so custom handlers can be imported
export PYTHONPATH="~/sglang_log:$PYTHONPATH"

# Set CUDA library paths for FlashInfer compilation
# Add common CUDA library locations
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Skip FlashInfer JIT compilation if it fails (use pre-compiled kernels)
export FLASHINFER_ALLOW_JIT_FAILURE=1

# Optional: Save attention patterns for debugging (comment out if not needed)
# export SGLANG_SAVE_ATTN=1
# export SGLANG_SAVE_ATTN_LAYERS=0,1,9,16,23
# export SGLANG_SAVE_ATTN_HEADS=0,1

# Create log directory if it doesn't exist
LOG_DIR="deepseek-v32-log"
mkdir -p ${LOG_DIR}

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/runtime_deepseek_v32_${TIMESTAMP}.log"

echo "=================================================="
echo "Starting SGLang with DeepSeek V3.2 (NSA attention)"
echo "Log file: ${LOG_FILE}"
echo "=================================================="

# Configuration variables
MODEL_PATH="${HF_MODELS}/deepseek-ai/DeepSeek-V3.2"
TP_SIZE=4  # Tensor parallel size (using all 4 GPUs)
DP_SIZE=1  # Data parallel size (disabled - only 4 GPUs available)
PORT=8000
HOST="0.0.0.0"
CONTEXT_LENGTH=10240  # DeepSeek V3.2 supports up to 128K context

# Start SGLang server with DeepSeek V3.2 and NSA attention
# NSA (Neural Sparse Attention) is enabled automatically for DeepSeek V3.2
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name DeepSeek-V3.2 \
  --host ${HOST} \
  --port ${PORT} \
  --context-length ${CONTEXT_LENGTH} \
  --tp ${TP_SIZE} \
  --dp ${DP_SIZE} \
  --chunked-prefill-size 512 \
  --trust-remote-code \
  --log-level info \
  --log-requests \
  --log-requests-max-input-len -1 \
  --model-loader-extra-config '{"enable_multithread_load": true,"num_threads": 64}' \
  --watchdog-timeout 600 > ${LOG_FILE} 2>&1 &

# Save the server PID
SERVER_PID=$!
echo "SGLang server started with PID: ${SERVER_PID}"

# Wait for server to be ready
echo "Waiting for server to be ready..."
MAX_WAIT=300  # Maximum wait time in seconds (5 minutes)
WAIT_TIME=0
SLEEP_INTERVAL=5

while [ ${WAIT_TIME} -lt ${MAX_WAIT} ]; do
  # Check if server is responding
  if curl -s http://${HOST}:${PORT}/health > /dev/null 2>&1; then
    echo "Server is ready!"
    break
  fi

  # Check if server process is still running
  if ! kill -0 ${SERVER_PID} 2>/dev/null; then
    echo "ERROR: Server process died! Check log file: ${LOG_FILE}"
    exit 1
  fi

  echo "Server not ready yet, waiting... (${WAIT_TIME}s elapsed)"
  sleep ${SLEEP_INTERVAL}
  WAIT_TIME=$((WAIT_TIME + SLEEP_INTERVAL))
done

if [ ${WAIT_TIME} -ge ${MAX_WAIT} ]; then
  echo "ERROR: Server did not become ready within ${MAX_WAIT} seconds"
  echo "Killing server process..."
  kill ${SERVER_PID}
  exit 1
fi

# Execute the Python request script
echo ""
echo "=================================================="
echo "Executing send_sglang_request.py"
echo "=================================================="
python send_sglang_request.py

# Keep the server running (optional: comment out if you want script to exit after request)
echo ""
echo "=================================================="
echo "Request completed. Server is still running."
echo "Server PID: ${SERVER_PID}"
echo "Log file: ${LOG_FILE}"
echo "To stop the server, run: kill ${SERVER_PID}"
echo "=================================================="

# Wait for the server process (keeps script running)
wait ${SERVER_PID}
# kill ${SERVER_PID}
# Note: NSA attention backend is automatically selected for DeepSeek V3.2
# Default kernel configurations:
#   - H200: flashmla_sparse (prefill), fa3 (decode), bf16 kv cache
#   - B200: flashmla_auto (prefill), flashmla_kv (decode), fp8_e4m3 kv cache
#
# To manually specify NSA kernels, add:
#   --nsa-prefill-backend flashmla_sparse \
#   --nsa-decode-backend fa3 \
#
# For Multi-Token Prediction (MTP) optimization, add:
#   --speculative-algorithm EAGLE \
#   --speculative-num-steps 3 \
#   --speculative-eagle-topk 1 \
#   --speculative-num-draft-tokens 4 \