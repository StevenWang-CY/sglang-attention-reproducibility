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
LOG_DIR="deepseek-v32-quant-log"
mkdir -p ${LOG_DIR}

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/runtime_deepseek_v32_quant_${TIMESTAMP}.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting SGLang with DeepSeek V3.2 (NSA attention)"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="

# Configuration variables
MODEL_PATH="${HF_MODELS}/QuantTrio/DeepSeek-V3.2-AWQ"

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
CONTEXT_LENGTH=10240  # DeepSeek V3.2 supports up to 128K context

# DP attention is automatically enabled for DeepSeek NSA , SGLang sets dp_size = number of GPUs when DP attention is enabled
# Start SGLang server with DeepSeek V3.2 AWQ quantized model
# NSA (Neural Sparse Attention) is enabled automatically for DeepSeek V3.2
# AWQ quantization reduces memory footprint while maintaining performance
# Using awq_marlin for faster inference than regular awq
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name DeepSeek-V3.2-AWQ \
  --host ${HOST} \
  --port ${PORT} \
  --context-length ${CONTEXT_LENGTH} \
  --tp ${TP_SIZE} \
  --chunked-prefill-size 512 \
  --trust-remote-code \
  --quantization awq_marlin \
  --dtype bfloat16 \
  --log-level info \
  --log-requests \
  --mem-fraction-static 0.7 \
  --disable-cuda-graph \
  --nsa-prefill-backend flashmla_sparse \
  --nsa-decode-backend flashmla_sparse \
  --kv-cache-dtype bfloat16 \
  --moe-runner-backend auto \
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
    # Extract the actual page_size used by SGLang from the log
    PAGE_SIZE_LOG=$(grep -oP "page_size[= ]+\K[0-9]+" ${LOG_FILE} 2>/dev/null | tail -1)
    if [ -n "${PAGE_SIZE_LOG}" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Actual page_size used by SGLang: ${PAGE_SIZE_LOG}"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] Actual page_size used by SGLang: 1 (default, no override logged)"
    fi
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

# Keep the server running (optional: comment out if you want script to exit after request)
echo ""
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Request completed. Server is still running."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Server PID: ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Log file: ${LOG_FILE}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] To stop the server, run: kill ${SERVER_PID}"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] =================================================="

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