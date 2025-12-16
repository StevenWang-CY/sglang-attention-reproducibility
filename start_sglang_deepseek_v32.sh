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

rm -f runtime.log

echo "=================================================="
echo "Starting SGLang with DeepSeek V3.2 (NSA attention)"
echo "=================================================="

# Configuration variables
MODEL_PATH="${HF_MODELS}/deepseek-ai/DeepSeek-V3.2"
TP_SIZE=6  # Tensor parallel size (adjust based on your GPU count)
DP_SIZE=6  # Data parallel size (adjust based on your setup)
PORT=8000
HOST="0.0.0.0"
CONTEXT_LENGTH=131072  # DeepSeek V3.2 supports up to 128K context

# Start SGLang server with DeepSeek V3.2 and NSA attention
# NSA (Neural Sparse Attention) is enabled automatically for DeepSeek V3.2
python -m sglang.launch_server \
  --model-path ${MODEL_PATH} \
  --served-model-name DeepSeek-V3.2-Exp \
  --host ${HOST} \
  --port ${PORT} \
  --context-length ${CONTEXT_LENGTH} \
  --tp ${TP_SIZE} \
  --dp ${DP_SIZE} \
  --enable-dp-attention \
  --chunked-prefill-size 512 \
  --trust-remote-code \
  --log-level info \
  --log-requests \
  --watchdog-timeout 600 > runtime.log 2>&1

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