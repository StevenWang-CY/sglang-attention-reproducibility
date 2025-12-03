#!/bin/bash

# Add the sglang_log directory to Python path so the custom handler can be imported
export PYTHONPATH="~/sglang_log:$PYTHONPATH"

# Set CUDA library paths for FlashInfer compilation
# Add common CUDA library locations
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

# Skip FlashInfer JIT compilation if it fails (use pre-compiled kernels)
export FLASHINFER_ALLOW_JIT_FAILURE=1
export SGLANG_SAVE_ATTN=1
rm -f server.log

echo "=================================================="
echo "Starting SGLang with torch_native attention backend"
echo "=================================================="

# Start SGLang server with logging
python -m sglang.launch_server \
  --model-path $HF_MODELS/Qwen/Qwen3-VL-30B-A3B-Instruct \
  --served-model-name Qwen3-VL-30B-A3B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --context-length 40960 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --log-level info \
  --log-requests \
  --enable-multimodal \
  --attention-backend torch_native \
  --mm-attention-backend sdpa
