# Report-9 vLLM env on phastform. FlashInfer-on-sm120 needs nvcc>=12.9; the only such nvcc here
# is the pip one in the venv (nvidia/cu13, nvcc 13.2). /usr/local/cuda-13.0 is ONLY ncu.
# The pip cuda-toolkit 13.0.2 is version-skewed (nvcc 13.2 vs headers 13.0); flashinfer's bundled
# cccl rejects the minor mismatch, so we use its documented escape hatch (nvcc 13.2 vs CTK 13.0
# headers is safe within CUDA-13). ninja (venv bin) must stay on PATH for the JIT.
export CUDA_HOME=$HOME/venvs/vllm/lib/python3.10/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$HOME/venvs/vllm/bin:$PATH
export TORCH_CUDA_ARCH_LIST=12.0
export VLLM_ATTENTION_BACKEND=FLASHINFER
export NVCC_APPEND_FLAGS="-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK"
