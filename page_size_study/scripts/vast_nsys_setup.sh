#!/bin/bash
# Setup a vast.ai pytorch:2.9.1-cuda12.8 container to replicate the phastform bench env
# (report 13 nsys experiments). Run as root inside the container, from /workspace.
set -ex
cd /workspace
nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader

apt-get update -q 2>&1 | tail -1
DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip python3-venv git wget 2>&1 | tail -1
pip install -q --upgrade pip
pip install -q torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install -q flashinfer-python==0.6.6 flashinfer-cubin==0.6.6 sgl-kernel==0.3.21 \
  transformers==4.57.1 ninja orjson uvloop pyzmq psutil einops "huggingface_hub[cli]" \
  fastapi uvicorn pydantic packaging pillow aiohttp requests tqdm setproctitle interegular

tar xzf sglang_CW_python.tgz
pip install -q -e python/ --no-deps

# driver-matched nsys (phastform lesson: older nsys traces silently empty on newer drivers)
wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb && apt-get update -q 2>&1 | tail -1
apt-get install -y -q cuda-nsight-systems-13-0 2>&1 | tail -1
NSYS=$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | sort -V | tail -1)
$NSYS --version

hf download Qwen/Qwen2.5-3B-Instruct --local-dir /workspace/Qwen2.5-3B-Instruct 2>&1 | tail -1

python - <<'EOF'
import torch, flashinfer
print("torch", torch.__version__, "| flashinfer", flashinfer.__version__)
import sglang
print("sglang", sglang.__version__)
EOF
echo SETUP_OK
