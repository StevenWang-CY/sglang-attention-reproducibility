#!/bin/bash
#SBATCH --partition=dgx-b200
#SBATCH --job-name=sglang-serve
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=00:15:00
#SBATCH --qos=dgx
#SBATCH --output=logs/%x-%j.out

# Optional: load modules / env
# module load cuda/12.2
# source ~/miniconda3/bin/activate env312

echo "Running on host: $(hostname)"

# Kill any existing Python/SGLang processes to free GPU memory
pkill -f "sglang.launch_server" || true
pkill -f "python.*sglang" || true
sleep 2

nvidia-smi

# export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Your actual commands here:
# python -m sglang.launch_server ...
LOG_DIR="qwen3vl-log"
mkdir -p ${LOG_DIR}

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/runtime_qwen3vl_${TIMESTAMP}_client.log"


./start_sglang_deepseek_v32_quant.sh > ${LOG_FILE} 2>&1
