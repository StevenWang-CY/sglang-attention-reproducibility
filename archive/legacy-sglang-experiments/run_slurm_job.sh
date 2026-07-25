#!/bin/bash
#SBATCH --partition=dgx-b200
#SBATCH --job-name=sglang-serve
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --time=00:06:00
#SBATCH --qos=dgx
#SBATCH --output=logs/%x-%j.out

# Optional: load modules / env
# module load cuda/12.2
# source ~/miniconda3/bin/activate env312

echo "Running on host: $(hostname)"
nvidia-smi

# Your actual commands here:
# python -m sglang.launch_server ...
./start_sglang_deepseek_v32.sh > runtime_deepseek_v32.log 2>&1
