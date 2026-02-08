curl -LsSf https://astral.sh/uv/install.sh | sh

echo 'export HF_MODELS=/data/hf_models' >> ~/.bashrc && source ~/.bashrc
echo $HF_MODELS
cd ~/sglang_my

# Install the python packages
uv pip install --upgrade pip
uv pip install -e "python"
