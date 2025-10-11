#!/bin/bash
# ==========================================
# Setup Script for Mobile-VideoGPT
# Based on: https://github.com/Amshaker/Mobile-VideoGPT#installation
# Tested for CUDA 11.8 / PyTorch 2.1+
# ==========================================

set -e  # stop on error

echo "🔧 Creating workspace..."

cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
conda init bash
source ~/.bashrc

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r



# python3 -m venv venv
# source venv/bin/activate

echo "📦 Cloning repositories..."


# --------------------------------------------------
# 1️⃣ Install base dependencies
# --------------------------------------------------
echo "🧱 Installing base Python packages..."
pip install --upgrade pip
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

pip install transformers==4.41.0

pip install -r requirements.txt

export PYTHONPATH="./:$PYTHONPATH"

# --------------------------------------------------
# 2️⃣ Install VideoMamba (video encoder backbone)
# --------------------------------------------------
echo "🎥 Installing VideoMamba..."
git clone https://github.com/OpenGVLab/VideoMamba.git
cd VideoMamba
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
which nvcc


python -c "import torch; print(f'torch version: {torch.__version__}')"

pip install -e causal-conv1d
pip install -e causal-conv1d --no-build-isolation
pip install causal-conv1d
python -c "import causal_conv1d; print(f'causal_conv1d version: {causal_conv1d.__version__}')"

pip install -e mamba
pip install mamba

cd ..

# --------------------------------------------------
# 3️⃣ Install FlashAttention for faster training
# --------------------------------------------------
echo "⚡ Installing FlashAttention..."
git clone https://github.com/HazyResearch/flash-attention.git

cd flash-attention
pip install ninja packaging wheel
python setup.py install
pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')"
cd ..

pip install mamba_ssm
# --------------------------------------------------
# 4️⃣ Final touches
# --------------------------------------------------
echo "🧩 Installing development tools and extras..."
pip install deepspeed accelerate bitsandbytes
pip install ipykernel notebook tqdm

echo "✅ Setup complete!"
echo "🚀 Mobile-VideoGPT environment is ready."
