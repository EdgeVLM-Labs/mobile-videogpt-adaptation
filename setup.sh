#!/bin/bash
# ==========================================
# Setup Script for Mobile-VideoGPT
# Based on: https://github.com/Amshaker/Mobile-VideoGPT#installation
# Tested for CUDA 11.8 / PyTorch 2.1+
# ==========================================

set -e  # stop on error

echo "🔧 Creating workspace..."
python3 -m venv mobile_videogpt
source mobile_videogpt/bin/activate

echo "📦 Cloning repositories..."
git clone https://github.com/OpenGVLab/VideoMamba.git
git clone https://github.com/HazyResearch/flash-attention.git

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
cd VideoMamba
pip install -e causal-conv1d
pip install -e mamba
cd ..

# --------------------------------------------------
# 3️⃣ Install FlashAttention for faster training
# --------------------------------------------------
echo "⚡ Installing FlashAttention..."
cd flash-attention
pip install ninja packaging wheel
python setup.py install
cd ..

# --------------------------------------------------
# 4️⃣ Final touches
# --------------------------------------------------
echo "🧩 Installing development tools and extras..."
pip install deepspeed accelerate bitsandbytes
pip install ipykernel notebook tqdm

echo "✅ Setup complete!"
echo "🚀 Mobile-VideoGPT environment is ready."
