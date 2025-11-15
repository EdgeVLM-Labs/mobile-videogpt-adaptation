#!/bin/bash
# ==========================================
# Setup Script for Mobile-VideoGPT
# Based on: https://github.com/Amshaker/Mobile-VideoGPT#installation
# ==========================================

echo "🔧 Creating workspace..."

cd /workspace
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p $HOME/miniconda
export PATH="$HOME/miniconda/bin:$PATH"
# conda init bash
# source ~/.bashrc
source $HOME/miniconda/etc/profile.d/conda.sh

conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

conda create --name=mobile_videogpt python=3.11
conda activate mobile_videogpt

pip install --upgrade pip

apt-get update

conda install -c nvidia cuda-toolkit=11.8 -y
# python3 -m venv venv
# source venv/bin/activate

echo "📦 Cloning repositories..."
# --------------------------------------------------
# 1️⃣ Install base dependencies
# --------------------------------------------------
echo "🧱 Installing base Python packages..."
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

pip install torch==2.1.2 torchvision==0.16.2 torchaudio --index-url https://download.pytorch.org/whl/cu118

echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
# source ~/.bashrc
which nvcc

conda activate mobile_videogpt

python -c "import torch; print(f'torch version: {torch.__version__}')"

# pip install -e causal-conv1d
# pip install -e causal-conv1d --no-build-isolation
pip install causal-conv1d
python -c "import causal_conv1d; print(f'causal_conv1d version: {causal_conv1d.__version__}')"

# pip install -e mamba
pip install mamba
pip install mamba_ssm

cd ..

# --------------------------------------------------
# 3️⃣ Install FlashAttention for faster training
# --------------------------------------------------
echo "⚡ Installing FlashAttention..."
git clone https://github.com/HazyResearch/flash-attention.git

cd flash-attention
pip install ninja packaging wheel
# python setup.py install

pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

pip install flash-attn --no-build-isolation
python -c "import flash_attn; print(f'flash_attn version: {flash_attn.__version__}')"
cd ..

echo "=== CUDA Check ==="
nvcc --version 2>/dev/null || echo "❌ nvcc not found"
nvidia-smi 2>/dev/null || echo "❌ nvidia-smi not found"

echo ""
echo "=== PyTorch CUDA Check ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('❌ PyTorch cannot see CUDA')
"

echo ""
echo "=== Flash Attention Check ==="
python -c "
try:
    import flash_attn
    print(f'✅ Flash Attention: {flash_attn.__version__}')
except ImportError:
    print('❌ Flash Attention not installed')
"

echo "✅ Setup complete!"
echo "🚀 Mobile-VideoGPT environment is ready."
