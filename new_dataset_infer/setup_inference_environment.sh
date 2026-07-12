#!/bin/bash

# Setup Script for New Dataset Inference
# This script installs all dependencies required to run inference scripts

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  Inference Environment Setup${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""

# Check if we're in the correct directory
if [ ! -f "setup.py" ]; then
    echo -e "${RED}❌ Error: Please run this script from the mobile-videogpt-adaptation root directory${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python
echo -e "${YELLOW}[1/6] Checking Python installation...${NC}"
if ! command_exists python; then
    echo -e "${RED}❌ Python not found. Please install Python 3.11 first.${NC}"
    exit 1
fi

PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}[2/6] Upgrading pip...${NC}"
python -m pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install core dependencies
echo -e "${YELLOW}[3/6] Installing core dependencies...${NC}"
echo "This may take several minutes..."
echo ""

# Install PyTorch (check CUDA version first)
if command_exists nvcc; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1.\2/p')
    echo "CUDA $CUDA_VERSION detected"
    
    # Force CUDA 12.1 for CUDA 12.x versions (most compatible)
    if [[ "$CUDA_VERSION" == "12."* ]]; then
        echo "Installing PyTorch for CUDA 12.1 (compatible with CUDA 12.x)..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    elif [[ "$CUDA_VERSION" == "11."* ]]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
    else
        echo -e "${YELLOW}⚠ Unknown CUDA version, installing PyTorch for CUDA 12.1${NC}"
        pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo -e "${YELLOW}⚠ CUDA not detected, installing CPU-only PyTorch${NC}"
    pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
fi

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install transformers and related packages
echo -e "${YELLOW}[4/6] Installing transformers and model dependencies...${NC}"
pip install transformers>=4.37.0
pip install accelerate
pip install peft  # For LoRA adapters
pip install bitsandbytes  # Optional but useful for quantization
echo -e "${GREEN}✓ Transformers packages installed${NC}"
echo ""

# Install HuggingFace Hub
echo -e "${YELLOW}[5/6] Installing HuggingFace Hub and utilities...${NC}"
pip install huggingface_hub
pip install datasets
echo -e "${GREEN}✓ HuggingFace packages installed${NC}"
echo ""

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch {torch.__version__} loaded successfully')" || {
    echo -e "${RED}❌ PyTorch verification failed${NC}"
    exit 1
}

# Install additional dependencies
echo "Installing additional dependencies..."
pip install numpy
pip install packaging
pip install ninja  # Required for building extensions
pip install tqdm
pip install pandas
pip install openpyxl  # For Excel output
pip install opencv-python
pip install pillow
pip install decord  # For video processing

# Install Triton (for Mamba SSM)
echo "Installing Triton..."
pip install triton>=2.1.0

# Install causal-conv1d and mamba-ssm (optional, may fail on some systems)
echo "Installing causal-conv1d and mamba-ssm..."
echo -e "${YELLOW}Note: These packages may take several minutes to build...${NC}"

# Try to install causal-conv1d
if pip install causal-conv1d>=1.1.0 --no-build-isolation 2>/dev/null; then
    echo -e "${GREEN}✓ causal-conv1d installed${NC}"
    
    # Try to install mamba-ssm
    if pip install mamba-ssm --no-build-isolation 2>/dev/null; then
        echo -e "${GREEN}✓ mamba-ssm installed${NC}"
    else
        echo -e "${YELLOW}⚠ Warning: mamba-ssm installation failed${NC}"
        echo -e "${YELLOW}  The model may not work if it requires VideoMamba${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Warning: causal-conv1d installation failed${NC}"
    echo -e "${YELLOW}  Skipping mamba-ssm (depends on causal-conv1d)${NC}"
    echo -e "${YELLOW}  The model may not work if it requires VideoMamba${NC}"
fi

echo -e "${GREEN}✓ Additional dependencies installed${NC}"
echo ""

# Install the mobilevideogpt package in development mode
echo "Installing mobilevideogpt package..."
pip install -e .
echo -e "${GREEN}✓ mobilevideogpt package installed${NC}"
echo ""

# Set HuggingFace token
echo -e "${YELLOW}[6/6] Setting up HuggingFace access token...${NC}"
echo ""
echo "You need a HuggingFace access token to download models from private repositories."
echo "If you don't have one, create it at: https://huggingface.co/settings/tokens"
echo ""
read -p "Do you want to set your HuggingFace token now? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    read -sp "Enter your HuggingFace token: " HF_TOKEN
    echo ""
    
    if [ -n "$HF_TOKEN" ]; then
        # Login using huggingface-cli
        echo "$HF_TOKEN" | huggingface-cli login --token "$HF_TOKEN"
        
        # Also set as environment variable for current session
        export HF_TOKEN="$HF_TOKEN"
        
        # Add to shell profile for persistence
        SHELL_PROFILE=""
        if [ -f "$HOME/.bashrc" ]; then
            SHELL_PROFILE="$HOME/.bashrc"
        elif [ -f "$HOME/.zshrc" ]; then
            SHELL_PROFILE="$HOME/.zshrc"
        fi
        
        if [ -n "$SHELL_PROFILE" ]; then
            if ! grep -q "export HF_TOKEN=" "$SHELL_PROFILE"; then
                echo "" >> "$SHELL_PROFILE"
                echo "# HuggingFace Token (added by setup_inference_environment.sh)" >> "$SHELL_PROFILE"
                echo "export HF_TOKEN=\"$HF_TOKEN\"" >> "$SHELL_PROFILE"
                echo -e "${GREEN}✓ Token saved to $SHELL_PROFILE${NC}"
            else
                echo -e "${YELLOW}⚠ Token already exists in $SHELL_PROFILE${NC}"
            fi
        fi
        
        echo -e "${GREEN}✓ HuggingFace token configured${NC}"
    else
        echo -e "${YELLOW}⚠ No token provided, skipping...${NC}"
    fi
else
    echo -e "${YELLOW}⚠ Skipping HuggingFace token setup${NC}"
    echo "You can set it later with: huggingface-cli login"
fi

echo ""
echo -e "${BLUE}=========================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${BLUE}=========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Download the dataset:"
echo "     bash new_dataset_infer/initialize_new_dataset.sh"
echo ""
echo "  2. Run inference:"
echo "     bash new_dataset_infer/run_new_inference.sh"
echo ""
echo "  3. (Optional) Test with limited videos:"
echo "     bash new_dataset_infer/run_new_inference.sh --limit 10"
echo ""
echo -e "${YELLOW}Note: If you set a HuggingFace token, restart your terminal or run:${NC}"
echo "      source ~/.bashrc   # or source ~/.zshrc"
echo ""
