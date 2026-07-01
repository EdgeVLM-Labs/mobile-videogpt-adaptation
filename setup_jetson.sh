#!/bin/bash
# ==========================================
# Mobile-VideoGPT Setup for Jetson Orin Nano
# Inference-only setup (polling + Gradio app)
#
# Tested on: JetPack 6.2 (L4T R36.4.7), CUDA 12.6, aarch64
# Uses NVIDIA official PyTorch 2.5 wheel (JP6.1, forward-compatible)
# ==========================================

set -e

# ---- Configuration ----
ENV_NAME="mvgpt"
PYTHON_VERSION="3.10"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# NVIDIA PyTorch wheel for JetPack 6.x (aarch64, CUDA 12.x)
TORCH_URL="https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"

# cuSPARSELt (required by PyTorch 2.5 on Jetson)
CUSPARSELT_VERSION="0.7.1.0"
CUSPARSELT_URL="https://developer.download.nvidia.com/compute/cusparselt/redist/libcusparse_lt/linux-aarch64/libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz"

# torchvision branch matching torch 2.5
TORCHVISION_BRANCH="release/0.20"
TORCHVISION_VERSION="0.20.0"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

step()    { echo -e "\n${GREEN}==> $1${NC}"; }
warn()    { echo -e "${YELLOW}    [warn] $1${NC}"; }
fail()    { echo -e "${RED}    [FAIL] $1${NC}"; exit 1; }
ok()      { echo -e "${GREEN}    [ok]${NC} $1"; }

# ==========================================
# STEP 1: CUDA environment
# ==========================================
step "Step 1/9: Checking CUDA environment"

export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

nvcc --version >/dev/null 2>&1 || fail "nvcc not found. Is CUDA toolkit installed?"
ok "nvcc found: $(nvcc --version 2>&1 | grep release | awk '{print $6}')"

# Persist CUDA paths
if ! grep -q '/usr/local/cuda/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
    ok "Added CUDA paths to ~/.bashrc"
fi

# ==========================================
# STEP 2: Install cuSPARSELt (PyTorch 2.5+ requires it)
# ==========================================
step "Step 2/9: Installing cuSPARSELt ${CUSPARSELT_VERSION}"

if [ -f /usr/local/cuda/lib64/libcusparseLt.so ]; then
    EXISTING_VER=$(ls /usr/local/cuda/lib64/libcusparseLt.so.* 2>/dev/null | grep -oP '\d+\.\d+\.\d+\.\d+' | head -1)
    ok "cuSPARSELt already installed (${EXISTING_VER})"

    # Upgrade if needed
    if [ "${EXISTING_VER}" != "${CUSPARSELT_VERSION}" ]; then
        warn "Upgrading cuSPARSELt from ${EXISTING_VER} to ${CUSPARSELT_VERSION}"
        warn "This requires sudo access"
        mkdir -p /tmp/cusparselt_install && cd /tmp/cusparselt_install
        curl --retry 3 -OLs "${CUSPARSELT_URL}"
        tar xf "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz"
        sudo cp -a "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/include/"* /usr/local/cuda/include/
        sudo cp -a "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/lib/"* /usr/local/cuda/lib64/
        sudo ldconfig
        cd "${PROJECT_DIR}"
        rm -rf /tmp/cusparselt_install
        ok "cuSPARSELt upgraded to ${CUSPARSELT_VERSION}"
    fi
else
    warn "cuSPARSELt not found, installing (requires sudo)"
    mkdir -p /tmp/cusparselt_install && cd /tmp/cusparselt_install
    curl --retry 3 -OLs "${CUSPARSELT_URL}"
    tar xf "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive.tar.xz"
    sudo cp -a "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/include/"* /usr/local/cuda/include/
    sudo cp -a "libcusparse_lt-linux-aarch64-${CUSPARSELT_VERSION}-archive/lib/"* /usr/local/cuda/lib64/
    sudo ldconfig
    cd "${PROJECT_DIR}"
    rm -rf /tmp/cusparselt_install
    ok "cuSPARSELt ${CUSPARSELT_VERSION} installed"
fi

# ==========================================
# STEP 3: Create conda environment
# ==========================================
step "Step 3/9: Creating conda environment: ${ENV_NAME} (Python ${PYTHON_VERSION})"

eval "$(conda shell.bash hook)" || fail "conda not found"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '${ENV_NAME}' already exists, reusing it"
    conda activate "${ENV_NAME}"
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
    conda activate "${ENV_NAME}"
fi
ok "Activated ${ENV_NAME} ($(python --version))"

# Verify conda activate actually put the right Python first in PATH
ACTIVE_PY=$(which python)
if [[ ! "$ACTIVE_PY" == *"${ENV_NAME}"* ]]; then
    warn "conda activate didn't set PATH correctly (using: ${ACTIVE_PY})"
    warn "Forcing PATH to use ${ENV_NAME} environment"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    ok "Fixed PATH — now using: $(which python)"
fi

pip install --upgrade pip

# ==========================================
# STEP 4: Install PyTorch (NVIDIA official wheel)
# ==========================================
step "Step 4/9: Installing PyTorch 2.5 from NVIDIA (this downloads ~800MB)"

# Check if torch is already installed with CUDA
if python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    ok "PyTorch already installed with CUDA — skipping"
    python -c "import torch; print(f'    PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"
else
    # numpy must be installed first with correct version
    pip install 'numpy==1.26.1'
    pip install --no-cache "${TORCH_URL}" || fail "Failed to download/install PyTorch wheel"

    # Verify CUDA works
    python -c "
import torch
assert torch.cuda.is_available(), 'PyTorch cannot see CUDA!'
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
" || fail "PyTorch CUDA verification failed"

    ok "PyTorch installed with CUDA support"
fi

# ==========================================
# STEP 5: Build torchvision from source
# ==========================================
step "Step 5/9: Building torchvision from source (this takes ~10-15 min on Jetson)"

# Install build dependencies (apt-get update may warn about stale PPAs — that's ok)
sudo apt-get update -qq 2>&1 | grep -v "^E:" || true
sudo apt-get install -y -qq libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev \
    libavcodec-dev libavformat-dev libswscale-dev 2>/dev/null

cd "${PROJECT_DIR}"

# Check if torchvision is already installed
if python -c "import torchvision; print(f'torchvision {torchvision.__version__}')" 2>/dev/null; then
    ok "torchvision already installed — skipping build"
else
    # Ensure setuptools has pkg_resources (needed by torchvision setup.py)
    pip install 'setuptools>=69,<72'

    if [ ! -d "torchvision_build" ]; then
        git clone --branch "${TORCHVISION_BRANCH}" --depth 1 https://github.com/pytorch/vision torchvision_build
    fi

    cd torchvision_build
    export BUILD_VERSION="${TORCHVISION_VERSION}"
    export TORCH_CUDA_ARCH_LIST="8.7"
    export FORCE_CUDA=1
    export MAX_JOBS=2  # Limit to avoid OOM on 8GB Jetson

    python setup.py install 2>&1 | tail -5
    cd "${PROJECT_DIR}"

    # Verify
    python -c "import torchvision; print(f'torchvision {torchvision.__version__}')" || fail "torchvision build failed"
    ok "torchvision built and installed"
fi

# ==========================================
# STEP 6: Install inference dependencies
# ==========================================
step "Step 6/9: Installing Python dependencies (inference-only)"

pip install -r "${PROJECT_DIR}/requirements_jetson.txt"

# Verify numpy stayed <2.0 (NVIDIA torch requires it)
python -c "
import numpy as np
v = tuple(int(x) for x in np.__version__.split('.')[:2])
assert v[0] < 2, f'numpy {np.__version__} is 2.x — NVIDIA torch needs <2.0'
print(f'numpy {np.__version__} ok')
" || {
    warn "numpy got upgraded to 2.x, downgrading..."
    pip install 'numpy>=1.26,<2.0'
}

ok "Python dependencies installed"

# ==========================================
# STEP 7: Install decord
# ==========================================
step "Step 7/9: Installing decord"

cd "${PROJECT_DIR}"

if python -c "from decord import VideoReader; print('decord: ok')" 2>/dev/null; then
    ok "decord already installed — skipping"
elif [ -d "${PROJECT_DIR}/decord/build" ] && [ -f "${PROJECT_DIR}/decord/build/libdecord.so" ]; then
    # Local build exists — install Python bindings
    cd "${PROJECT_DIR}/decord/python"
    pip install .
    cd "${PROJECT_DIR}"
    ok "decord installed from local build"
elif [ -d "${PROJECT_DIR}/decord/python" ]; then
    # Source exists but not built yet — build it
    warn "decord source found but not built. Building from source..."
    cd "${PROJECT_DIR}/decord"
    mkdir -p build && cd build
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    make -j2
    cd "${PROJECT_DIR}/decord/python"
    pip install .
    cd "${PROJECT_DIR}"
    ok "decord built and installed from source"
else
    warn "decord directory not found. Cloning and building from source..."
    cd "${PROJECT_DIR}"
    git clone --recursive https://github.com/dmlc/decord.git decord_src
    cd decord_src
    mkdir -p build && cd build
    cmake .. -DUSE_CUDA=OFF -DCMAKE_BUILD_TYPE=Release
    make -j2
    cd "${PROJECT_DIR}/decord_src/python"
    pip install .
    cd "${PROJECT_DIR}"
    ok "decord cloned, built, and installed"
fi

# Verify
python -c "from decord import VideoReader; print('decord: ok')" || fail "decord import failed"

# ==========================================
# STEP 8: Install Mamba dependencies (VideoMamba encoder)
# ==========================================
step "Step 8/9: Installing causal-conv1d and mamba-ssm (compiling CUDA kernels for SM 8.7)"

export TORCH_CUDA_ARCH_LIST="8.7"
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
export MAMBA_FORCE_BUILD=TRUE
export CUDA_HOME=/usr/local/cuda
export MAX_JOBS=1  # Single compile job to avoid OOM on 7.4GB Jetson

# Verify we're using the right torch before spending 30+ min compiling
TORCH_VER=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
if [[ ! "$TORCH_VER" == 2.5* ]]; then
    fail "Wrong torch version: $TORCH_VER (expected 2.5.x). Check conda activation."
fi
ok "Using torch ${TORCH_VER}"

# Check each package separately so we don't rebuild what already works
CC1D_OK=false
if python -c "import causal_conv1d; import causal_conv1d_cuda; print('causal-conv1d: ok')" 2>/dev/null; then
    CC1D_OK=true
    ok "causal-conv1d already installed with CUDA extension"
fi

MAMBA_OK=false
if python -c "import mamba_ssm; print('mamba-ssm: ok')" 2>/dev/null; then
    MAMBA_OK=true
    ok "mamba-ssm already installed"
fi

if [ "${CC1D_OK}" = true ] && [ "${MAMBA_OK}" = true ]; then
    ok "Both packages already installed — skipping"
else
    pip cache purge

    # ---- Increase swap to prevent OOM during CUDA compilation ----
    CURRENT_SWAP_MB=$(free -m | awk '/^Swap:/ {print $2}')
    if [ "${CURRENT_SWAP_MB}" -lt 8000 ]; then
        warn "Swap is ${CURRENT_SWAP_MB}MB — increasing to 8GB for safe compilation"
        if [ -f /swapfile_build ]; then
            sudo swapoff /swapfile_build 2>/dev/null || true
            sudo rm -f /swapfile_build
        fi
        sudo fallocate -l 8G /swapfile_build
        sudo chmod 600 /swapfile_build
        sudo mkswap /swapfile_build
        sudo swapon /swapfile_build
        ok "Swap increased to $(free -m | awk '/^Swap:/ {print $2}')MB"
    else
        ok "Swap is ${CURRENT_SWAP_MB}MB — sufficient"
    fi

    BUILD_DIR="/tmp/jetson_mamba_build"
    rm -rf "${BUILD_DIR}"
    mkdir -p "${BUILD_DIR}"

    # ---- Build causal-conv1d from patched source (if needed) ----
    if [ "${CC1D_OK}" = false ]; then
    # We download source and patch setup.py to compile ONLY for SM 8.7 (Orin Nano)
    # instead of 7+ architectures. This cuts compile memory by ~7x.
    pip uninstall -y causal-conv1d 2>/dev/null || true
    rm -rf "${CONDA_PREFIX}/lib/python3.10/site-packages/causal_conv1d"* 2>/dev/null || true
    rm -f "${CONDA_PREFIX}/lib/python3.10/site-packages/causal_conv1d_cuda"*.so 2>/dev/null || true

    echo "    Downloading causal-conv1d source..."

    wget -q -O "${BUILD_DIR}/causal_conv1d-1.6.0.tar.gz" \
        "https://files.pythonhosted.org/packages/source/c/causal-conv1d/causal_conv1d-1.6.0.tar.gz"
    cd "${BUILD_DIR}"
    tar xzf causal_conv1d-1.6.0.tar.gz
    cd causal_conv1d-1.6.0

    # Patch setup.py: only compile SM 8.7
    python -c "
import re
with open('setup.py', 'r') as f:
    content = f.read()

content = content.replace('    cc_flag = []', '    cc_flag = [\"-gencode\", \"arch=compute_87,code=sm_87\"]')
content = re.sub(r'\n\s+cc_flag\.append\(\"-gencode\"\)\n\s+cc_flag\.append\(\"arch=compute_\d+,code=sm_\d+\"\)', '', content)
content = re.sub(r'\s+if bare_metal_version >= Version\([^)]+\):\s*\n(\s*\n)*', '\n', content)
content = re.sub(r'\s+if bare_metal_version <= Version\([^)]+\):\s*\n(\s*\n)*', '\n', content)

with open('setup.py', 'w') as f:
    f.write(content)
compile(content, 'setup.py', 'exec')
print('Patched setup.py: only SM 8.7 (syntax verified)')
"

    echo "    Building causal-conv1d (CUDA kernels for SM 8.7 only, MAX_JOBS=1)..."
    pip install . --no-build-isolation --no-cache-dir

    # Verify causal-conv1d before proceeding to mamba-ssm
    python -c "import causal_conv1d; import causal_conv1d_cuda; print(f'causal_conv1d {causal_conv1d.__version__}: OK')" \
        || fail "causal-conv1d CUDA extension failed to load"
    ok "causal-conv1d built successfully"
    fi  # end CC1D_OK check

    # ---- Build mamba-ssm from patched source (if needed) ----
    if [ "${MAMBA_OK}" = false ]; then
    pip uninstall -y mamba-ssm 2>/dev/null || true
    rm -rf "${CONDA_PREFIX}/lib/python3.10/site-packages/mamba_ssm"* 2>/dev/null || true
    rm -f "${CONDA_PREFIX}/lib/python3.10/site-packages/selective_scan_cuda"*.so 2>/dev/null || true
    echo "    Cloning mamba-ssm from GitHub (PyPI tarball lacks csrc/)..."
    MAMBA_VER="2.2.4"
    MAMBA_DIR="${BUILD_DIR}/mamba"
    rm -rf "${BUILD_DIR}"/mamba*
    cd "${BUILD_DIR}"
    git clone --branch "v${MAMBA_VER}" --depth 1 https://github.com/state-spaces/mamba.git
    cd "${MAMBA_DIR}"

    # Patch setup.py: only compile SM 8.7
    # Simple approach: replace cc_flag=[] with pre-initialized list,
    # remove all other gencode appends, remove empty version-check blocks
    python -c "
import re
with open('setup.py', 'r') as f:
    content = f.read()

# 1. Replace cc_flag = [] with pre-initialized list containing only SM 8.7
content = content.replace('    cc_flag = []', '    cc_flag = [\"-gencode\", \"arch=compute_87,code=sm_87\"]')

# 2. Remove all cc_flag.append gencode pairs
content = re.sub(r'\n\s+cc_flag\.append\(\"-gencode\"\)\n\s+cc_flag\.append\(\"arch=compute_\d+,code=sm_\d+\"\)', '', content)

# 3. Remove empty 'if bare_metal_version >=' blocks (their gencode content was removed)
content = re.sub(r'\s+if bare_metal_version >= Version\([^)]+\):\s*\n(\s*\n)*', '\n', content)

# 4. Remove empty 'if bare_metal_version <=' blocks similarly
content = re.sub(r'\s+if bare_metal_version <= Version\([^)]+\):\s*\n(\s*\n)*', '\n', content)

with open('setup.py', 'w') as f:
    f.write(content)

# Verify syntax
compile(content, 'setup.py', 'exec')
print('Patched setup.py: only SM 8.7 (syntax verified)')
"

    echo "    Building mamba-ssm (CUDA kernels for SM 8.7 only, MAX_JOBS=1)..."
    echo "    This may take 20-40 minutes..."
    pip install . --no-build-isolation --no-cache-dir \
        || fail "mamba-ssm build failed. Build dir preserved at ${BUILD_DIR} for debugging."

    # Patch distributed_utils.py: NVIDIA PyTorch 2.5 removed _all_gather_base
    # but mamba-ssm tries to use it as a fallback. Wrap in try/except since
    # we only need inference (no distributed training).
    DIST_UTILS="${CONDA_PREFIX}/lib/python3.10/site-packages/mamba_ssm/distributed/distributed_utils.py"
    if [ -f "${DIST_UTILS}" ]; then
        python -c "
content = open('${DIST_UTILS}').read()
old = '''if \"all_gather_into_tensor\" not in dir(torch.distributed):
    torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
if \"reduce_scatter_tensor\" not in dir(torch.distributed):
    torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base'''
new = '''try:
    if \"all_gather_into_tensor\" not in dir(torch.distributed):
        torch.distributed.all_gather_into_tensor = torch.distributed._all_gather_base
except AttributeError:
    pass
try:
    if \"reduce_scatter_tensor\" not in dir(torch.distributed):
        torch.distributed.reduce_scatter_tensor = torch.distributed._reduce_scatter_base
except AttributeError:
    pass'''
if old in content:
    content = content.replace(old, new)
    open('${DIST_UTILS}', 'w').write(content)
    print('Patched distributed_utils.py (NVIDIA torch compat)')
else:
    print('distributed_utils.py already patched or different layout — skipping')
"
    fi

    fi  # end MAMBA_OK check

    cd "${PROJECT_DIR}"
    rm -rf "${BUILD_DIR}"

    # ---- Remove extra swap ----
    if [ -f /swapfile_build ]; then
        sudo swapoff /swapfile_build 2>/dev/null || true
        sudo rm -f /swapfile_build
        ok "Removed temporary build swap"
    fi

    # Verify both
    python -c "
import causal_conv1d
import causal_conv1d_cuda
import mamba_ssm
import platform
assert platform.machine() == 'aarch64', 'Expected aarch64 platform'
print(f'causal_conv1d: {causal_conv1d.__version__}')
print('mamba_ssm: ok')
print('CUDA extensions loaded successfully (aarch64)')
" || fail "Mamba dependencies verification failed"
fi

ok "Mamba dependencies installed"

# ==========================================
# STEP 9: Set PYTHONPATH & verify
# ==========================================
step "Step 9/9: Final setup and verification"

export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH}"

if ! grep -q "mobile-videogpt-adaptation" ~/.bashrc 2>/dev/null; then
    echo "export PYTHONPATH=\"${PROJECT_DIR}:\${PYTHONPATH}\"" >> ~/.bashrc
    ok "Added project to PYTHONPATH in ~/.bashrc"
fi

python -c "
import sys
import torch
import transformers
import peft
import decord
import causal_conv1d
import mamba_ssm
import cv2
import gradio

print()
print('=== Jetson Inference Environment ===')
print(f'  Python:        {sys.version.split()[0]}')
print(f'  PyTorch:       {torch.__version__}')
print(f'  CUDA:          {torch.version.cuda}')
print(f'  GPU:           {torch.cuda.get_device_name(0)}')
print(f'  VRAM:          {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
print(f'  Transformers:  {transformers.__version__}')
print(f'  PEFT:          {peft.__version__}')
print(f'  Decord:        ok')
print(f'  Causal-Conv1D: {causal_conv1d.__version__}')
print(f'  Mamba-SSM:     ok')
print(f'  OpenCV:        {cv2.__version__}')
print(f'  Gradio:        {gradio.__version__}')
try:
    import flash_attn
    print(f'  Flash-Attn:    {flash_attn.__version__}')
except ImportError:
    print(f'  Flash-Attn:    not installed (optional)')
print()
" || fail "Verification failed"

echo ""
echo -e "${GREEN}============================================${NC}"
echo -e "${GREEN}  Setup complete!${NC}"
echo -e "${GREEN}============================================${NC}"
echo ""
echo "To use the environment:"
echo "  conda activate ${ENV_NAME}"
echo ""
echo "To run inference on a video:"
echo "  python polling/run_polling.py sample_videos/00000340.mp4"
echo ""
echo "To launch the Gradio UI:"
echo "  python polling/gradio_app.py"
echo ""
