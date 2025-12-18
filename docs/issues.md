# Encountered Issues

## ``Issue #1``: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120) Incompatibility

The NVIDIA RTX PRO 6000 Blackwell Server Edition GPU with CUDA capability sm_120 is **not compatible** with the current environment required by Mobile-VideoGPT. The Blackwell architecture is too new and many critical CUDA-compiled packages do not yet have kernel support for sm_120.

<details>
<summary><b>Error Messages</b></summary>

#### 1. PyTorch CUDA Incompatibility

```
UserWarning: NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA capability sm_120
is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_60 sm_70 sm_75 sm_80 sm_86 sm_90.
```

#### 2. Flash Attention Symbol Error (after PyTorch upgrade)

```
ImportError: flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so: undefined symbol: _ZNK3c106SymInt6sym_neERKS0_
```

#### 3. Mamba SSM / Causal Conv1d Kernel Error

```
torch.AcceleratorError: CUDA error: no kernel image is available for execution on the device
Search for `cudaErrorNoKernelImageForDevice' in CUDA documentation
```

This error occurs in:

- `mamba_ssm/ops/selective_scan_interface.py`
- `causal_conv1d/cpp_functions.py`

</details>
</br>
<details>
<summary><b>Attempted Fixes</b></summary>

#### 1. Upgrade PyTorch to CUDA 12.8

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Result:** Partially successful - PyTorch itself worked, but CUDA-compiled extensions broke.

#### 2. Uninstall Flash Attention

```bash
pip uninstall flash-attn -y
```

**Result:** Fixed flash attention error. Transformers falls back to PyTorch's native SDPA.

#### 3. Set `attn_implementation="sdpa"` in model loading

Added `attn_implementation="sdpa"` to `from_pretrained()` calls or set environment variable:

```bash
export TRANSFORMERS_ATTN_IMPLEMENTATION="sdpa"
```

**Result:** Fixed the flash attention import requirement.

#### 4. Rebuild mamba-ssm and causal-conv1d from source

```bash
pip uninstall mamba-ssm causal-conv1d -y
export TORCH_CUDA_ARCH_LIST="12.0"
pip install causal-conv1d --no-build-isolation --no-cache-dir
pip install mamba-ssm --no-build-isolation --no-cache-dir
```

**Result:** Failed - source code does not support sm_120 architecture yet.

#### 5. Build from GitHub repositories

```bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d && pip install . --no-build-isolation

git clone https://github.com/state-spaces/mamba.git
cd mamba && pip install . --no-build-isolation
```

**Result:** Failed - CUDA kernels in these repos do not include sm_120 support.
</details>

### Root Cause

The core issue is that **VideoMamba** (the vision encoder used by Mobile-VideoGPT) relies on:

- `mamba-ssm` - State Space Model implementation with custom CUDA kernels
- `causal-conv1d` - Causal convolution with custom CUDA kernels

Both packages use pre-compiled CUDA kernels that only support up to sm_90 (Hopper architecture). The Blackwell architecture (sm_120) is too new and requires:

1. Updated CUDA kernel code with sm_120 support
2. Recompilation of all CUDA extensions

### References

- [PyTorch CUDA Compatibility](https://pytorch.org/get-started/locally/)
- [Flash Attention GitHub](https://github.com/Dao-AILab/flash-attention)
- [Mamba SSM GitHub](https://github.com/state-spaces/mamba)
- [Causal Conv1d GitHub](https://github.com/Dao-AILab/causal-conv1d)
- [CUDA Compute Capabilities](https://developer.nvidia.com/cuda-gpus)
