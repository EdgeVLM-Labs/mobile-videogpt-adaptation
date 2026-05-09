# Install Notes — Why Each Step Exists

> Why `setup_jetson.sh` looks the way it does. This explains the rationale
> behind each install decision so a future engineer (or paper reviewer) can
> understand why standard pip installs don't work on Jetson and what
> custom workarounds are required.
>
> The actual install commands live in [`setup_jetson.sh`](../../setup_jetson.sh).
> This document explains *why* each step is needed.

---

## 🎯 Platform Reality

The Jetson Orin Nano Super is **aarch64 (ARM)** with NVIDIA's customized
JetPack stack. Almost every package in the inference pipeline ships
prebuilt wheels for **x86_64 only**, so they either fail to install or
silently install a CPU-only / broken version. Half the install effort is
**building from source against the right CUDA + ARM toolchain**.

| Setting | Value |
|---|---|
| Board | Jetson Orin Nano Super |
| Architecture | aarch64 |
| OS | Ubuntu 22.04 (JetPack 6.2 / L4T R36.4.7) |
| CUDA | 12.6 |
| Python | 3.10 |
| GPU compute capability | SM 8.7 (Ampere, Jetson variant) |

The SM 8.7 is the critical detail — most prebuilt CUDA wheels target SM
8.0 / 8.6 / 9.0 (server GPUs) and silently miss kernels for the Jetson
variant. **CUDA libraries with custom kernels must be rebuilt with
`TORCH_CUDA_ARCH_LIST=8.7`** for them to actually run on this device.

---

## 📦 Step-by-Step Rationale

### 1. CUDA environment check

Verify `nvcc` is on `$PATH` and persist `/usr/local/cuda/bin` and
`/usr/local/cuda/lib64` into `~/.bashrc`. Without these, every later
build step fails to find CUDA headers/libraries.

### 2. cuSPARSELt 0.7.1+ installed manually

PyTorch 2.5 expects `libcusparse_lt.so` ≥ 0.7.1, but JetPack 6.2 ships
an older version (or none). The official PyTorch wheel from NVIDIA fails
at import with `OSError: cusparseLtCreate not found` if this isn't
upgraded first.

**Workaround**: download the aarch64 cuSPARSELt archive from
`developer.download.nvidia.com`, extract, and copy headers + .so files
into `/usr/local/cuda/{include,lib64}` then run `ldconfig`.

There is no pip equivalent for this — it has to be installed at the
system level before PyTorch import succeeds.

### 3. Conda environment

Plain venv works too, but conda makes it easier to pin Python 3.10
(matches the wheel ABI tag `cp310`). The environment is named `mvgpt`
throughout the docs and run scripts.

### 4. PyTorch 2.5.0a0 (NVIDIA Jetson wheel)

**Why we can't use `pip install torch`**:
- The PyPI wheel for ARM (aarch64) is CPU-only — `torch.cuda.is_available()`
  returns `False` because it was never built against CUDA.
- We need a wheel that was *built specifically for Jetson*.

**Source**: NVIDIA's developer download page hosts pre-built Jetson
wheels at `developer.download.nvidia.com/compute/redist/jp/v61/pytorch/`.
We use `torch-2.5.0a0+872d972e41.nv24.08...cp310-linux_aarch64.whl`.

**Forward compatibility**: this wheel was compiled for JetPack 6.1 +
CUDA 12.x but works on JetPack 6.2 + CUDA 12.6 because Jetson's CUDA
ABI is forward-compatible within major versions.

**Note about** `pypi.jetson-ai-lab.dev`: an alternative Jetson wheel
mirror that **does not resolve via DNS from this board**, so we don't
use it. The NVIDIA developer URL works fine.

### 5. torchvision built from source

Same problem as PyTorch — the PyPI torchvision is CPU-only on aarch64.
Even if it imports, ops like `nms` and `roi_align` aren't compiled
against CUDA. We build from source against our installed PyTorch:

```bash
git clone -b release/0.20 https://github.com/pytorch/vision torchvision_build
cd torchvision_build
python setup.py install
```

Branch `release/0.20` matches `torch 2.5`. Build takes ~10–15 min on
Jetson because of CUDA kernel compilation; mostly limited by
`/tmp/build` I/O.

### 6. Python deps from `requirements_jetson.txt`

Standard `pip install -r requirements_jetson.txt` for everything that
**doesn't need CUDA kernels**: `transformers`, `peft`, `accelerate`,
`gradio`, `optimum-quanto`, `onnx`, `onnxruntime`, etc.

These are all pure-Python or have pre-built aarch64 wheels.

### 7. decord (video reader) — built from C++ source

Decord doesn't ship aarch64 wheels at all. The repo bundles a
pre-built `decord/build/libdecord.so` binary; the setup script:

1. Confirms the C++ library exists (already in repo)
2. Installs the Python bindings via the bundled `decord/python/setup.py`

Why ship the C++ binary in the repo? Because building decord from
source requires `cmake`, `ffmpeg-dev`, and ~5 minutes — way faster to
ship the precompiled `.so` than to make every install rebuild it.

If the binary is corrupted or missing, the script can rebuild from the
decord submodule, but this is the slow path.

### 8. causal-conv1d + mamba-ssm — built from source for SM 8.7

These libraries provide custom CUDA kernels for the Mamba state-space
operations VideoMamba uses. Pre-built wheels target server GPU compute
capabilities (SM 8.0 / 8.6 / 9.0) and silently lack SM 8.7 kernels.

If you `pip install causal-conv1d` and try to use it, it imports fine
but raises `RuntimeError: named symbol not found` at the first kernel
call.

**Fix**: clone each repo and build with the correct CUDA arch:

```bash
TORCH_CUDA_ARCH_LIST=8.7 pip install --no-build-isolation -e .
```

Build takes ~15–20 min total because each `.cu` file is compiled for
SM 8.7. The `--no-build-isolation` is required so the build picks up
our installed PyTorch (not a fresh isolated one that would have a
different CUDA version).

Additionally, `mamba_ssm.distributed.distributed_utils` references
torch APIs that NVIDIA's PyTorch wheel exposes differently. The setup
script auto-patches this file post-install to use a try/except pattern
that works on both stock and Jetson PyTorch.

### 9. Final verification

After all builds, the script imports `torch`, `torchvision`, `decord`,
`causal_conv1d`, `mamba_ssm` and confirms:

- `torch.cuda.is_available()` returns `True`
- `torch.cuda.get_device_properties(0).total_memory` reports the
  expected ~7.4 GB unified RAM (note attribute name is `total_memory`,
  not `total_mem` — the latter raises AttributeError on NVIDIA's
  wheel).

If any import fails, the script prints which step probably caused it,
making it easier to retry only the failing step.

---

## 🚫 What we explicitly skip

| Library | Why skipped |
|---|---|
| **flash-attn** | Pre-built wheels don't exist for SM 8.7; building from source would take ~30 min. Inference engine handles its absence by falling back to PyTorch SDPA, which works fine. |
| **bitsandbytes** | Pre-built kernels miss SM 8.7. We did try; runtime raised "named symbol not found in ops.cu". See [JOURNEY § Phase 6](./JETSON_OPTIMIZATION_JOURNEY.md) for details. |
| **TensorRT-LLM** | Multi-week integration effort to wire it into the multimodal wrapper. Listed as future work for getting INT8 acceleration of the Qwen2 backbone. |

---

## ✅ How to run the install

```bash
git clone https://github.com/EdgeVLM-Labs/mobile-videogpt-adaptation.git
cd mobile-videogpt-adaptation
git checkout jetson-inference
bash setup_jetson.sh
```

Total time on a fresh Jetson: ~30–45 minutes (mostly CUDA kernel
compilation for steps 5, 7, 8).

Once finished:

```bash
conda activate mvgpt
python -c "import torch; print(torch.cuda.is_available())"  # expect True
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py
```

---

## 🆘 Common install failures

| Symptom | Likely cause | Fix |
|---|---|---|
| `torch.cuda.is_available()` is False | Got the x86 PyPI wheel | Re-run setup; manually verify Step 4 used the NVIDIA URL |
| `cusparseLtCreate not found` on `import torch` | cuSPARSELt < 0.7.1 | Re-run Step 2 with sudo |
| `ImportError: triton_key` | torch._inductor warning, harmless | Ignore — doesn't block inference |
| `named symbol not found` in `ops.cu` | Library built for wrong CUDA arch | Rebuild that library with `TORCH_CUDA_ARCH_LIST=8.7` |
| Build hangs at `nvcc` for >10 min | Normal — Jetson kernels take a long time | Be patient; check `top` to confirm `nvcc` is running |
| `pypi.jetson-ai-lab.dev` DNS fail | Mirror not resolvable from this board | Don't use that mirror; use `developer.download.nvidia.com` instead |
