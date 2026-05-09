# Jetson Orin Nano — Inference Fixes & Latency Report

> ⚠️ **This document is superseded.** It captures the Phase-1 fix list as
> originally written. The current canonical reference is
> [`JETSON_OPTIMIZATION_JOURNEY.md`](./JETSON_OPTIMIZATION_JOURNEY.md),
> which covers all phases (1 through 8) with up-to-date measurements.
> This file is kept for archive.

This document covers all changes made to get Mobile-VideoGPT inference running on the Jetson Orin Nano (8GB unified RAM), along with latency benchmarks and remaining issues.

## Platform

| Spec | Value |
|---|---|
| Board | Jetson Orin Nano |
| JetPack | 6.2 (L4T R36.4.7) |
| CUDA | 12.6 |
| RAM | 7.4GB unified (CPU + GPU shared) |
| Python | 3.10.19 |
| PyTorch | 2.5.0a0 (Jetson wheel) |

## The Core Problem

The Jetson Orin Nano has only **8GB of unified memory** shared between CPU and GPU. Mobile-VideoGPT requires:

- **Main model** (Qwen2 0.5B + VideoMamba + projectors): ~1.3GB
- **CLIP vision tower** (ViT-Base-Patch16): ~600MB
- **Inference activations** (16 frames through encoders + LLM generation): ~1-2GB
- **System/OS overhead**: ~2-3GB

Total: ~5-6GB minimum, leaving almost no headroom on an 8GB board. The original code assumed a desktop GPU with separate VRAM and crashed with CUDA OOM errors.

---

## Changes Made

### 1. Main Model Loading — `device_map="auto"`

**File**: `polling/inference_engine.py`

**Problem**: Original code loaded the entire model with `.to('cuda')`, which tried to allocate all 1.3GB on CUDA at once and crashed when combined with other allocations.

**Fix**: Used HuggingFace Accelerate's `device_map="auto"` with a controlled CUDA budget (40% of free CUDA memory). This automatically splits model layers between CUDA and CPU based on available memory.

```python
# Before
model = MobileVideoGPTQwenForCausalLM.from_pretrained(base_model_path, ...)
model.to('cuda')

# After
cuda_budget = max(int(free_mem * 0.4), 512 * 1024 * 1024)
max_memory = {0: cuda_budget, "cpu": "2GiB"}
model = MobileVideoGPTQwenForCausalLM.from_pretrained(
    base_model_path,
    device_map="auto",
    max_memory=max_memory,
    low_cpu_mem_usage=True,
    ...
)
```

The 40% budget (~2GB) is enough for the model weights on CUDA while leaving ~3GB for inference activations.

---

### 2. CLIP Vision Tower — CPU Offload

**File**: `mobilevideogpt/model/multimodal_encoder/clip_encoder.py`

**Problem**: CLIP (599MB) was hardcoded with `.to('cuda')` in `load_model()`. This loaded the model to CPU first (599MB), then copied to CUDA (another 599MB), consuming ~1.2GB peak on top of the already-loaded main model.

**Fix**: Removed `.to('cuda')` and kept CLIP entirely on CPU. On the Jetson's unified memory, CPU inference is slower but avoids exhausting the CUDA allocator's pool.

```python
# Before
self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name).to('cuda')

# After
self.vision_tower = CLIPVisionModel.from_pretrained(
    self.vision_tower_name,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
)
```

This saves ~600MB of CUDA allocator space for inference activations.

---

### 3. Device Mismatch Fix — CPU/CUDA Tensor Bridging

**File**: `mobilevideogpt/model/arch.py`

**Problem**: With CLIP on CPU, the attention-based frame selection produces indices on CPU. These indices are then used to index into video frames which are on CUDA. PyTorch requires index tensors to be on the same device as the data tensor for advanced indexing.

**Fix**: Move indices to the video tensor's device before indexing, and ensure context features are transferred to CUDA before being returned for downstream use.

```python
# Before
batch_indices = torch.arange(num_chunks).unsqueeze(1).repeat(1, topK).to(seleted_indices.device)
select_video = video_batch[batch_indices, seleted_indices]

# After
batch_indices = torch.arange(num_chunks).unsqueeze(1).repeat(1, topK).to(video_batch.device)
seleted_indices = seleted_indices.to(video_batch.device)
select_video = video_batch[batch_indices, seleted_indices]
```

Also added device transfer for the return value:

```python
if context_image_features.device != video_features.device:
    context_image_features = context_image_features.to(video_features.device)
return video_features, context_image_features
```

---

### 4. causal_conv1d API Compatibility

**File**: `mamba_ssm/ops/selective_scan_interface.py` (installed package, not project code)

**Problem**: `mamba_ssm` was compiled against `causal_conv1d` 1.2.x which had a 7-argument `causal_conv1d_fwd()` C++ function. The installed version (1.6.0) added a new `out` parameter at position 5, making it 8 arguments.

**Error**: `causal_conv1d_fwd(): incompatible function arguments`

**Fix**: Patched the 2 call sites in `selective_scan_interface.py` to pre-allocate an output tensor and pass it as the new 6th argument:

```python
# Before (7 args — old API)
conv1d_out = causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, None, None, True
)

# After (8 args — new API)
conv1d_out = torch.empty_like(x)
causal_conv1d_cuda.causal_conv1d_fwd(
    x, conv1d_weight, conv1d_bias, None, None, conv1d_out, None, True
)
```

---

### 5. VideoMamba Build — Skip Pretrained Download

**File**: `mobilevideogpt/model/videomamba/build_videomamba.py`

**Problem**: `build_videomamba()` tried to download pretrained weights from a hardcoded URL (`/mnt/petrelfs/.../videomamba_m16_25M_f8_res224.pth`) which doesn't exist externally. This caused a crash during model loading.

**Fix**: Set `pretrained=None` since the VideoMamba weights are already included in the main model's safetensors file and get loaded by `from_pretrained()`.

```python
# Before
model = videomamba_middle(pretrained=True, ...)

# After
model = videomamba_middle(pretrained=None, ...)
```

---

### 6. CUDA Memory Budget Tuning

**File**: `polling/inference_engine.py`

**Problem**: Initial budget of 60% CUDA left insufficient headroom for VideoMamba + Qwen inference activations (attention matrices, intermediate tensors, KV cache).

**Fix**: Lowered to 40% CUDA budget with increased CPU budget (2GB). This puts more model layers on CPU (slightly slower) but ensures enough CUDA memory for the compute-intensive inference kernels.

```python
# Before
cuda_budget = max(int(free_mem * 0.6), 512 * 1024 * 1024)
max_memory = {0: cuda_budget, "cpu": "1GiB"}

# After
cuda_budget = max(int(free_mem * 0.4), 512 * 1024 * 1024)
max_memory = {0: cuda_budget, "cpu": "2GiB"}
```

---

## Latency Benchmarks

**Test video**: `sample_videos/00000340.mp4` (4.83s, 30fps, 145 frames)
**Configuration**: 16 frames, base model only (no LoRA), FP16

### Model Loading

| Stage | Time |
|---|---|
| Config + tokenizer | 4s |
| Base model (device_map=auto) | 22s |
| CLIP vision tower (CPU) | 12s |
| **Total load time** | **41.28s** |

### Inference Per Poll

| Metric | Poll #1 | Poll #2 | Average |
|---|---|---|---|
| Frame extraction | 1.32s | 1.90s | 1.61s |
| Total inference | 57.98s | 45.15s | **51.57s** |
| Output tokens | ~37 | ~41 | ~39 |

Poll #2 is ~22% faster than Poll #1 due to CUDA kernel warmup and caching.

### Breakdown (estimated)

| Stage | Time (est.) |
|---|---|
| CLIP encoding (16 frames, CPU) | ~15-20s |
| Frame selection (attention, CPU) | ~2-3s |
| VideoMamba encoding (8 frames, CUDA) | ~10-15s |
| Qwen2 text generation (~40 tokens) | ~15-20s |
| **Total per poll** | **~50s** |

### Session Summary

| Metric | Value |
|---|---|
| Total session duration | 106.15s |
| Total polls | 2 |
| Success rate | 100% |
| Mean latency | 51.57s |

---

## Model Response Quality

Running on the **base model only** (LoRA adapter failed to load — see remaining issues):

**Poll #1**: "The exercise form shown is incorrect, as indicated by the text 'Incorrect Form' and a red arrow pointing to the wrong position. A correction should be made to ensure proper alignment of the torso with the hips and arms."

**Poll #2**: "The exercise form shown is incorrect, as indicated by the text 'Incorrect Form' and a red checkmark. A correction to the correct form would be: 'Lift your knees off the ground.'"

The model correctly identifies incorrect exercise form and provides basic correction suggestions.

---

## Remaining Issues

### 1. LoRA Weights Not Loading (401 Unauthorized)

The HuggingFace repo `EdgeVLM-Labs/mobile-videogpt-finetune-2000` returns a 401 error. The model runs on base weights only.

**Fix**: Run `huggingface-cli login` with the correct access token, or make the repo public.

### 2. Decord Fallback to OpenCV

Decord fails with `'Tensor' object has no attribute 'asnumpy'`, falling back to OpenCV for frame extraction. This adds ~300ms per extraction.

**Fix**: Reinstall decord or fix the tensor type mismatch in the extraction code.

### 3. Duplicate Log Lines

Every log message appears twice. Likely caused by two logging handlers being attached to the same logger.

### 4. Triton Import Warning

```
ImportError: cannot import name 'triton_key' from 'triton.compiler.compiler'
```

Non-blocking warning from torch inductor. Does not affect inference.

### 5. Power Throttling

The Jetson may show "device throttled due to overcurrent" warnings during inference. This can be mitigated by:

```bash
sudo nvpmodel -m 0       # Set max power mode (15W)
sudo jetson_clocks        # Lock clocks to max frequency
```

Requires a sufficient power supply (5V/3A+ barrel jack, not USB).

---

## Tips for Running on Jetson

1. **Close VSCode** before running inference — it uses ~2.3GB RAM
2. **Clear page cache** before runs: `sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'`
3. **Run from terminal** (not IDE) to maximize available memory
4. **Set max power mode** for best performance: `sudo nvpmodel -m 0 && sudo jetson_clocks`
5. **Monitor memory** during runs: `tegrastats` or `watch -n 1 free -h`
