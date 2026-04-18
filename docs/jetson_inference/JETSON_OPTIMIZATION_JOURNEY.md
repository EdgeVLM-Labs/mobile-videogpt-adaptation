# Mobile-VideoGPT on Jetson Orin Nano — Optimization Journey

> **Living document** — updated continuously as we iterate on optimizations.
> Last updated: 2026-04-18

---

## 📌 Project Overview

**Goal**: Run Mobile-VideoGPT (0.5B parameter Vision-Language Model) on Jetson Orin Nano for real-time exercise form evaluation.

**Challenge**: The model was designed for desktop GPUs (24-80GB VRAM). Jetson Orin Nano has only **8GB unified RAM shared between CPU and GPU**.

**Model Architecture**:
- Qwen2 LLM (0.5B params) — text generation
- VideoMamba — video encoder
- CLIP ViT-Base — context image encoder
- Custom projectors

---

## 🎯 Summary Table (Current Status)

| Metric | Original | Now | Improvement |
|---|---|---|---|
| Model loading | ❌ OOM crash | ✅ ~40s (one-time when using Gradio) | Works |
| Inference latency (warm) | ❌ OOM crash | **~13s / poll** | ~4x faster than first working version |
| Back-to-back reliability | ❌ 2nd run crashes | **✅ 4+ runs without failure** | Production-viable |
| Memory usage | Exceeds 8GB | ~6.1GB | Fits |
| Output quality | N/A | ✅ Correct evaluations | Baseline preserved |

---

## 🏗️ Platform Details

| Spec | Value |
|---|---|
| Board | Jetson Orin Nano |
| JetPack | 6.2 (L4T R36.4.7) |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| Python | 3.10.19 |
| PyTorch | 2.5.0a0 (Jetson wheel) |
| GPU Architecture | Ampere SM 8.7 |
| RAM | 7.4GB unified (CPU + GPU shared) |

---

## 🔧 Optimizations Applied

### Stage 1: Core Fixes — Make It Run

| # | Problem | Solution | File |
|---|---|---|---|
| 1.1 | **Entire model loaded to GPU → OOM** | `device_map="auto"` splits model between GPU/CPU based on available memory | [`polling/inference_engine.py`](../../polling/inference_engine.py) |
| 1.2 | **CLIP ViT-Base hardcoded `.to('cuda')`** | Moved CLIP entirely to **CPU** (saves 600MB CUDA) | [`clip_encoder.py`](../../mobilevideogpt/model/multimodal_encoder/clip_encoder.py) |
| 1.3 | **CPU/CUDA device mismatch** when CLIP (CPU) features used by video encoder (CUDA) | Added device-bridging to transfer tensors between devices | [`arch.py`](../../mobilevideogpt/model/arch.py) |
| 1.4 | **VideoMamba downloads redundant 25MB checkpoint** → 1.2GB peak memory waste | `pretrained=None` — weights already in safetensors | [`build_videomamba.py`](../../mobilevideogpt/model/videomamba/build_videomamba.py) |
| 1.5 | **`bfloat16` not supported on Jetson Ampere SM 8.7** | Switched all dtype: `bfloat16` → `float16` | [`inference_engine.py`](../../polling/inference_engine.py) |
| 1.6 | **`torch.compile()` crashes on Jetson** (Triton aarch64 incompatibility) | Removed `torch.compile()` | [`inference_engine.py`](../../polling/inference_engine.py) |
| 1.7 | **`causal_conv1d` API mismatch** (7 args → 8 args) | Monkey-patched at runtime | [`inference_engine.py`](../../polling/inference_engine.py) |
| 1.8 | **`mamba_ssm` distributed utils** NVIDIA PyTorch incompatibility | Auto-patched in setup script | [`setup_jetson.sh`](../../setup_jetson.sh) |
| 1.9 | **CUDA allocator fragmentation** on 8GB device | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | [`inference_engine.py`](../../polling/inference_engine.py) |
| 1.10 | **Triton autotuner allocates 256MB probe** that fails on fragmented Jetson | Monkey-patched `get_empty_cache_for_benchmark` to return a tiny buffer | [`inference_engine.py`](../../polling/inference_engine.py) |

**Result**: Model loads and runs inference without crashing.

---

### Stage 2: Latency Optimizations — Make It Faster

| # | Change | Rationale | Impact |
|---|---|---|---|
| 2.1 | `max_new_tokens`: 512 → 128 | Exercise feedback is 2-3 sentences | ~15-20s saved |
| 2.2 | CUDA budget: 40% → 55% of free memory | More Qwen layers on GPU = faster per-token generation | ~5-10s saved |
| 2.3 | Kept `num_frames` at 16 | Model architecture requires it (2 chunks × 4 topK) | — |

**Result**: Latency dropped from ~50s → **~31s per poll** (40% faster).

---

### Stage 3: Installation & Portability

| # | Change | Why |
|---|---|---|
| 3.1 | Created `setup_jetson.sh` (504 lines) | Automates entire Jetson setup |
| 3.2 | Created `requirements_jetson.txt` | Jetson-specific Python deps |
| 3.3 | Builds `causal-conv1d` + `mamba-ssm` from source with `TORCH_CUDA_ARCH_LIST=8.7` | Pre-built wheels don't support Jetson Ampere |
| 3.4 | Uses NVIDIA Jetson wheels for PyTorch | JP6.1 wheel works on JP6.2/CUDA 12.6 |
| 3.5 | Builds torchvision from source | Pip PyTorch (x86) shows `CUDA: False` on Jetson |

**Result**: Repo can be cloned to fresh Jetson → `bash setup_jetson.sh` → works end-to-end.

---

### Stage 4: TensorRT CLIP Infrastructure (opt-in)

| # | Work | File |
|---|---|---|
| 4.1 | ONNX export pipeline for CLIP ViT-Base (300MB file, FP32) | [`scripts/export_clip_to_onnx.py`](../../scripts/export_clip_to_onnx.py) |
| 4.2 | `trtexec`-based engine builder with FP16/FP32/fp16-safe modes | [`scripts/build_clip_tensorrt.sh`](../../scripts/build_clip_tensorrt.sh) |
| 4.3 | Python-API builder with layer-level precision control | [`scripts/build_clip_tensorrt_safe.py`](../../scripts/build_clip_tensorrt_safe.py) |
| 4.4 | PyTorch-compatible TRT runtime wrapper + preload cache | [`clip_trt.py`](../../mobilevideogpt/model/multimodal_encoder/clip_trt.py) |
| 4.5 | Accuracy validation vs PyTorch FP32 golden reference | [`scripts/test_clip_tensorrt.py`](../../scripts/test_clip_tensorrt.py) |

**Benchmarks (16 frames)**:

| Backend | Latency | Accuracy | Status |
|---|---|---|---|
| PyTorch CPU FP16 (original) | 37,280ms | baseline | — |
| PyTorch CPU FP32 (reference) | 6,500ms | baseline | — |
| **TensorRT GPU FP32** | **180ms** | **cosine = 1.0006** ✅ | Works, memory-tight |
| TensorRT GPU FP16 | 79ms | ❌ output saturated ±512 | **Deferred** — TRT 10.3 + Ampere SM 8.7 bug |

**Memory reality on 8GB Jetson**: TRT FP32 CLIP (~500MB CUDA) + Qwen2 on CUDA = tight fit. Sometimes OOMs at `lm_head` during generation. **Disabled by default**, enable with `USE_TRT_CLIP=1` on boards with more RAM.

---

### Stage 5: ONNX Runtime CPU CLIP (shipped default)

| # | Work | File |
|---|---|---|
| 5.1 | ORT session wrapper using the same exported ONNX file | [`clip_ort.py`](../../mobilevideogpt/model/multimodal_encoder/clip_ort.py) |
| 5.2 | Wire into backend selection priority (TRT → ORT → PyTorch) | [`clip_encoder.py`](../../mobilevideogpt/model/multimodal_encoder/clip_encoder.py) |

**Why ORT CPU wins**: ONNX Runtime has graph fusion + multi-threaded CPU kernels tuned for ARM NEON. No GPU memory usage at all, so it's rock-solid regardless of other allocations.

| Metric | PyTorch CPU FP16 | **ONNX Runtime CPU** | Speedup |
|---|---|---|---|
| CLIP 16 frames | 37,280ms | **5,064ms** | **7.4x** |
| CUDA memory | 0 | 0 | — |
| Accuracy | baseline | same (FP32 internally) | ✅ preserved |

**Result**: Full pipeline drops from ~31s → **~23s per poll** (cold) or **~13s per poll** (warm).

---

### Stage 6: Reliability — Make It Production-Viable

The critical issue that blocked production use: **running inference twice in the same process would OOM on the 2nd run** due to CUDA allocator fragmentation.

| # | Problem | Solution | File |
|---|---|---|---|
| 6.1 | **2nd back-to-back inference OOMs** at `lm_head` | Inter-poll `gc.collect()` + `torch.cuda.empty_cache()` between polls | [`inference_engine.py`](../../polling/inference_engine.py) |
| 6.2 | **Rare transient OOMs** from temporary fragmentation | One-shot retry: catch OOM → clear cache + sync → retry once (usually succeeds) | [`inference_engine.py`](../../polling/inference_engine.py) |
| 6.3 | **40s reload per inference script invocation** | Use [`gradio_app.py`](../../polling/gradio_app.py) — long-running server: model loads once, inference requests come via UI/HTTP |

**Verified with 4 back-to-back runs through the Gradio UI**: all 4 succeeded. ✅

---

## 📊 Latency Breakdown (Current)

```
Total per poll (warm): ~13s

┌──────────────────────────────────────────────┐
│ Qwen2 LLM (mixed CPU/GPU)   ~6-8s  ████████ │  60% ← new bottleneck
│ CLIP (ORT CPU, 16 frames)   ~3-5s  ████     │  30%
│ VideoMamba (CUDA)           ~2-3s  ██       │
│ Frame extraction            ~1s    █        │
└──────────────────────────────────────────────┘
```

The bottleneck has shifted from CLIP (60% of time) to **Qwen2 LLM generation** (60%). Any further wins need to target Qwen2.

---

## 🎬 Demo — Actual Output

**Input**: Exercise video with "Incorrect Form" text overlay

**Model Response**:
> *"The exercise form shown is incorrect, as indicated by the text 'Incorrect Form' and a red arrow pointing to the wrong position. A correction should be made to ensure proper alignment of the torso with the hips and arms."*

✅ Correctly identifies incorrect form
✅ Provides reasonable correction suggestion
✅ Generated in ~13s after warmup

---

## 🚀 Future Roadmap

### ✅ Done

- TensorRT CLIP pipeline (infrastructure ready, opt-in on higher-memory boards)
- ONNX Runtime CPU CLIP (default, reliable)
- Server-mode architecture for back-to-back reliability
- Inter-poll memory cleanup + OOM retry

### 🎯 Tier 1 — Next Targets (Qwen2 is the bottleneck)

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 7.1 | **Qwen2 INT8 via AWQ** (Activation-aware Weight Quantization) | 2x on LLM stage → ~6-7s total | <1% loss |
| 7.2 | **ONNX Runtime for Qwen2** (not just CLIP) | 1.5-2x on LLM | Zero |
| 7.3 | **KV cache reuse across polls** — same video, similar context | 30-50% on repeated polls | Zero |
| 7.4 | **Fix TRT FP16 CLIP** — solve the ±512 saturation bug | Saves another ~5s | Zero |

### ⚡ Tier 2 — Quick Wins

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 7.5 | `sudo nvpmodel -m 0` + `jetson_clocks` | 20-30% overall | Zero |
| 7.6 | Headless boot (no GNOME desktop) | Frees ~500MB → can re-enable TRT CLIP | Zero |
| 7.7 | `max_new_tokens` 128 → 64 | ~3s saved | Zero |

### 🧠 Tier 3 — Architectural

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 7.8 | Async frame extraction (pipeline parallel) | ~1-2s saved | Zero |
| 7.9 | Speculative decoding with draft LLM | 2x on LLM stage | Zero (verified against main) |
| 7.10 | Model distillation to smaller student | 3-5x on LLM | Medium (needs retraining) |

### ❌ Avoided

- **INT4 quantization** — Too lossy for a 0.5B model (2-5% accuracy loss)
- **Model pruning** — Risk of breaking VideoMamba custom kernels

---

## 🎯 Realistic Target Trajectory

```
Phase 1 (DONE):   ~50s/poll  — Make it work
Phase 2 (DONE):   ~31s/poll  — Quick latency wins (max_new_tokens, CUDA budget)
Phase 3 (DONE):   ~23s/poll  — ONNX Runtime CLIP
Phase 4 (DONE):   ~13s/poll  — Server mode + reliability (current)
Phase 5 (Next):   ~5-7s/poll — Qwen2 INT8 / ORT / KV cache reuse
Phase 6:          ~2-3s/poll — Speculative decoding, true real-time
```

---

## 🛠️ How to Run

### Fresh Setup
```bash
git clone https://github.com/EdgeVLM-Labs/mobile-videogpt-adaptation.git
cd mobile-videogpt-adaptation
git checkout jetson-inference
bash setup_jetson.sh

# Optional — build TensorRT engine for CLIP (not required, ORT CPU works)
python scripts/export_clip_to_onnx.py
bash scripts/build_clip_tensorrt.sh fp32
```

### Recommended: Gradio Server (persistent, reliable)

```bash
conda activate mvgpt
python polling/gradio_app.py
# Then open the URL printed in console
```

Model loads once, inference runs are back-to-back reliable (no OOM between runs, no 40s reload).

### One-Shot CLI (for testing)

```bash
conda activate mvgpt
python polling/run_polling.py sample_videos/00000340.mp4 --max-polls 1

# With LoRA finetuning:
python polling/run_polling.py sample_videos/00000340.mp4 \
  --max-polls 1 \
  --lora-weights "EdgeVLM-Labs/mvgpt-14_2000-pool-exercise_feedback-20260320_201254"
```

### Max Performance Mode

```bash
sudo nvpmodel -m 0          # 15W power profile
sudo jetson_clocks          # Lock max GPU clocks
```

### Enable TensorRT CLIP (opt-in — requires headroom)

```bash
# Build engine first (if not already built)
python scripts/export_clip_to_onnx.py
bash scripts/build_clip_tensorrt.sh fp32

# Then run with flag
USE_TRT_CLIP=1 python polling/gradio_app.py
```

⚠️ On 8GB Jetson Orin Nano this is memory-tight and may OOM during Qwen2 generation. Viable on Orin NX 8GB+ or Orin Nano Super.

---

## 📝 Change Log

| Date | Change | Latency |
|---|---|---|
| 2026-03-19 | **Phase 1** — Initial Jetson adaptation (device_map, CLIP→CPU, bf16→fp16, etc.) | First working: ~50s |
| 2026-03-21 | **Phase 2** — `max_new_tokens` 512→128, CUDA budget 40%→55% | ~31s |
| 2026-03-21 | Experimented with `num_frames=8` — **failed** (VideoMamba has hardcoded 8-frame pos embedding) | — |
| 2026-03-21 | Experimented with CUDA budget 60% — OOM during lm_head generation | Reverted to 55% |
| 2026-04-18 | **Phase 3** — TensorRT CLIP infrastructure shipped (opt-in, memory-limited) | — |
| 2026-04-18 | **Phase 4a** — ONNX Runtime CPU CLIP becomes default (7.4x faster than PyTorch CPU) | ~23s |
| 2026-04-18 | **Phase 4b** — Inter-poll cleanup + OOM retry: reliable back-to-back inference | ~13s (warm) |
| 2026-04-18 | **Phase 4c** — Validated Gradio server with 4 consecutive runs (no crashes) | — |

---

## 📚 Related Docs

- [`jetson_inference_fixes.md`](./jetson_inference_fixes.md) — Original fixes documentation
- [`setup_jetson.sh`](../../setup_jetson.sh) — Setup script with detailed comments
- [`models/tensorrt/README.md`](../../models/tensorrt/README.md) — TensorRT engine build instructions
- [Stage logs](./) — Raw inference logs from each optimization stage

---

## 👥 Contributors

Work done collaboratively on the `jetson-inference` branch.

## 🤝 Contributing

When you make new optimizations:
1. Update the **Change Log** table at the bottom
2. Add a new row to the relevant **Tier** table (or move from "Planned" to "Applied")
3. Update the **Latency Breakdown** if the bottleneck shifts
4. Add before/after numbers to the **Summary Table** at the top
