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

| Metric | Original | Current | Improvement |
|---|---|---|---|
| Model loading | ❌ OOM crash | ✅ ~41s | Works |
| Inference latency | ❌ OOM crash | ~31s/poll | Works |
| Memory usage | Exceeds 8GB | ~6.1GB | Fits |
| Output quality | N/A | ✅ Correct evaluations | Baseline preserved |

---

## 🏗️ Platform Details

| Spec | Value |
|---|---|
| Board | Jetson Orin Nano |
| JetPack | 6.2 (L4T R36.4.7) |
| CUDA | 12.6 |
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

## 📊 Latency Breakdown (Current)

```
Total per poll: ~31s

┌────────────────────────────────────────────────┐
│ CLIP (CPU, 16 frames)     ~15-20s  ████████   │ 60% ← biggest bottleneck
│ Qwen2 LLM (mixed CPU/GPU) ~10-15s  ██████     │ 35%
│ VideoMamba (CUDA)         ~3-5s    ██         │
│ Frame extraction          ~1-2s    █          │
└────────────────────────────────────────────────┘
```

---

## 🎬 Demo — Actual Output

**Input**: Exercise video with "Incorrect Form" text overlay

**Model Response**:
> *"The exercise form shown is incorrect, as indicated by the text 'Incorrect Form' and a red circle. The correct form should be to extend your arms forward while bending at the hips and knees."*

✅ Correctly identifies incorrect form
✅ Provides reasonable correction suggestion
✅ Generated in ~25s (Poll #2, after warmup)

---

## 🚀 Future Roadmap

### 🎯 Tier 1 — High Impact (Planned)

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 4.1 | **TensorRT for CLIP** (FP16) | 15s → 1s on CLIP stage | **Zero** (same precision) |
| 4.2 | **Move CLIP back to GPU** (after freeing memory) | Eliminates CPU bottleneck | **Zero** |
| 4.3 | **INT8 CLIP + INT8 Qwen2** | Model fits fully on GPU | <1% loss |

### ⚡ Tier 2 — Quick Wins

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 4.4 | `sudo nvpmodel -m 0` + `jetson_clocks` | 20-30% overall | **Zero** |
| 4.5 | Headless boot (no GNOME desktop) | Frees ~500MB memory | **Zero** |
| 4.6 | `max_new_tokens` 128 → 64 | ~5s saved | **Zero** (same quality, shorter output) |

### 🧠 Tier 3 — Architectural

| # | Optimization | Expected Gain | Accuracy Risk |
|---|---|---|---|
| 4.7 | KV cache reuse across polls | 30-50% on repeated polls | **Zero** |
| 4.8 | Async frame extraction (pipeline parallel) | ~2s saved | **Zero** |
| 4.9 | Speculative decoding with draft LLM | 2x on LLM stage | **Zero** (verified against main) |

### ❌ Avoided

- **INT4 quantization** — Too lossy for a 0.5B model (2-5% accuracy loss, could break short structured outputs)
- **Model pruning** — Risk of breaking VideoMamba custom kernels

---

## 🎯 Realistic Target Trajectory

```
Phase 1 (DONE):   31s/poll  — Makes it work
Phase 2 (Next):   Tier 2 wins →    ~22s/poll
Phase 3:          TensorRT CLIP →  ~6-8s/poll
Phase 4:          KV cache + async ~3-4s/poll  ← near real-time
Phase 5:          Tier 3 complete   ~1-2s/poll  ← true real-time
```

---

## 🛠️ How to Run

### Fresh Setup
```bash
git clone https://github.com/EdgeVLM-Labs/mobile-videogpt-adaptation.git
cd mobile-videogpt-adaptation
git checkout jetson-inference
bash setup_jetson.sh
```

### Run Inference
```bash
# Activate env
conda activate mvgpt

# Single inference
python polling/run_polling.py sample_videos/00000340.mp4 --max-polls 1

# With LoRA finetuning
python polling/run_polling.py sample_videos/00000340.mp4 \
  --max-polls 1 \
  --lora-weights "EdgeVLM-Labs/mvgpt-14_2000-pool-exercise_feedback-20260320_201254"
```

### Max Performance Mode
```bash
sudo nvpmodel -m 0          # 15W power profile
sudo jetson_clocks           # Lock max GPU clocks
```

---

## 📝 Change Log

| Date | Change | Latency |
|---|---|---|
| 2026-03-19 | **Phase 1** — Initial Jetson adaptation (device_map, CLIP→CPU, bf16→fp16, etc.) | First working: ~50s |
| 2026-03-21 | **Phase 2** — `max_new_tokens` 512→128, CUDA budget 40%→55% | ~31s |
| 2026-03-21 | Experimented with `num_frames=8` — **failed** (VideoMamba has hardcoded 8-frame pos embedding) | — |
| 2026-03-21 | Experimented with CUDA budget 60% — OOM during lm_head generation | Reverted to 55% |
| 2026-04-18 | **Next**: Tier 2 quick wins + TensorRT CLIP exploration | — |

---

## 📚 Related Docs

- [`jetson_inference_fixes.md`](./jetson_inference_fixes.md) — Original fixes documentation
- [`setup_jetson.sh`](../../setup_jetson.sh) — Setup script with detailed comments
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
