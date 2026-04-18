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
| Inference latency (warm) | ❌ OOM crash | **~11s / poll** | ~4.5x faster than first working version |
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
Total per poll (warm): ~11s

┌──────────────────────────────────────────────┐
│ Qwen2 LLM (mixed CPU/GPU)   ~5-6s  ███████  │  55% ← current bottleneck
│ CLIP (ORT CPU, 16 frames)   ~3s    ████     │  27%
│ VideoMamba (CUDA)           ~2s    ██       │  18%
│ Frame extraction            ~1s    █        │
└──────────────────────────────────────────────┘
```

The bottleneck is **Qwen2 LLM generation** — half on GPU, half on CPU due to device_map="auto" split. Any further wins need to target this stage.

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
- Power mode MAXN_SUPER (GPU 612 → 1020 MHz, ~11% gain)
- `max_new_tokens` 128 → 64

### 🔍 Architectural Findings (from codebase review)

**Where our inference time goes (breakdown by code path)**:
- Qwen2 LLM on mixed CPU/GPU device_map: **largest portion** (~6s of 11s)
- ORT CLIP encoding on CPU: ~3s
- VideoMamba + projectors on GPU: ~2s
- Preprocessing + miscellaneous: ~1s

**KV cache reuse is architecturally limited** in this model:
- Sequence order: `[system tokens] → [video embeddings] → [prompt text] → [assistant prefix]`
- Video embeddings come **before** the prompt text in the sequence
- K/V cache only helps for tokens **before** any changing content
- Since video changes every poll, only the initial ~5-10 system tokens are truly cacheable
- Expected gain: **~5-10%**, not the "30-50%" originally hoped

**Vision feature caching is conditional**:
- Only beneficial if video content is **near-static** between polls
- For exercise videos (active movement every frame), frames DO change
- Would need motion-detection heuristic to be useful
- Expected gain: **highly variable** (0% for moving, 30%+ for static)

### 🎯 Tier 1 — Realistic High-Value Targets

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 7.1 | **Qwen2 INT8 via AWQ/SmoothQuant** | **2x on LLM** → ~7-8s total | <0.3% loss (structured outputs) | **XL** (Jetson aarch64 tooling) |
| 7.2 | **Fix TRT FP16 CLIP overflow bug** — solve ±512 saturation | Saves ~3-5s IF memory allows | Zero | L |
| 7.3 | **Qwen2 via ONNX Runtime** (not just CLIP) | 1.5-2x on LLM | Zero | L (complex: KV cache in ONNX) |

### ⚡ Tier 2 — Small Cumulative Wins

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 7.4 | **Warmup run at server startup** — amortize first-call JIT | Makes 1st poll as fast as subsequent | Zero | S |
| 7.5 | **Async frame extraction** — decode next frames during current inference | ~0.5-1s saved | Zero | M |
| 7.6 | **KV cache for system-prefix tokens** (limited gain given architecture) | ~5-10% per poll | Zero | M |
| 7.7 | **Vision feature caching with motion detection** — skip CLIP on static frames | Variable (0-30%) | Zero | M |
| 7.8 | **Move `lm_head` to CPU** — frees ~272MB GPU, may enable TRT CLIP to fit | Neutral (trade-off) | Zero | S |

### 🧠 Tier 3 — Architectural (bigger bets)

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 7.9 | **Speculative decoding with draft LLM** | 2x on LLM | Zero (verified) | XL |
| 7.10 | **Model distillation** (smaller student model) | 3-5x on LLM | Medium (retraining) | XL |

### ❌ Avoided

- **INT4 quantization** — 2-5% loss on 0.5B is too much for structured output
- **Model pruning** — Risk of breaking VideoMamba custom kernels
- **torch.compile** — Triton broken on Jetson aarch64 (tried, failed)

---

## 🎯 Realistic Target Trajectory

```
Phase 1 (DONE):   ~50s/poll  — Make it work
Phase 2 (DONE):   ~31s/poll  — Quick latency wins (max_new_tokens, CUDA budget)
Phase 3 (DONE):   ~23s/poll  — ONNX Runtime CLIP
Phase 4 (DONE):   ~13s/poll  — Server mode + reliability
Phase 5 (DONE):   ~11s/poll  — MAXN_SUPER, max_tokens→64 (current)
Phase 6 (Next):   ~7-8s/poll — Qwen2 INT8 OR Fix TRT FP16 CLIP
Phase 7:          ~3-5s/poll — Both #6 approaches combined, possibly + specdec
```

**Note**: Further latency wins now require significant engineering effort
(quantization tooling on Jetson aarch64, FP16 overflow debugging). The
current 11s baseline with the server architecture is already viable
for polling-style applications where feedback every ~3-5s is acceptable.

---

## 🧪 INT8 Quantization Investigation (Phase 6, 2026-04-19)

**Hypothesis**: INT8 quantization of Qwen2 could fit the full model on GPU
(eliminating CPU↔GPU bounces) AND speed up matmul via INT8 tensor cores.
Expected: ~2x LLM speedup → ~5-6s total.

**Reality on Jetson Orin Nano Ampere SM 8.7**:

| Library | Path | Latency vs FP16 | Memory | Accuracy |
|---|---|---|---|---|
| optimum-quanto 0.2.7 | Post-load `quantize()` int8 | **0.22x (slower)** | -50% | ✅ cosine 0.9999 |
| torchao 0.1 | `apply_weight_only_int8_quant` | **0.21x (slower)** | -50% | ✅ cosine 0.9999 |
| torchao 0.1 | `apply_dynamic_quant` (W8A8) | **0.008x (117x slower)** | -50% | ✅ preserved |

**Why every Python-level INT8 path is slower on Jetson:**
1. These libraries dequantize INT8 → FP16 on every matmul (no fast path)
2. Jetson SM 8.7 INT8 tensor cores exist but PyTorch's generic INT8 kernels aren't
   optimized for this specific architecture (generic codegen, not specialized)
3. torchao 0.17+ has proper kernels but requires torch >= 2.11 — we're pinned
   to NVIDIA's 2.5.0a0 Jetson wheel

**End-to-end test with USE_QUANTO=1 in inference_engine.py**:
- FP16 baseline: ~11s/poll
- Quanto INT8:  ~14-21s/poll (slower despite full GPU fit)

**Paths that WOULD work but are multi-week engineering**:
- TensorRT-LLM — proper INT8 for Jetson, complex multimodal integration
- bitsandbytes custom build for SM 8.7 — previously failed (kernel errors)
- llama.cpp GGUF + custom multimodal wrapper — untested complexity

**Decision**: Keep quanto integration as opt-in (`USE_QUANTO=1`) for future
scenarios (memory-constrained, or combined with TRT-LLM later). Do NOT make
it default — it regresses latency on current hardware.

**Implication for roadmap**: The 11s baseline is the practical floor for
this model+hardware combination without major engineering. Further wins
now have to come from architectural changes (two-tier feedback, distilled
smaller model, hardware upgrade), not kernel-level optimization.

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
| 2026-04-18 | **Phase 5a** — `max_new_tokens` 128 → 64 (responses are 30-40 tokens) | ~12.6s |
| 2026-04-18 | **Phase 5b** — MAXN_SUPER power mode (GPU 612 → 1020 MHz) + `jetson_clocks` | ~11.2s |
| 2026-04-18 | **Phase 5c** — Codebase review: KV cache reuse has architectural limits (see "Architectural Findings") | — |
| 2026-04-19 | **Phase 6** — INT8 quantization dead-end on Jetson (see Section "INT8 Quantization Investigation") | 11.2s (no change) |

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
