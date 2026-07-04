# Mobile-VideoGPT on Jetson Orin Nano — Optimization Journey

> **Living document** — updated continuously as we iterate on optimizations.
> Last updated: 2026-05-04

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
| **TTFT** (time-to-first-word) | ❌ OOM | **~2.3s** | **5s-feedback target hit** |
| Full-response latency | ❌ OOM | ~5-10s (length-dependent) | — |
| Back-to-back reliability | ❌ 2nd run crashes | ✅ 4+ runs without failure | Production-viable |
| Memory usage | Exceeds 8GB | ~7GB | Fits |
| Output quality | N/A | ✅ Correct evaluations | Baseline preserved |

**Recommended demo invocation**:
```bash
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py
```

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

The Jetson aarch64 + custom CUDA stack means **almost no library
installs cleanly with `pip install`** — most CUDA-dependent packages
need to be built from source against the NVIDIA Jetson PyTorch wheel
and `TORCH_CUDA_ARCH_LIST=8.7` to actually run on this device.

We package this complexity into [`setup_jetson.sh`](../../setup_jetson.sh)
(9 steps, ~30–45 min on a fresh board), so a fresh clone reaches a
working inference state with one command:

```bash
git clone … && cd mobile-videogpt-adaptation
git checkout jetson-inference
bash setup_jetson.sh
```

**For the rationale behind each install step** (why pip wheels fail,
why we build cuSPARSELt / torchvision / causal-conv1d / mamba-ssm from
source, why we patch `mamba_ssm.distributed`, and the
`pypi.jetson-ai-lab.dev` DNS quirk), see
[**INSTALL_NOTES.md**](./INSTALL_NOTES.md).

**Result**: Repo can be cloned to a fresh Jetson, run
`bash setup_jetson.sh`, and reach a working inference setup end-to-end.

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

### Stage 7: Real-Time Feel — Hit the 5s Target (TTFT 2.3s)

The breakthrough that made the pipeline demo-ready. Four changes combined:

| # | Optimization | What it does | Saves |
|---|---|---|---|
| 7.1 | **`lm_head` patch** ([`qwen.py`](../../mobilevideogpt/model/language_model/qwen.py)) | During prefill, only compute logits for the LAST position (not all 450). Eliminates a 137MB temp allocation that was causing OOMs. | **Unlocks 7.2 + 7.3** |
| 7.2 | **Full-GPU Qwen2** (`USE_FULL_GPU=1`) | 75% CUDA budget forces all 24 layers on GPU. No more CPU↔GPU tensor transfers between layers during autoregressive generation. | ~2-3s |
| 7.3 | **TRT CLIP viable** (`USE_TRT_CLIP=1`) | Previously OOMed alongside Qwen2. Now safe thanks to 7.1. CLIP: ~3s (ORT CPU) → ~0.2s (TRT GPU). | ~2.8s |
| 7.4 | **Token streaming** ([`gradio_app.py`](../../polling/gradio_app.py)) | TextIteratorStreamer + progress stages in UI. First word appears at prefill+first-token, rest streams live. | 0s real; converts wait into visible progress |

**Result**:
- **TTFT: 11s → 2.3s** (perceived latency for user)
- **Full-response latency: 11s → 5-10s** (actual compute)
- Panel sees first word in ~2s, full coaching feedback during exercise

---

### Stage 8: Real-Time Frame Buffering — Ground the Pipeline in "Now"

While the engineering pipeline up to Phase 7 was fast, the **content** the
model evaluated was not strictly real-time. The original buffering strategy
was inherited from offline video benchmarks:

- **Capture rate**: 1 fps
- **Buffer**: 64 frames (≈ 64 s of history)
- **Per-poll sampling**: uniform across the full buffer

For a patient doing **multiple exercises in sequence** (squats → push-ups →
lunges), this meant a single poll could mix frames from three different
exercises — confusing the model and producing generic feedback. Even within
one long exercise (≥ 60 s), frames from the start and end were averaged,
hiding form drift over time.

| # | Change | File |
|---|---|---|
| 8.1 | **Tail sampling** instead of uniform sampling — each poll uses the most-recent `num_frames` from the buffer, not a sample spread across all of it | [`stream_handler.py`](../../polling/stream_handler.py) |
| 8.2 | **Capture FPS 1 → 4** — finer temporal resolution; 16 frames now span 4 seconds instead of 16 | [`config.py`](../../polling/config.py) |
| 8.3 | **Buffer size 64 → 32** — keeps only ~8 s of history, so old-exercise frames purge within ~8 s of switching | [`config.py`](../../polling/config.py) |
| 8.4 | **Cold-start guard** — first inference waits for buffer to hold a full window of real frames; UI shows a `0/16 → 16/16` countdown so the user knows the system is alive | [`gradio_app.py`](../../polling/gradio_app.py) |

**Effect on temporal semantics**:
| | Before Phase 8 | After Phase 8 |
|---|---|---|
| Active window per poll | uniform sample over last ~64 s | last 4 s contiguously |
| Exercise switch contamination | up to 64 s of stale frames mixed in | fully purged in ≤ 8 s |
| Cold-start first response | ran on partial + zero-padded buffer | waits 4 s, then runs on full buffer |
| Camera native FPS used | 1 of every 30 frames kept | 4 of every 30 frames kept |

**No effect on inference compute time** — the model still receives 16
frames per poll. Phase 8 is about *what the 16 frames represent* (the most
recent 4 s of activity) rather than how fast the model processes them.

---

### Stage 9: Input Gating — Don't Talk When Nobody's Exercising

Up to Stage 8 the pipeline was fast and temporally grounded, but it was
**unconditionally generative**: every poll (~3 s) ran the VLM and the VLM
*always* produced text — because it has no "abstain" option and the prompt
itself presupposes an exercise is happening
([`config.py`](../../polling/config.py) → *"Please evaluate the exercise form
shown…"*). The result: an empty room, a person standing idle, or someone doing
an **untrained** exercise all still got confident-sounding coaching.

This is not a model bug — it's a missing **gate** in front of the model. The fix
is a cheap pre-check that decides whether to invoke the VLM at all. We split it
into two tiers by what they detect:

| Tier | Detects | Status |
|---|---|---|
| **Tier 1 — Motion gate** | "Is anything happening?" (empty / idle scene) | ✅ **Shipped** |
| **Tier 2 — Exercise gate** | "Is this a *trained* exercise?" (rejects untrained movement) | 📝 Designed — see [TIER2_EXERCISE_GATE_DESIGN.md](./TIER2_EXERCISE_GATE_DESIGN.md) |

#### Tier 1 — Motion gate (shipped)

| # | Change | File |
|---|---|---|
| 9.1 | `compute_motion_score()` — mean absolute inter-frame pixel delta over the most-recent window, on 64×64 grayscale thumbnails (a few ms, noise-robust) | [`stream_handler.py`](../../polling/stream_handler.py) |
| 9.2 | `enable_motion_gate` / `motion_threshold` config, **env-driven and OFF by default** (`MOTION_GATE=1 MOTION_THRESHOLD=2.5`) | [`config.py`](../../polling/config.py) |
| 9.3 | Loop hook **before** metrics/inference start — on a static scene, show *"⏸ Waiting for exercise…"* and skip the poll (no compute, no poll-number consumed) | [`gradio_app.py`](../../polling/gradio_app.py), [`inference_engine.py`](../../polling/inference_engine.py) |
| 9.4 | **Hysteresis** (`motion_idle_polls`, default 2) — only declare idle after N *consecutive* below-threshold polls; any active poll resets the streak. Stops a single low-motion poll (slow rep phase, brief pause) from flipping the UI to "waiting" mid-exercise | [`config.py`](../../polling/config.py), both loops |

**Properties:**
- **Latency:** the gate itself is ~10–20 ms. On a *passing* poll the visible path
  is unchanged (TTFT still ~2.3 s). On a *gated* poll it skips the whole ~5–10 s
  VLM — so on average it makes the system faster and lower-power.
- **Safety:** purely additive. With `MOTION_GATE` unset the pipeline behaves
  exactly as before. Applies to the live (direct-webcam) path only — browser
  webcam replicates a single frame (no motion to measure) and video-file mode
  doesn't use the live buffer (`compute_motion_score` returns +inf → never
  gates).
- **Tuning:** `motion_threshold` is on a 0–255 scale; typical 1.5–4.0 depending
  on camera/lighting. The gated/idle motion score is logged each poll so it can
  be tuned from real session logs. If it still flips to "waiting" mid-exercise,
  raise `MOTION_IDLE_POLLS` (debounce) and/or lower `MOTION_THRESHOLD`.

> ⚠️ **Known limitation — isometric / static-hold exercises.** A motion gate
> keys on *movement*, so a held plank, wall-sit, or any near-still hold reads as
> "no activity" and gets gated — even though the person is actively exercising.
> Motion gating fundamentally cannot cover these. The fix is **presence/posture
> detection** rather than motion: the Tier 2 exercise gate matches the *posture*
> embedding (a plank has a distinctive pose) and so handles static holds
> correctly — see [TIER2_EXERCISE_GATE_DESIGN.md](./TIER2_EXERCISE_GATE_DESIGN.md).
> Until Tier 2 ships, either leave the motion gate off when demoing holds, or
> rely on it only for dynamic exercises.

> **Note on confidence-based suppression.** We *also* have a confidence module
> (`is_confident()` in [`calculate_confidence.py`](../../utils/confidence_scoring/calculate_confidence.py)),
> but it currently only *labels* output `(NOT CONFIDENT)` and is computed from
> generation scores that exist only on the **non-streaming** path. Using it to
> *suppress* would force the demo off token-streaming (losing the 2.3 s TTFT
> feel), so it is intentionally left off for the live demo. The motion gate is
> the preferred Tier-1 mechanism because it's free and streaming-compatible.

#### Tier 2 — Exercise-relevance gate (designed, not yet built)

Handles the harder case (moving, but untrained exercise) via image-embedding
**prototype matching** that reuses the CLIP features the VLM already computes.
Full spec — including the key constraint that the deployed engine emits
penultimate-layer *patch* features (not CLIP's text-image space, so text
zero-shot doesn't apply), the two implementation options (standalone +0.2 s vs
embedding-reuse ~free), and the calibration plan — is in
[**TIER2_EXERCISE_GATE_DESIGN.md**](./TIER2_EXERCISE_GATE_DESIGN.md).

**Result:** the live demo no longer narrates to an empty stage. Combined with
the designed Tier 2, the pipeline will also decline gracefully on untrained
movements instead of bluffing.

---

## 📊 Latency Breakdown (Current — Phase 7)

```
Before Phase 7 (FP16 + ORT CPU CLIP):     After Phase 7 (+ TRT CLIP + Full-GPU + patch):
───────────────────────────────────       ────────────────────────────────────────
Total per poll (warm): ~11s                Total per poll (warm): 5-10s
                                           TTFT (first word shown): ~2.3s
Qwen2 LLM (mixed)      ~5-6s  ███████     Qwen2 LLM (all on GPU)   ~3-4s  █████
CLIP (ORT CPU)         ~3s    ████        CLIP (TRT GPU)           ~0.2s  ▏
VideoMamba (CUDA)      ~2s    ██          VideoMamba (CUDA)        ~2s    ██
Frame extraction       ~1s    █           Frame extraction         ~1s    █
```

The Qwen2 time drop (5-6s → 3-4s) comes from eliminating CPU offload bouncing once we have budget headroom.

---

## 🎬 Demo — Actual Output

**Input**: Exercise video with "Incorrect Form" text overlay

**Model Response**:
> *"The exercise form shown is incorrect, as indicated by the text 'Incorrect Form' and a red arrow pointing to the wrong position. A correction should be made to ensure proper alignment of the torso with the hips and arms."*

✅ Correctly identifies incorrect form
✅ Provides reasonable correction suggestion
✅ **First word visible at 2.3s**, full response 5-10s later (streaming)

---

## 🚀 Future Roadmap

### ✅ Done

- TensorRT CLIP pipeline (engine built, validated, **now production-default**)
- ONNX Runtime CPU CLIP (automatic fallback if TRT CLIP OOMs)
- Server-mode architecture for back-to-back reliability
- Inter-poll memory cleanup + OOM retry
- Power mode MAXN_SUPER (GPU 612 → 1020 MHz)
- `max_new_tokens` 128 → 64
- **`lm_head` patch** (only compute last-position logits during prefill)
- **Full-GPU Qwen2** (all 24 layers on CUDA, no CPU bounce)
- **Token streaming** (progressive UI updates, perceived TTFT = 2.3s)
- **Real-time frame buffering** (tail sampling + 4 fps capture + cold-start guard) — Phase 8
- **Motion gate** (Tier 1 input gating — skip the VLM on an empty/idle scene, env-driven, off by default) — Phase 9

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

### 🎯 Tier 1 — Realistic High-Value Targets (post-Phase 7)

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 8.1 | **Qwen2 INT8 via TensorRT-LLM** | **2x on LLM** → ~3-5s total | <0.3% loss (structured outputs) | **XL** (Jetson aarch64 tooling) |
| 8.2 | **Fix TRT FP16 CLIP overflow bug** — solve ±512 saturation | Saves ~3-5s on cold polls | Zero | L |
| 8.3 | **Qwen2 via ONNX Runtime** (not just CLIP) | 1.5-2x on LLM | Zero | L (complex: KV cache in ONNX) |

### ⚡ Tier 2 — Small Cumulative Wins

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 8.4 | **Warmup run at server startup** — amortize first-call JIT | Makes 1st poll as fast as subsequent | Zero | S |
| 8.5 | **Async frame extraction** — decode next frames during current inference | ~0.5-1s saved | Zero | M |
| 8.6 | **KV cache for system-prefix tokens** (limited gain given architecture) | ~5-10% per poll | Zero | M |
| 8.7 | **Vision feature caching with motion detection** — skip CLIP on static frames | Variable (0-30%) | Zero | M |

### 🧠 Tier 3 — Architectural (bigger bets)

| # | Optimization | Expected Gain | Accuracy Risk | Effort |
|---|---|---|---|---|
| 8.8 | **Speculative decoding with draft LLM** | 2x on LLM | Zero (verified) | XL |
| 8.9 | **Model distillation** (smaller student model) | 3-5x on LLM | Medium (retraining) | XL |

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
Phase 5 (DONE):   ~11s/poll  — MAXN_SUPER, max_tokens→64
Phase 6 (DONE):   INT8 quant dead-end (kept as opt-in)
Phase 7 (DONE):   🎯 TTFT 2.3s / Full 5-10s — Streaming + full-GPU + TRT CLIP
Phase 8 (DONE):   Real-time frame buffering (each poll = last 4 s of activity)
Phase 9 (DONE):   Input gating — Tier 1 motion gate (don't run the VLM on an idle scene); Tier 2 exercise gate designed

— Phases 7-9 together are our shipping configuration. Further wins below are post-demo —

Phase 10 (Future): ~3-5s actual    — TensorRT-LLM Qwen2 (multi-week work)
Phase 11 (Future): ~1-2s perceived — KV cache + speculative decoding
```

**Note**: Phase 7 hits the 5s-feedback target through real GPU optimization
(saves 5-6s) plus streaming UI (gives the perception of immediate response).
For deeper actual-latency wins below 5s on this hardware, the next steps
require significant engineering effort (TensorRT-LLM integration, KV cache
plumbing, distillation) — appropriate post-demo if higher throughput is needed.

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

### 🎯 Recommended for Demo / Production Use

This is the configuration that hits TTFT 2.3s:

```bash
# One-time (persists across reboots if nvpmodel set)
sudo nvpmodel -m 2          # MAXN_SUPER mode (full clock budget)
sudo jetson_clocks          # Lock clocks at max

# Per-session: clear memory + launch
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'
conda activate mvgpt
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/gradio_app.py

# Optional — add the motion gate so it stays quiet on an empty/idle stage
# (Stage 9, Tier 1). Off unless MOTION_GATE=1; tune MOTION_THRESHOLD per camera.
# USE_FULL_GPU=1 USE_TRT_CLIP=1 MOTION_GATE=1 MOTION_THRESHOLD=2.5 python polling/gradio_app.py
```

Open the printed URL in a browser. Model loads once (~40s), then every
inference runs in a warm server. First word appears at ~2.3s, full
response streams over 5-10s.

**Health checks in the startup logs**:
- `Loading TensorRT CLIP engine: .../clip_vit_base_fp32.engine`
- `CUDA budget: X.XXGb (75% of 5.XX GB free, mode=FULL-GPU+TRT-CLIP)`
- `Qwen2 layer placement: {'cuda:0': 24}  (lm_head: cuda:0)`

### Fallback: Default Safe Mode (no env vars)

If `USE_TRT_CLIP` triggers OOM (e.g., high desktop activity eating GPU memory):

```bash
conda activate mvgpt
python polling/gradio_app.py
```

This runs CLIP on CPU via ONNX Runtime — slower (~11s total per poll) but
bulletproof reliable on any 8GB Jetson.

### One-Shot CLI (for testing, no UI)

```bash
conda activate mvgpt
USE_FULL_GPU=1 USE_TRT_CLIP=1 python polling/run_polling.py \
  sample_videos/00000340.mp4 --max-polls 1 --lora-weights ""
```

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
| 2026-04-19 | **Phase 7** — 🎯 5s-feedback target HIT via combined attack: | |
| 2026-04-19 | • Token streaming (TextIteratorStreamer + progress UI stages) | — |
| 2026-04-19 | • `lm_head` patch: only compute last-position logits during prefill (saves 137MB OOM) | — |
| 2026-04-19 | • Full GPU Qwen2 (USE_FULL_GPU=1, all 24 layers on CUDA, no CPU bounce) | — |
| 2026-04-19 | • TRT CLIP FP32 viable (USE_TRT_CLIP=1, lm_head patch removes OOM risk) | — |
| 2026-04-19 | **Result**: TTFT **2.3s** (was 11s), Poll #0 total latency **5.6s** | **TTFT 2.3s** |
| 2026-05-02 | Power profile measured (avg 8.7 W, peak 19.0 W, idle 7.0 W) | — |
| 2026-05-04 | **Phase 8** — Real-time frame buffering: tail sampling, 4 fps capture, buffer 32, cold-start guard. Each poll now covers the most recent 4 seconds of activity (no exercise contamination across polls) | TTFT unchanged; semantics fixed |
| 2026-06-22 | **Phase 9** — Input gating. Tier 1 motion gate shipped (skip the VLM on an empty/idle scene; `MOTION_GATE=1`, off by default; ~10–20 ms gate, skips ~5–10 s VLM when idle). Tier 2 exercise-relevance gate designed ([TIER2_EXERCISE_GATE_DESIGN.md](./TIER2_EXERCISE_GATE_DESIGN.md)) | TTFT unchanged on passing polls |

---

## 📚 Related Docs

- [`README.md`](./README.md) — entry point and reading-order index for this folder
- [`TIER2_EXERCISE_GATE_DESIGN.md`](./TIER2_EXERCISE_GATE_DESIGN.md) — design for the Tier 2 exercise-relevance gate (Stage 9)
- [`INSTALL_NOTES.md`](./INSTALL_NOTES.md) — why each step in `setup_jetson.sh` exists
- [`HEADLESS_DEMO_SETUP.md`](./HEADLESS_DEMO_SETUP.md) — SSH + headless launch guide
- [`POWER_MEASUREMENT.md`](./POWER_MEASUREMENT.md) — How to capture power numbers
- [`REVIEWER_RESPONSE.md`](./REVIEWER_RESPONSE.md) — Answers to IEEE AIIoT reviewer feedback
- [`jetson_inference_fixes.md`](./jetson_inference_fixes.md) — Initial Phase-1 fix list (superseded by this doc; archived)
- [`setup_jetson.sh`](../../setup_jetson.sh) — Setup script with detailed comments
- [`models/tensorrt/README.md`](../../models/tensorrt/README.md) — TensorRT engine build instructions
- [`stage_logs/`](./stage_logs/) — raw inference logs from each optimization stage
- [`screenshots/`](./screenshots/) — demo screenshots from inference runs

---

## 👥 Contributors

Work done collaboratively on the `jetson-inference` branch.

## 🤝 Contributing

When you make new optimizations:
1. Update the **Change Log** table at the bottom
2. Add a new row to the relevant **Tier** table (or move from "Planned" to "Applied")
3. Update the **Latency Breakdown** if the bottleneck shifts
4. Add before/after numbers to the **Summary Table** at the top
