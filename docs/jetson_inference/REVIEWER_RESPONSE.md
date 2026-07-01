# Reviewer Response — Jetson Deployment Details

> Working doc to address the Jetson-related feedback from the IEEE AIIoT
> reviewers. Covers on-device deployment metrics only — latency, memory,
> power, quantization stance, and per-watt / per-parameter efficiency.
> Score-related variance and accuracy-metric concerns are tracked
> separately. Will be edited iteratively before being merged into the
> paper / rebuttal letter.
>
> **Hardware**: NVIDIA Jetson Orin Nano Super (8 GB unified RAM, Ampere SM 8.7)
> **Model deployed**: Mobile-VideoGPT-0.5B (FP16, fine-tuned with LoRA)
> **Model evaluated in paper**: Mobile-VideoGPT-1.5B (same family)
> **Software**: JetPack 6.2, PyTorch 2.5.0a0 (NVIDIA Jetson wheel),
> TensorRT 10.3, ONNX Runtime 1.23

---

## 1. Inference Latency

| Metric | Value |
|---|---|
| Cold-start model load (one-time) | ~41 s |
| Time-to-first-token (TTFT) | **~2.3 s** |
| Full response (32 tokens, warm) | ~10 s |
| Per-token generation rate | ~250 ms / token (~4 tokens/s) |
| Effective polling cadence | every 3 s (configurable) |

**Notes**
- TTFT is what the user perceives as latency (response begins streaming).
- Measured with USE_FULL_GPU=1 + USE_TRT_CLIP=1 with MAXN_SUPER power mode and `jetson_clocks` locked.
- Without these optimizations, TTFT was ~11 s.

---

## 2. Peak Memory During Inference

| Component | Peak GPU memory |
|---|---|
| Qwen2 LLM weights (FP16) | ~1.0 GB |
| VideoMamba video encoder | ~0.4 GB |
| TensorRT CLIP engine (FP32) | ~0.5 GB |
| Activations + KV cache during generation | ~1.5 GB |
| **Working set during inference** | **~3.5 GB** |
| OS / Gradio / other overhead | ~2.5 GB |
| **Total system memory used** | **~6.0 GB / 8.0 GB** |

Model load completes within the 8 GB envelope; inference completes with ~2 GB headroom.

---

## 3. Power Consumption

Measured with `tegrastats --interval 1000` over a 253-second session
covering multiple inference polls (USE_FULL_GPU=1, USE_TRT_CLIP=1,
MAXN_SUPER mode, `jetson_clocks` locked).

| Rail | Min | Avg | Peak |
|---|---|---|---|
| **VDD_IN** (total board power) | 7.00 W | **8.69 W** | 18.96 W |
| VDD_CPU_GPU_CV (compute only) | 0.56 W | 1.57 W | 8.22 W |
| VDD_SOC (SoC fixed) | 3.12 W | 3.43 W | 5.39 W |
| Power envelope (MAXN_SUPER) | — | — | 25 W (hardware ceiling) |

**For the paper**:
- **Idle (model loaded, between polls): ~7 W**
- **Average during inference: ~8.7 W**
- **Peak during inference: ~19 W** (during prefill / vision encoding)
- Operates well below the 25 W MAXN_SUPER envelope (≈76% of peak budget at peak,
  ≈35% on average)

Method: `tegrastats --interval 1000 --logfile diagnostic_frames/power_log.txt`
running concurrently with Gradio inference; analyzed via
[`scripts/analyze_power.sh`](../../scripts/analyze_power.sh). Raw log
preserved in repository for reproducibility.

---

## 4. Quantization

**Did we use quantization?** **No.** The deployed model
(Mobile-VideoGPT-0.5B) operates at native FP16 precision throughout,
with TensorRT FP32 used for the CLIP encoder.

**Why no quantization was necessary on this deployment**:

| Constraint | Whether FP16 0.5B already meets it |
|---|---|
| **Memory feasibility (8 GB envelope)** | Yes — peak working set ~3.5 GB / 8 GB during inference; weights 1 GB. |
| **Latency target (real-time-feeling feedback)** | Yes — 2.3 s time-to-first-token, full response 5–10 s. |
| **Power budget (25 W envelope)** | Yes — average 8.7 W, peak 19 W during inference. |

Since native FP16 already satisfies all three deployment constraints,
quantization was unnecessary, and we did not introduce it as an
additional engineering variable that could perturb generation quality —
a particularly relevant concern for corrective exercise feedback, where
output fidelity directly affects the user-facing guidance.

**Note on the 1.5 B variant**: The comparative evaluation reported in
the paper uses the 1.5 B parameter variant of Mobile-VideoGPT, which is
the highest-scoring candidate among those that passed our memory
feasibility filter. For on-device deployment we used the 0.5 B variant
of the same model family — sufficient to satisfy hardware constraints at
native FP16 without compression, while remaining within the same
architectural family as the evaluated model. Quantizing the 1.5 B
variant (e.g., via TensorRT-LLM with proper SM 8.7 kernels) and
benchmarking it against FP16 0.5 B is a natural next step we leave as
future work.

---

## 5. Performance-per-Watt / Performance-per-Parameter

We report on-device efficiency figures **only for the model we fully
deployed and benchmarked** (Mobile-VideoGPT-0.5B).

Energy per inference (deployed model):
`Avg_power × Avg_latency = 8.69 W × 10 s ≈ 87 J`.

| Property | Mobile-VideoGPT-0.5B (deployed, on-device) |
|---|---|
| Parameters | 0.5 B |
| Avg power during inference | 8.69 W |
| Peak power during inference | 18.96 W |
| Avg full-response latency | ~10 s |
| **Energy per inference** | **~87 J** |
| Latency / parameter | ~20 µs/M params |
| Energy / parameter | ~174 nJ/param |

**Why not for the other candidates?** The remaining three candidates
(VideoLLaMA3-2B, NVILA-Lite-2B, Gemma3n-E2B) were eliminated during the
**feasibility filter** stage of the methodology, before any on-device
deployment was attempted. We therefore do not report on-device latency,
power, or energy figures for them — doing so would require a fully
engineered deployment per model (TensorRT engines, memory budgeting,
etc.) which is out of scope for this paper.

This is a deliberate methodological choice: the paper's contribution is
the **selection pipeline** itself, not exhaustive on-device profiling of
every candidate. Profiling the rejected candidates — possibly with
aggressive quantization to make them deployable — is a natural extension
we list as future work.
