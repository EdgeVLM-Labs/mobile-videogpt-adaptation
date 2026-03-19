# MobileVideoGPT: Load-in-4bit Integration

Integration of bitsandbytes NF4 quantization into the MobileVideoGPT polling inference pipeline. The model uses a multi-component architecture (Qwen2 LLM + VideoMamba + CLIP + LoRA adapters) that required resolving three compatibility issues before 4-bit loading could work end-to-end.

**Default precision:** `torch.bfloat16` (hardcoded in `polling/inference_engine.py:249`, despite `polling/config.py:40` declaring `torch_dtype: "float16"` which is never read).

## Architecture

```
MobileVideoGPTQwenForCausalLM (mobilevideogpt/model/language_model/qwen.py)
├── Qwen2ForCausalLM (base LLM - 0.5B/1.5B)        ← quantized by bitsandbytes
├── VideoMamba vision tower (mobilevideogpt/model/videomamba/utils.py)  ← NOT quantized
├── CLIP vision tower (mobilevideogpt/model/multimodal_encoder/clip_encoder.py) ← NOT quantized
├── Video projector (mobilevideogpt/model/multimodal_projector/builder.py) ← NOT quantized
├── Image projector                                   ← NOT quantized
└── LoRA adapters (loaded via peft)                   ← NOT quantized
```

Only the Qwen2 linear layers are quantized to 4-bit. Vision encoders, projectors, and LoRA adapters remain in their original precision.

## Issues & Resolutions

### Issue 1: `merge_and_unload()` incompatible with quantized weights

**Error:** LoRA adapter merging fails because quantized 4-bit weights cannot be modified in-place.

**Location:** `polling/inference_engine.py:238`

**Original code:**
```python
self.model = self.model.merge_and_unload()  # always called
```

**Fix:** Skip merging when quantized; keep LoRA as a separate PEFT adapter (inference works via the PEFT forward pass: `output = quantized_base(x) + lora_B(lora_A(x))`).

```python
if is_quantized:
    self.logger.info("Skipping merge_and_unload (incompatible with quantized model)")
else:
    self.model = self.model.merge_and_unload()
```

### Issue 2: `.to(device, dtype)` fails on quantized parameters

**Error:** `RuntimeError` — bitsandbytes `Params4bit` cannot be cast to a different dtype.

**Location:** `polling/inference_engine.py:249`

**Original code:**
```python
self.model = self.model.to(device=self.config.device, dtype=torch.bfloat16)  # always called
```

**Fix (attempt 1 — failed):** Skip `.to()` entirely for quantized models.
```python
if is_quantized:
    pass  # skip entirely
```
This caused `Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same` because non-quantized submodules (VideoMamba, projectors) remained on CPU.

**Root cause:** The VideoMamba vision tower (`mobilevideogpt/model/videomamba/utils.py:66-71`) only casts dtype but never calls `.to(device)`. The CLIP tower explicitly calls `.to('cuda')` (`mobilevideogpt/model/multimodal_encoder/clip_encoder.py:29`), but VideoMamba does not. In the non-4bit path, the blanket `.to(device, dtype)` call moved everything to CUDA. Skipping it entirely left VideoMamba and projectors on CPU.

**Fix (final):** Call `.to(device)` without dtype cast. Quantized parameters handle `.to(device)` as a no-op (already placed by `device_map`), while non-quantized submodules get moved to CUDA.
```python
if is_quantized:
    self.model = self.model.to(device=self.config.device)  # device only, no dtype
else:
    self.model = self.model.to(device=self.config.device, dtype=torch.bfloat16)
```

### Issue 3: `torch.compile()` incompatible with bitsandbytes layers

**Location:** `polling/inference_engine.py:267`

**Fix:** Skip compilation for quantized models.
```python
if is_quantized:
    self.logger.info("Skipping torch.compile (incompatible with quantized model)")
else:
    self.model = torch.compile(self.model, mode="reduce-overhead")
```

### Issue 4: Missing `device_map` for `from_pretrained()`

bitsandbytes requires `device_map` to place quantized weights on GPU during loading. Without it, the model stays on CPU.

**Location:** `polling/inference_engine.py:136`

**Fix:** Add `device_map='auto'` and set `low_cpu_mem_usage=True` (required when using `device_map`).
```python
kwargs['device_map'] = 'auto'
# ...
self.model = MobileVideoGPTQwenForCausalLM.from_pretrained(
    self.config.base_model_path,
    low_cpu_mem_usage=is_quantized,  # was False
    **kwargs
)
```

## Files Modified

| File | Change |
|------|--------|
| `polling/inference_engine.py` | Added `device_map='auto'`, `low_cpu_mem_usage` toggle, conditional `merge_and_unload`, device-only `.to()`, conditional `torch.compile` |
| `polling/config.py` | Updated comment on `load_4bit` flag |
| `polling/gradio_app.py` | Added "Load in 4-bit (NF4)" checkbox, wired `load_4bit` to config, added VRAM to per-poll and session metrics display |
| `polling/metrics.py` | Added `vram_allocated_gb` / `vram_reserved_gb` to `InferenceMetrics`, VRAM stats to session summary |

## Benchmark Results (0.5B Model)

Test setup: `Amshaker/Mobile-VideoGPT-0.5B` + `EdgeVLM-Labs/mobile-videogpt-finetune-2000` LoRA, 25 polls per run, same video/settings.

| Metric | bf16 (run 1) | bf16 (run 2) | 4-bit NF4 (run 1) | 4-bit NF4 (run 2) |
|--------|-------------|-------------|-------------------|-------------------|
| **VRAM** | 1.70 GB | 1.53 GB | 1.74 GB | 1.74 GB |
| **Mean Latency** | 865.8 ms | 890.6 ms | 1146.2 ms | 1128.6 ms |
| **Mean TTFT** | 43.5 ms | 42.6 ms | 59.5 ms | 58.5 ms |
| **Mean Tokens/s** | 21.1 | 20.9 | 15.6 | 15.8 |

**4-bit uses more VRAM and is slower on the 0.5B model.** Three reasons:

### 1. Unmerged LoRA doubles parameter overhead

Without 4-bit, `merge_and_unload()` folds LoRA weights into base weights — one set of parameters. With 4-bit, merging is impossible, so both the quantized base weights AND full LoRA adapter weights coexist:

```
bf16 path:  base_weights (merged with LoRA) → ~1.0 GB
4-bit path: quantized_base (~0.25 GB) + lora_A + lora_B (~0.5 GB) + quant_state → ~1.0+ GB
```

For a 0.5B model, the LoRA adapter overhead roughly cancels the quantization savings.

### 2. Dequantization overhead per forward pass

Every linear layer must dequantize 4-bit → bf16 for matrix multiplication, adding ~30-40% latency per layer. Additionally, the unmerged LoRA path computes `output = base(x) + B(A(x))` instead of a single fused matmul.

### 3. No `torch.compile()` optimization

The bf16 path benefits from PyTorch graph-level fusion and kernel optimization via `torch.compile(mode="reduce-overhead")`. This is skipped for 4-bit due to bitsandbytes incompatibility.

### When 4-bit would help

The 1.5B variant has ~3 GB of LLM weights in bf16. Quantizing to 4-bit would reduce this to ~0.75 GB — a ~2.25 GB saving that decisively outweighs the LoRA/dequantization overhead. The 0.5B model is simply too small for the tradeoff to be favorable.
