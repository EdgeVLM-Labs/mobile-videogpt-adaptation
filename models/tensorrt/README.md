# TensorRT Engines

Engine files (`*.engine`) and ONNX files (`*.onnx`) are **not committed** to the repo
because they are large (~150–300MB each) and Jetson-specific.

Build them locally with:

```bash
# 1. Export PyTorch CLIP → ONNX (~300MB)
python scripts/export_clip_to_onnx.py

# 2. Build TensorRT engine from ONNX
bash scripts/build_clip_tensorrt.sh fp32    # recommended — accuracy guaranteed
bash scripts/build_clip_tensorrt.sh fp16    # faster but has overflow on TRT 10.3 + Jetson
```

## Current status

- **FP32 engine**: Works correctly, 180ms for batch=16, 206x faster than CPU PyTorch.
- **FP16 engine**: Builds but produces saturated `[-512, +512]` output on TRT 10.3 / Jetson
  Orin Nano (Ampere SM 8.7). Not usable until this is fixed.

## Opt-in usage

On Jetson Orin Nano (8GB), TensorRT CLIP + Qwen2 is too tight — the main model's
`lm_head` during generation OOMs. Use only on boards with more memory:

```bash
USE_TRT_CLIP=1 python polling/run_polling.py sample_videos/00000340.mp4
```

Without the env var, CLIP stays on CPU (~37s for CLIP stage, but works reliably).
