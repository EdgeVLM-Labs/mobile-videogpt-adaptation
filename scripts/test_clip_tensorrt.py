"""
Validate TensorRT CLIP engine vs original PyTorch CLIP.

Uses FP32 PyTorch as reference (the "golden" output) and compares against
TensorRT FP16 on GPU. This is more robust than FP16 CPU comparison since
CPU FP16 support is poor.

Run:
    python scripts/test_clip_tensorrt.py
"""
import os
import sys
import time
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

from mobilevideogpt.model.multimodal_encoder.clip_trt import (
    TensorRTCLIPVisionModel,
    find_best_engine,
)


MODEL_NAME = "openai/clip-vit-base-patch16"
BATCH_SIZE = 16
NUM_RUNS = 5


def build_test_input():
    """
    Build a realistic CLIP input (normalized image tensor).
    Uses the same preprocessing CLIP expects so FP16 numerics are stable.
    """
    # Create a synthetic image — gradient pattern (not random noise)
    imgs = []
    for i in range(BATCH_SIZE):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
        img[:, :, 2] = (i * 16) % 256
        imgs.append(img)

    processor = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    out = processor(images=imgs, return_tensors="pt")
    return out["pixel_values"]  # (16, 3, 224, 224) in FP32, normalized


def main():
    print("=" * 60)
    print("TensorRT CLIP Validation + Benchmark")
    print("=" * 60)

    # Accept engine path as CLI arg, else auto-detect
    if len(sys.argv) > 1:
        engine_path = sys.argv[1]
        if not os.path.isabs(engine_path):
            engine_path = os.path.join(PROJECT_ROOT, engine_path)
    else:
        engine_path = find_best_engine()
    if engine_path is None or not os.path.exists(engine_path):
        print("❌ No TensorRT engine found. Build one first:")
        print("   bash scripts/build_clip_tensorrt.sh fp32    # or fp16")
        return
    print(f"Using engine: {engine_path}")

    print("Building normalized test input...")
    pixel_values = build_test_input()
    print(f"  Input shape: {pixel_values.shape}, dtype: {pixel_values.dtype}")
    print(f"  Input range: [{pixel_values.min():.2f}, {pixel_values.max():.2f}]")

    # ---------------------------------------------------------
    # Reference: PyTorch CLIP on CPU (FP32 — the "golden" output)
    # ---------------------------------------------------------
    print()
    print("Loading PyTorch CLIP (FP32, CPU) as reference...")
    t0 = time.time()
    pt_model = CLIPVisionModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    pt_model.eval()
    for p in pt_model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded in {time.time() - t0:.1f}s")

    print()
    print(f"Running PyTorch CPU FP32 — 1 warmup + {NUM_RUNS} timed runs...")
    with torch.no_grad():
        _ = pt_model(pixel_values, output_hidden_states=True)

    pt_times = []
    for _ in range(NUM_RUNS):
        t0 = time.time()
        with torch.no_grad():
            pt_out = pt_model(pixel_values, output_hidden_states=True)
        pt_times.append(time.time() - t0)
    pt_features = pt_out.hidden_states[-2].float()
    pt_avg = sum(pt_times) / len(pt_times)
    print(f"  PyTorch CPU FP32 avg: {pt_avg * 1000:.0f}ms per call")
    print(f"  Output shape: {pt_features.shape}")
    print(f"  Output range: [{pt_features.min():.2f}, {pt_features.max():.2f}]")

    # Also get the FP16 CPU result (what the current pipeline uses) for comparison
    print()
    print("Running PyTorch CPU FP16 (current pipeline baseline)...")
    pt_model_fp16 = pt_model.half()
    pt_input_fp16 = pixel_values.half()
    fp16_times = []
    for _ in range(NUM_RUNS):
        t0 = time.time()
        with torch.no_grad():
            pt_out_fp16 = pt_model_fp16(pt_input_fp16, output_hidden_states=True)
        fp16_times.append(time.time() - t0)
    pt_features_fp16 = pt_out_fp16.hidden_states[-2].float()
    pt_avg_fp16 = sum(fp16_times) / len(fp16_times)
    print(f"  PyTorch CPU FP16 avg: {pt_avg_fp16 * 1000:.0f}ms per call")

    # Free PyTorch models before loading TensorRT
    del pt_model, pt_model_fp16, pt_out, pt_out_fp16
    import gc
    gc.collect()

    # ---------------------------------------------------------
    # TensorRT engine on GPU (FP16)
    # ---------------------------------------------------------
    print()
    print(f"Loading TensorRT engine on GPU: {engine_path}")
    t0 = time.time()
    trt_model = TensorRTCLIPVisionModel(engine_path)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    trt_dtype = trt_model.dtype
    print(f"  Engine dtype: {trt_dtype}")

    print()
    print(f"Running TensorRT GPU ({trt_dtype}) — 2 warmup + {NUM_RUNS} timed runs...")
    trt_input = pixel_values.to(device="cuda", dtype=trt_dtype).contiguous()
    for _ in range(2):
        _ = trt_model(trt_input)

    trt_times = []
    for _ in range(NUM_RUNS):
        torch.cuda.synchronize()
        t0 = time.time()
        trt_out = trt_model(trt_input)
        torch.cuda.synchronize()
        trt_times.append(time.time() - t0)
    trt_features = trt_out.hidden_states[-2].float().cpu()
    trt_avg = sum(trt_times) / len(trt_times)
    print(f"  TensorRT GPU FP16 avg: {trt_avg * 1000:.0f}ms per call")
    print(f"  Output shape: {trt_features.shape}")
    print(f"  Output range: [{trt_features.min():.2f}, {trt_features.max():.2f}]")

    # ---------------------------------------------------------
    # Accuracy comparison: TRT-FP16 vs PT-FP32 (golden)
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("Accuracy: TensorRT FP16 vs PyTorch FP32 (golden)")
    print("=" * 60)

    assert pt_features.shape == trt_features.shape
    print(f"  Output shape: {pt_features.shape}  ✅ match")

    abs_diff = (pt_features - trt_features).abs()
    rel_diff = abs_diff / (pt_features.abs() + 1e-3)
    cosine = torch.nn.functional.cosine_similarity(
        pt_features.flatten(), trt_features.flatten(), dim=0
    ).item()

    print(f"  Max absolute diff:  {abs_diff.max():.4f}")
    print(f"  Mean absolute diff: {abs_diff.mean():.4f}")
    print(f"  Mean relative diff: {rel_diff.mean() * 100:.2f}%")
    print(f"  Cosine similarity:  {cosine:.6f}  (1.0 = identical)")

    if cosine >= 0.999:
        verdict = "✅ Essentially identical (cosine ≥ 0.999)"
    elif cosine >= 0.99:
        verdict = "✅ Acceptable precision (cosine ≥ 0.99)"
    elif cosine >= 0.95:
        verdict = "⚠️  Minor drift (cosine ≥ 0.95)"
    else:
        verdict = "❌ Significant drift — investigate"
    print(f"  {verdict}")

    # ---------------------------------------------------------
    # Speedup summary
    # ---------------------------------------------------------
    print()
    print("=" * 60)
    print("Latency comparison (batch=16 frames)")
    print("=" * 60)
    print(f"  PyTorch CPU FP16 (current):  {pt_avg_fp16 * 1000:8.0f} ms")
    print(f"  PyTorch CPU FP32 (ref):      {pt_avg * 1000:8.0f} ms")
    print(f"  TensorRT GPU FP16 (new):     {trt_avg * 1000:8.0f} ms")
    print()
    print(f"  Speedup vs current (FP16 CPU): {pt_avg_fp16 / trt_avg:.1f}x  🚀")


if __name__ == "__main__":
    main()
