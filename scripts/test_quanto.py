"""
Quick smoke test: verify optimum-quanto works for INT8 quantization
of a Qwen2-style transformer block on Jetson.

Checks:
  1. Quantization doesn't crash
  2. Memory footprint drops as expected
  3. Forward pass still runs (on CPU and CUDA)
  4. Accuracy is preserved at INT8 (for simple inputs)
"""
import os
import warnings
import time

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from optimum.quanto import quantize, qint8, qint4, freeze


def size_mb(model):
    """Total param bytes in MB."""
    total = 0
    for p in model.parameters():
        # For quanto modules, weight might be int8
        total += p.element_size() * p.numel()
    return total / (1024 * 1024)


def make_fake_qwen_block(hidden_size=896, inter_size=4864):
    """A mini Qwen2-like FFN block for testing."""
    return nn.Sequential(
        nn.Linear(hidden_size, inter_size, bias=False),
        nn.SiLU(),
        nn.Linear(inter_size, hidden_size, bias=False),
    )


def bench(model, x, n=20, device='cpu'):
    model.eval()
    x = x.to(device)
    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
    # Time
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n):
            _ = model(x)
        if device == 'cuda':
            torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000  # ms


def main():
    print("=" * 60)
    print("optimum-quanto smoke test (Qwen2-style FFN block)")
    print("=" * 60)

    torch.manual_seed(0)
    hidden = 896
    seq_len = 16

    # FP16 baseline
    fp16_model = make_fake_qwen_block().half()
    fp16_size = size_mb(fp16_model)
    print(f"\nFP16 baseline:")
    print(f"  Model size:  {fp16_size:.1f} MB")

    # Identical weights for apples-to-apples
    torch.manual_seed(0)
    int8_model = make_fake_qwen_block().half()

    # Quantize to INT8
    quantize(int8_model, weights=qint8)
    freeze(int8_model)
    int8_size = size_mb(int8_model)
    print(f"\nINT8 quantized:")
    print(f"  Model size:  {int8_size:.1f} MB ({int8_size/fp16_size*100:.0f}% of FP16)")
    print(f"  Savings:     {fp16_size - int8_size:.1f} MB")

    # Accuracy comparison
    x = torch.randn(1, seq_len, hidden, dtype=torch.float16)
    with torch.no_grad():
        fp16_out = fp16_model(x)
        int8_out = int8_model(x)

    abs_diff = (fp16_out.float() - int8_out.float()).abs()
    cos = torch.nn.functional.cosine_similarity(
        fp16_out.float().flatten(), int8_out.float().flatten(), dim=0
    ).item()
    print(f"\nAccuracy (vs FP16):")
    print(f"  Max abs diff:     {abs_diff.max().item():.4f}")
    print(f"  Mean abs diff:    {abs_diff.mean().item():.4f}")
    print(f"  Cosine similarity: {cos:.6f}")

    # Speed — CPU
    print(f"\nLatency (CPU, batch=1 seq=16):")
    fp16_cpu_ms = bench(fp16_model.float(), x.float(), device='cpu')
    int8_cpu_ms = bench(int8_model, x, device='cpu')
    print(f"  FP32 (CPU): {fp16_cpu_ms:.2f}ms")
    print(f"  INT8 (CPU): {int8_cpu_ms:.2f}ms  ({fp16_cpu_ms/int8_cpu_ms:.2f}x)")

    # Speed — CUDA
    if torch.cuda.is_available():
        print(f"\nLatency (CUDA, batch=1 seq=16):")
        fp16_cuda = make_fake_qwen_block().half().cuda()
        int8_cuda_model = make_fake_qwen_block().half()
        quantize(int8_cuda_model, weights=qint8)
        freeze(int8_cuda_model)
        int8_cuda_model = int8_cuda_model.cuda()
        x_cuda = x.cuda()

        fp16_gpu_ms = bench(fp16_cuda, x_cuda, device='cuda')
        int8_gpu_ms = bench(int8_cuda_model, x_cuda, device='cuda')
        print(f"  FP16 (CUDA): {fp16_gpu_ms:.2f}ms")
        print(f"  INT8 (CUDA): {int8_gpu_ms:.2f}ms  ({fp16_gpu_ms/int8_gpu_ms:.2f}x)")

    print()
    print("=" * 60)
    print("Verdict:")
    if cos > 0.99 and int8_size < fp16_size:
        print(f"  ✅ INT8 quantization works on Jetson")
        print(f"  ✅ Memory: {int8_size/fp16_size*100:.0f}% of FP16")
        print(f"  ✅ Accuracy preserved (cosine {cos:.4f})")
    else:
        print(f"  ⚠️  Issues detected — see output above")


if __name__ == "__main__":
    main()
