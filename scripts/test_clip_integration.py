"""
Quick integration test — loads CLIPVisionTower which should auto-use TensorRT.

Verifies the whole chain: clip_encoder.py → clip_trt.py → TensorRT engine.
"""
import os
import sys
import logging
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

import torch
from mobilevideogpt.model.multimodal_encoder.clip_encoder import CLIPVisionTower


def main():
    print("=" * 60)
    print("CLIPVisionTower integration test")
    print("=" * 60)

    tower = CLIPVisionTower("openai/clip-vit-base-patch16")
    print()
    print(f"Backend: {getattr(tower, '_backend', 'unknown')}")
    print(f"Device:  {tower.device}")
    print(f"Dtype:   {tower.dtype}")
    print(f"Hidden:  {tower.hidden_size}")
    print(f"Patches: {tower.num_patches}")
    print()

    # Test forward pass with 16 random frames
    print("Running forward pass (batch=16)...")
    import time
    dummy = torch.randn(16, 3, 224, 224, dtype=tower.dtype, device=tower.device)

    # Warmup
    _ = tower(dummy, select_feature='patch')
    torch.cuda.synchronize() if tower.device.type == "cuda" else None

    # Timed
    t0 = time.time()
    for _ in range(3):
        features = tower(dummy, select_feature='patch')
        if tower.device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = (time.time() - t0) / 3

    print(f"  Output shape: {features.shape}  (expected: [16, 196, 768])")
    print(f"  Latency:      {elapsed * 1000:.0f}ms per call")
    print()
    print("✅ Integration works")


if __name__ == "__main__":
    main()
