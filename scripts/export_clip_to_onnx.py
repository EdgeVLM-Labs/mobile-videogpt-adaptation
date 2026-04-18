"""
Export CLIP ViT-Base vision tower to ONNX for TensorRT conversion.

This is a one-time setup step. Output: models/tensorrt/clip_vit_base.onnx

Memory-aware: CLIP is small (~600MB) and loaded on CPU only, so export
should work fine on Jetson 8GB even with other processes running.

Run:
    python scripts/export_clip_to_onnx.py
"""
import os
import sys
import gc
import time

os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore")

import torch
from transformers import CLIPVisionModel

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "models", "tensorrt")
OUT_PATH = os.path.join(OUT_DIR, "clip_vit_base.onnx")

# CLIP model used by Mobile-VideoGPT
MODEL_NAME = "openai/clip-vit-base-patch16"

# Input shape — matches what the pipeline feeds:
# 16 context frames × 3 channels × 224 × 224
# We use batch dim as dynamic so we can feed fewer frames if needed
BATCH_SIZE = 16
NUM_CHANNELS = 3
IMAGE_SIZE = 224


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 60)
    print("CLIP ViT-Base → ONNX Export")
    print("=" * 60)
    print(f"Model:  {MODEL_NAME}")
    print(f"Output: {OUT_PATH}")
    print(f"Shape:  ({BATCH_SIZE}, {NUM_CHANNELS}, {IMAGE_SIZE}, {IMAGE_SIZE})")
    print()

    # Memory check
    print(f"[Memory] Before load: {_rss_mb():.0f} MB")

    # Load CLIP on CPU in FP32 (export works best in FP32;
    # TensorRT will convert to FP16 later)
    print(f"Loading {MODEL_NAME} on CPU...")
    t0 = time.time()
    model = CLIPVisionModel.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    model.eval()
    # Disable gradient tracking to save memory during export
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"  Loaded in {time.time() - t0:.1f}s")
    print(f"[Memory] After load:  {_rss_mb():.0f} MB")

    # Create dummy input
    dummy_input = torch.randn(
        BATCH_SIZE, NUM_CHANNELS, IMAGE_SIZE, IMAGE_SIZE, dtype=torch.float32
    )

    # Wrap the model to return only what we need: hidden_states from select_layer=-2
    # This matches the pipeline's feature_select() behavior but we let the
    # pipeline handle that step — here we return ALL hidden states so the pipeline
    # can pick any layer.
    class CLIPWrapper(torch.nn.Module):
        def __init__(self, clip):
            super().__init__()
            self.clip = clip

        def forward(self, pixel_values):
            # Request hidden states from all layers
            out = self.clip(pixel_values, output_hidden_states=True)
            # Return the 2nd-to-last layer (what the pipeline uses: select_layer = -2)
            return out.hidden_states[-2]

    wrapped = CLIPWrapper(model)
    wrapped.eval()

    print()
    print("Running test forward pass...")
    with torch.no_grad():
        test_out = wrapped(dummy_input)
    print(f"  Output shape: {test_out.shape}  (expected: [16, 197, 768])")

    # Export to ONNX
    print()
    print("Exporting to ONNX...")
    t0 = time.time()
    torch.onnx.export(
        wrapped,
        dummy_input,
        OUT_PATH,
        input_names=["pixel_values"],
        output_names=["hidden_states"],
        dynamic_axes={
            "pixel_values":  {0: "batch"},
            "hidden_states": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
        export_params=True,
    )
    print(f"  Export took {time.time() - t0:.1f}s")

    # Cleanup PyTorch model to free memory
    del wrapped, model
    gc.collect()

    # Verify the ONNX file
    print()
    print("Verifying ONNX file...")
    import onnx
    onnx_model = onnx.load(OUT_PATH)
    onnx.checker.check_model(onnx_model)
    size_mb = os.path.getsize(OUT_PATH) / (1024 * 1024)
    print(f"  ✅ Valid ONNX model")
    print(f"  📦 File size: {size_mb:.1f} MB")
    print(f"  🔧 Opset: {onnx_model.opset_import[0].version}")
    print(f"  📥 Inputs:  {[i.name for i in onnx_model.graph.input]}")
    print(f"  📤 Outputs: {[o.name for o in onnx_model.graph.output]}")

    print()
    print("=" * 60)
    print("✅ ONNX export complete!")
    print(f"   Saved to: {OUT_PATH}")
    print()
    print("Next step: Build TensorRT engine")
    print(f"   bash scripts/build_clip_tensorrt.sh")
    print("=" * 60)


def _rss_mb():
    """Return current process RSS memory in MB."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 0.0


if __name__ == "__main__":
    main()
