"""
Validate the ONNX file independent of TensorRT.

Runs the ONNX file through ONNX Runtime (CPU) and compares with PyTorch.
If these match, our ONNX export is correct and any accuracy issue is
TensorRT-specific (likely FP16 overflow during conversion).
"""
import os
import sys
import warnings

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
import onnxruntime as ort
from transformers import CLIPVisionModel, CLIPImageProcessor

ONNX_PATH = os.path.join(PROJECT_ROOT, "models", "tensorrt", "clip_vit_base.onnx")
MODEL_NAME = "openai/clip-vit-base-patch16"
BATCH = 16


def main():
    # Use a real normalized image input
    imgs = []
    for i in range(BATCH):
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        img[:, :, 0] = np.linspace(0, 255, 224, dtype=np.uint8)[None, :]
        img[:, :, 1] = np.linspace(0, 255, 224, dtype=np.uint8)[:, None]
        img[:, :, 2] = (i * 16) % 256
        imgs.append(img)

    proc = CLIPImageProcessor.from_pretrained(MODEL_NAME)
    pixel_values = proc(images=imgs, return_tensors="pt")["pixel_values"]

    # PyTorch FP32 reference
    print("Running PyTorch FP32 reference...")
    model = CLIPVisionModel.from_pretrained(MODEL_NAME).eval()
    with torch.no_grad():
        out = model(pixel_values, output_hidden_states=True)
    pt_features = out.hidden_states[-2].numpy()
    print(f"  PyTorch output range: [{pt_features.min():.2f}, {pt_features.max():.2f}]")
    del model

    # ONNX Runtime CPU
    print()
    print("Running ONNX Runtime (CPU)...")
    sess = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
    ort_features = sess.run(None, {"pixel_values": pixel_values.numpy()})[0]
    print(f"  ONNX output range: [{ort_features.min():.2f}, {ort_features.max():.2f}]")

    # Compare
    print()
    abs_diff = np.abs(pt_features - ort_features)
    cosine = (pt_features.flatten() * ort_features.flatten()).sum() / (
        np.linalg.norm(pt_features) * np.linalg.norm(ort_features) + 1e-8
    )
    print(f"Max abs diff: {abs_diff.max():.6f}")
    print(f"Cosine similarity: {cosine:.6f}")

    if cosine >= 0.999:
        print()
        print("✅ ONNX file is CORRECT — matches PyTorch FP32.")
        print("   Any accuracy drift is happening during TensorRT conversion.")
        print("   Next: try FP32 TensorRT engine to confirm, then find FP16 fix.")
    else:
        print()
        print("❌ ONNX file is WRONG — doesn't match PyTorch!")
        print("   Need to fix the ONNX export (scripts/export_clip_to_onnx.py).")


if __name__ == "__main__":
    main()
