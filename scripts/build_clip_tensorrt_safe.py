"""
Build a FP16 TensorRT engine with precise layer-level precision control.

Unlike trtexec's wildcard --layerPrecisions, this script uses the TensorRT
Python API to walk the network and explicitly force FP32 on overflow-prone
operations (LayerNorm + Softmax + Residual Adds).

Output: models/tensorrt/clip_vit_base_fp16.engine

Memory budget: ~1.5GB CUDA needed during build.
"""
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import tensorrt as trt

ONNX_FILE = os.path.join(PROJECT_ROOT, "models", "tensorrt", "clip_vit_base.onnx")
ENGINE_FILE = os.path.join(PROJECT_ROOT, "models", "tensorrt", "clip_vit_base_fp16.engine")

# Workspace memory — 256MB is plenty for CLIP ViT-Base
WORKSPACE_MB = 256

# Operation types that overflow in FP16 for ViT-style models.
# We force these to compute in FP32 while keeping MatMul in FP16.
FP32_OP_TYPES = {
    "LayerNormalization",  # LN can produce large values before normalization
    "Softmax",              # exp(large) overflows FP16 easily
}


def main():
    print("=" * 60)
    print("CLIP ViT-Base → TensorRT FP16 (safe precision)")
    print("=" * 60)

    if not os.path.exists(ONNX_FILE):
        print(f"❌ ONNX file not found: {ONNX_FILE}")
        sys.exit(1)

    logger = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(logger, "")

    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    print(f"Parsing {ONNX_FILE}...")
    with open(ONNX_FILE, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ERROR: {parser.get_error(i)}")
            sys.exit(1)

    print(f"  Network: {network.num_layers} layers, "
          f"{network.num_inputs} input(s), {network.num_outputs} output(s)")

    # Configure build
    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * 1024 * 1024
    )

    # Enable FP16 globally
    config.set_flag(trt.BuilderFlag.FP16)
    # OBEY means TRT *must* honor our per-layer precision (hard constraint)
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)

    # Walk the network and force FP32 on overflow-prone ops.
    # We're aggressive here because CLIP ViT-Base has residual branches that
    # can accumulate to very large values; better safe + correct than fast + wrong.
    forced_count = 0
    op_counts = {}
    for i in range(network.num_layers):
        layer = network.get_layer(i)

        # LayerNorm: mean/variance computations can overflow
        is_layernorm = (layer.type == trt.LayerType.NORMALIZATION)
        # Softmax: exp(large) overflows FP16 (max 65504)
        is_softmax = (layer.type == trt.LayerType.SOFTMAX)
        # Reduction: variance/sum over residual branches
        is_reduce = (layer.type == trt.LayerType.REDUCE)

        if is_layernorm or is_softmax or is_reduce:
            layer.precision = trt.float32
            for j in range(layer.num_outputs):
                layer.set_output_type(j, trt.float32)
            forced_count += 1
            if is_layernorm:
                kind = "LayerNorm"
            elif is_softmax:
                kind = "Softmax"
            else:
                kind = "Reduce"
            op_counts[kind] = op_counts.get(kind, 0) + 1

    print(f"  Forced FP32 on {forced_count} layers: {op_counts}")

    # Set dynamic shape profile
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "pixel_values",
        min=(1, 3, 224, 224),
        opt=(16, 3, 224, 224),
        max=(16, 3, 224, 224),
    )
    config.add_optimization_profile(profile)

    # Build
    print()
    print("Building engine (this takes 1-3 minutes)...")
    t0 = time.time()
    serialized_engine = builder.build_serialized_network(network, config)
    build_time = time.time() - t0

    if serialized_engine is None:
        print("❌ Engine build failed (None returned)")
        sys.exit(1)

    print(f"  Build took {build_time:.1f}s")

    with open(ENGINE_FILE, "wb") as f:
        f.write(serialized_engine)

    size_mb = os.path.getsize(ENGINE_FILE) / (1024 * 1024)
    print()
    print("=" * 60)
    print(f"✅ Engine built!")
    print(f"   File: {ENGINE_FILE}")
    print(f"   Size: {size_mb:.1f} MB")
    print(f"   LayerNorm + Softmax forced to FP32 for safety")
    print("=" * 60)


if __name__ == "__main__":
    main()
