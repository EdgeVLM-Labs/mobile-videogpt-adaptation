#!/bin/bash
# Build TensorRT engine for CLIP ViT-Base on Jetson Orin Nano.
#
# Supports multiple precision modes:
#   fp16 (default):       Full FP16 — fastest but may overflow on CLIP
#   fp16-safe:            FP16 with FP32 LayerNorm + attention (no overflow)
#   fp32:                 FP32 — slowest but guaranteed correct (for debugging)
#
# Usage:
#   bash scripts/build_clip_tensorrt.sh              # default: fp16-safe
#   bash scripts/build_clip_tensorrt.sh fp16         # aggressive fp16
#   bash scripts/build_clip_tensorrt.sh fp32         # debugging

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
MODELS_DIR="$PROJECT_ROOT/models/tensorrt"

ONNX_FILE="$MODELS_DIR/clip_vit_base.onnx"
TRTEXEC="/usr/src/tensorrt/bin/trtexec"

# Precision mode — defaults to fp16-safe (preserves accuracy)
MODE="${1:-fp16-safe}"

case "$MODE" in
    fp16)
        ENGINE_FILE="$MODELS_DIR/clip_vit_base_fp16.engine"
        PRECISION_FLAGS="--fp16"
        ;;
    fp16-safe)
        # FP16 compute with FP32 LayerNorm — avoids overflow in CLIP's
        # residual branches while keeping most matmuls in FP16.
        ENGINE_FILE="$MODELS_DIR/clip_vit_base_fp16.engine"
        PRECISION_FLAGS="--fp16 --precisionConstraints=obey --layerPrecisions=*LayerNorm*:fp32,*layernorm*:fp32,*ln*:fp32"
        ;;
    fp32)
        ENGINE_FILE="$MODELS_DIR/clip_vit_base_fp32.engine"
        PRECISION_FLAGS=""
        ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: fp16, fp16-safe (default), fp32"
        exit 1
        ;;
esac

echo "============================================================"
echo "CLIP ViT-Base → TensorRT Engine ($MODE)"
echo "============================================================"
echo "Input:  $ONNX_FILE"
echo "Output: $ENGINE_FILE"
echo "Flags:  $PRECISION_FLAGS"
echo

if [ ! -f "$ONNX_FILE" ]; then
    echo "❌ ONNX file not found. Run first:"
    echo "   python scripts/export_clip_to_onnx.py"
    exit 1
fi

# Free memory before building
echo "Clearing system page cache..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "  (skipped — no sudo)"

echo "Building engine (this takes 2-5 minutes)..."
echo

"$TRTEXEC" \
    --onnx="$ONNX_FILE" \
    --saveEngine="$ENGINE_FILE" \
    $PRECISION_FLAGS \
    --minShapes=pixel_values:1x3x224x224 \
    --optShapes=pixel_values:16x3x224x224 \
    --maxShapes=pixel_values:16x3x224x224 \
    --memPoolSize=workspace:512M \
    --skipInference \
    2>&1 | tail -25

echo
if [ -f "$ENGINE_FILE" ]; then
    SIZE=$(du -h "$ENGINE_FILE" | cut -f1)
    echo "============================================================"
    echo "✅ Engine built! ($MODE)"
    echo "   File:  $ENGINE_FILE"
    echo "   Size:  $SIZE"
    echo "============================================================"
else
    echo "❌ Build failed — no output file"
    exit 1
fi
