#!/usr/bin/env bash
# Launch the Coach web app with the SAME GPU fast-path the Gradio app uses.
#
# Why this script exists: launching `python -m polling.webapp.server` directly
# (no env) makes CLIP fall back to ONNX-Runtime on CPU (~3.8s/poll) instead of
# the TensorRT engine on GPU (~0.2s). That alone pushes TTFT from ~2.5s to ~6s.
# These two env vars are the documented known-good config (see
# docs/jetson_inference/JETSON_OPTIMIZATION_JOURNEY.md).
#
#   USE_TRT_CLIP=1  -> TensorRT CLIP on GPU  (needs models/tensorrt/clip_vit_base_*.engine)
#   USE_FULL_GPU=1  -> keep the whole model (incl. lm_head) on GPU, no CPU spill
#
# Run from the repo root:  bash polling/webapp/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."   # repo root (mobile-videogpt-adaptation/)

ENGINE_GLOB=(models/tensorrt/clip_vit_base_fp16.engine models/tensorrt/clip_vit_base_fp32.engine)
if ! ls "${ENGINE_GLOB[@]}" >/dev/null 2>&1; then
  echo "WARNING: no TensorRT CLIP engine found under models/tensorrt/." >&2
  echo "         CLIP will fall back to CPU (~3.8s/poll). Build it once with:" >&2
  echo "             bash scripts/build_clip_tensorrt.sh" >&2
fi

export USE_FULL_GPU=1
export USE_TRT_CLIP=1
exec python -m polling.webapp.server "$@"
