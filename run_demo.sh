#!/bin/bash
#
# run_demo.sh — one-command launcher for the Mobile-VideoGPT Jetson demo.
#
# Runs a preflight check (conda env, TensorRT engine, webcam, power mode, free
# memory), then launches the Gradio inference server in the fast configuration
# (USE_FULL_GPU=1 USE_TRT_CLIP=1). Open the printed URL in a laptop browser.
#
# Usage:
#   ./run_demo.sh                 # preflight + launch (fast mode)
#   ./run_demo.sh --check         # preflight only, don't launch
#   ./run_demo.sh --motion-gate   # also enable the Tier-1 motion gate
#   ./run_demo.sh --threshold 0.7 # motion gate + custom threshold (implies --motion-gate)
#   ./run_demo.sh --safe          # CPU-CLIP fallback (no TRT/full-GPU) — bulletproof, slower
#   ./run_demo.sh --share         # also publish a public *.gradio.live URL (any device/network)
#   ./run_demo.sh --help
#
# See docs/jetson_inference/HEADLESS_DEMO_SETUP.md for the full runbook.

set -u

# ── Config ──────────────────────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-mvgpt}"
ENGINE="models/tensorrt/clip_vit_base_fp32.engine"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Flags ───────────────────────────────────────────────────────────────────
CHECK_ONLY=0
MOTION_GATE=0
MOTION_THRESHOLD=""
SAFE_MODE=0
SHARE=0

while [ $# -gt 0 ]; do
  case "$1" in
    --check)        CHECK_ONLY=1 ;;
    --motion-gate)  MOTION_GATE=1 ;;
    --threshold)    MOTION_GATE=1; MOTION_THRESHOLD="${2:-}"; shift ;;
    --safe)         SAFE_MODE=1 ;;
    --share)        SHARE=1 ;;
    --help|-h)
      sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *) echo "Unknown option: $1 (try --help)"; exit 1 ;;
  esac
  shift
done

green() { printf '\033[0;32m%s\033[0m\n' "$1"; }
red()   { printf '\033[0;31m%s\033[0m\n' "$1"; }
yellow(){ printf '\033[0;33m%s\033[0m\n' "$1"; }

FAIL=0

echo "═══════════════════════════════════════════════════════════"
echo "  Mobile-VideoGPT Jetson demo — preflight"
echo "═══════════════════════════════════════════════════════════"

# 1) conda env exists
if conda env list 2>/dev/null | grep -qE "^${CONDA_ENV}\s|/${CONDA_ENV}\$|/${CONDA_ENV}\s"; then
  green "✓ conda env '${CONDA_ENV}' found"
else
  red "✗ conda env '${CONDA_ENV}' not found — run setup_jetson.sh"
  FAIL=1
fi

# 2) TensorRT engine present (only needed in fast mode)
if [ "$SAFE_MODE" -eq 1 ]; then
  yellow "• safe mode: skipping TensorRT engine (CLIP runs on CPU via ONNX Runtime)"
elif [ -f "$ENGINE" ]; then
  green "✓ TensorRT CLIP engine present ($(du -h "$ENGINE" | cut -f1))"
else
  red "✗ $ENGINE missing — build it or use --safe"
  FAIL=1
fi

# 3) webcam
if ls /dev/video0 >/dev/null 2>&1; then
  green "✓ webcam present (/dev/video0)"
else
  yellow "• no /dev/video0 — plug in the USB webcam (or use Video File / Browser Webcam in the UI)"
fi

# 4) power mode (best effort — nvpmodel may need sudo)
PM="$(nvpmodel -q 2>/dev/null | grep -i 'power mode' | head -1)"
if echo "$PM" | grep -qi 'MAXN'; then
  green "✓ power mode: ${PM}"
elif [ -n "$PM" ]; then
  yellow "• power mode: ${PM} — for best speed run: sudo nvpmodel -m 2 && sudo jetson_clocks"
else
  yellow "• could not read power mode (nvpmodel needs sudo) — recommend MAXN_SUPER (mode 2)"
fi

# 5) free memory
FREE_GB="$(free -g | awk '/^Mem:/{print $7}')"
if [ "${FREE_GB:-0}" -ge 5 ]; then
  green "✓ available memory: ${FREE_GB} GB"
else
  AVAIL="$(free -h | awk '/^Mem:/{print $7}')"
  yellow "• available memory: ${AVAIL} — fast mode wants ~5-6 GB. Consider headless mode + drop caches."
fi

echo "───────────────────────────────────────────────────────────"
if [ "$FAIL" -ne 0 ]; then
  red "Preflight FAILED — fix the ✗ items above before the demo."
  exit 1
fi
green "Preflight OK."

if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "(--check: not launching)"
  exit 0
fi

# ── Free page cache (best effort; needs sudo) ────────────────────────────────
echo "Dropping page caches (may prompt for sudo)..."
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null \
  && green "✓ caches dropped" \
  || yellow "• skipped drop_caches (no sudo) — fine, just a bit less free RAM"

# ── Activate conda ───────────────────────────────────────────────────────────
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# ── Build env + launch ───────────────────────────────────────────────────────
LAUNCH_ENV=()
if [ "$SAFE_MODE" -eq 1 ]; then
  yellow "Launching in SAFE mode (CPU CLIP, ~10s/poll, bulletproof)..."
else
  LAUNCH_ENV+=("USE_FULL_GPU=1" "USE_TRT_CLIP=1")
fi
if [ "$MOTION_GATE" -eq 1 ]; then
  LAUNCH_ENV+=("MOTION_GATE=1")
  [ -n "$MOTION_THRESHOLD" ] && LAUNCH_ENV+=("MOTION_THRESHOLD=${MOTION_THRESHOLD}")
  green "Motion gate ENABLED${MOTION_THRESHOLD:+ (threshold ${MOTION_THRESHOLD})}"
fi
if [ "$SHARE" -eq 1 ]; then
  LAUNCH_ENV+=("GRADIO_SHARE=1")
  green "Public share link ENABLED — a *.gradio.live URL will print below (needs internet)"
fi

echo "═══════════════════════════════════════════════════════════"
echo "  Launching: ${LAUNCH_ENV[*]} python polling/gradio_app.py"
echo "  Model loads once (~40s). Then open the printed URL in a browser."
echo "═══════════════════════════════════════════════════════════"

exec env "${LAUNCH_ENV[@]}" python polling/gradio_app.py
