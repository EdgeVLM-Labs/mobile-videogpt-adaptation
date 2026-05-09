"""
End-to-end webcam pipeline diagnostic.

Captures frames from the same code path the inference engine uses,
saves them to disk so you can verify they look correct BEFORE feeding
them through the model.

What this checks:
  1. Camera opens and produces frames
  2. Frames are RGB (not BGR or weird color cast)
  3. Frame buffer fills correctly over time
  4. The 16 frames sampled for inference look properly exposed
  5. Per-frame shape and dtype match what the model expects

Outputs (in ~/Documents/mobile-videogpt-adaptation/diagnostic_frames/):
  - raw_001.jpg ... raw_005.jpg          : 5 frames straight from camera
  - sampled_for_video_NN.jpg             : 16 frames sampled for VideoMamba
  - sampled_for_context_NN.jpg           : 16 frames sampled for CLIP
  - report.txt                           : shape/dtype/min/max per stage

Usage:
  conda activate mvgpt
  python scripts/diagnose_webcam_pipeline.py [camera_index]
  # default camera_index = 0 (i.e. /dev/video0)
"""
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
OUT_DIR = PROJECT_ROOT / "diagnostic_frames"
OUT_DIR.mkdir(exist_ok=True)

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cv2

from polling.stream_handler import VideoStreamHandler


def save_frame(frame, path):
    """Save an RGB numpy frame as JPG (cv2 expects BGR)."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def main():
    camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print("=" * 60)
    print("Webcam Pipeline Diagnostic")
    print("=" * 60)
    print(f"Camera index:  /dev/video{camera_index}")
    print(f"Output dir:    {OUT_DIR}")
    print()

    report_lines = []
    def log(msg):
        print(msg)
        report_lines.append(msg)

    # --- Step 1: Test that camera opens ---
    log("[1/4] Testing camera availability...")
    if not VideoStreamHandler.test_webcam_availability(camera_index):
        log(f"  FAIL: cannot open /dev/video{camera_index}")
        return
    log(f"  PASS: camera opens")

    # --- Step 2: Start streaming (uses same code as Gradio) ---
    log("")
    log("[2/4] Starting stream capture (same as Gradio)...")
    handler = VideoStreamHandler(
        buffer_size=64,
        num_frames=16,
        fps=1,
        image_resolution=224,
    )
    handler.start_stream_capture(str(camera_index))

    # Wait for buffer to fill
    log("  Waiting 5 seconds for buffer to fill (1 fps × 5 = ~5 frames)...")
    time.sleep(5)
    buf_size = len(handler.frame_buffer)
    log(f"  Buffer size after 5s:  {buf_size} frames")

    if buf_size == 0:
        log("  FAIL: no frames captured. Camera may be in use or broken.")
        handler.close()
        return

    # --- Step 3: Save 5 raw frames straight from buffer ---
    log("")
    log("[3/4] Saving 5 raw frames from buffer to verify quality...")
    raw_samples = list(handler.frame_buffer)[-5:]
    for i, fd in enumerate(raw_samples, 1):
        path = OUT_DIR / f"raw_{i:03d}.jpg"
        save_frame(fd.frame, path)
        log(f"  raw_{i:03d}.jpg  shape={fd.frame.shape} dtype={fd.frame.dtype} "
            f"min={fd.frame.min()} max={fd.frame.max()} mean={fd.frame.mean():.1f}")

    # --- Step 4: Run the SAME pipeline get_frames_for_inference uses ---
    log("")
    log("[4/4] Running get_frames_for_inference (same as Gradio polling)...")

    # Wait a bit longer to have at least 16 frames
    log("  Waiting 12 more seconds to fill buffer to 16+ frames...")
    time.sleep(12)
    log(f"  Buffer size now:  {len(handler.frame_buffer)} frames")

    # Use the model's actual processors
    log("  Loading CLIP image processor + VideoMamba processor...")
    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # VideoMamba uses a callable-style preprocessor — minimal stub for diagnostic
    class _DummyVideoProcessor:
        """Mimics video_processor.preprocess() — returns list of (3,224,224) tensors."""
        def preprocess(self, frames_list):
            import torch
            from PIL import Image
            tensors = []
            for f in frames_list:
                # f is RGB numpy (H,W,3)
                pil = Image.fromarray(f).resize((224, 224))
                arr = np.array(pil, dtype=np.float32) / 255.0
                # CHW + normalize like ImageNet (rough)
                tensor = torch.from_numpy(arr).permute(2, 0, 1)
                tensors.append(tensor)
            return {"pixel_values": tensors}

    video_processor = _DummyVideoProcessor()

    video_frames, context_frames, slice_len = handler.get_frames_for_inference(
        image_processor=image_processor,
        video_processor=video_processor,
        num_video_frames=16,
        num_context_images=16,
        polling_interval=3.0,
    )

    log(f"  slice_len:         {slice_len}")
    log(f"  video_frames len:  {len(video_frames)}")
    log(f"  context_frames len:{len(context_frames)}")

    if len(video_frames) > 0:
        v0 = video_frames[0]
        log(f"  video_frames[0]:   shape={tuple(v0.shape)} dtype={v0.dtype} "
            f"range=[{v0.min().item():.3f}, {v0.max().item():.3f}]")
    if len(context_frames) > 0:
        c0 = context_frames[0]
        log(f"  context_frames[0]: shape={tuple(c0.shape)} dtype={c0.dtype} "
            f"range=[{c0.min().item():.3f}, {c0.max().item():.3f}]")

    # Save the model-input frames as visualizable JPGs (de-normalize roughly)
    log("")
    log("  Saving model-input frames to disk (de-normalized for visualization)...")
    import torch

    def tensor_to_jpg(tensor, path):
        """Convert (3,224,224) tensor in CLIP-normalized space back to viewable image."""
        # CLIP normalization: mean=[0.481,0.458,0.408], std=[0.269,0.261,0.276]
        mean = torch.tensor([0.481, 0.458, 0.408]).view(3, 1, 1)
        std = torch.tensor([0.269, 0.261, 0.276]).view(3, 1, 1)
        denorm = tensor * std + mean
        denorm = (denorm.clamp(0, 1) * 255).byte()
        # CHW -> HWC
        np_img = denorm.permute(1, 2, 0).numpy()
        bgr = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(path), bgr)

    for i, t in enumerate(video_frames[:16]):
        tensor_to_jpg(t, OUT_DIR / f"sampled_for_video_{i:02d}.jpg")
    for i, t in enumerate(context_frames[:16]):
        tensor_to_jpg(t, OUT_DIR / f"sampled_for_context_{i:02d}.jpg")

    log(f"  Wrote {min(16, len(video_frames))} video frames + "
        f"{min(16, len(context_frames))} context frames to {OUT_DIR}/")

    # Cleanup
    handler.close()

    # Write report
    report_path = OUT_DIR / "report.txt"
    report_path.write_text("\n".join(report_lines))

    print()
    print("=" * 60)
    print("✅ Diagnostic complete!")
    print()
    print(f"Inspect the images at:  {OUT_DIR}")
    print(f"Report saved at:         {report_path}")
    print()
    print("To copy to your laptop:")
    print(f"  scp -r edgevlm@<jetson-ip>:{OUT_DIR} ~/Downloads/")
    print()
    print("What to check:")
    print("  - raw_*.jpg should look like normal photos of what the camera sees")
    print("  - sampled_for_video_*.jpg should look like the same scene at 224x224")
    print("  - sampled_for_context_*.jpg should look like the same scene at 224x224")
    print("  - If they're all dark, washed out, or blank → camera issue")
    print("  - If they look correct → frames are flowing to the model properly")
    print("=" * 60)


if __name__ == "__main__":
    main()
