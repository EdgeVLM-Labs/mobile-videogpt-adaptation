"""
Configuration for polling-based streaming inference.
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PollingConfig:
    """Configuration for polling-based video stream inference."""

    # Model configuration
    base_model_path: str = "Amshaker/Mobile-VideoGPT-0.5B"
    lora_weights_path: str = "EdgeVLM-Labs/mobile-videogpt-finetune-v2-mixed"

    # Polling configuration
    polling_interval: float = 3.0  # Seconds between inference calls
    max_polling_duration: float = 300.0  # Maximum total polling duration (5 minutes default)

    # Video processing
    # Model constraints (cannot be changed without retraining):
    #   1. num_chunks × num_select_k_frames_in_chunk == 8   (VideoMamba t=8)
    #   2. (num_select_k_frames_in_chunk × 49) must be a perfect square
    #      (mm_projector reshapes to 2D spatial grid — sqrt(num_tokens))
    # Only valid combo: num_chunks=2, k=4 → 4×49=196=14² ✓
    # So num_frames must stay at 16 (2 chunks × 8 frames/chunk from CHUNK_SIZE).
    num_frames: int = 16
    num_context_images: int = 16
    chunk_size: int = 8  # VideoMamba chunk size
    # Capture rate — controls PREVIEW smoothness only. Decoupled from the model:
    # frames are captured at `fps` for a smooth live preview, while each inference
    # uniformly samples num_frames across the last `inference_window_seconds` of
    # buffer. So raising fps smooths the preview without changing what the model sees.
    # 10 keeps the preview smooth while cutting capture/encode CPU vs 15. Must stay
    # >= ceil(num_frames / inference_window_seconds) (=4) so the 4s window always has
    # >=16 real frames to sample — 10 leaves a wide margin (40 frames), no padding,
    # so the model input is unchanged.
    fps: int = 10
    # Temporal window the model's num_frames cover. 16 frames over 4s = the 4 fps
    # density used during fine-tuning — keep at 4.0 to match training.
    inference_window_seconds: float = 4.0
    image_resolution: int = 224  # Frame resolution

    # Frame buffer configuration
    # Must hold >= inference_window_seconds * fps frames. At 10 fps over a 4s window
    # that is 40; 96 (~9.6s) leaves margin and still evicts old frames quickly so a
    # previous exercise does not contaminate the current poll.
    frame_buffer_size: int = 96
    frame_overlap: float = 0.5  # Overlap ratio between polling windows (0.0 - 1.0)

    # Inference configuration
    prompt: str = "Watch the video. Identify the exercise and give short feedback on the form."
    max_new_tokens: int = 64  # Reduced from 128 — responses are 30-40 tokens
    do_sample: bool = False
    num_beams: int = 1
    use_cache: bool = True

    # Model loading options
    load_4bit: bool = False  # 4-bit incompatible with custom model architecture
    load_8bit: bool = False
    # Fixed: num_chunks × num_select_k_frames_in_chunk must = 8, and k×49 must be
    # a perfect square. Only valid value: k=4 (with num_chunks=2).
    num_select_k_frames_in_chunk: int = 4
    topk: bool = True

    # Device configuration
    device: str = "cuda"
    torch_dtype: str = "float16"

    # Logging configuration
    log_level: str = "INFO"
    log_dir: str = "logs/polling"
    save_metrics: bool = True

    # Output configuration
    output_dir: str = "results/polling"
    save_responses: bool = True

    # Confidence scoring configuration
    enable_confidence_scoring: bool = False  # Toggle confidence-based filtering

    # Motion gate (Tier 1): when enabled, skip the VLM on a static scene
    # (empty room / person standing idle) so the model doesn't emit feedback
    # when nobody is exercising. Purely additive and OFF by default — the
    # existing always-on polling path is unchanged unless MOTION_GATE=1.
    #
    # `motion_threshold` is the mean absolute inter-frame pixel delta on a
    # 0-255 grayscale scale (computed on 64x64 thumbnails); scenes below it
    # are treated as "no activity". Tune per camera/lighting (typical 1.5-4.0).
    # Both are env-driven so a fresh launch can opt in without code changes:
    #   MOTION_GATE=1 MOTION_THRESHOLD=2.5 python polling/gradio_app.py
    enable_motion_gate: bool = field(
        default_factory=lambda: os.getenv("MOTION_GATE", "0") == "1"
    )
    motion_threshold: float = field(
        default_factory=lambda: float(os.getenv("MOTION_THRESHOLD", "2.5"))
    )
    # Hysteresis / debounce for the motion gate. A single low-motion poll (the
    # slow bottom of a rep, a brief pause between reps) should NOT flip the UI to
    # "waiting" mid-exercise. We only declare the scene idle after this many
    # *consecutive* below-threshold polls; any active poll resets the streak.
    # 2 → ~2 polls (~6 s at a 3 s interval) of genuine stillness before going
    # quiet. Raise it if it still flips mid-exercise; lower it (1) to go quiet
    # faster on a truly empty stage.
    motion_idle_polls: int = field(
        default_factory=lambda: int(os.getenv("MOTION_IDLE_POLLS", "2"))
    )


    def __post_init__(self):
        """Create necessary directories. num_frames is fixed at 16 due to model constraints."""
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        # Force matched num_context_images (pipeline requires equal counts)
        self.num_context_images = self.num_frames

    @classmethod
    def from_env(cls) -> "PollingConfig":
        """Create config from environment variables."""
        return cls(
            base_model_path=os.getenv("BASE_MODEL_PATH", cls.base_model_path),
            lora_weights_path=os.getenv("LORA_WEIGHTS_PATH", cls.lora_weights_path),
            polling_interval=float(os.getenv("POLLING_INTERVAL", cls.polling_interval)),
            max_polling_duration=float(os.getenv("MAX_POLLING_DURATION", cls.max_polling_duration)),
            num_frames=int(os.getenv("NUM_FRAMES", cls.num_frames)),
            device=os.getenv("DEVICE", cls.device),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )
