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
    lora_weights_path: str = "EdgeVLM-Labs/mobile-videogpt-finetune-2000"

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
    # Capture rate. With num_frames=16 and tail-sampling, each poll covers
    # (num_frames / fps) seconds of activity. fps=4 gives a 4-second window
    # per inference — roughly one exercise rep — and prevents stale frames
    # from a previous exercise contaminating the current poll.
    fps: int = 4
    image_resolution: int = 224  # Frame resolution

    # Frame buffer configuration
    # Sized to ~2× the active window so old frames evict quickly when the
    # patient transitions between exercises. With fps=4 and num_frames=16,
    # buffer_size=32 holds ~8 seconds of recent history.
    frame_buffer_size: int = 32
    frame_overlap: float = 0.5  # Overlap ratio between polling windows (0.0 - 1.0)

    # Inference configuration
    prompt: str = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
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
