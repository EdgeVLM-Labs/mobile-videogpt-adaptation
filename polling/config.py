"""Polling inference configuration."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PollingConfig:
    """Configuration for polling-based video stream inference."""

    base_model_path: str = "Amshaker/Mobile-VideoGPT-0.5B"
    lora_weights_path: str = "EdgeVLM-Labs/mobile-videogpt-finetune-2000"

    polling_interval: float = 3.0  # seconds between inference calls
    max_polling_duration: float = 300.0

    num_frames: int = 16
    num_context_images: int = 16
    chunk_size: int = 8
    fps: int = 1
    image_resolution: int = 224

    frame_buffer_size: int = 64
    frame_overlap: float = 0.5

    prompt: str = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
    max_new_tokens: int = 512
    temperature: float = 0.0  # 0 = greedy decoding
    do_sample: bool = False
    num_beams: int = 1
    use_cache: bool = True

    load_4bit: bool = False  # 4-bit quantization via bitsandbytes (NF4)
    load_8bit: bool = False
    num_select_k_frames_in_chunk: int = 4
    topk: bool = True

    device: str = "cuda"
    torch_dtype: str = "float16"

    log_level: str = "INFO"
    log_dir: str = "logs/polling"
    save_metrics: bool = True

    output_dir: str = "results/polling"
    save_responses: bool = True

    enable_confidence_scoring: bool = False


    def __post_init__(self):
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    @classmethod
    def from_env(cls) -> "PollingConfig":
        return cls(
            base_model_path=os.getenv("BASE_MODEL_PATH", cls.base_model_path),
            lora_weights_path=os.getenv("LORA_WEIGHTS_PATH", cls.lora_weights_path),
            polling_interval=float(os.getenv("POLLING_INTERVAL", cls.polling_interval)),
            max_polling_duration=float(os.getenv("MAX_POLLING_DURATION", cls.max_polling_duration)),
            num_frames=int(os.getenv("NUM_FRAMES", cls.num_frames)),
            device=os.getenv("DEVICE", cls.device),
            log_level=os.getenv("LOG_LEVEL", cls.log_level),
        )
