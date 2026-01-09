"""
Utility functions for streaming inference.

Provides helper functions for configuration loading, logging setup,
and common operations.
"""

import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import numpy as np
from PIL import Image


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is malformed
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True,
) -> logging.Logger:
    """
    Setup logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path to save logs
        console: Whether to log to console

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger("streaming")
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers
    logger.handlers = []

    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def preprocess_frame(
    frame: np.ndarray,
    target_size: tuple = (224, 224),
    normalize: bool = True,
) -> np.ndarray:
    """
    Preprocess a single video frame.

    Args:
        frame: Input frame (H, W, C) in RGB format
        target_size: Target (height, width) for resizing
        normalize: Whether to normalize to [0, 1] range

    Returns:
        Preprocessed frame
    """
    # Resize
    if frame.shape[:2] != target_size:
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((target_size[1], target_size[0]), Image.BILINEAR)
        frame = np.array(pil_image)

    # Normalize
    if normalize and frame.dtype == np.uint8:
        frame = frame.astype(np.float32) / 255.0

    return frame


def frames_to_tensor(
    frames: list,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Convert list of frames to tensor batch.

    Args:
        frames: List of numpy arrays (H, W, C)
        device: Target device
        dtype: Target dtype

    Returns:
        Tensor of shape (B, C, H, W)
    """
    # Stack frames
    frames_array = np.stack(frames, axis=0)  # (B, H, W, C)

    # Convert to tensor and transpose to (B, C, H, W)
    tensor = torch.from_numpy(frames_array).permute(0, 3, 1, 2)

    # Move to device and convert dtype
    tensor = tensor.to(device=device, dtype=dtype)

    return tensor


def add_special_tokens(tokenizer, model, tokens: Dict[str, str]) -> Dict[str, int]:
    """
    Add special action tokens to tokenizer and model.

    Args:
        tokenizer: HuggingFace tokenizer
        model: Model to resize embeddings
        tokens: Dict mapping token names to token strings

    Returns:
        Dict mapping token strings to token IDs
    """
    # Get token strings
    special_tokens = list(tokens.values())

    # Add to tokenizer
    num_added = tokenizer.add_tokens(special_tokens, special_tokens=True)

    if num_added > 0:
        # Resize model embeddings
        model.resize_token_embeddings(len(tokenizer))

        # Initialize new embeddings (copy from EOS token)
        with torch.no_grad():
            eos_embedding = model.get_input_embeddings().weight[tokenizer.eos_token_id]
            for token_str in special_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                model.get_input_embeddings().weight[token_id] = eos_embedding.clone()

        logging.info(f"Added {num_added} special tokens: {special_tokens}")
    else:
        logging.info("Special tokens already in vocabulary")

    # Return token ID mapping
    token_ids = {
        token_str: tokenizer.convert_tokens_to_ids(token_str)
        for token_str in special_tokens
    }

    return token_ids


def format_feedback(
    action: str,
    text: Optional[str],
    confidence: float,
    timestamp: float,
) -> str:
    """
    Format feedback output for display.

    Args:
        action: Action token (<next>, <feedback>, <correct>)
        text: Generated feedback text (None for <next>)
        confidence: Prediction confidence
        timestamp: Time since start

    Returns:
        Formatted string
    """
    if action == "<next>":
        return f"[{timestamp:.1f}s] Observing... (confidence: {confidence:.2f})"
    elif action == "<feedback>":
        return f"[{timestamp:.1f}s] FEEDBACK: {text} (confidence: {confidence:.2f})"
    elif action == "<correct>":
        return f"[{timestamp:.1f}s] ✓ {text} (confidence: {confidence:.2f})"
    else:
        return f"[{timestamp:.1f}s] {action}: {text}"


class PerformanceMonitor:
    """Monitor and report performance metrics."""

    def __init__(self, window_size: int = 30):
        """
        Initialize performance monitor.

        Args:
            window_size: Number of samples for moving average
        """
        self.window_size = window_size
        self.frame_times = []
        self.chunk_times = []
        self.inference_times = []
        self.start_time = None

    def start(self):
        """Start monitoring."""
        import time
        self.start_time = time.time()

    def record_frame_time(self, duration: float):
        """Record frame processing time."""
        self.frame_times.append(duration)
        if len(self.frame_times) > self.window_size:
            self.frame_times.pop(0)

    def record_chunk_time(self, duration: float):
        """Record chunk processing time."""
        self.chunk_times.append(duration)
        if len(self.chunk_times) > self.window_size:
            self.chunk_times.pop(0)

    def record_inference_time(self, duration: float):
        """Record inference time."""
        self.inference_times.append(duration)
        if len(self.inference_times) > self.window_size:
            self.inference_times.pop(0)

    def get_fps(self) -> float:
        """Get current FPS."""
        if not self.frame_times:
            return 0.0
        avg_time = np.mean(self.frame_times)
        return 1.0 / avg_time if avg_time > 0 else 0.0

    def get_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        import time
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "elapsed_time": elapsed,
            "avg_frame_time": np.mean(self.frame_times) if self.frame_times else 0.0,
            "avg_chunk_time": np.mean(self.chunk_times) if self.chunk_times else 0.0,
            "avg_inference_time": np.mean(self.inference_times) if self.inference_times else 0.0,
            "current_fps": self.get_fps(),
            "frames_processed": len(self.frame_times),
        }

    def print_stats(self):
        """Print performance statistics."""
        stats = self.get_stats()
        print(f"\n{'='*50}")
        print(f"Performance Statistics:")
        print(f"{'='*50}")
        print(f"Elapsed Time: {stats['elapsed_time']:.2f}s")
        print(f"Current FPS: {stats['current_fps']:.1f}")
        print(f"Avg Frame Time: {stats['avg_frame_time']*1000:.1f}ms")
        print(f"Avg Chunk Time: {stats['avg_chunk_time']*1000:.1f}ms")
        print(f"Avg Inference Time: {stats['avg_inference_time']*1000:.1f}ms")
        print(f"Frames Processed: {stats['frames_processed']}")
        print(f"{'='*50}\n")
