"""
Streaming inference module for Mobile-VideoGPT.

This module provides real-time video processing and feedback generation
for exercise form correction using a streaming adaptation of Mobile-VideoGPT.
"""

from .buffer import VideoFrameBuffer
from .context import TemporalContextManager, KVCacheManager
from .predictor import ActionTokenPredictor
from .engine import StreamingMobileVideoGPT
from .utils import load_config, setup_logging

__version__ = "0.1.0"
__all__ = [
    "VideoFrameBuffer",
    "TemporalContextManager",
    "KVCacheManager",
    "ActionTokenPredictor",
    "StreamingMobileVideoGPT",
    "load_config",
    "setup_logging",
]
