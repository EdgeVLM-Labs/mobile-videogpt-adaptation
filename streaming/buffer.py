"""
Video frame buffer with sliding window for chunk extraction.

This module implements a circular buffer that accumulates video frames
and extracts fixed-size chunks with configurable overlap for processing
by the VideoMamba encoder.
"""

from collections import deque
from typing import List, Optional, Tuple
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class VideoFrameBuffer:
    """
    Circular buffer for video frames with sliding window chunk extraction.

    The buffer maintains a queue of frames and extracts overlapping chunks
    when enough frames are available. This enables pseudo-streaming by
    processing video in small overlapping windows.

    Attributes:
        chunk_size: Number of frames per chunk (fixed by VideoMamba = 8)
        overlap: Number of frames to overlap between consecutive chunks
        stride: Number of frames to advance after each chunk (chunk_size - overlap)
        max_buffer_size: Maximum frames to keep in buffer
        buffer: Deque storing frame tensors
        frame_count: Total number of frames processed
    """

    def __init__(
        self,
        chunk_size: int = 8,
        overlap: int = 4,
        max_buffer_size: Optional[int] = None,
    ):
        """
        Initialize the video frame buffer.

        Args:
            chunk_size: Number of frames per chunk (must be 8 for VideoMamba)
            overlap: Number of frames to overlap between chunks (0 to chunk_size-1)
            max_buffer_size: Maximum frames to store (None = chunk_size + overlap)

        Raises:
            ValueError: If chunk_size is not 8 or overlap >= chunk_size
        """
        if chunk_size != 8:
            logger.warning(f"VideoMamba requires chunk_size=8, got {chunk_size}. "
                          f"This may cause encoding errors.")

        if overlap >= chunk_size:
            raise ValueError(f"Overlap ({overlap}) must be less than chunk_size ({chunk_size})")

        if overlap < 0:
            raise ValueError(f"Overlap must be non-negative, got {overlap}")

        self.chunk_size = chunk_size
        self.overlap = overlap
        self.stride = chunk_size - overlap

        # Set buffer size to hold one chunk plus overlap
        if max_buffer_size is None:
            max_buffer_size = chunk_size + overlap
        self.max_buffer_size = max_buffer_size

        # Initialize circular buffer
        self.buffer: deque = deque(maxlen=max_buffer_size)
        self.context_buffer: deque = deque(maxlen=max_buffer_size)  # For context images

        # Tracking
        self.frame_count = 0
        self.chunk_count = 0

        logger.info(f"Initialized VideoFrameBuffer: chunk_size={chunk_size}, "
                   f"overlap={overlap}, stride={stride}, max_size={max_buffer_size}")

    def add_frame(self, frame: np.ndarray) -> bool:
        """
        Add a single frame to the buffer.

        Args:
            frame: Frame as numpy array (H, W, C) in RGB format

        Returns:
            True if a chunk is ready to be extracted, False otherwise
        """
        if frame.ndim != 3:
            raise ValueError(f"Frame must be 3D (H, W, C), got shape {frame.shape}")

        # Add to buffer (automatically removes oldest if at max capacity)
        self.buffer.append(frame)
        self.context_buffer.append(frame.copy())  # Separate buffer for context
        self.frame_count += 1

        # Check if we have enough frames for a chunk
        chunk_ready = len(self.buffer) >= self.chunk_size

        if chunk_ready and self.frame_count % self.stride == 0:
            logger.debug(f"Chunk ready: {len(self.buffer)} frames in buffer "
                        f"(frame {self.frame_count})")
            return True

        return False

    def get_chunk(self) -> Optional[Tuple[List[np.ndarray], List[np.ndarray]]]:
        """
        Extract a chunk of frames from the buffer.

        Returns:
            Tuple of (video_frames, context_frames) where:
                - video_frames: List of chunk_size frames for VideoMamba
                - context_frames: List of frames for CLIP encoder
            Returns None if not enough frames available
        """
        if len(self.buffer) < self.chunk_size:
            return None

        # Extract video chunk (first chunk_size frames)
        video_chunk = list(self.buffer)[:self.chunk_size]

        # Extract context frames (sample from buffer)
        context_frames = self._sample_context_frames()

        # Remove processed frames (keeping overlap)
        for _ in range(self.stride):
            if len(self.buffer) > self.overlap:
                self.buffer.popleft()

        self.chunk_count += 1
        logger.debug(f"Extracted chunk {self.chunk_count}: {len(video_chunk)} video frames, "
                    f"{len(context_frames)} context frames")

        return video_chunk, context_frames

    def _sample_context_frames(self, num_context: int = 16) -> List[np.ndarray]:
        """
        Sample context frames uniformly from the context buffer.

        Args:
            num_context: Number of context images to sample

        Returns:
            List of sampled frames
        """
        buffer_list = list(self.context_buffer)
        buffer_len = len(buffer_list)

        if buffer_len == 0:
            return []

        if buffer_len <= num_context:
            # Return all frames if we have fewer than needed
            return buffer_list

        # Uniform sampling
        indices = np.linspace(0, buffer_len - 1, num_context, dtype=int)
        context_frames = [buffer_list[i] for i in indices]

        return context_frames

    def reset(self):
        """Clear the buffer and reset counters."""
        self.buffer.clear()
        self.context_buffer.clear()
        self.frame_count = 0
        self.chunk_count = 0
        logger.info("Buffer reset")

    def __len__(self) -> int:
        """Return current number of frames in buffer."""
        return len(self.buffer)

    def is_ready(self) -> bool:
        """Check if buffer has enough frames for chunk extraction."""
        return len(self.buffer) >= self.chunk_size

    @property
    def stats(self) -> dict:
        """Get buffer statistics."""
        return {
            "frames_processed": self.frame_count,
            "chunks_extracted": self.chunk_count,
            "current_buffer_size": len(self.buffer),
            "chunk_size": self.chunk_size,
            "overlap": self.overlap,
            "stride": self.stride,
        }
