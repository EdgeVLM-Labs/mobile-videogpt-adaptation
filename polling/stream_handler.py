"""
Video stream handler for polling-based inference.
Handles frame extraction from video files and live streams.
"""

import os
import time
import logging
import threading
from typing import List, Tuple, Optional, Generator
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu


@dataclass
class FrameData:
    """Container for frame data with metadata."""
    frame: np.ndarray
    timestamp: float
    frame_index: int


class VideoStreamHandler:
    """
    Handles video stream input for polling-based inference.
    Supports both video files and live camera/RTSP streams.
    """

    def __init__(
        self,
        buffer_size: int = 64,
        num_frames: int = 16,
        fps: int = 1,
        image_resolution: int = 224,
    ):
        self.buffer_size = buffer_size
        self.num_frames = num_frames
        self.fps = fps
        self.image_resolution = image_resolution

        self.logger = logging.getLogger("VideoStreamHandler")

        # Frame buffer (thread-safe deque)
        self.frame_buffer: deque = deque(maxlen=buffer_size)

        # Stream state
        self._is_running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_reader: Optional[VideoReader] = None

        # Video file state
        self._video_path: Optional[str] = None
        self._video_fps: float = 30.0
        self._total_frames: int = 0
        self._current_frame_idx: int = 0

    def open_video_file(self, video_path: str) -> bool:
        """Open a video file for frame extraction."""
        if not os.path.exists(video_path):
            self.logger.error(f"Video file not found: {video_path}")
            return False

        try:
            self._video_reader = VideoReader(video_path, ctx=cpu(0), num_threads=1)
            self._video_fps = self._video_reader.get_avg_fps()
            self._total_frames = len(self._video_reader)
            self._video_path = video_path
            self._current_frame_idx = 0

            self.logger.info(
                f"Opened video: {video_path}, "
                f"FPS: {self._video_fps:.2f}, "
                f"Total frames: {self._total_frames}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to open video: {e}")
            return False

    def open_stream(self, source: str) -> bool:
        """Open a video stream (camera index or RTSP URL)."""
        try:
            # Try to parse as camera index
            if source.isdigit():
                source = int(source)

            self._cap = cv2.VideoCapture(source)
            if not self._cap.isOpened():
                self.logger.error(f"Failed to open stream: {source}")
                return False

            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.logger.info(f"Opened stream: {source}, FPS: {self._video_fps:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to open stream: {e}")
            return False

    def start_stream_capture(self, source: str):
        """Start background thread to capture frames from stream."""
        if not self.open_stream(source):
            return

        self._is_running = True
        self._stream_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._stream_thread.start()
        self.logger.info("Started stream capture thread")

    def _capture_loop(self):
        """Background loop to capture frames from stream."""
        frame_interval = 1.0 / self.fps
        last_capture = 0
        frame_idx = 0

        while self._is_running and self._cap is not None:
            current_time = time.time()

            # Sample at target FPS
            if current_time - last_capture >= frame_interval:
                ret, frame = self._cap.read()
                if not ret:
                    self.logger.warning("Failed to read frame from stream")
                    continue

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                frame_data = FrameData(
                    frame=frame_rgb,
                    timestamp=current_time,
                    frame_index=frame_idx,
                )
                self.frame_buffer.append(frame_data)

                frame_idx += 1
                last_capture = current_time

            time.sleep(0.001)  # Small sleep to prevent CPU spinning

    def stop_stream(self):
        """Stop the stream capture."""
        self._is_running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.logger.info("Stopped stream capture")

    def extract_frames_from_file(
        self,
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        advance_by: Optional[float] = None,
    ) -> Tuple[List[np.ndarray], int]:
        """
        Extract frames from video file for a time window.

        Args:
            start_time: Start time in seconds (None = current position)
            duration: Duration in seconds (None = num_frames / fps)
            advance_by: How many seconds to advance position after extraction
                       (None = same as duration for no overlap)

        Returns:
            Tuple of (list of frames, number of frames extracted)
        """
        if self._video_reader is None:
            self.logger.error("No video file opened")
            return [], 0

        # Calculate frame indices
        if start_time is not None:
            start_frame = int(start_time * self._video_fps)
        else:
            start_frame = self._current_frame_idx

        # Default duration: enough for num_frames at specified fps
        if duration is None:
            duration = self.num_frames / self.fps

        end_frame = min(
            start_frame + int(duration * self._video_fps),
            self._total_frames
        )

        # Calculate stride for uniform sampling
        num_available = end_frame - start_frame
        if num_available <= 0:
            return [], 0

        # Sample frames uniformly to get exactly num_frames
        if num_available >= self.num_frames:
            # Uniform sampling across the window
            frame_indices = [
                start_frame + int(i * num_available / self.num_frames)
                for i in range(self.num_frames)
            ]
        else:
            # Not enough frames, take what we have
            frame_indices = list(range(start_frame, end_frame))

        try:
            frames = self._video_reader.get_batch(frame_indices).asnumpy()
            # Move position forward by advance_by (or duration if not specified)
            # advance_by < duration creates overlapping windows
            if advance_by is None:
                advance_by = duration
            self._current_frame_idx = start_frame + int(advance_by * self._video_fps)
            return list(frames), len(frames)
        except Exception as e:
            self.logger.error(f"Error extracting frames: {e}")
            return [], 0

    def get_frames_for_inference(
        self,
        image_processor,
        video_processor,
        num_video_frames: int = 16,
        num_context_images: int = 16,
        polling_interval: Optional[float] = None,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], int]:
        """
        Get processed frames ready for model inference.

        For video files: extracts next window of frames
        For streams: gets frames from buffer

        Args:
            polling_interval: Polling interval in seconds (controls position advancement)

        Returns:
            Tuple of (video_frames, context_frames, slice_len)
        """
        # Get raw frames
        if self._video_reader is not None:
            # Video file mode - advance by polling_interval to allow continuous polling
            raw_frames, slice_len = self.extract_frames_from_file(advance_by=polling_interval)
        elif len(self.frame_buffer) > 0:
            # Stream mode - get frames from buffer
            buffer_frames = list(self.frame_buffer)
            if len(buffer_frames) >= num_video_frames:
                # Uniform sample from buffer
                step = len(buffer_frames) // num_video_frames
                raw_frames = [buffer_frames[i * step].frame for i in range(num_video_frames)]
            else:
                raw_frames = [f.frame for f in buffer_frames]
            slice_len = len(raw_frames)
        else:
            return [], [], 0

        if not raw_frames:
            return [], [], 0

        # Uniform sample to target counts
        video_frames_raw = self._uniform_sample(raw_frames, min(num_video_frames, len(raw_frames)))
        context_frames_raw = self._uniform_sample(raw_frames, min(num_context_images, len(raw_frames)))

        # Process for video encoder
        video_frames = video_processor.preprocess(video_frames_raw)['pixel_values']

        # Process for image encoder
        context_frames = [
            image_processor.preprocess(frame, return_tensors='pt')['pixel_values'][0]
            for frame in context_frames_raw
        ]

        # Pad if needed
        while len(video_frames) < num_video_frames:
            video_frames.append(torch.zeros((3, self.image_resolution, self.image_resolution)))

        while len(context_frames) < num_context_images:
            context_frames.append(
                torch.zeros((3, image_processor.crop_size['height'], image_processor.crop_size['width']))
            )

        return video_frames, context_frames, slice_len

    def _uniform_sample(self, lst: List, n: int) -> List:
        """Uniformly sample n items from list."""
        if n >= len(lst):
            return lst
        step = len(lst) // n
        return [lst[i * step] for i in range(n)]

    def get_remaining_duration(self) -> float:
        """Get remaining video duration in seconds."""
        if self._video_reader is None:
            return float('inf')  # Stream mode

        remaining_frames = self._total_frames - self._current_frame_idx
        return remaining_frames / self._video_fps

    def reset(self):
        """Reset to beginning of video."""
        self._current_frame_idx = 0
        self.frame_buffer.clear()

    def close(self):
        """Close all resources."""
        self.stop_stream()
        if self._video_reader:
            del self._video_reader
            self._video_reader = None

    @property
    def is_exhausted(self) -> bool:
        """Check if video file is exhausted."""
        if self._video_reader is None:
            return False  # Stream never exhausted
        return self._current_frame_idx >= self._total_frames

    @property
    def current_position(self) -> float:
        """Get current position in seconds."""
        if self._video_reader is None:
            return 0.0
        return self._current_frame_idx / self._video_fps

    @property
    def total_duration(self) -> float:
        """Get total video duration in seconds."""
        if self._video_reader is None:
            return float('inf')
        return self._total_frames / self._video_fps
