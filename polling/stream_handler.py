"""
Video stream handler for polling-based inference.
Handles frame extraction from video files and live streams.
"""

import os
import sys
import time
import logging
import threading
import subprocess
import warnings
from typing import List, Tuple, Optional, Generator
from collections import deque
from dataclasses import dataclass
from contextlib import contextmanager

# Suppress FFmpeg warnings
os.environ['OPENCV_FFMPEG_LOGLEVEL'] = '-8'
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['AV_LOG_LEVEL'] = 'quiet'

import cv2
import numpy as np
import torch
from PIL import Image
from decord import VideoReader, cpu

# Set decord log level to error only
try:
    from decord import bridge
    bridge.set_bridge('torch')
except:
    pass

# Suppress Python warnings
warnings.filterwarnings('ignore')


@contextmanager
def suppress_stderr():
    """Context manager to suppress stderr output (FFmpeg warnings)."""
    stderr_fd = sys.stderr.fileno()
    with open(os.devnull, 'w') as devnull:
        old_stderr = os.dup(stderr_fd)
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(old_stderr, stderr_fd)
            os.close(old_stderr)


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

    @staticmethod
    def test_webcam_availability(index: int = 0) -> bool:
        """Test if a webcam is available at the given index."""
        try:
            cap = cv2.VideoCapture(index)
            is_opened = cap.isOpened()
            cap.release()
            # Just check if it opens, don't try to read (which can timeout)
            return is_opened
        except:
            return False

    @staticmethod
    def is_wsl2() -> bool:
        """Check if running in WSL2."""
        try:
            with open('/proc/version', 'r') as f:
                return 'microsoft' in f.read().lower()
        except:
            return False

    def __init__(
        self,
        buffer_size: int = 64,
        num_frames: int = 16,
        fps: int = 1,
        image_resolution: int = 224,
        inference_window_seconds: float = 0.0,
    ):
        self.buffer_size = buffer_size
        self.num_frames = num_frames
        self.fps = fps
        self.image_resolution = image_resolution
        # If > 0, webcam inference samples num_frames across this many seconds of
        # recent buffer (decoupled from capture fps). 0 = legacy frame-count tail.
        self.inference_window_seconds = inference_window_seconds

        self.logger = logging.getLogger("VideoStreamHandler")

        # Frame buffer (thread-safe deque)
        self.frame_buffer: deque = deque(maxlen=buffer_size)

        # Stream state
        self._is_running = False
        self._stream_thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None
        self._video_reader: Optional[VideoReader] = None

        # FFmpeg process for WSL2 cameras
        self._ffmpeg_proc: Optional[subprocess.Popen] = None
        self._use_ffmpeg_capture = False
        self._frame_width = 640
        self._frame_height = 480

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
            # Suppress FFmpeg output (no log files)
            os.environ['FFREPORT'] = 'file=:level=-8'

            with suppress_stderr():
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
                self.logger.info(f"Attempting to open webcam with index: {source}")

            # For webcam on WSL2, use ffmpeg subprocess (much more reliable)
            if isinstance(source, int) and self.is_wsl2():
                self.logger.info("WSL2 detected - using ffmpeg subprocess for camera capture")
                return self._open_stream_ffmpeg(source)

            # For webcam, use FFMPEG backend which is more reliable
            if isinstance(source, int):
                self.logger.info("Using FFMPEG backend for webcam...")
                self._cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)

                if not self._cap.isOpened():
                    self.logger.warning("FFMPEG failed, trying default V4L2...")
                    self._cap = cv2.VideoCapture(source)
            else:
                self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                # Try different backends for better compatibility
                if isinstance(source, int):
                    self.logger.warning(f"Trying CAP_V4L2 explicitly...")
                    self._cap = cv2.VideoCapture(source, cv2.CAP_V4L2)

                if not self._cap.isOpened():
                    self.logger.error(
                        f"Failed to open stream: {source}. "
                        f"Possible issues: No camera detected, camera in use by another app, "
                        f"or permission denied."
                    )
                    return False

            # Set buffer size to 1 for low latency (webcam only)
            if isinstance(source, int):
                self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.logger.info("Set camera buffer size to 1 for low latency")

            self._video_fps = self._cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.logger.info(f"Opened stream: {source}, FPS: {self._video_fps:.2f}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to open stream: {e}")
            return False

    def _open_stream_ffmpeg(self, camera_index: int) -> bool:
        """Open camera using ffmpeg subprocess - works better on WSL2."""
        try:
            device = f"/dev/video{camera_index}"

            # Build ffmpeg command to output raw RGB frames to stdout
            cmd = [
                'ffmpeg',
                '-f', 'v4l2',
                '-framerate', '30',
                '-video_size', f'{self._frame_width}x{self._frame_height}',
                '-i', device,
                '-f', 'rawvideo',
                '-pix_fmt', 'rgb24',
                '-loglevel', 'error',
                '-'  # Output to stdout
            ]

            self.logger.info(f"Starting ffmpeg: {' '.join(cmd)}")

            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=self._frame_width * self._frame_height * 3 * 2  # Buffer 2 frames
            )

            # Give ffmpeg a moment to start
            time.sleep(0.5)

            # Check if process is still running
            if self._ffmpeg_proc.poll() is not None:
                stderr = self._ffmpeg_proc.stderr.read().decode()
                self.logger.error(f"ffmpeg failed to start: {stderr}")
                return False

            self._use_ffmpeg_capture = True
            self._video_fps = 30.0
            self.logger.info(f"ffmpeg capture started for {device}, FPS: {self._video_fps:.2f}")
            return True

        except FileNotFoundError:
            self.logger.error("ffmpeg not found! Install with: sudo apt install ffmpeg")
            return False
        except Exception as e:
            self.logger.error(f"Failed to start ffmpeg capture: {e}")
            return False

    def start_stream_capture(self, source: str):
        """Start background thread to capture frames from stream."""
        if not self.open_stream(source):
            return

        # Skip warmup for ffmpeg mode (it handles initialization internally)
        if not self._use_ffmpeg_capture:
            # Give camera time to initialize (V4L2 needs this)
            if isinstance(source, int) or (isinstance(source, str) and source.isdigit()):
                self.logger.info("Warming up camera...")
                time.sleep(0.5)  # 500ms warmup for V4L2 cameras
                # Try to grab a few frames to flush camera buffer
                for _ in range(5):
                    self._cap.grab()
                self.logger.info("Camera warmup complete")

        self._is_running = True

        # Choose appropriate capture loop
        if self._use_ffmpeg_capture:
            self._stream_thread = threading.Thread(target=self._capture_loop_ffmpeg, daemon=True)
        else:
            self._stream_thread = threading.Thread(target=self._capture_loop, daemon=True)

        self._stream_thread.start()
        self.logger.info("Started stream capture thread")

    def _capture_loop(self):
        """Background loop to capture frames from stream."""
        frame_interval = 1.0 / self.fps
        last_capture = time.time()  # Initialize to current time
        frame_idx = 0
        consecutive_failures = 0
        max_consecutive_failures = 30  # Stop after 30 consecutive failures

        self.logger.info(f"Capture loop started: target FPS={self.fps}, interval={frame_interval:.4f}s")

        # Try to capture first frame immediately
        ret, frame = self._cap.read()
        if ret:
            self.logger.info(f"First frame captured successfully! Shape: {frame.shape}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_data = FrameData(
                frame=frame_rgb,
                timestamp=time.time(),
                frame_index=frame_idx,
            )
            self.frame_buffer.append(frame_data)
            frame_idx += 1
            last_capture = time.time()
        else:
            self.logger.warning("Failed to capture first frame, will retry...")

        while self._is_running and self._cap is not None:
            current_time = time.time()

            # Sample at target FPS
            if current_time - last_capture >= frame_interval:
                ret, frame = self._cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures <= 5:  # Only log first few failures
                        self.logger.warning(f"Failed to read frame from stream (attempt {consecutive_failures})")

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"Failed to read {max_consecutive_failures} consecutive frames, stopping capture")
                        self._is_running = False
                        break

                    time.sleep(0.1)  # Wait a bit before retrying
                    continue

                # Successfully read a frame
                if consecutive_failures > 0:
                    self.logger.info(f"Successfully resumed frame capture after {consecutive_failures} failures")
                consecutive_failures = 0

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

    def _capture_loop_ffmpeg(self):
        """Background loop to capture frames from ffmpeg subprocess (for WSL2)."""
        frame_size = self._frame_width * self._frame_height * 3  # RGB24
        frame_interval = 1.0 / self.fps
        last_capture = time.time()
        frame_idx = 0
        consecutive_failures = 0
        max_consecutive_failures = 30

        self.logger.info(f"FFmpeg capture loop started: target FPS={self.fps}, frame size={frame_size} bytes")

        while self._is_running and self._ffmpeg_proc is not None:
            current_time = time.time()

            # Sample at target FPS
            if current_time - last_capture >= frame_interval:
                try:
                    # Read one frame worth of raw RGB data from ffmpeg stdout
                    raw_frame = self._ffmpeg_proc.stdout.read(frame_size)

                    if len(raw_frame) != frame_size:
                        consecutive_failures += 1
                        if consecutive_failures <= 5:
                            self.logger.warning(f"FFmpeg: incomplete frame read ({len(raw_frame)}/{frame_size} bytes)")

                        if consecutive_failures >= max_consecutive_failures:
                            self.logger.error(f"FFmpeg: {max_consecutive_failures} consecutive failures, stopping")
                            self._is_running = False
                            break

                        # Check if ffmpeg process died
                        if self._ffmpeg_proc.poll() is not None:
                            stderr = self._ffmpeg_proc.stderr.read().decode()
                            self.logger.error(f"FFmpeg process died: {stderr}")
                            self._is_running = False
                            break

                        time.sleep(0.01)
                        continue

                    # Convert raw bytes to numpy array (already RGB)
                    frame_rgb = np.frombuffer(raw_frame, dtype=np.uint8).reshape(
                        (self._frame_height, self._frame_width, 3)
                    )

                    # Log first successful frame
                    if frame_idx == 0:
                        self.logger.info(f"FFmpeg: First frame captured! Shape: {frame_rgb.shape}")

                    if consecutive_failures > 0:
                        self.logger.info(f"FFmpeg: Resumed after {consecutive_failures} failures")
                    consecutive_failures = 0

                    frame_data = FrameData(
                        frame=frame_rgb.copy(),  # Copy to prevent buffer overwrite
                        timestamp=current_time,
                        frame_index=frame_idx,
                    )
                    self.frame_buffer.append(frame_data)

                    frame_idx += 1
                    last_capture = current_time

                    # Log progress periodically
                    if frame_idx % 100 == 0:
                        self.logger.info(f"FFmpeg: Captured {frame_idx} frames, buffer size: {len(self.frame_buffer)}")

                except Exception as e:
                    consecutive_failures += 1
                    if consecutive_failures <= 5:
                        self.logger.warning(f"FFmpeg capture error: {e}")

                    if consecutive_failures >= max_consecutive_failures:
                        self.logger.error(f"FFmpeg: Too many errors, stopping")
                        self._is_running = False
                        break

            time.sleep(0.001)  # Small sleep to prevent CPU spinning

        self.logger.info(f"FFmpeg capture loop ended after {frame_idx} frames")

    def stop_stream(self):
        """Stop the stream capture."""
        self._is_running = False
        if self._stream_thread:
            self._stream_thread.join(timeout=2.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        if self._ffmpeg_proc:
            self._ffmpeg_proc.terminate()
            try:
                self._ffmpeg_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._ffmpeg_proc.kill()
            self._ffmpeg_proc = None
            self._use_ffmpeg_capture = False
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

        # Ensure we don't go past video end
        if start_frame >= self._total_frames:
            self.logger.warning(f"Start frame {start_frame} >= total frames {self._total_frames}")
            return [], 0

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
            self.logger.warning(f"No frames available: start={start_frame}, end={end_frame}")
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

        # Ensure all indices are valid
        frame_indices = [idx for idx in frame_indices if 0 <= idx < self._total_frames]
        if not frame_indices:
            return [], 0

        try:
            # Try to extract frames with suppressed stderr
            with suppress_stderr():
                frames = self._video_reader.get_batch(frame_indices).asnumpy()

            # Move position forward by advance_by (or duration if not specified)
            # advance_by < duration creates overlapping windows
            if advance_by is None:
                advance_by = duration
            self._current_frame_idx = start_frame + int(advance_by * self._video_fps)

            self.logger.debug(f"Extracted {len(frames)} frames from indices {frame_indices[0]}-{frame_indices[-1]}")
            return list(frames), len(frames)

        except Exception as e:
            # If batch extraction fails, use OpenCV as robust fallback
            self.logger.warning(f"Decord extraction failed, using OpenCV fallback: {e}")

            try:
                # Use OpenCV for sequential frame reading (more robust but slower)
                cap = cv2.VideoCapture(self._video_path)
                if not cap.isOpened():
                    raise Exception("Failed to open video with OpenCV")

                frames = []
                failed_indices = []

                for idx in frame_indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    if ret:
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    else:
                        failed_indices.append(idx)
                        # Pad with last successful frame or black frame
                        if len(frames) > 0:
                            frames.append(frames[-1].copy())  # Duplicate last frame
                        else:
                            # Create black frame with correct dimensions
                            black_frame = np.zeros((self.image_resolution, self.image_resolution, 3), dtype=np.uint8)
                            frames.append(black_frame)

                cap.release()

                if len(failed_indices) > 0:
                    self.logger.warning(f"Padded {len(failed_indices)} corrupted frames with duplicates")

                if len(frames) >= self.num_frames // 2:  # Accept if we got at least half the frames
                    # Advance position
                    if advance_by is None:
                        advance_by = duration
                    self._current_frame_idx = start_frame + int(advance_by * self._video_fps)

                    self.logger.info(f"OpenCV fallback: extracted {len(frames)} frames ({len(frames) - len(failed_indices)} good, {len(failed_indices)} padded)")
                    return frames, len(frames)
                else:
                    raise Exception(f"Too many corrupted frames: only {len(frames) - len(failed_indices)}/{len(frame_indices)} readable")

            except Exception as e2:
                self.logger.error(f"OpenCV fallback also failed: {e2}")

            # If all else fails, advance position and return empty
            if advance_by is None:
                advance_by = duration
            self._current_frame_idx = start_frame + int(advance_by * self._video_fps)
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
            # Video file mode - use polling_interval as both duration and advance_by
            # This ensures we sample from a window equal to polling_interval and advance by the same amount
            # For continuous non-overlapping polling throughout the entire video
            raw_frames, slice_len = self.extract_frames_from_file(
                duration=polling_interval,
                advance_by=polling_interval
            )
        elif len(self.frame_buffer) > 0:
            # Stream mode (real-time camera): use the MOST RECENT num_video_frames
            # frames from the buffer, not a uniform sample across the whole buffer.
            #
            # Why: for live exercise feedback the model must reason about "what
            # the patient is doing right now". Uniform sampling across a 64s
            # buffer would mix multiple exercises (squats + push-ups + lunges)
            # into a single inference, producing confused/generic responses.
            # Capture runs fast (smooth preview), but the model needs a fixed
            # TEMPORAL window of "what the patient is doing right now". Select the
            # frames from the last `inference_window_seconds`; the uniform sample
            # below then picks num_video_frames evenly across that window — so the
            # model input is independent of capture fps. Falls back to a frame-count
            # tail if no window is configured or too few frames are present.
            buffer_frames = list(self.frame_buffer)
            window = getattr(self, "inference_window_seconds", 0.0)
            if window and window > 0:
                cutoff = buffer_frames[-1].timestamp - window
                windowed = [f for f in buffer_frames if f.timestamp >= cutoff]
                if len(windowed) < num_video_frames:
                    windowed = buffer_frames[-num_video_frames:]
                raw_frames = [f.frame for f in windowed]
            else:
                raw_frames = [f.frame for f in buffer_frames[-num_video_frames:]]
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
