#!/usr/bin/env python3
"""
Gradio Interface for Mobile-VideoGPT Polling Inference
Real-time exercise form evaluation with LoRA adapters
"""

import os
import sys
import json
import time
import glob
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Generator
import threading
import queue
from io import StringIO

import gradio as gr
import numpy as np
import torch
from PIL import Image

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
from polling.stream_handler import VideoStreamHandler
from polling.metrics import MetricsTracker
try:
    from utils.naturalizer.feedback_naturalizer import FeedbackNaturalizer
except ImportError:
    FeedbackNaturalizer = None


class LogCapture(logging.Handler):
    """Custom logging handler to capture logs for display"""
    def __init__(self):
        super().__init__()
        self.logs = []

    def emit(self, record):
        log_entry = self.format(record)
        self.logs.append(log_entry)

    def get_logs(self):
        return "\n".join(self.logs[-100:])  # Last 100 lines

    def clear(self):
        self.logs = []


class GradioPollingApp:
    """Gradio interface for polling inference"""

    def __init__(self):
        self.engine: Optional[PollingInferenceEngine] = None
        self.naturalizer: Optional[FeedbackNaturalizer] = None
        self.is_running = False
        self.current_session_id = None
        self.poll_results = []
        self.metrics_history = []
        self.log_capture = LogCapture()
        self.log_capture.setLevel(logging.INFO)
        self.log_capture.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))
        self.temp_dir = None
        self.current_video_path = None

        # Browser webcam state
        self._browser_frame: Optional[np.ndarray] = None
        self._use_browser_webcam = False

        # Add log capture to root logger
        logging.getLogger().addHandler(self.log_capture)
        logging.getLogger().setLevel(logging.INFO)

    def update_browser_frame(self, frame: Optional[np.ndarray]) -> None:
        """Update the browser frame from Gradio's streaming webcam component.

        This is called continuously when browser webcam is streaming.
        """
        if frame is not None and isinstance(frame, np.ndarray):
            self._browser_frame = frame
            # Log occasionally to avoid spam
            if not hasattr(self, '_frame_log_counter'):
                self._frame_log_counter = 0
            self._frame_log_counter += 1
            if self._frame_log_counter % 30 == 0:  # Log every ~30 frames
                logging.debug(f"Browser webcam frame updated: shape={frame.shape}")

    def get_latest_webcam_frame(self) -> Optional[np.ndarray]:
        """Get the latest frame from webcam buffer for preview."""
        # For browser webcam mode, return the browser frame
        if self._use_browser_webcam and self._browser_frame is not None:
            return self._browser_frame
        # For direct webcam mode, get from buffer
        if self.engine and self.engine.stream_handler and len(self.engine.stream_handler.frame_buffer) > 0:
            latest_frame_data = self.engine.stream_handler.frame_buffer[-1]
            return latest_frame_data.frame  # Already RGB
        return None

    @staticmethod
    def get_available_cameras() -> List[Tuple[str, int]]:
        """Get list of available cameras with their names and indices.

        Returns:
            List of tuples (display_name, video_index)
        """
        cameras = []

        # Try to get camera info from /dev/v4l/by-id/ (Linux)
        try:
            v4l_path = Path("/dev/v4l/by-id/")
            if v4l_path.exists():
                for symlink in v4l_path.iterdir():
                    if symlink.is_symlink():
                        # Get the actual video device it points to
                        target = symlink.resolve()
                        video_num = int(target.name.replace('video', ''))

                        # Parse camera name from symlink
                        name_parts = symlink.name.replace('usb-', '').replace('_', ' ').split('-video-index')[0]
                        # Clean up the name
                        name = ' '.join(name_parts.split()).title()

                        # Only add video-index0 (main video stream, not metadata)
                        if 'video-index0' in symlink.name:
                            cameras.append((f"{name} (video{video_num})", video_num))
        except Exception as e:
            logging.warning(f"Could not read /dev/v4l/by-id/: {e}")

        # Fallback: try numeric indices 0-5
        if not cameras:
            for i in range(6):
                cameras.append((f"Camera {i} (video{i})", i))

        return cameras

    def extract_video_segment(self, video_path: str, start_time: float, duration: float) -> str:
        """Extract video segment starting at specific time using FFmpeg"""
        try:
            # Create output path in temp directory
            output_path = os.path.join(self.temp_dir, f"segment_{start_time:.1f}s.mp4")

            # Use FFmpeg to extract segment
            cmd = [
                'ffmpeg',
                '-y',  # Overwrite output file
                '-ss', str(start_time),  # Start time
                '-i', video_path,  # Input file
                '-t', str(duration),  # Duration
                '-c', 'copy',  # Copy codec (fast)
                '-avoid_negative_ts', 'make_zero',  # Fix timestamp issues
                '-loglevel', 'error',  # Suppress output
                output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return output_path

        except subprocess.CalledProcessError as e:
            logging.warning(f"FFmpeg copy failed, trying re-encode: {e}")
            try:
                # Fallback: re-encode if copy fails
                cmd = [
                    'ffmpeg',
                    '-y',
                    '-ss', str(start_time),
                    '-i', video_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',  # Re-encode video
                    '-preset', 'ultrafast',  # Fast encoding
                    '-c:a', 'aac',  # Re-encode audio
                    '-loglevel', 'error',
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path
            except Exception as e2:
                logging.error(f"Failed to extract segment: {e2}")
                return video_path  # Return original on failure
        except Exception as e:
            logging.error(f"Error extracting segment: {e}")
            return video_path

    def get_sample_videos(self) -> List[str]:
        """Get list of videos from sample_videos folder"""
        project_root = Path(__file__).parent.parent
        sample_videos_dir = project_root / "sample_videos"

        if not sample_videos_dir.exists():
            return []

        videos = []
        for ext in ['*.mp4', '*.avi', '*.mov']:
            videos.extend(glob.glob(str(sample_videos_dir / ext)))

        return sorted([str(Path(v).name) for v in videos])

    def format_metrics(self, metrics: dict) -> str:
        """Format metrics dictionary as readable text"""
        if not metrics:
            return "No metrics available"

        output = []
        output.append("**Poll Metrics**\n")
        output.append(f"**Latency:** {metrics.get('latency_ms', 0):.1f} ms\n")
        output.append(f"**TTFT:** {metrics.get('ttft_ms', 0):.1f} ms\n")
        output.append(f"**Tokens/s:** {metrics.get('tokens_per_second', 0):.1f}\n")
        output.append(f"**Frames:** {metrics.get('frames_processed', 0)}\n")
        output.append(f"**Output Tokens:** {metrics.get('output_tokens', 0)}")

        return "".join(output)

    def format_session_metrics(self, session_metrics: dict) -> str:
        """Format session-level metrics"""
        if not session_metrics:
            return "No session metrics available"

        output = []
        output.append("**Session Summary**\n")
        output.append(f"**Session ID:** {session_metrics.get('session_id', 'N/A')}")
        output.append(f"**Duration:** {session_metrics.get('duration_seconds', 0):.2f}s")
        output.append(f"**Total Polls:** {session_metrics.get('total_polls', 0)}")
        output.append(f"**Success Rate:** {session_metrics.get('success_rate', 0):.1f}%\n")

        latency_ms = session_metrics.get('latency_ms', {})
        if latency_ms:
            output.append("**Latency Statistics (ms):**")
            output.append(f"  Mean: {latency_ms.get('mean', 0):.1f}")
            output.append(f"  Median: {latency_ms.get('median', 0):.1f}")
            output.append(f"  Min: {latency_ms.get('min', 0):.1f}")
            output.append(f"  Max: {latency_ms.get('max', 0):.1f}")

        return "\n".join(output)

    def format_all_responses(self, results: List[dict]) -> str:
        """Format all poll responses"""
        if not results:
            return "No responses yet"

        output = []
        output.append("**All Poll Responses**\n")
        output.append("=" * 60 + "\n")

        for i, result in enumerate(results, 1):
            output.append(f"**Poll #{i}** (Position: {result.get('position', 'N/A')}s)")
            output.append(f"_{result.get('response', 'No response')}_\n")

        return "\n".join(output)

    def run_inference(
        self,
        video_source: str,
        webcam_mode: str,
        camera_name: str,
        browser_frame: Optional[np.ndarray],
        base_model: str,
        lora_weights: str,
        polling_interval: float,
        num_frames: int,
        fps: int,
        max_new_tokens: int,
        warmup_runs: int,
        prompt: str,
        use_naturalizer: bool,
        naturalizer_threshold: float,
        progress=gr.Progress()
    ) -> Generator[Tuple[str, str, str, str, str, str], None, None]:
        """Run polling inference"""

        # Determine mode
        use_browser_webcam = webcam_mode == "Browser Webcam"
        use_direct_webcam = webcam_mode == "Direct Webcam (Linux only)"
        use_webcam = use_browser_webcam or use_direct_webcam

        try:
            # Reset state
            self.poll_results = []
            self.metrics_history = []
            self.is_running = True
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Store browser frame for browser webcam mode
            self._browser_frame = browser_frame
            self._use_browser_webcam = use_browser_webcam

            # Initialize naturalizer if enabled (reuse existing instance if available)
            if use_naturalizer:
                if not self.naturalizer:
                    logging.info(f"Initializing Feedback Naturalizer (threshold={naturalizer_threshold})")
                    self.naturalizer = FeedbackNaturalizer(threshold=naturalizer_threshold)
                else:
                    logging.info("Reusing existing Feedback Naturalizer")
                    self.naturalizer.reset()
            else:
                # Clean up naturalizer if it exists but is not needed
                if self.naturalizer:
                    logging.info("Cleaning up unused Feedback Naturalizer")
                    try:
                        self.naturalizer.cleanup()
                    except:
                        pass
                self.naturalizer = None

            # Create temp directory for video segments
            if self.temp_dir:
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
            self.temp_dir = tempfile.mkdtemp(prefix="polling_segments_")

            # Determine video path based on mode
            if use_browser_webcam:
                # Browser webcam mode - frames come from Gradio's webcam component
                if browser_frame is None:
                    yield (
                        None,
                        "❌ **Browser Webcam Error**",
                        "**No frame from browser webcam!**\n\nMake sure to:\n1. Allow camera access in your browser\n2. Click the webcam preview to start capturing",
                        "Enable browser webcam",
                        "",
                        self.log_capture.get_logs()
                    )
                    return

                video_path = "browser_webcam"
                self.current_video_path = video_path
                logging.info("Using browser webcam mode (frames from Gradio)")
                progress(0.05, desc="Browser webcam ready...")

            elif use_direct_webcam:
                # Direct webcam mode - use V4L2/ffmpeg (Linux only)
                import re
                match = re.search(r'video(\d+)', camera_name)
                if match:
                    camera_index = int(match.group(1))
                else:
                    camera_index = 0  # Fallback

                video_path = str(camera_index)
                self.current_video_path = video_path
                progress(0, desc=f"Testing {camera_name}...")

                # Test webcam before proceeding
                logging.info(f"Testing {camera_name} (index {camera_index})...")
                if not VideoStreamHandler.test_webcam_availability(camera_index):
                    error_msg = (
                        f"⚠️ **{camera_name} detected but cannot access frames**\n\n"
                        f"**This appears to be a virtual/USB-forwarded camera (vhci_hcd)**\n\n"
                        f"**On WSL2, use 'Browser Webcam' mode instead!**\n\n"
                        f"Direct webcam only works on native Linux.\n"
                    )
                    logging.error(error_msg)
                    yield (
                        None,
                        "❌ **Direct Webcam Error**",
                        error_msg,
                        "Cannot access webcam - try Browser Webcam mode",
                        "",
                        self.log_capture.get_logs()
                    )
                    return

                logging.info(f"{camera_name} test successful!")
                progress(0.05, desc="Opening webcam...")
            else:
                # Video file mode
                project_root = Path(__file__).parent.parent
                video_path = str(project_root / "sample_videos" / video_source)
                self.current_video_path = video_path

                if not os.path.exists(video_path):
                    yield (
                        video_path,
                        "❌ **Error**",
                        "❌ Video file not found",
                        "Error: Video file does not exist",
                        "",
                        self.log_capture.get_logs()
                    )
                    return

                progress(0, desc=f"Loading video: {video_source}")

            # Clear previous logs
            self.log_capture.clear()

            # Create config
            config = PollingConfig(
                base_model_path=base_model,
                lora_weights_path=lora_weights,
                polling_interval=polling_interval,
                num_frames=num_frames,
                fps=fps,
                max_new_tokens=max_new_tokens,
                prompt=prompt
            )

            logging.info(f"Starting inference with video: {video_path}")
            logging.info(f"Config: interval={polling_interval}s, frames={num_frames}, fps={fps}")

            # Initialize or reuse engine
            progress(0.1, desc="Loading model...")

            # Check if we can reuse existing engine
            if self.engine and self.engine._is_loaded:
                # Check if config matches
                if (self.engine.config.base_model_path == base_model and
                    self.engine.config.lora_weights_path == lora_weights):
                    logging.info("Reusing existing engine (config matches)")
                else:
                    logging.info("Config changed, cleaning up old engine and creating new one")
                    try:
                        self.engine.cleanup()
                        self.engine.metrics.cleanup()
                    except:
                        pass
                    self.engine = None

            # Create new engine if needed
            if not self.engine:
                logging.info("Creating new inference engine")
                self.engine = PollingInferenceEngine(config)
            else:
                # Update config for existing engine (update runtime params only)
                self.engine.config.polling_interval = polling_interval
                self.engine.config.num_frames = num_frames
                self.engine.config.fps = fps
                self.engine.config.max_new_tokens = max_new_tokens
                self.engine.config.prompt = prompt

            # Load model to initialize processors
            if not self.engine.load_model():
                logging.error("Failed to load model")
                yield (
                    video_path if use_webcam else video_path,
                    "**Error**",
                    "Failed to load model",
                    "Error: Could not load model",
                    "",
                    self.log_capture.get_logs()
                )
                return

            # Aggressive memory cleanup before starting a new session.
            # Previous sessions can leave the CUDA caching allocator fragmented,
            # which causes OOM on the 2nd+ inference. Reset state here.
            import gc
            gc.collect()
            gc.collect()  # 2 passes: first collects refs, second collects what those referenced
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                free_mb = torch.cuda.mem_get_info()[0] / (1024 * 1024)
                logging.info(f"Pre-session CUDA free: {free_mb:.0f}MB")

            # Start metrics session
            self.engine.metrics.start_session(
                video_source=video_path,
                prompt=prompt,
                polling_interval=polling_interval,
                naturalizer_enabled=use_naturalizer
            )

            # Run warmup
            if warmup_runs > 0:
                progress(0.2, desc=f"Running {warmup_runs} warmup runs...")
                self.engine.warmup(warmup_runs)

            # Start polling
            progress(0.3, desc="Starting polling...")

            # Open video or stream
            if use_browser_webcam:
                # Browser webcam mode - frames come from Gradio component
                logging.info("Using browser webcam mode - frames streamed from browser")
                logging.info("Make sure browser has camera access and webcam is visible in the UI")
                if self._browser_frame is not None:
                    logging.info(f"Initial browser frame shape: {self._browser_frame.shape}")
                else:
                    logging.warning("No browser frame received yet - click 'Start Webcam' in the browser component")
                total_duration = float('inf')

            elif use_direct_webcam:
                # Direct webcam mode - use V4L2/ffmpeg
                self.engine.stream_handler.start_stream_capture(video_path)

                # Check if stream opened successfully (for non-ffmpeg mode)
                if not self.engine.stream_handler._use_ffmpeg_capture:
                    if not self.engine.stream_handler._cap or not self.engine.stream_handler._cap.isOpened():
                        logging.error("Failed to open webcam stream")
                        yield (
                            None,
                            "**Error**",
                            "Failed to open webcam",
                            "Error: Could not open webcam. Try 'Browser Webcam' mode instead.",
                            "",
                            self.log_capture.get_logs()
                        )
                        return

                logging.info(f"Direct webcam stream started (FPS: {self.engine.stream_handler._video_fps:.2f})")
                total_duration = float('inf')  # Infinite duration for webcam
            else:
                # For video files, open file
                if not self.engine.stream_handler.open_video_file(video_path):
                    logging.error("Failed to open video source")
                    yield (
                        video_path,
                        "**Error**",
                        "Failed to open video",
                        "Error: Could not open video source",
                        "",
                        self.log_capture.get_logs()
                    )
                    return

                logging.info(f"Video opened: duration={self.engine.stream_handler.total_duration:.2f}s")
                total_duration = self.engine.stream_handler.total_duration

            poll_index = 0
            motion_idle_streak = 0  # consecutive below-threshold polls (motion gate hysteresis)

            # Yield initial state with video loaded
            if use_webcam:
                timestamp_display = "🟢 **Live Webcam** — ready to analyze"
                initial_video = None  # No video player for webcam
            else:
                timestamp_display = f"🟡 **Analyzing video** — 0:00 / {int(total_duration//60)}:{int(total_duration%60):02d}"
                initial_video = video_path

            yield (
                initial_video,
                timestamp_display,
                "Starting polling...",
                "Initializing...",
                "",
                self.log_capture.get_logs()
            )

            # COLD-START GUARD (webcam mode only):
            # Wait for the buffer to hold a full window of real frames before
            # the first inference. Otherwise the first poll runs on a partially-
            # filled buffer (fewer real frames + zero-padding) and produces a
            # weak/confused response. Show a friendly countdown so the user knows
            # the system is alive while it warms up.
            if use_webcam and self.engine.stream_handler is not None:
                target_frames = config.num_frames
                expected_seconds = target_frames / max(config.fps, 1)
                warmup_start = time.time()
                warmup_timeout = max(expected_seconds * 2.0, 6.0)  # safety cap

                while self.is_running:
                    have = len(self.engine.stream_handler.frame_buffer)
                    elapsed = time.time() - warmup_start
                    if have >= target_frames:
                        break
                    if elapsed > warmup_timeout:
                        logging.warning(
                            f"Cold-start timeout — proceeding with {have}/{target_frames} frames"
                        )
                        break
                    yield (
                        None,                 # video player (webcam)
                        timestamp_display,
                        f"🎬 **Warming up camera...** {have}/{target_frames} frames "
                        f"buffered ({elapsed:.1f}/{expected_seconds:.0f}s)",
                        f"Filling buffer at {config.fps} fps...",
                        "",
                        self.log_capture.get_logs(),
                    )
                    time.sleep(0.25)

            while self.is_running:
                # Check if video exhausted (only for video files)
                if not use_webcam and self.engine.stream_handler.current_position >= total_duration:
                    break

                # Update progress (only for video files)
                if not use_webcam:
                    position = self.engine.stream_handler.current_position
                    progress_pct = 0.3 + (position / total_duration) * 0.6
                    progress(progress_pct, desc=f"Poll #{poll_index + 1} at {position:.1f}s / {total_duration:.1f}s")
                else:
                    # For webcam, just calculate elapsed time without progress bar update
                    session_start = self.engine.metrics.current_session.start_time if self.engine.metrics.current_session else time.time()
                    position = time.time() - session_start

                # MOTION GATE (Tier 1, optional — MOTION_GATE=1):
                # On a live webcam, skip the whole VLM when the scene is static
                # (empty room / person standing idle) so the model never narrates
                # to an empty stage. Checked before metrics/inference start, so a
                # gated tick consumes no poll number and no compute. Direct-webcam
                # only: browser-webcam replicates a single frame (no motion to
                # measure) and video files don't use the live buffer.
                #
                # Hysteresis: a single low-motion poll (slow part of a rep, brief
                # pause) must NOT flip the UI to "waiting" mid-exercise. Only go
                # idle after `motion_idle_polls` consecutive below-threshold polls;
                # any active poll resets the streak.
                if config.enable_motion_gate and use_direct_webcam:
                    motion = self.engine.stream_handler.compute_motion_score(config.num_frames)
                    if motion < config.motion_threshold:
                        motion_idle_streak += 1
                    else:
                        motion_idle_streak = 0

                    if motion_idle_streak >= config.motion_idle_polls:
                        logging.info(
                            f"Motion gate: score {motion:.2f} < {config.motion_threshold:.2f} "
                            f"for {motion_idle_streak} polls — idle scene, skipping"
                        )
                        yield (
                            None,
                            "⏸ **Waiting for exercise**",
                            "⏸ **Waiting for exercise…**",
                            f"Idle — motion {motion:.1f} < {config.motion_threshold:.1f}",
                            self.format_all_responses(self.poll_results),
                            self.log_capture.get_logs(),
                        )
                        time.sleep(config.polling_interval)
                        continue

                # Start metrics
                self.engine.metrics.start_inference(poll_index)

                try:
                    # Extract frames - different paths for different modes
                    if use_browser_webcam:
                        # Browser webcam mode - get frames from browser
                        if self._browser_frame is not None:
                            # Convert browser frame to model input
                            # Browser frame is a numpy array (H, W, 3) in RGB
                            frame = self._browser_frame
                            if isinstance(frame, np.ndarray) and len(frame.shape) == 3:
                                # Convert to PIL Image for processing
                                pil_frame = Image.fromarray(frame)

                                # Replicate single frame to create video input
                                raw_frames = [pil_frame] * config.num_frames
                                context_frames_raw = [pil_frame] * config.num_context_images

                                # Process for video encoder
                                video_frames = self.engine.video_processor.preprocess(raw_frames)['pixel_values']

                                # Process for image encoder
                                context_frames = [
                                    self.engine.image_processor.preprocess(f, return_tensors='pt')['pixel_values'][0]
                                    for f in context_frames_raw
                                ]

                                slice_len = 1  # Single frame from browser
                            else:
                                logging.warning(f"Invalid browser frame format: type={type(frame)}, shape={getattr(frame, 'shape', 'N/A')}")
                                video_frames, context_frames, slice_len = [], [], 0
                        else:
                            logging.debug("Browser frame is None - waiting for webcam stream")
                            video_frames, context_frames, slice_len = [], [], 0
                    else:
                        # Video file or direct webcam mode - use stream handler
                        video_frames, context_frames, slice_len = self.engine.stream_handler.get_frames_for_inference(
                            self.engine.image_processor,
                            self.engine.video_processor,
                            num_video_frames=config.num_frames,
                            num_context_images=config.num_context_images,
                            polling_interval=config.polling_interval,
                        )

                    if slice_len == 0:
                        # Log and skip this poll, but continue to next position
                        if use_browser_webcam:
                            logging.warning(f"Poll #{poll_index + 1}: No browser frame available, skipping")
                        else:
                            buffer_size = len(self.engine.stream_handler.frame_buffer) if self.engine.stream_handler else 0
                            logging.warning(f"Poll #{poll_index + 1}: No frames extracted (buffer size: {buffer_size}), skipping")
                        poll_index += 1
                        time.sleep(config.polling_interval)
                        continue

                    # Run inference with token streaming — yield partial responses to
                    # the UI as they're generated. Keeps the panel's "real-time feel"
                    # even when total generation takes ~10s.
                    response = ""
                    ttft = 0.0
                    input_tokens = 0
                    output_tokens = 0
                    partial_for_ui = ""
                    is_repeat = False
                    repeat_info = ""
                    display_response = ""

                    # Pre-compute everything the UI needs that doesn't depend on response
                    if use_webcam:
                        elapsed_min_pre = int(position // 60)
                        elapsed_sec_pre = int(position % 60)
                        timestamp_pre = f"🟡 **Live Webcam** — analyzing · {elapsed_min_pre}:{elapsed_sec_pre:02d} elapsed · poll #{poll_index + 1}"
                        # Don't push numpy frames to gr.Video (it expects file paths).
                        # Sending None keeps the video player on the previous segment.
                        video_display_pre = None
                    else:
                        current_min_pre = int(position // 60)
                        current_sec_pre = int(position % 60)
                        total_min_pre = int(total_duration // 60)
                        total_sec_pre = int(total_duration % 60)
                        timestamp_pre = f"🟡 **Analyzing video** — {current_min_pre}:{current_sec_pre:02d} / {total_min_pre}:{total_sec_pre:02d} · poll #{poll_index + 1}"
                        video_display_pre = self.extract_video_segment(
                            self.current_video_path, position, config.polling_interval
                        )

                    # Immediate first update — UI transitions from previous state to
                    # "processing" the moment the poll starts (before any tokens).
                    yield (
                        video_display_pre,
                        timestamp_pre,
                        f"**Poll #{poll_index + 1}** (Position: {position:.2f}s) 🎬 Starting analysis...",
                        "Preparing — first word arrives in ~9s",
                        self.format_all_responses(self.poll_results),
                        self.log_capture.get_logs(),
                    )

                    # Progress stages shown during prefill (no tokens yet)
                    progress_stages = [
                        (0.0,  "🎬 Extracting & preprocessing 16 frames..."),
                        (1.5,  "👁️ CLIP encoding context images..."),
                        (5.0,  "🎞️ VideoMamba processing video frames..."),
                        (7.0,  "🧠 Qwen2 analyzing movement patterns..."),
                        (9.0,  "💬 Generating coaching feedback..."),
                    ]

                    # Stream generation — receive heartbeats during prefill (None token),
                    # then actual tokens once decoding starts.
                    stream = self.engine.run_single_inference_streaming(
                        video_frames, context_frames, prompt, slice_len
                    )
                    for partial_text, is_final, elapsed, final_metrics in stream:
                        if is_final:
                            response = partial_text
                            if final_metrics is not None:
                                ttft = final_metrics.get("ttft", 0.0)
                                input_tokens = final_metrics.get("input_tokens", 0)
                                output_tokens = final_metrics.get("output_tokens", 0)
                            break

                        if partial_text is None:
                            # Heartbeat during prefill — pick the latest progress stage
                            stage_msg = progress_stages[0][1]
                            for t_stage, msg in progress_stages:
                                if elapsed >= t_stage:
                                    stage_msg = msg
                            current_response_streaming = (
                                f"**Poll #{poll_index + 1}** (Position: {position:.2f}s) "
                                f"{stage_msg}\n\n_Analyzing video — please wait..._"
                            )
                            yield (
                                video_display_pre,
                                timestamp_pre,
                                current_response_streaming,
                                f"{stage_msg}  ({elapsed:.1f}s)",
                                self.format_all_responses(self.poll_results),
                                self.log_capture.get_logs(),
                            )
                            continue

                        # Token arrived — stream it to the UI
                        partial_for_ui = partial_text
                        current_response_streaming = (
                            f"**Poll #{poll_index + 1}** (Position: {position:.2f}s) "
                            f"✍️ writing...\n\n{partial_for_ui}"
                        )
                        yield (
                            video_display_pre,
                            timestamp_pre,
                            current_response_streaming,
                            f"Streaming tokens — {elapsed:.1f}s elapsed",
                            self.format_all_responses(self.poll_results),
                            self.log_capture.get_logs(),
                        )

                    # Post-process the final response
                    if self.naturalizer:
                        nat_result = self.naturalizer.process(response)
                        display_response = nat_result['display']
                        is_repeat = nat_result['is_repeat']
                        repeat_info = f" 🔄 Repeat #{nat_result['repeat_count']}" if is_repeat else " ✨ New"
                    else:
                        display_response = response

                    # Validate token counts (ensure non-negative)
                    input_tokens = max(0, input_tokens) if input_tokens else 0
                    output_tokens = max(0, output_tokens) if output_tokens else 0
                    ttft = max(0.0, ttft) if ttft else 0.0

                    # Record metrics
                    metrics_obj = self.engine.metrics.end_inference(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        frames_processed=slice_len,
                        buffer_size=0,
                        response=response,
                        time_to_first_token=ttft,
                        naturalizer_response=display_response if self.naturalizer else ""
                    )

                    # Convert to dict for display
                    metrics = {
                        'latency_ms': metrics_obj.total_inference_time * 1000,
                        'ttft_ms': metrics_obj.time_to_first_token * 1000,
                        'tokens_per_second': metrics_obj.tokens_per_second,
                        'frames_processed': metrics_obj.frames_processed,
                        'output_tokens': metrics_obj.output_tokens
                    }

                    # Store results
                    result = {
                        'poll': poll_index + 1,
                        'position': f"{position:.2f}",
                        'response': response,
                        'metrics': metrics
                    }
                    self.poll_results.append(result)
                    self.metrics_history.append(metrics)

                    # Log poll completion
                    logging.info(f"Poll #{poll_index + 1} complete: latency={metrics.get('latency_ms', 0):.1f}ms{repeat_info}")

                    # Format outputs
                    current_response = f"**Poll #{poll_index + 1}** (Position: {position:.2f}s){repeat_info}\n\n{display_response}"
                    current_metrics = self.format_metrics(metrics)
                    all_responses = self.format_all_responses(self.poll_results)

                    # Format timestamp
                    if use_webcam:
                        elapsed_min = int(position // 60)
                        elapsed_sec = int(position % 60)
                        timestamp = f"🟢 **Live Webcam** — {elapsed_min}:{elapsed_sec:02d} elapsed · poll #{poll_index + 1} ready"
                        segment_path = None  # No video segment for webcam
                    else:
                        current_min = int(position // 60)
                        current_sec = int(position % 60)
                        total_min = int(total_duration // 60)
                        total_sec = int(total_duration % 60)
                        timestamp = f"🟢 **Video** — {current_min}:{current_sec:02d} / {total_min}:{total_sec:02d} · poll #{poll_index + 1} ready"

                        # Extract video segment for this poll (video files only)
                        segment_path = self.extract_video_segment(
                            self.current_video_path,
                            position,
                            config.polling_interval
                        )

                    # For webcam, skip video update (gr.Video can't take numpy arrays).
                    # The video player keeps showing the previous segment / stays empty.
                    video_display = None if use_webcam else segment_path

                    yield (
                        video_display,
                        timestamp,
                        current_response,
                        current_metrics,
                        all_responses,
                        self.log_capture.get_logs()
                    )

                    poll_index += 1

                    # Wait for next poll
                    if poll_index < 100:  # Safety limit
                        time.sleep(config.polling_interval)

                except Exception as e:
                    logging.error(f"Error in poll #{poll_index + 1}: {str(e)}")

                    if use_webcam:
                        elapsed_min = int(position // 60) if 'position' in locals() else 0
                        elapsed_sec = int(position % 60) if 'position' in locals() else 0
                        timestamp = f"❌ **Error** at {elapsed_min}:{elapsed_sec:02d}"
                        error_video = None
                    else:
                        current_min = int(position // 60) if 'position' in locals() else 0
                        current_sec = int(position % 60) if 'position' in locals() else 0
                        total_min = int(total_duration // 60)
                        total_sec = int(total_duration % 60)
                        timestamp = f"❌ **Error** at {current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}"
                        error_video = video_path

                    yield (
                        error_video,
                        timestamp,
                        f"Error in poll #{poll_index + 1}: {str(e)}",
                        "Error occurred",
                        self.format_all_responses(self.poll_results),
                        self.log_capture.get_logs()
                    )
                    break

            # Final summary
            progress(1.0, desc="Complete!")
            logging.info(f"Polling complete: {poll_index} polls processed")

            # Stop webcam stream if active
            if use_webcam:
                self.engine.stream_handler.stop_stream()
                logging.info("Webcam stream stopped")

            # End metrics session and save
            if self.engine:
                summary = self.engine.metrics.end_session()
                logging.info("Metrics and summary saved successfully")

            if use_webcam:
                timestamp = f"✅ **Complete** — {poll_index} polls from webcam"
                final_video = None
            else:
                total_min = int(total_duration // 60)
                total_sec = int(total_duration % 60)
                timestamp = f"✅ **Complete** — {total_min}:{total_sec:02d} / {total_min}:{total_sec:02d} · {poll_index} polls"
                final_video = video_path

            # For webcam mode, gr.Video can't take numpy arrays — leave empty.
            # For video file mode, show the file.
            final_display = None if use_webcam else final_video

            yield (
                final_display,
                timestamp,
                f"**Polling Complete**\n\nProcessed {poll_index} polls successfully",
                f"**Final Stats:**\n{poll_index} polls completed",
                self.format_all_responses(self.poll_results),
                self.log_capture.get_logs()
            )

        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")

            # End session even on error
            if self.engine:
                try:
                    self.engine.metrics.end_session()
                except Exception as metrics_error:
                    logging.error(f"Failed to save metrics: {metrics_error}")

            yield (
                video_path if 'video_path' in locals() else None,
                "❌ **Fatal Error**",
                f"**Error:** {str(e)}",
                "Error occurred during inference",
                "",
                self.log_capture.get_logs()
            )

        finally:
            self.is_running = False

            # Stop webcam stream if active
            if self.engine and hasattr(self.engine, 'stream_handler'):
                try:
                    if hasattr(self.engine.stream_handler, '_is_running') and self.engine.stream_handler._is_running:
                        self.engine.stream_handler.stop_stream()
                        logging.info("Webcam stream stopped in cleanup")
                    self.engine.stream_handler.close()
                except Exception as e:
                    logging.error(f"Error closing stream: {e}")

                # Clean up metrics log handlers
                try:
                    self.engine.metrics.cleanup()
                except Exception as e:
                    logging.error(f"Error cleaning up metrics: {e}")

            # Clean up temp directory
            if self.temp_dir and os.path.exists(self.temp_dir):
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    logging.warning(f"Failed to clean up temp directory: {e}")

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def stop_inference(self):
        """Stop current inference"""
        self.is_running = False

        # Stop webcam stream if active
        if self.engine and hasattr(self.engine, 'stream_handler'):
            try:
                if hasattr(self.engine.stream_handler, '_is_running') and self.engine.stream_handler._is_running:
                    self.engine.stream_handler.stop_stream()
                    logging.info("Webcam stream stopped via stop button")
            except Exception as e:
                logging.error(f"Error stopping stream: {e}")

        return "Stopping inference..."


def create_interface():
    """Create Gradio interface."""
    app = GradioPollingApp()

    available_cameras = app.get_available_cameras()
    camera_labels = [name for name, _ in available_cameras]
    sample_videos = app.get_sample_videos()

    custom_theme = gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.violet,
        neutral_hue=gr.themes.colors.slate,
    ).set(
        body_background_fill="#0b0d12",
        body_background_fill_dark="#0b0d12",
        body_text_color="#e8eaed",
        body_text_color_dark="#e8eaed",
        background_fill_primary="#11141a",
        background_fill_primary_dark="#11141a",
        background_fill_secondary="#161a22",
        background_fill_secondary_dark="#161a22",
        block_background_fill="#11141a",
        block_background_fill_dark="#11141a",
        block_border_width="1px",
        block_border_color="#252a35",
        block_border_color_dark="#252a35",
        block_label_text_color="#9ca3af",
        block_label_text_color_dark="#9ca3af",
        block_title_text_color="#e8eaed",
        block_title_text_color_dark="#e8eaed",
        input_background_fill="#161a22",
        input_background_fill_dark="#161a22",
        input_border_color="#252a35",
        input_border_color_dark="#252a35",
        button_primary_background_fill="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
        button_primary_background_fill_dark="linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%)",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        button_secondary_background_fill="#1f242e",
        button_secondary_background_fill_dark="#1f242e",
        button_secondary_text_color="#e8eaed",
        button_secondary_text_color_dark="#e8eaed",
        border_color_accent="#4f46e5",
        border_color_primary="#252a35",
        border_color_primary_dark="#252a35",
    )

    PAGE_CSS = """
    .gradio-container, .gradio-container > .main {
        max-width: 1240px !important;
        margin: 24px auto !important;
        padding: 0 24px !important;
    }
    body, .gradio-container { background: #0b0d12 !important; }
    footer { display: none !important; }

    #status-badge {
        padding: 14px 20px !important;
        background: #11141a !important;
        border: 1px solid #252a35 !important;
        border-radius: 12px !important;
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #e8eaed !important;
        margin-bottom: 18px !important;
    }
    #status-badge * { color: #e8eaed !important; background: transparent !important; }
    #status-badge p { margin: 0 !important; }

    #preview-card {
        border: 1px solid #252a35 !important;
        border-radius: 14px !important;
        padding: 8px !important;
        background: #0d1015 !important;
        overflow: hidden !important;
        margin-bottom: 14px !important;
    }
    #preview-card > * { background: transparent !important; }
    #preview-card video, #preview-card img { border-radius: 10px !important; }

    #feedback-card {
        border: 1px solid #252a35 !important;
        border-radius: 14px !important;
        padding: 0 !important;
        background: linear-gradient(180deg, #161a22 0%, #11141a 100%) !important;
        min-height: 460px;
        overflow: hidden;
    }
    #response-md {
        padding: 22px 24px !important;
        font-size: 15px !important;
        line-height: 1.65 !important;
        color: #e8eaed !important;
        background: transparent !important;
    }
    #response-md * { color: #e8eaed !important; background: transparent !important; }

    #metrics-card, #history-card {
        padding: 14px 16px !important;
        border: 1px solid #252a35 !important;
        border-radius: 10px !important;
        background: #161a22 !important;
        color: #d4d4d4 !important;
        min-height: 90px;
        max-height: 320px;
        overflow-y: auto !important;
    }
    #metrics-card *, #history-card * { color: #d4d4d4 !important; background: transparent !important; }

    .tab-nav, div[role="tablist"] {
        background: transparent !important;
        border-bottom: 1px solid #252a35 !important;
        gap: 4px !important;
        padding: 4px !important;
    }
    button[role="tab"] {
        font-weight: 600 !important;
        color: #9ca3af !important;
        background: transparent !important;
        border: none !important;
        padding: 10px 18px !important;
        border-radius: 8px !important;
        transition: all 0.15s ease !important;
    }
    button[role="tab"]:hover {
        color: #e8eaed !important;
        background: rgba(99, 102, 241, 0.1) !important;
    }
    button[role="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: linear-gradient(135deg, rgba(99,102,241,0.25), rgba(139,92,246,0.25)) !important;
        box-shadow: inset 0 -2px 0 #6366f1 !important;
    }

    button.primary, button.lg.primary, .controls-row button {
        height: 48px !important;
        font-weight: 700 !important;
        letter-spacing: 0.4px !important;
        border-radius: 10px !important;
    }

    /* radio/checkbox excluded so the native selected dot stays visible */
    input:not([type="radio"]):not([type="checkbox"]), select, textarea {
        background: #161a22 !important;
        color: #e8eaed !important;
        border-color: #252a35 !important;
    }
    input[type="radio"], input[type="checkbox"] {
        accent-color: #8b5cf6 !important;
        width: 18px !important;
        height: 18px !important;
        cursor: pointer;
    }

    fieldset.gr-radio, .gr-radio { border: none !important; }
    .gr-radio label, .gr-form label.svelte {
        background: #161a22 !important;
        border: 1px solid #252a35 !important;
        border-radius: 10px !important;
        padding: 10px 14px !important;
        margin-right: 8px !important;
        transition: border-color 0.15s, background 0.15s;
    }
    .gr-radio label:has(input:checked),
    .gr-form label.svelte:has(input:checked) {
        border-color: #8b5cf6 !important;
        background: rgba(139, 92, 246, 0.12) !important;
    }

    .block > label > span, .block > .label-wrap > .label-text {
        color: #9ca3af !important;
        font-weight: 600 !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .logs-box textarea {
        font-family: 'JetBrains Mono', 'Courier New', monospace !important;
        font-size: 12px !important;
        background: #0a0c10 !important;
        color: #c4c4c4 !important;
        border-radius: 10px !important;
        border: 1px solid #252a35 !important;
    }
    """

    with gr.Blocks(
        title="Mobile-VideoGPT — Real-time Exercise Coach",
        theme=custom_theme,
        css=PAGE_CSS,
    ) as demo:
        gr.HTML(
            """
            <div style="
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 24px 28px;
                background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%);
                border-radius: 18px;
                box-shadow: 0 14px 36px rgba(79, 70, 229, 0.32);
                margin-bottom: 18px;
                color: #ffffff;
                font-family: 'Inter', system-ui, -apple-system, sans-serif;
                flex-wrap: wrap;
                gap: 16px;
            ">
              <div style="display: flex; align-items: center; gap: 16px;">
                <div style="
                    width: 48px; height: 48px;
                    border-radius: 14px;
                    background: rgba(255,255,255,0.18);
                    border: 1px solid rgba(255,255,255,0.28);
                    display: flex; align-items: center; justify-content: center;
                    font-weight: 800; font-size: 17px; letter-spacing: 1px; color: #fff;
                    box-shadow: 0 4px 12px rgba(0,0,0,0.18);
                ">MV</div>
                <div>
                  <div style="font-size: 22px; font-weight: 800; line-height: 1.1; color: #fff;">
                    Mobile-VideoGPT
                  </div>
                  <div style="font-size: 13px; opacity: 0.92; margin-top: 4px; color: #fff;">
                    Real-time exercise form coach · Jetson Orin Nano Super
                  </div>
                </div>
              </div>
              <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                <span style="padding: 6px 12px; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.24); border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #fff;">0.5B PARAMS</span>
                <span style="padding: 6px 12px; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.24); border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #fff;">VIDEOMAMBA + QWEN2</span>
                <span style="padding: 6px 12px; background: rgba(255,255,255,0.16); border: 1px solid rgba(255,255,255,0.24); border-radius: 999px; font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #fff;">FP16</span>
              </div>
            </div>
            """
        )

        with gr.Group():
            video_timestamp = gr.Markdown(
                value="⏸ &nbsp;**Idle** — choose a source and press **Start**",
                elem_id="status-badge",
            )

        with gr.Row(equal_height=False):
            with gr.Column(scale=3, min_width=460):
                webcam_mode = gr.Radio(
                    choices=["Video File", "Browser Webcam", "Direct Webcam (Linux only)"],
                    value="Video File",
                    label="INPUT SOURCE",
                )

                gr.HTML(
                    '<div style="font-size: 11px; font-weight: 700; letter-spacing: 1.4px; '
                    'color: #9ca3af; text-transform: uppercase; margin: 14px 2px 8px;">Preview</div>'
                )
                with gr.Group(elem_id="preview-card"):
                    video_player = gr.Video(
                        label="Video",
                        autoplay=True,
                        loop=True,
                        show_label=False,
                        height=400,
                        visible=True,
                    )
                    browser_webcam = gr.Image(
                        sources=["webcam"],
                        type="numpy",
                        label="Browser Webcam",
                        visible=False,
                        streaming=True,
                        show_label=False,
                        height=400,
                    )
                    webcam_preview = gr.Image(
                        value=app.get_latest_webcam_frame,
                        every=0.5,
                        label="Live Webcam",
                        show_label=False,
                        height=400,
                        interactive=False,
                        visible=False,
                    )

                with gr.Row():
                    start_btn = gr.Button("▶  Start", variant="primary", scale=2)
                    stop_btn = gr.Button("■  Stop", variant="secondary", scale=1)

            with gr.Column(scale=2, min_width=320):
                with gr.Row(elem_id="feedback-header"):
                    gr.HTML(
                        '<div style="font-size: 11px; font-weight: 700; letter-spacing: 1.4px; '
                        'color: #9ca3af; text-transform: uppercase; margin: 0 2px 8px; flex: 1;">'
                        'Coaching feedback</div>'
                    )
                    voice_enabled = gr.Checkbox(
                        label="🔊 Voice",
                        value=True,
                        elem_id="voice-toggle",
                        scale=0,
                        min_width=110,
                    )
                with gr.Group(elem_id="feedback-card"):
                    current_response = gr.Markdown(
                        value="*Waiting for first poll. Pick a source on the left and press Start.*",
                        elem_id="response-md",
                    )

        gr.HTML(
            '<div style="font-size: 11px; font-weight: 700; letter-spacing: 1.4px; '
            'color: #9ca3af; text-transform: uppercase; margin: 24px 2px 8px;">Advanced</div>'
        )
        with gr.Tabs():
            with gr.Tab("Source"):
                video_dropdown = gr.Dropdown(
                    choices=sample_videos,
                    label="Sample video",
                    value=sample_videos[0] if sample_videos else None,
                    info="Videos from sample_videos/",
                    visible=True,
                )
                camera_selector = gr.Dropdown(
                    choices=camera_labels,
                    label="Camera device (Direct mode)",
                    value=camera_labels[0] if camera_labels else None,
                    info="Native Linux V4L2 only",
                    visible=False,
                )
                gr.HTML(
                    '<div style="font-size: 12px; color: #9ca3af; margin-top: 10px; '
                    'padding: 12px 14px; background: #161a22; border-radius: 8px; '
                    'border-left: 3px solid #6366f1;">'
                    '<strong style="color: #c7d2fe;">Browser Webcam</strong> works on WSL2 / remote browsers. '
                    '<strong style="color: #c7d2fe;">Direct Webcam</strong> is for native Linux V4L2.'
                    '</div>'
                )

            with gr.Tab("Model"):
                with gr.Row():
                    base_model = gr.Dropdown(
                        choices=[
                            "Amshaker/Mobile-VideoGPT-0.5B",
                            "Amshaker/Mobile-VideoGPT-1.5B",
                        ],
                        label="Base model",
                        value="Amshaker/Mobile-VideoGPT-0.5B",
                        info="0.5B is the deployed default on Jetson",
                    )
                    lora_weights = gr.Dropdown(
                        choices=[
                            "EdgeVLM-Labs/mobile-videogpt-finetune-v2-mixed",
                            "EdgeVLM-Labs/mobile-videogpt-finetune-2000",
                            "EdgeVLM-Labs/qved-finetune-20260110_155349",
                        ],
                        label="LoRA adapter",
                        value="EdgeVLM-Labs/mobile-videogpt-finetune-v2-mixed",
                    )

            with gr.Tab("Inference"):
                with gr.Row():
                    polling_interval = gr.Slider(
                        minimum=1, maximum=10, value=3, step=0.5,
                        label="Polling interval (s)",
                    )
                    num_frames = gr.Slider(
                        minimum=8, maximum=32, value=16, step=8,
                        label="Frames per poll",
                    )
                with gr.Row():
                    fps = gr.Slider(
                        minimum=1, maximum=30, value=1, step=1,
                        label="Sample FPS",
                        info="1 for files · auto-set to 30 for webcam",
                    )
                    max_new_tokens = gr.Slider(
                        minimum=32, maximum=256, value=64, step=32,
                        label="Max new tokens",
                    )
                with gr.Row():
                    warmup_runs = gr.Slider(
                        minimum=0, maximum=5, value=1, step=1,
                        label="Warmup runs",
                    )
                prompt = gr.Textbox(
                    label="Prompt",
                    value="Watch the video. Identify the exercise and give short feedback on the form.",
                    lines=3,
                )

            with gr.Tab("Naturalizer"):
                use_naturalizer = gr.Checkbox(
                    label="Enable feedback naturalizer",
                    value=False,
                    info="Detect repeated feedback and rephrase",
                )
                naturalizer_threshold = gr.Slider(
                    minimum=0.5, maximum=0.95, value=0.70, step=0.05,
                    label="Similarity threshold",
                    info="Higher = stricter repeat detection",
                )

            with gr.Tab("Metrics"):
                gr.HTML(
                    '<div style="font-size: 12px; color: #9ca3af; font-weight: 600; '
                    'margin: 4px 2px 6px 2px;">Current poll</div>'
                )
                current_metrics = gr.Markdown(
                    value="*No metrics yet.*",
                    elem_id="metrics-card",
                )
                gr.HTML(
                    '<div style="font-size: 12px; color: #9ca3af; font-weight: 600; '
                    'margin: 14px 2px 6px 2px;">Session history</div>'
                )
                all_responses = gr.Markdown(
                    value="*No responses yet.*",
                    elem_id="history-card",
                )

            with gr.Tab("Logs"):
                live_logs = gr.Textbox(
                    value="No logs yet",
                    lines=18,
                    max_lines=24,
                    elem_classes=["logs-box"],
                    interactive=False,
                    show_label=False,
                )

        def toggle_source_mode(mode):
            is_browser = mode == "Browser Webcam"
            is_direct = mode == "Direct Webcam (Linux only)"
            is_file = mode == "Video File"
            return (
                gr.update(visible=is_file),         # video_player
                gr.update(visible=is_browser),      # browser_webcam
                gr.update(visible=is_direct),       # webcam_preview
                gr.update(visible=is_file),         # video_dropdown
                gr.update(visible=is_direct),       # camera_selector
                gr.update(value=4 if (is_browser or is_direct) else 1),  # fps (webcam: 16 frames / 4 fps = 4s window, matches training)
            )

        webcam_mode.change(
            fn=toggle_source_mode,
            inputs=[webcam_mode],
            outputs=[
                video_player, browser_webcam, webcam_preview,
                video_dropdown, camera_selector, fps,
            ],
        )

        browser_webcam.stream(
            fn=app.update_browser_frame,
            inputs=[browser_webcam],
            outputs=[],
        )

        start_btn.click(
            fn=app.run_inference,
            inputs=[
                video_dropdown, webcam_mode, camera_selector, browser_webcam,
                base_model, lora_weights,
                polling_interval, num_frames, fps, max_new_tokens, warmup_runs,
                prompt, use_naturalizer, naturalizer_threshold,
            ],
            outputs=[
                video_player, video_timestamp, current_response,
                current_metrics, all_responses, live_logs,
            ],
        )

        stop_btn.click(fn=app.stop_inference, outputs=current_response)

        TTS_JS = r"""
        (text, enabled) => {
            if (!enabled) {
                if (window.speechSynthesis) window.speechSynthesis.cancel();
                window._mvg_lastPollId = '';
                return;
            }
            if (!text || !window.speechSynthesis) return;

            const skipMarkers = [
                'writing...', 'Starting analysis', 'Extracting',
                'CLIP encoding', 'VideoMamba', 'Qwen2 analyzing',
                'Generating coaching', 'Warming up camera',
                'Analyzing video', 'Initializing',
                'Starting polling', 'Waiting for first poll',
                'Waiting to start', 'Polling Complete',
            ];
            for (const m of skipMarkers) {
                if (text.indexOf(m) !== -1) return;
            }

            // Dedup by poll id, not by speech text — back-to-back polls with
            // identical bodies must still be spoken.
            const pollIdMatch = text.match(/Poll #\d+\s*\(Position:\s*[\d.]+s\)/);
            const pollId = pollIdMatch ? pollIdMatch[0] : text.slice(0, 60);
            if (pollId === window._mvg_lastPollId) return;
            window._mvg_lastPollId = pollId;

            let body = text;
            const idx = text.indexOf('\n\n');
            if (idx >= 0) body = text.slice(idx + 2);

            const speech = body
                .replace(/\[(.*?)\]\(.*?\)/g, '$1')
                .replace(/[*_`#>]/g, '')
                .replace(/\s+/g, ' ')
                .trim();
            if (!speech) return;

            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance(speech);
            u.rate = 1.05;
            u.pitch = 1.0;
            u.volume = 1.0;
            const voices = window.speechSynthesis.getVoices();
            const en = voices.find(v => /en[-_]/i.test(v.lang) && /samantha|google|jenny|aria|zira|natural/i.test(v.name))
                    || voices.find(v => /en[-_]/i.test(v.lang));
            if (en) u.voice = en;
            window.speechSynthesis.speak(u);

            // Chrome cuts off utterances >~15s without a periodic pause/resume.
            if (!window._mvg_keepAlive) {
                window._mvg_keepAlive = setInterval(() => {
                    if (window.speechSynthesis.speaking && !window.speechSynthesis.paused) {
                        window.speechSynthesis.pause();
                        window.speechSynthesis.resume();
                    }
                }, 10000);
            }
        }
        """
        current_response.change(
            fn=None,
            inputs=[current_response, voice_enabled],
            outputs=[],
            js=TTS_JS,
        )

        voice_enabled.change(
            fn=None,
            inputs=[voice_enabled],
            outputs=[],
            js=r"""
            (enabled) => {
                if (!enabled && window.speechSynthesis) {
                    window.speechSynthesis.cancel();
                    window._mvg_lastPollId = '';
                }
            }
            """,
        )

        gr.HTML(
            "<script>"
            "if (window.speechSynthesis) {"
            "  window.speechSynthesis.getVoices();"
            "  window.speechSynthesis.onvoiceschanged = () => window.speechSynthesis.getVoices();"
            "}"
            "</script>"
        )

    return demo


if __name__ == "__main__":
    # Show detected cameras at startup
    print("\n" + "="*60)
    print("Mobile-VideoGPT Polling Inference")
    print("="*60)

    app_instance = GradioPollingApp()
    cameras = app_instance.get_available_cameras()

    if cameras:
        print("\n Detected Cameras:")
        for name, idx in cameras:
            print(f"  • {name}")
    else:
        print("\n  No cameras detected via /dev/v4l/by-id/")
        print("   Will show generic camera indices")

    demo = create_interface()
    # share=True publishes a public *.gradio.live tunnel URL that works from any
    # device on any network (needs internet on the Jetson) — useful when the demo
    # WiFi uses client/AP isolation or a firewall blocks port 7860. Opt-in via
    # GRADIO_SHARE=1; default stays LAN-only on 0.0.0.0:7860.
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=share,
        show_error=True
    )
