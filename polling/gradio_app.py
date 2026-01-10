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
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
import threading
import queue
from io import StringIO

import gradio as gr
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
from polling.stream_handler import VideoStreamHandler
from polling.metrics import MetricsTracker


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
        self.is_running = False
        self.current_session_id = None
        self.poll_results = []
        self.metrics_history = []
        self.log_capture = LogCapture()
        self.log_capture.setLevel(logging.INFO)
        self.log_capture.setFormatter(logging.Formatter('%(asctime)s | %(levelname)s | %(message)s'))

        # Add log capture to root logger
        logging.getLogger().addHandler(self.log_capture)
        logging.getLogger().setLevel(logging.INFO)

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
        output.append("📊 **Poll Metrics**\n")
        output.append(f"**Latency:** {metrics.get('latency_ms', 0):.1f} ms")
        output.append(f"**TTFT:** {metrics.get('ttft_ms', 0):.1f} ms")
        output.append(f"**Tokens/s:** {metrics.get('tokens_per_second', 0):.1f}")
        output.append(f"**Frames:** {metrics.get('frames_processed', 0)}")
        output.append(f"**Output Tokens:** {metrics.get('output_tokens', 0)}")

        return "\n".join(output)

    def format_session_metrics(self, session_metrics: dict) -> str:
        """Format session-level metrics"""
        if not session_metrics:
            return "No session metrics available"

        output = []
        output.append("📈 **Session Summary**\n")
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
        output.append("💬 **All Poll Responses**\n")
        output.append("=" * 60 + "\n")

        for i, result in enumerate(results, 1):
            output.append(f"**Poll #{i}** (Position: {result.get('position', 'N/A')}s)")
            output.append(f"_{result.get('response', 'No response')}_\n")

        return "\n".join(output)

    def run_inference(
        self,
        video_source: str,
        use_webcam: bool,
        base_model: str,
        lora_weights: str,
        polling_interval: float,
        num_frames: int,
        fps: int,
        max_new_tokens: int,
        warmup_runs: int,
        prompt: str,
        progress=gr.Progress()
    ) -> tuple:
        """Run polling inference"""
        try:
            # Reset state
            self.poll_results = []
            self.metrics_history = []
            self.is_running = True
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Determine video path
            if use_webcam:
                video_path = "0"  # Webcam
                progress(0, desc="Opening webcam...")
            else:
                project_root = Path(__file__).parent.parent
                video_path = str(project_root / "sample_videos" / video_source)

                if not os.path.exists(video_path):
                    yield (
                        video_path,
                        "❌ Video file not found",
                        "Error: Video file does not exist",
                        "",
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

            # Initialize engine
            progress(0.1, desc="Loading model...")
            self.engine = PollingInferenceEngine(config)

            # Load model to initialize processors
            if not self.engine.load_model():
                logging.error("Failed to load model")
                yield (
                    video_path if use_webcam else video_path,
                    "❌ Failed to load model",
                    "Error: Could not load model",
                    "",
                    "",
                    self.log_capture.get_logs()
                )
                return

            # Run warmup
            if warmup_runs > 0:
                progress(0.2, desc=f"Running {warmup_runs} warmup runs...")
                self.engine.warmup(warmup_runs)

            # Start polling
            progress(0.3, desc="Starting polling...")

            # Open video
            if not self.engine.stream_handler.open_video_file(video_path):
                logging.error("Failed to open video source")
                yield (
                    video_path,
                    "❌ Failed to open video",
                    "Error: Could not open video source",
                    "",
                    "",
                    self.log_capture.get_logs()
                )
                return

            logging.info(f"Video opened: duration={self.engine.stream_handler.total_duration:.2f}s")

            poll_index = 0
            total_duration = self.engine.stream_handler.total_duration

            while self.is_running:
                # Check if video exhausted
                if self.engine.stream_handler.current_position >= total_duration:
                    break

                # Update progress
                position = self.engine.stream_handler.current_position
                progress_pct = 0.3 + (position / total_duration) * 0.6
                progress(progress_pct, desc=f"Poll #{poll_index + 1} at {position:.1f}s / {total_duration:.1f}s")

                # Start metrics
                self.engine.metrics.start_inference(poll_index)

                try:
                    # Extract frames
                    video_frames, context_frames, slice_len = self.engine.stream_handler.get_frames_for_inference(
                        self.engine.image_processor,
                        self.engine.video_processor,
                        num_video_frames=config.num_frames,
                        num_context_images=config.num_context_images,
                        polling_interval=config.polling_interval,
                    )

                    if slice_len == 0:
                        break

                    # Run inference
                    response, ttft, input_tokens, output_tokens = self.engine.run_single_inference(
                        video_frames, context_frames, prompt, slice_len
                    )

                    # Record metrics
                    metrics = self.engine.metrics.end_inference(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        frames_processed=slice_len,
                        buffer_size=0
                    )

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
                    logging.info(f"Poll #{poll_index + 1} complete: latency={metrics.get('latency_ms', 0):.1f}ms")

                    # Format outputs
                    current_response = f"**Poll #{poll_index + 1}** (Position: {position:.2f}s)\n\n{response}"
                    current_metrics = self.format_metrics(metrics)
                    all_responses = self.format_all_responses(self.poll_results)

                    # Get session metrics
                    session_metrics = self.engine.metrics.current_session.get_summary()
                    session_summary = self.format_session_metrics(session_metrics)

                    yield (
                        video_path,
                        current_response,
                        current_metrics,
                        all_responses,
                        session_summary,
                        self.log_capture.get_logs()
                    )

                    poll_index += 1

                    # Wait for next poll
                    if poll_index < 100:  # Safety limit
                        time.sleep(config.polling_interval)

                except Exception as e:
                    logging.error(f"Error in poll #{poll_index + 1}: {str(e)}")
                    yield (                        video_path,                        f"❌ Error in poll #{poll_index + 1}: {str(e)}",
                        "Error occurred",
                        self.format_all_responses(self.poll_results),
                        self.format_session_metrics(self.engine.metrics.current_session.get_summary()) if self.engine else "",
                        self.log_capture.get_logs()
                    )
                    break

            # Final summary
            progress(1.0, desc="Complete!")
            logging.info(f"Polling complete: {poll_index} polls processed")

            session_metrics = self.engine.metrics.current_session.get_summary()
            final_summary = self.format_session_metrics(session_metrics)

            yield (
                video_path,
                f"✅ **Polling Complete**\n\nProcessed {poll_index} polls successfully",
                f"**Final Stats:**\n{poll_index} polls completed",
                self.format_all_responses(self.poll_results),
                final_summary,
                self.log_capture.get_logs()
            )

        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")
            yield (
                video_source if 'video_path' not in locals() else video_path,
                f"❌ **Error:** {str(e)}",
                "Error occurred during inference",
                "",
                "",
                self.log_capture.get_logs()
            )

        finally:
            self.is_running = False
            if self.engine:
                self.engine.cleanup()

    def stop_inference(self):
        """Stop current inference"""
        self.is_running = False
        return "Stopping inference..."


def create_interface():
    """Create Gradio interface"""
    app = GradioPollingApp()

    with gr.Blocks(title="Mobile-VideoGPT Polling Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🎬 Mobile-VideoGPT Polling Inference
        Real-time exercise form evaluation with LoRA adapters
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📹 Video Source")

                use_webcam = gr.Checkbox(
                    label="Use Webcam",
                    value=False,
                    info="Check to use webcam instead of video file"
                )

                video_dropdown = gr.Dropdown(
                    choices=app.get_sample_videos(),
                    label="Select Video",
                    value=app.get_sample_videos()[0] if app.get_sample_videos() else None,
                    info="Videos from sample_videos/ folder"
                )

                gr.Markdown("### 🤖 Model Configuration")

                base_model = gr.Textbox(
                    label="Base Model",
                    value="Amshaker/Mobile-VideoGPT-0.5B",
                    info="HuggingFace model ID"
                )

                lora_weights = gr.Textbox(
                    label="LoRA Weights",
                    value="EdgeVLM-Labs/mobile-videogpt-finetune-2000",
                    info="HuggingFace LoRA adapter ID"
                )

                gr.Markdown("### ⚙️ Inference Parameters")

                polling_interval = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=0.5,
                    label="Polling Interval (seconds)",
                    info="Time between polls"
                )

                num_frames = gr.Slider(
                    minimum=8,
                    maximum=32,
                    value=16,
                    step=8,
                    label="Number of Frames",
                    info="Frames per poll (must be multiple of 8)"
                )

                fps = gr.Slider(
                    minimum=1,
                    maximum=30,
                    value=1,
                    step=1,
                    label="FPS",
                    info="Frames per second sampling rate"
                )

                max_new_tokens = gr.Slider(
                    minimum=32,
                    maximum=256,
                    value=64,
                    step=32,
                    label="Max New Tokens",
                    info="Maximum tokens to generate"
                )

                warmup_runs = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=1,
                    step=1,
                    label="Warmup Runs",
                    info="Number of warmup iterations"
                )

                prompt = gr.Textbox(
                    label="Prompt",
                    value="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
                    lines=3,
                    info="Evaluation prompt"
                )

                with gr.Row():
                    start_btn = gr.Button("▶️ Start Polling", variant="primary", size="lg")
                    stop_btn = gr.Button("⏹️ Stop", variant="stop", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### 🎬 Video Player")
                video_player = gr.Video(
                    label="Current Video",
                    autoplay=True,
                    show_label=False,
                    height=300
                )

                gr.Markdown("### 📊 Real-time Results")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Current Poll")
                        current_response = gr.Markdown(
                            value="Waiting to start...",
                            elem_classes=["response-box"]
                        )

                    with gr.Column():
                        gr.Markdown("#### Current Metrics")
                        current_metrics = gr.Markdown(
                            value="No metrics yet",
                            elem_classes=["metrics-box"]
                        )

                gr.Markdown("### 📝 All Responses")
                all_responses = gr.Markdown(
                    value="No responses yet",
                    elem_classes=["all-responses-box"]
                )

                gr.Markdown("### 📈 Session Summary")
                session_summary = gr.Markdown(
                    value="No session data yet",
                    elem_classes=["summary-box"]
                )

                gr.Markdown("### 📋 Live Logs")
                live_logs = gr.Textbox(
                    value="No logs yet",
                    lines=15,
                    max_lines=20,
                    elem_classes=["logs-box"],
                    interactive=False,
                    show_label=False
                )

        # Event handlers
        start_btn.click(
            fn=app.run_inference,
            inputs=[
                video_dropdown,
                use_webcam,
                base_model,
                lora_weights,
                polling_interval,
                num_frames,
                fps,
                max_new_tokens,
                warmup_runs,
                prompt
            ],
            outputs=[
                video_player,
                current_response,
                current_metrics,
                all_responses,
                session_summary,
                live_logs
            ]
        )

        stop_btn.click(
            fn=app.stop_inference,
            outputs=current_response
        )

        # Custom CSS
        demo.css = """
        .response-box, .metrics-box, .all-responses-box, .summary-box {
            min-height: 200px;
            max-height: 400px;
            overflow-y: auto;
            padding: 15px;
            border: 1px solid #333;
            border-radius: 8px;
            background-color: #2d2d2d !important;
        }

        .response-box *, .metrics-box *, .all-responses-box *, .summary-box * {
            color: #e0e0e0 !important;
            background-color: transparent !important;
        }

        .response-box, .metrics-box, .all-responses-box, .summary-box {
            color: #e0e0e0 !important;
        }

        .logs-box {
            font-family: 'Courier New', monospace;
            font-size: 12px;
            background-color: #1e1e1e;
            color: #d4d4d4 !important;
            padding: 10px;
            border-radius: 8px;
            overflow-y: auto;
        }

        .logs-box textarea {
            background-color: #1e1e1e !important;
            color: #d4d4d4 !important;
            font-family: 'Courier New', monospace !important;
        }
        """

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
