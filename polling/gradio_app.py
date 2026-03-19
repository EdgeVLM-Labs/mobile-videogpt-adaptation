#!/usr/bin/env python3
"""Gradio interface for Mobile-VideoGPT polling inference."""

import os
import sys
import time
import glob
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Generator

import gradio as gr
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine
from utils.naturalizer.feedback_naturalizer import FeedbackNaturalizer


class GradioPollingApp:
    def __init__(self):
        self.engine: Optional[PollingInferenceEngine] = None
        self.naturalizer: Optional[FeedbackNaturalizer] = None
        self.is_running = False
        self.current_session_id = None
        self.poll_results = []
        self.metrics_history = []
        self.temp_dir = None
        self.current_video_path = None

        logging.getLogger().setLevel(logging.INFO)

    def extract_video_segment(self, video_path: str, start_time: float, duration: float) -> str:
        """Extract video segment with FFmpeg. Falls back to re-encode if copy fails."""
        try:
            output_path = os.path.join(self.temp_dir, f"segment_{start_time:.1f}s.mp4")

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-i', video_path,
                '-t', str(duration),
                '-c', 'copy',
                '-avoid_negative_ts', 'make_zero',
                '-loglevel', 'error',
                output_path
            ]

            subprocess.run(cmd, check=True, capture_output=True)
            return output_path

        except subprocess.CalledProcessError as e:
            logging.warning(f"FFmpeg copy failed, trying re-encode: {e}")
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', video_path,
                    '-t', str(duration),
                    '-c:v', 'libx264',
                    '-preset', 'ultrafast',
                    '-c:a', 'aac',
                    '-loglevel', 'error',
                    output_path
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                return output_path
            except Exception as e2:
                logging.error(f"Failed to extract segment: {e2}")
                return video_path
        except Exception as e:
            logging.error(f"Error extracting segment: {e}")
            return video_path

    def get_sample_videos(self) -> List[str]:
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

        lines = [
            "**Poll Metrics**",
            f"- **Latency:** {metrics.get('latency_ms', 0):.1f} ms",
            f"- **TTFT:** {metrics.get('ttft_ms', 0):.1f} ms",
            f"- **Tokens/s:** {metrics.get('tokens_per_second', 0):.1f}",
            f"- **Output Tokens:** {metrics.get('output_tokens', 0)}",
            f"- **VRAM:** {metrics.get('vram_allocated_gb', 0):.2f} GB",
        ]
        return "\n".join(lines)

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

        vram = session_metrics.get('vram_gb', {})
        if vram:
            output.append("")
            output.append("**VRAM Usage (GB):**")
            output.append(f"  Avg Allocated: {vram.get('mean_allocated', 0):.2f}")
            output.append(f"  Peak Allocated: {vram.get('peak_allocated', 0):.2f}")
            output.append(f"  Peak Reserved: {vram.get('peak_reserved', 0):.2f}")

        return "\n".join(output)

    def format_all_responses(self, results: List[dict], separate_exercise: bool = False) -> str:
        """Format all poll responses"""
        if not results:
            return "No responses yet"

        output = []
        output.append("**All Poll Responses**\n")
        output.append("=" * 60 + "\n")

        for i, result in enumerate(results, 1):
            response = result.get('response', 'No response')
            if separate_exercise and " - " in response:
                exercise_part, feedback_part = response.split(" - ", 1)
                output.append(f"**Poll #{i}** (Position: {result.get('position', 'N/A')}s) — {exercise_part.strip()}")
                output.append(f"_{feedback_part.strip()}_\n")
            else:
                output.append(f"**Poll #{i}** (Position: {result.get('position', 'N/A')}s)")
                output.append(f"_{response}_\n")

        return "\n".join(output)

    def run_inference(
        self,
        video_source: str,
        use_webcam: bool,
        base_model: str,
        lora_weights: str,
        polling_interval: float,
        fps: int,
        max_new_tokens: int,
        temperature: float,
        warmup_runs: int,
        prompt: str,
        separate_exercise: bool,
        use_naturalizer: bool,
        naturalizer_threshold: float,
        load_4bit: bool,
        progress=gr.Progress()
    ) -> Generator[Tuple[str, str, str, str, str], None, None]:
        try:
            num_frames = 16
            self.poll_results = []
            self.metrics_history = []
            self.is_running = True
            self.current_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Reset UI at start
            yield (
                None,
                "**Analysis Position:** 0:00 / 0:00",
                "Initializing...",
                "No metrics yet",
                "No responses yet",
            )

            if use_naturalizer:
                if not self.naturalizer:
                    logging.info(f"Initializing Feedback Naturalizer (threshold={naturalizer_threshold})")
                    self.naturalizer = FeedbackNaturalizer(threshold=naturalizer_threshold)
                else:
                    logging.info("Reusing existing Feedback Naturalizer")
                    self.naturalizer.reset()
            else:
                if self.naturalizer:
                    try:
                        self.naturalizer.cleanup()
                    except:
                        pass
                self.naturalizer = None

            if self.temp_dir:
                try:
                    import shutil
                    shutil.rmtree(self.temp_dir)
                except:
                    pass
            self.temp_dir = tempfile.mkdtemp(prefix="polling_segments_")

            if use_webcam:
                video_path = "0"
                self.current_video_path = video_path
                progress(0, desc="Opening webcam...")
            else:
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
                    )
                    return

                progress(0, desc=f"Loading video: {video_source}")

            config = PollingConfig(
                base_model_path=base_model,
                lora_weights_path=lora_weights,
                polling_interval=polling_interval,
                num_frames=num_frames,
                fps=fps,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                prompt=prompt,
                load_4bit=load_4bit,
            )

            logging.info(f"Starting inference with video: {video_path}")
            logging.info(f"Config: interval={polling_interval}s, frames={num_frames}, fps={fps}")

            progress(0.1, desc="Loading model...")

            # Reuse engine if config matches, else recreate
            if self.engine and self.engine._is_loaded:
                if (self.engine.config.base_model_path == base_model and
                    self.engine.config.lora_weights_path == lora_weights and
                    self.engine.config.load_4bit == load_4bit):
                    logging.info("Reusing existing engine")
                else:
                    logging.info("Config changed, recreating engine")
                    try:
                        self.engine.cleanup()
                        self.engine.metrics.cleanup()
                    except:
                        pass
                    self.engine = None

            if not self.engine:
                logging.info("Creating new inference engine")
                self.engine = PollingInferenceEngine(config)
            else:
                # Update runtime params only
                self.engine.config.polling_interval = polling_interval
                self.engine.config.num_frames = num_frames
                self.engine.config.fps = fps
                self.engine.config.max_new_tokens = max_new_tokens
                self.engine.config.temperature = temperature
                self.engine.config.prompt = prompt

            if not self.engine.load_model():
                logging.error("Failed to load model")
                yield (
                    video_path,
                    "**Error**",
                    "Failed to load model",
                    "Error: Could not load model",
                    "",
                )
                return

            self.engine.metrics.start_session(
                video_source=video_path,
                prompt=prompt,
                polling_interval=polling_interval,
                naturalizer_enabled=use_naturalizer
            )

            # Warmup with per-run progress
            if warmup_runs > 0:
                for i in range(warmup_runs):
                    warmup_pct = 0.1 + (i / warmup_runs) * 0.1
                    progress(warmup_pct, desc=f"Warmup {i + 1}/{warmup_runs}...")
                    self.engine.warmup(1)
                progress(0.2, desc="Warmup complete")

            progress(0.3, desc="Starting polling...")

            if not self.engine.stream_handler.open_video_file(video_path):
                logging.error("Failed to open video source")
                yield (
                    video_path,
                    "**Error**",
                    "Failed to open video",
                    "Error: Could not open video source",
                    "",
                )
                return

            logging.info(f"Video opened: duration={self.engine.stream_handler.total_duration:.2f}s")

            total_duration = self.engine.stream_handler.total_duration
            poll_index = 0

            yield (
                video_path,
                f"**Analysis Position:** 0:00 / {int(total_duration//60)}:{int(total_duration%60):02d}",
                "Starting polling...",
                "Initializing...",
                "",
            )

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
                        # Log and skip this poll, but continue to next position
                        logging.warning(f"Poll #{poll_index + 1}: No frames extracted, skipping")
                        poll_index += 1
                        time.sleep(config.polling_interval)
                        continue

                    # Run inference
                    response, ttft, input_tokens, output_tokens = self.engine.run_single_inference(
                        video_frames, context_frames, prompt, slice_len
                    )

                    # Process through naturalizer
                    if self.naturalizer:
                        nat_result = self.naturalizer.process(response)
                        display_response = nat_result['display']
                        is_repeat = nat_result['is_repeat']
                        repeat_info = f" 🔄 Repeat #{nat_result['repeat_count']}" if is_repeat else " ✨ New"
                    else:
                        display_response = response
                        is_repeat = False
                        repeat_info = ""

                    input_tokens = max(0, input_tokens) if input_tokens else 0
                    output_tokens = max(0, output_tokens) if output_tokens else 0
                    ttft = max(0.0, ttft) if ttft else 0.0

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
                        'output_tokens': metrics_obj.output_tokens,
                        'vram_allocated_gb': metrics_obj.vram_allocated_gb,
                    }

                    result = {
                        'poll': poll_index + 1,
                        'position': f"{position:.2f}",
                        'response': response,
                        'metrics': metrics
                    }
                    self.poll_results.append(result)
                    self.metrics_history.append(metrics)

                    logging.info(f"Poll #{poll_index + 1} complete: latency={metrics.get('latency_ms', 0):.1f}ms{repeat_info}")

                    # Handle separate exercise mode
                    exercise_label = ""
                    if separate_exercise and " - " in display_response:
                        exercise_part, feedback_part = display_response.split(" - ", 1)
                        exercise_label = f"\n**Exercise:** {exercise_part.strip()}"
                        display_response = feedback_part.strip()

                    current_response = f"**Poll #{poll_index + 1}** (Position: {position:.2f}s){repeat_info}\n\n{display_response}"
                    current_metrics = self.format_metrics(metrics)
                    if exercise_label:
                        current_metrics = exercise_label + "\n\n" + current_metrics
                    all_responses = self.format_all_responses(self.poll_results, separate_exercise)
                    current_min = int(position // 60)
                    current_sec = int(position % 60)
                    total_min = int(total_duration // 60)
                    total_sec = int(total_duration % 60)
                    timestamp = f"**Analysis Position:** {current_min}:{current_sec:02d} / {total_min}:{total_sec:02d} (Poll #{poll_index + 1})"

                    segment_path = self.extract_video_segment(
                        self.current_video_path, position, config.polling_interval
                    )

                    yield (
                        segment_path,  # Show segment at poll position
                        timestamp,
                        current_response,
                        current_metrics,
                        all_responses,
                    )

                    poll_index += 1

                    if poll_index < 100:  # Safety limit
                        time.sleep(config.polling_interval)

                except Exception as e:
                    logging.error(f"Error in poll #{poll_index + 1}: {str(e)}")
                    current_min = int(position // 60) if 'position' in locals() else 0
                    current_sec = int(position % 60) if 'position' in locals() else 0
                    total_min = int(total_duration // 60)
                    total_sec = int(total_duration % 60)
                    timestamp = f"**Error at:** {current_min}:{current_sec:02d} / {total_min}:{total_sec:02d}"

                    yield (
                        video_path,  # Reload video
                        timestamp,
                        f"Error in poll #{poll_index + 1}: {str(e)}",
                        "Error occurred",
                        self.format_all_responses(self.poll_results, separate_exercise),
                    )
                    break

            progress(1.0, desc="Complete!")
            logging.info(f"Polling complete: {poll_index} polls processed")

            # Session summary
            session_summary = ""
            if self.engine:
                summary = self.engine.metrics.end_session()
                logging.info("Metrics and summary saved successfully")
                if summary:
                    session_summary = self.format_session_metrics(summary)

            total_min = int(total_duration // 60)
            total_sec = int(total_duration % 60)
            timestamp = f"**Complete:** {total_min}:{total_sec:02d} / {total_min}:{total_sec:02d} ({poll_index} polls)"
            final_metrics = session_summary if session_summary else f"**Final Stats:**\n{poll_index} polls completed"

            yield (
                video_path,  # Keep video loaded
                timestamp,
                f"**Polling Complete**\n\nProcessed {poll_index} polls successfully",
                final_metrics,
                self.format_all_responses(self.poll_results, separate_exercise),
            )

        except Exception as e:
            logging.error(f"Fatal error: {str(e)}")

            # End session even on error
            if self.engine:
                try:
                    self.engine.metrics.end_session()
                except Exception as e2:
                    logging.error(f"Failed to save metrics: {e2}")

            yield (
                video_path if 'video_path' in locals() else None,  # Keep video loaded
                "**Fatal Error**",
                f"**Error:** {str(e)}",
                "Error occurred during inference",
                "",
            )

        finally:
            self.is_running = False

            if self.engine:
                try:
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
        self.is_running = False
        return None, "Stopping inference..."


def create_interface():
    app = GradioPollingApp()

    with gr.Blocks(title="Mobile-VideoGPT Polling Inference", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # Mobile-VideoGPT Polling Inference
        Real-time exercise form evaluation with LoRA adapters
        """)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Video Source")

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

                gr.Markdown("### Model Configuration")

                base_model = gr.Dropdown(
                    choices=[
                        "Amshaker/Mobile-VideoGPT-0.5B",
                        "Amshaker/Mobile-VideoGPT-1.5B"
                    ],
                    label="Base Model",
                    value="Amshaker/Mobile-VideoGPT-0.5B",
                    info="Select base model size (0.5B or 1.5B)"
                )

                lora_weights = gr.Dropdown(
                    choices=[
                        "EdgeVLM-Labs/mobile-videogpt-finetune-2000",
                        "EdgeVLM-Labs/mvgpt-14_1000-pool-exercise_feedback-20260317_124726",
                        "EdgeVLM-Labs/qved-finetune-20260110_155349",
                        "EdgeVLM-Labs/mobile-videogpt-finetune-20260208_082050",
                        "EdgeVLM-Labs/mvgpt-1000-fit-300k-20260223_032309",
                        "EdgeVLM-Labs/mvgpt-1000-generated-exercise-fb-20260220_110059",
                    ],
                    label="LoRA Weights",
                    value="EdgeVLM-Labs/mobile-videogpt-finetune-2000",
                    info="Select LoRA adapter"
                )

                gr.Markdown("### Inference Parameters")

                polling_interval = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=0.5,
                    label="Polling Interval (seconds)",
                    info="Time between polls"
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

                temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=0.0,
                    step=0.1,
                    label="Temperature",
                    info="0 = greedy decoding; >0 = sampling (higher = more creative)"
                )

                warmup_runs = gr.Slider(
                    minimum=0,
                    maximum=5,
                    value=2,
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

                gr.Markdown("### Quantization")

                load_4bit = gr.Checkbox(
                    label="Load in 4-bit (NF4)",
                    value=False,
                    info="Reduces VRAM usage via bitsandbytes 4-bit quantization"
                )

                gr.Markdown("### Feedback Naturalizer")

                separate_exercise = gr.Checkbox(
                    label="Separate Exercise",
                    value=False,
                    info="If output is 'exercise - feedback', show them separately"
                )

                use_naturalizer = gr.Checkbox(
                    label="Enable Naturalizer",
                    value=False,
                    info="Detect repetitive feedback and provide varied responses"
                )

                naturalizer_threshold = gr.Slider(
                    minimum=0.5,
                    maximum=0.95,
                    value=0.70,
                    step=0.05,
                    label="Similarity Threshold",
                    info="Higher = stricter repeat detection (0.70 recommended)",
                    visible=False
                )

                with gr.Row():
                    start_btn = gr.Button("Start Polling", variant="primary", size="lg")
                    stop_btn = gr.Button("Stop", variant="stop", size="lg")

            with gr.Column(scale=2):
                gr.Markdown("### Video Player")

                # Video timestamp indicator
                video_timestamp = gr.Markdown(
                    value="**Analysis Position:** 0:00 / 0:00",
                    elem_classes=["timestamp-box"]
                )

                video_player = gr.Video(
                    label="Current Video",
                    autoplay=True,
                    loop=True,
                    show_label=False,
                    height=300
                )

                gr.Markdown("### Real-time Results")

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

                gr.Markdown("### All Responses")
                all_responses = gr.Markdown(
                    value="No responses yet",
                    elem_classes=["all-responses-box"]
                )

        # Collect all input controls for disable/enable during inference
        input_controls = [
            use_webcam, video_dropdown, base_model, lora_weights,
            polling_interval, fps, max_new_tokens, temperature, warmup_runs,
            prompt, load_4bit, separate_exercise, use_naturalizer,
            naturalizer_threshold, start_btn
        ]

        def disable_inputs():
            return [gr.update(interactive=False)] * len(input_controls)

        def enable_inputs():
            return [gr.update(interactive=True)] * len(input_controls)

        # Toggle naturalizer threshold visibility
        use_naturalizer.change(
            fn=lambda enabled: gr.update(visible=enabled),
            inputs=use_naturalizer,
            outputs=naturalizer_threshold
        )

        # Event handlers: disable controls -> run inference -> re-enable controls
        start_btn.click(
            fn=disable_inputs,
            outputs=input_controls
        ).then(
            fn=app.run_inference,
            inputs=[
                video_dropdown,
                use_webcam,
                base_model,
                lora_weights,
                polling_interval,
                fps,
                max_new_tokens,
                temperature,
                warmup_runs,
                prompt,
                separate_exercise,
                use_naturalizer,
                naturalizer_threshold,
                load_4bit
            ],
            outputs=[
                video_player,
                video_timestamp,
                current_response,
                current_metrics,
                all_responses,
            ]
        ).then(
            fn=enable_inputs,
            outputs=input_controls
        )

        stop_btn.click(
            fn=app.stop_inference,
            outputs=[video_player, current_response]
        )

        # Custom CSS
        demo.css = """
        .timestamp-box {
            padding: 10px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 8px;
            text-align: center;
            font-size: 16px;
            font-weight: bold;
            color: white !important;
            margin-bottom: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        .timestamp-box * {
            color: white !important;
        }

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
