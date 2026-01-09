#!/usr/bin/env python3
"""
Real-time webcam demo for streaming Mobile-VideoGPT.

This script demonstrates real-time exercise form feedback using a webcam
or video file. It displays the video feed with overlaid feedback and FPS counter.

Usage:
    python demo_streaming.py --config streaming_config.yaml
    python demo_streaming.py --video path/to/video.mp4
    python demo_streaming.py --model Amshaker/Mobile-VideoGPT-0.5B
"""

import argparse
import os
import cv2
import time
import numpy as np
from pathlib import Path
import sys
import gc
import torch

# Set headless backend for OpenCV if no display is available
if os.environ.get('QT_QPA_PLATFORM') == 'offscreen':
    # Use headless backend
    cv2.setNumThreads(1)

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from streaming.engine import StreamingMobileVideoGPT
from streaming.utils import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Streaming Mobile-VideoGPT Demo")

    parser.add_argument(
        "--config",
        type=str,
        default="streaming_config.yaml",
        help="Path to configuration YAML file",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Amshaker/Mobile-VideoGPT-0.5B",
        help="Path to pretrained model or LoRA adapter",
    )

    parser.add_argument(
        "--base-model",
        type=str,
        default=None,
        help="Base model path (required if --model is a LoRA adapter)",
    )

    parser.add_argument(
        "--lora-adapter",
        type=str,
        default=None,
        help="LoRA adapter path (if provided, --model becomes the base model)",
    )

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to video file (default: webcam)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Don't display video feed (headless mode)",
    )

    parser.add_argument(
        "--save-output",
        type=str,
        default=None,
        help="Save annotated video to file",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum frames to process (None = infinite)",
    )

    return parser.parse_args()


class StreamingDemo:
    """Demo application for streaming inference."""

    def __init__(
        self,
        engine: StreamingMobileVideoGPT,
        video_source: int | str = 0,
        display: bool = True,
        save_path: str = None,
    ):
        """
        Initialize demo.

        Args:
            engine: StreamingMobileVideoGPT instance
            video_source: Webcam ID or video file path
            display: Whether to display video feed
            save_path: Path to save output video (None = don't save)
        """
        self.engine = engine
        self.video_source = video_source
        self.display = display
        self.save_path = save_path

        # Video capture
        self.cap = cv2.VideoCapture(video_source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {video_source}")

        # Get video properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))

        print(f"Video source: {video_source}")
        print(f"Resolution: {self.width}x{self.height} @ {self.fps} FPS")

        # Video writer
        self.writer = None
        if save_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.writer = cv2.VideoWriter(
                save_path,
                fourcc,
                self.fps,
                (self.width, self.height)
            )
            print(f"Saving output to: {save_path}")

        # State
        self.current_feedback = "Starting..."
        self.feedback_timestamp = 0
        self.frame_count = 0
        self.start_time = time.time()

    def run(self, max_frames: int = None):
        """
        Run streaming demo.

        Args:
            max_frames: Maximum frames to process (None = infinite)
        """
        print("\n" + "="*60)
        print("Starting Streaming Demo")
        print("="*60)
        print(f"DEBUG: Entered run() method, display={self.display}")

        if self.display:
            print("Press 'q' to quit, 's' to show stats, 'r' to reset\n")
        else:
            print("Running in HEADLESS mode - feedback will be printed to terminal")
            print("Press Ctrl+C to stop\n")

        print(f"DEBUG: About to start main loop...")

        try:
            frame_num = 0
            while True:
                if frame_num == 0:
                    print(f"DEBUG: Reading first frame...")

                # Read frame
                ret, frame = self.cap.read()

                if frame_num == 0:
                    print(f"DEBUG: First frame read: ret={ret}, shape={frame.shape if ret else None}")

                if not ret:
                    print("End of video or camera disconnected")
                    break

                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process frame
                process_start = time.time()
                result = self.engine.process_frame(frame_rgb)
                process_time = time.time() - process_start

                # Update feedback if received
                if result is not None:
                    self.current_feedback = result["feedback_text"]
                    self.feedback_timestamp = result["timestamp"]
                    feedback_msg = f"\n[{result['timestamp']:.1f}s] 🎯 FEEDBACK: {self.current_feedback}\n"
                    print(feedback_msg)
                    # Always log feedback to make it visible in headless mode
                    if not self.display:
                        print(f"Chunk #{result.get('chunk_id', '?')} | Confidence: {result.get('confidence', 0):.2f}")
                        print("-" * 80)

                # Annotate frame
                annotated = self._annotate_frame(frame, process_time)

                # Save frame
                if self.writer:
                    self.writer.write(annotated)

                # Display
                if self.display:
                    try:
                        cv2.imshow("Streaming Mobile-VideoGPT", annotated)

                        # Handle keys
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Quit requested")
                            break
                        elif key == ord('s'):
                            self.engine.print_stats()
                        elif key == ord('r'):
                            print("Resetting engine...")
                            self.engine.reset()
                            self.current_feedback = "Reset - Starting..."
                    except cv2.error as e:
                        print(f"Display error (running in headless mode?): {e}")
                        print("Continuing without display...")
                        self.display = False  # Disable display for subsequent frames

                self.frame_count += 1

                # Progress indicator for headless mode
                if not self.display and self.frame_count % 30 == 0:
                    elapsed = time.time() - self.start_time
                    fps = self.frame_count / elapsed if elapsed > 0 else 0
                    print(f"Processed {self.frame_count} frames | {fps:.1f} FPS | Elapsed: {elapsed:.1f}s")

                # Check max frames
                if max_frames and self.frame_count >= max_frames:
                    print(f"Reached maximum frames: {max_frames}")
                    break

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            self._cleanup()

    def _annotate_frame(self, frame: np.ndarray, process_time: float) -> np.ndarray:
        """
        Add text overlays to frame.

        Args:
            frame: Input frame (BGR format)
            process_time: Time taken to process this frame

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Calculate FPS
        elapsed = time.time() - self.start_time
        fps = self.frame_count / elapsed if elapsed > 0 else 0

        # Font settings
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        # Top-left: FPS and frame count
        fps_text = f"FPS: {fps:.1f} | Frame: {self.frame_count}"
        cv2.putText(
            annotated,
            fps_text,
            (10, 30),
            font,
            font_scale,
            (0, 255, 0),
            thickness,
        )

        # Top-left: Processing time
        time_text = f"Process: {process_time*1000:.1f}ms"
        cv2.putText(
            annotated,
            time_text,
            (10, 60),
            font,
            font_scale,
            (0, 255, 255),
            thickness,
        )

        # Bottom: Feedback text
        feedback_display = self.current_feedback[:80]  # Truncate if too long

        # Multi-line feedback (wrap text)
        words = feedback_display.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_w, _), _ = cv2.getTextSize(test_line, font, font_scale, thickness)

            if text_w < w - 40:  # Leave margin
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Draw feedback box
        box_height = 20 + len(lines) * 35
        cv2.rectangle(
            annotated,
            (10, h - box_height - 10),
            (w - 10, h - 10),
            (0, 0, 0),
            -1,  # Filled
        )

        # Draw feedback text
        y_offset = h - box_height + 5
        for i, line in enumerate(lines):
            cv2.putText(
                annotated,
                line,
                (20, y_offset + i * 35),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )

        return annotated

    def _cleanup(self):
        """Clean up resources."""
        print("\n" + "="*60)
        print("Cleaning up...")

        # Release video capture and writer
        self.cap.release()
        if self.writer:
            self.writer.release()

        # Close windows
        if self.display:
            cv2.destroyAllWindows()

        # Print final stats
        print("\nFinal Statistics:")
        self.engine.print_stats()

        elapsed = time.time() - self.start_time
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {self.frame_count / elapsed:.2f}")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    args = parse_args()

    # Load config
    if Path(args.config).exists():
        config = load_config(args.config)
        print(f"Loaded config from: {args.config}")
    else:
        print(f"Config file not found: {args.config}, using defaults")
        config = None

    # Determine video source
    video_source = args.video if args.video else 0

    # Update config with command line args
    if config:
        if args.video:
            config["demo"]["video_source"] = args.video
        config["demo"]["display_video"] = not args.no_display

    # Initialize streaming engine
    model_path = args.model
    base_model = args.base_model
    lora_adapter = args.lora_adapter

    print(f"\nInitializing Mobile-VideoGPT ({model_path})...")
    if lora_adapter:
        print(f"  Base model: {model_path}")
        print(f"  LoRA adapter: {lora_adapter}")
    elif base_model:
        print(f"  Base model: {base_model}")
        print(f"  LoRA adapter: {model_path}")
        lora_adapter = model_path
        model_path = base_model

    engine = StreamingMobileVideoGPT(
        model_path=model_path,
        config_dict=config,
        device=args.device,
        lora_adapter=lora_adapter,
    )

    print("Engine created - synchronizing CUDA...")
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    print("Synchronization complete")

    # Run demo
    print("Creating demo object...")
    demo = StreamingDemo(
        engine=engine,
        video_source=video_source,
        display=not args.no_display,
        save_path=args.save_output,
    )
    print("Demo object created successfully")

    print("Starting demo.run()...")
    demo.run(max_frames=args.max_frames)


if __name__ == "__main__":
    main()
