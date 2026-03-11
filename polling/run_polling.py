#!/usr/bin/env python3
"""Main entry point for polling-based streaming inference."""

import os
import sys
import argparse
import logging
import json
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from polling.config import PollingConfig
from polling.inference_engine import PollingInferenceEngine


def setup_global_logging(log_level: str = "INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    logging.getLogger('mmengine').setLevel(logging.WARNING)
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('transformers.modeling_utils').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Polling-based streaming inference for Mobile-VideoGPT",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "video_source", type=str,
        help="Path to video file or stream URL (e.g., 0 for webcam, rtsp://...)",
    )

    parser.add_argument("--base-model", type=str, default="Amshaker/Mobile-VideoGPT-0.5B")
    parser.add_argument("--lora-weights", type=str, default="EdgeVLM-Labs/mobile-videogpt-finetune-2000")
    parser.add_argument("--polling-interval", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=300.0)
    parser.add_argument("--max-polls", type=int, default=None)
    parser.add_argument(
        "--prompt", type=str,
        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--fps", type=int, default=1)
    parser.add_argument("--load-4bit", action="store_true")
    parser.add_argument("--load-8bit", action="store_true")
    parser.add_argument("--warmup-runs", type=int, default=0)
    parser.add_argument("--enable-confidence-scoring", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/polling")
    parser.add_argument("--log-dir", type=str, default="logs/polling")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    return parser.parse_args()


def print_banner():
    """Print startup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════════╗
║         Mobile-VideoGPT Polling Inference Engine                 ║
║                                                                  ║
║  Real-time Exercise Form Evaluation with LoRA Adapters           ║
╚══════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def on_response_callback(poll_index: int, response: str, metrics):
    """Callback for each inference response."""
    print(f"\n{'─'*60}")
    print(f"📊 Poll #{poll_index + 1} Complete")
    print(f"   Latency: {metrics.total_inference_time*1000:.1f}ms")
    print(f"   TTFT: {metrics.time_to_first_token*1000:.1f}ms")
    print(f"   Tokens/s: {metrics.tokens_per_second:.1f}")
    print(f"{'─'*60}\n")


def main():
    args = parse_args()
    setup_global_logging(args.log_level)
    print_banner()

    config = PollingConfig(
        base_model_path=args.base_model,
        lora_weights_path=args.lora_weights,
        polling_interval=args.polling_interval,
        max_polling_duration=args.max_duration,
        num_frames=args.num_frames,
        fps=args.fps,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        load_4bit=args.load_4bit,
        load_8bit=args.load_8bit,
        enable_confidence_scoring=args.enable_confidence_scoring,
        log_dir=args.log_dir,
        output_dir=args.output_dir,
        log_level=args.log_level,
    )

    print("\n📋 Configuration:")
    print(f"   Base Model: {config.base_model_path}")
    print(f"   LoRA Weights: {config.lora_weights_path}")
    print(f"   Video Source: {args.video_source}")
    print(f"   Polling Interval: {config.polling_interval}s")
    print(f"   Max Duration: {config.max_polling_duration}s")
    print(f"   Num Frames: {config.num_frames}")
    print(f"   Confidence Scoring: {'Enabled' if config.enable_confidence_scoring else 'Disabled'}")
    print(f"   Prompt: {config.prompt[:80]}...")
    print()

    engine = PollingInferenceEngine(config)

    try:
        print("🔄 Loading model...")
        if not engine.load_model():
            print("❌ Failed to load model")
            return 1

        print("✅ Model loaded successfully\n")

        if args.warmup_runs > 0:
            print(f"🔥 Running {args.warmup_runs} warmup run(s)...")
            engine.warmup(num_runs=args.warmup_runs)
            print("✅ Warmup complete\n")

        print(f"🎬 Starting polling on: {args.video_source}")
        print(f"   Press Ctrl+C to stop\n")

        summary = engine.run_polling_loop(
            video_source=args.video_source,
            on_response=on_response_callback,
            max_polls=args.max_polls,
        )

        if "error" not in summary:
            print("\n" + "="*60)
            print("📊 SESSION SUMMARY")
            print("="*60)
            print(f"   Session ID: {summary['session_id']}")
            print(f"   Duration: {summary['duration_seconds']:.2f}s")
            print(f"   Total Polls: {summary['total_polls']}")
            print(f"   Success Rate: {summary['success_rate']:.1f}%")
            print()
            print("   Latency (ms):")
            print(f"      Mean: {summary['latency_ms']['mean']:.2f}")
            print(f"      Median: {summary['latency_ms']['median']:.2f}")
            print(f"      Min: {summary['latency_ms']['min']:.2f}")
            print(f"      Max: {summary['latency_ms']['max']:.2f}")
            print()
            print("   Time to First Token (ms):")
            print(f"      Mean: {summary['time_to_first_token_ms']['mean']:.2f}")
            print(f"      Median: {summary['time_to_first_token_ms']['median']:.2f}")
            print()
            print(f"   Throughput: {summary['tokens_per_second']['mean']:.2f} tokens/s")
            print("="*60)

            summary_file = os.path.join(config.output_dir, f"summary_{summary['session_id']}.json")
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"\n💾 Summary saved to: {summary_file}")

        return 0

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
        return 0

    finally:
        engine.cleanup()


if __name__ == "__main__":
    sys.exit(main())
