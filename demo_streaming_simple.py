#!/usr/bin/env python3
"""Simplified streaming demo to isolate segfault issue."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import cv2
from streaming.engine import StreamingMobileVideoGPT

def main():
    print("Loading config...")
    config = {
        'video': {'chunk_size': 8, 'overlap': 4, 'num_context_images': 16},
        'temporal': {'max_history': 3, 'aggregation': 'concatenate'},
        'action_prediction': {'strategy': 'rule_based', 'interval': 5},
        'special_tokens': {'next': '<next>', 'feedback': '<feedback>', 'correct': '<correct>'},
        'generation': {'max_new_tokens': 256}
    }

    print("Creating engine...")
    engine = StreamingMobileVideoGPT(
        model_path='Amshaker/Mobile-VideoGPT-0.5B',
        config_dict=config,
        device='cuda',
        lora_adapter='EdgeVLM-Labs/mobile-videogpt-finetune-2000'
    )
    print("Engine created successfully!")

    print("Opening video...")
    cap = cv2.VideoCapture('sample_videos/test_stream.mp4')
    if not cap.isOpened():
        raise RuntimeError("Failed to open video")
    print("Video opened!")

    print("Processing frames...")
    frame_count = 0
    while frame_count < 20:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = engine.process_frame(frame_rgb)

        if result:
            print(f"Frame {frame_count}: {result.get('feedback_text', 'No feedback')}")

        frame_count += 1
        if frame_count % 5 == 0:
            print(f"Processed {frame_count} frames...")

    cap.release()
    print(f"\nCompleted! Processed {frame_count} frames")

if __name__ == "__main__":
    main()
