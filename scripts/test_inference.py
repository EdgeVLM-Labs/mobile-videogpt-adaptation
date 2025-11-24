#!/usr/bin/env python3
"""
Test Inference Script for QVED Dataset

This script runs inference on videos from the QVED test set using a finetuned model.
It loads videos from qved_test.json and generates predictions.

Usage:
    python scripts/test_inference.py --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-150
    python scripts/test_inference.py --model_path results/qved_finetune_mobilevideogpt_0.5B --output results.json
"""

import json
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from mobilevideogpt.model import MobileVideoGPTQwenForCausalLM
from mobilevideogpt.mm_utils import tokenizer_image_token
from mobilevideogpt.conversation import conv_templates
from mobilevideogpt.constants import *
from eval.video_encoding import _get_rawvideo_dec
from transformers import AutoTokenizer


def load_model(model_path: str, device: str = "cuda"):
    """Load the finetuned model and tokenizer."""
    print(f"Loading model from: {model_path}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )

    # Load model
    model = MobileVideoGPTQwenForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True
    )
    model.eval()

    # Load vision towers
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    print("✓ Model loaded successfully")
    return model, tokenizer, video_processor, image_processor


def run_inference(model, tokenizer, video_processor, image_processor,
                  video_path: str, prompt: str, device: str = "cuda"):
    """Run inference on a single video."""

    # Process video frames
    video_frames, context_frames, slice_len = _get_rawvideo_dec(
        video_path,
        image_processor,
        video_processor,
        max_frames=NUM_FRAMES,
        image_resolution=224,
        num_video_frames=NUM_FRAMES,
        num_context_images=NUM_CONTEXT_IMAGES,
    )

    # Prepare prompt
    qs = prompt
    if model.config.mm_use_im_start_end:
        qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + qs
    else:
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs

    conv = conv_templates["qwen2_instruct"].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt_text = conv.get_prompt()

    # Tokenize
    input_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors='pt'
    ).unsqueeze(0).to(device)

    # Move frames to device
    if video_frames is not None:
        if isinstance(video_frames, list):
            video_frames = torch.stack(video_frames)
        video_frames = video_frames.to(device, dtype=torch.bfloat16)
    if context_frames is not None:
        if isinstance(context_frames, list):
            context_frames = torch.stack(context_frames)
        context_frames = context_frames.to(device, dtype=torch.bfloat16)

    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=[video_frames, context_frames],
            do_sample=False,
            temperature=0.0,
            max_new_tokens=512,
            use_cache=True,
        )

    # Decode output
    response = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0].strip()

    return response


def main():
    parser = argparse.ArgumentParser(description="Run inference on QVED test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model checkpoint")
    parser.add_argument("--test_json", type=str, default="dataset/qved_test.json",
                        help="Path to test set JSON")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Base path for video files")
    parser.add_argument("--output", type=str, default="test_predictions.json",
                        help="Output file for predictions")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    # Load model
    model, tokenizer, video_processor, image_processor = load_model(
        args.model_path,
        args.device
    )

    # Load test data
    print(f"\nLoading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Total test samples: {len(test_data)}")

    # Run inference
    results = []
    print("\nRunning inference...")

    for item in tqdm(test_data, desc="Processing videos"):
        video_rel_path = item['video']
        video_path = str(Path(args.data_path) / video_rel_path)

        # Extract prompt and ground truth
        conversations = item['conversations']
        prompt = conversations[0]['value']
        ground_truth = conversations[1]['value']

        try:
            # Run inference
            prediction = run_inference(
                model, tokenizer, video_processor, image_processor,
                video_path, prompt, args.device
            )

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "status": "success"
            })

        except Exception as e:
            print(f"\n✗ Error processing {video_rel_path}: {str(e)}")
            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": "",
                "status": "error",
                "error": str(e)
            })

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    # Print summary
    successful = sum(1 for r in results if r['status'] == 'success')
    failed = len(results) - successful

    print(f"\n{'='*60}")
    print("Inference Complete!")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
