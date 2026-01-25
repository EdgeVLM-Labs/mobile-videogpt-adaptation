#!/usr/bin/env python3
"""
Test Inference Script for QVED Dataset

This script runs inference on videos from the QVED test set using a finetuned model.
It loads videos from qved_test.json and generates predictions.

Usage:
    python utils/test_inference.py --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70
    python utils/test_inference.py --model_path results/qved_finetune_mobilevideogpt_0.5B --output test_predictions.json
"""

import sys
import os
import warnings
import logging
import argparse
import json

os.environ['PYTHONWARNINGS'] = 'ignore'

warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mobilevideogpt.utils import preprocess_input


def load_model(pretrained_path: str, device: str = "cuda", base_model: str = "Amshaker/Mobile-VideoGPT-0.5B"):
    """Loads the pre-trained model and tokenizer.

    Args:
        pretrained_path: Path to finetuned model (can be checkpoint or base dir with LoRA adapters)
        device: Device to load model on
        base_model: Base model to use when loading LoRA adapters
    """
    # Check if this is a LoRA checkpoint or full model
    is_lora_checkpoint = False
    adapter_path = pretrained_path

    # If it's a checkpoint-* directory, it contains LoRA adapters
    if "checkpoint-" in pretrained_path:
        is_lora_checkpoint = True
    # If it's the base finetuning dir, check for adapter files
    elif os.path.exists(os.path.join(pretrained_path, "adapter_config.json")):
        is_lora_checkpoint = True

    if is_lora_checkpoint:
        print(f"Loading LoRA adapters from: {adapter_path}")
        print(f"Base model: {base_model}")

        # Load base model first
        config = AutoConfig.from_pretrained(base_model)
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            config=config,
            torch_dtype=torch.float16
        )

        # Load LoRA adapters
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()  # Merge LoRA weights into base model
    else:
        # Load full model directly
        config = AutoConfig.from_pretrained(pretrained_path)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            config=config,
            torch_dtype=torch.float16
        )

    model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, video_path: str, prompt: str, device: str = "cuda", max_new_tokens: int = 512):
    """Runs inference on the given video file and returns prediction with throughput metrics."""
    import time

    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, video_path, prompt
    )

    # Get input token count
    input_token_count = input_ids.shape[1]

    with torch.inference_mode():
        # Time the generation
        start_time = time.time()

        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().to(device),
            context_images=torch.stack(context_frames, dim=0).half().to(device),
            do_sample=False,  # Use greedy decoding
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,  # KV cache enabled
        )

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

    # Calculate metrics
    output_token_count = output_ids.shape[1]
    generated_token_count = output_token_count - input_token_count  # Only new tokens
    tokens_per_second = generated_token_count / generation_time if generation_time > 0 else 0

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    return outputs, {
        'generated_tokens': generated_token_count,
        'generation_time': generation_time,
        'tokens_per_second': tokens_per_second
    }


def warmup_gpu(model, tokenizer, warmup_videos: list, device: str = "cuda", max_new_tokens: int = 512):
    """Warm up GPU with sample videos before actual inference."""
    print("\n🔥 Warming up GPU...")
    for video_path in warmup_videos[:3]:  # Use up to 3 videos for warmup
        if not os.path.exists(video_path):
            continue
        try:
            prompt = "Please evaluate this exercise form."
            _ = run_inference(model, tokenizer, video_path, prompt, device, max_new_tokens)
        except Exception as e:
            print(f"  ⚠ Warmup warning for {video_path}: {e}")
    print("✓ GPU warmup complete")


def main():
    parser = argparse.ArgumentParser(description="Run inference on QVED test set")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model checkpoint")
    parser.add_argument("--test_json", type=str, default="dataset/qved_test.json",
                        help="Path to test set JSON")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Base path for video files")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file for predictions (default: saves to model directory)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--base_model", type=str, default="Amshaker/Mobile-VideoGPT-0.5B",
                        help="Base model to use when loading LoRA adapters")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of samples to process (for testing)")

    args = parser.parse_args()

    # Set default output path to model directory if not provided
    if args.output is None:
        # Extract base directory (remove checkpoint-XX if present)
        if "checkpoint-" in args.model_path:
            base_dir = str(Path(args.model_path).parent)
        else:
            base_dir = args.model_path
        args.output = str(Path(base_dir) / "test_predictions.json")
        print(f"Output will be saved to: {args.output}")

    # Load model
    print(f"📦 Loading model from: {args.model_path}")
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        base_model=args.base_model
    )

    # Load test data
    print(f"\n📋 Loading test data from: {args.test_json}")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    print(f"Total test samples: {len(test_data)}")

    # GPU warmup with sample videos
    warmup_dir = Path("sample_videos")
    if warmup_dir.exists() and args.device == "cuda":
        warmup_videos = [str(f) for f in warmup_dir.glob("*.mp4")]
        if warmup_videos:
            warmup_gpu(model, tokenizer, warmup_videos, args.device, args.max_new_tokens)

    # Run inference
    results = []
    throughput_stats = []
    print("\n🎬 Running inference...")

    for item in tqdm(test_data, desc="Processing videos"):
        video_rel_path = item['video']
        video_path = str(Path(args.data_path) / video_rel_path)

        # Extract prompt and ground truth
        conversations = item['conversations']
        prompt = conversations[0]['value']
        ground_truth = conversations[1]['value']

        try:
            # Run inference
            prediction, metrics = run_inference(
                model, tokenizer,
                video_path, prompt,
                args.device, args.max_new_tokens
            )

            throughput_stats.append(metrics['tokens_per_second'])

            results.append({
                "video_path": video_rel_path,
                "prompt": prompt,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "generated_tokens": metrics['generated_tokens'],
                "generation_time": round(metrics['generation_time'], 4),
                "tokens_per_second": round(metrics['tokens_per_second'], 2),
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

    # Calculate throughput statistics
    if throughput_stats:
        avg_throughput = np.mean(throughput_stats)
        median_throughput = np.median(throughput_stats)
        min_throughput = np.min(throughput_stats)
        max_throughput = np.max(throughput_stats)

    print(f"\n{'='*60}")
    print("✅ Inference Complete!")
    print(f"{'='*60}")
    print(f"Total: {len(results)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    if throughput_stats:
        print(f"\n📊 Throughput Statistics (Tokens/Second):")
        print(f"  Mean:   {avg_throughput:.2f}")
        print(f"  Median: {median_throughput:.2f}")
        print(f"  Min:    {min_throughput:.2f}")
        print(f"  Max:    {max_throughput:.2f}")

    print(f"\nResults saved to: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
