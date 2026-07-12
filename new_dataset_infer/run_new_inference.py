#!/usr/bin/env python3
"""
New Dataset Inference Script

This script runs inference on videos from the new dataset using a finetuned model.
It scans the dataset/ folder for all videos and generates predictions.

NOTE: This script is designed to be called from run_new_inference.sh
      Update configuration (prompt, model path) in the bash script, not here.

Usage:
    bash new_dataset_infer/run_new_inference.sh [options]
"""

import sys
import os
import warnings
import logging
import argparse
import json
from huggingface_hub import hf_hub_download

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
import pandas as pd

# Add workspace root to path for imports
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
    # If it's a local path, check for adapter files
    elif os.path.exists(os.path.join(pretrained_path, "adapter_config.json")):
        is_lora_checkpoint = True
    # If it's a HuggingFace repo, check if it contains LoRA adapters
    else:
        try:
            # Try to download adapter_config.json from HF Hub
            hf_hub_download(pretrained_path, "adapter_config.json")
            is_lora_checkpoint = True
            print(f"Detected LoRA adapters in HuggingFace repo: {pretrained_path}")
        except:
            # Not a LoRA checkpoint, treat as full model
            is_lora_checkpoint = False

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
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
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


def warmup_gpu(model, tokenizer, warmup_videos: list, device: str = "cuda", max_new_tokens: int = 512, prompt: str = None):
    """Warm up GPU with sample videos before actual inference."""
    print("\n🔥 Warming up GPU...")
    for video_path in warmup_videos[:3]:  # Use up to 3 videos for warmup
        if not os.path.exists(video_path):
            continue
        try:
            _ = run_inference(model, tokenizer, video_path, prompt, device, max_new_tokens)
        except Exception as e:
            print(f"  ⚠ Warmup warning for {video_path}: {e}")
    print("✓ GPU warmup complete")


def scan_dataset_folder(data_path: str):
    """Scans the dataset folder for all video files (.mov, .mp4).
    
    Returns:
        List of dicts with video_path and exercise_class
    """
    data_path = Path(data_path)
    video_extensions = ['.mov', '.mp4']
    videos = []
    
    print(f"📂 Scanning {data_path} for videos...")
    
    # Walk through dataset directory
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in video_extensions):
                full_path = Path(root) / file
                relative_path = full_path.relative_to(data_path)
                
                # Determine exercise class (parent folder)
                if len(relative_path.parts) >= 2:
                    exercise_class = relative_path.parts[0]
                else:
                    exercise_class = "unknown"
                
                videos.append({
                    'video_path': str(full_path),
                    'video_filename': file,
                    'exercise_class': exercise_class
                })
    
    # Sort videos: first by exercise class, then by filename
    videos.sort(key=lambda x: (x['exercise_class'], x['video_filename'].lower()))
    
    print(f"✅ Found {len(videos)} videos")
    return videos


def save_results_to_excel(results: list, output_path: str):
    """Saves results to an Excel file with Video Filename and Model Prediction columns."""
    
    # Prepare data for Excel
    excel_data = []
    for result in results:
        excel_data.append({
            'Video Filename': result['video_filename'],
            'Model Prediction': result['prediction'] if result['status'] == 'success' else f"ERROR: {result.get('error', 'Unknown error')}"
        })
    
    # Create DataFrame and save to Excel
    df = pd.DataFrame(excel_data)
    df.to_excel(output_path, index=False, sheet_name='Inference Results')
    print(f"📊 Excel report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run inference on new dataset videos")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to finetuned model (set in bash script)")
    parser.add_argument("--data_path", type=str, default="dataset",
                        help="Base path for video files (default: dataset)")
    parser.add_argument("--output_dir", type=str, default="new_dataset_infer/results",
                        help="Output directory for results (default: new_dataset_infer/results)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda/cpu)")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--base_model", type=str, default="Amshaker/Mobile-VideoGPT-0.5B",
                        help="Base model to use when loading LoRA adapters")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of videos to process (for testing)")
    parser.add_argument("--prompt", type=str, required=True,
                        help="Prompt to use for inference (set in bash script)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_output = output_dir / "inference_results.json"
    excel_output = output_dir / "inference_results.xlsx"

    print("=" * 60)
    print("  New Dataset Inference")
    print("=" * 60)
    print()

    # Load model
    print(f"📦 Loading model from: {args.model_path}")
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        base_model=args.base_model
    )

    # Scan dataset folder for videos
    print(f"\n📋 Scanning dataset folder: {args.data_path}")
    videos = scan_dataset_folder(args.data_path)

    if not videos:
        print("❌ No videos found in dataset folder!")
        sys.exit(1)

    if args.limit:
        videos = videos[:args.limit]
        print(f"Limited to {args.limit} videos")

    print(f"Total videos to process: {len(videos)}")

    # GPU warmup with first few videos
    if args.device == "cuda" and len(videos) > 0:
        warmup_videos = [v['video_path'] for v in videos[:3]]
        warmup_gpu(model, tokenizer, warmup_videos, args.device, args.max_new_tokens, args.prompt)

    # Run inference
    results = []
    throughput_stats = []
    print("\n🎬 Running inference...")

    for video_info in tqdm(videos, desc="Processing videos"):
        video_path = video_info['video_path']
        video_filename = video_info['video_filename']
        exercise_class = video_info['exercise_class']

        try:
            # Run inference
            prediction, metrics = run_inference(
                model, tokenizer,
                video_path, args.prompt,
                args.device, args.max_new_tokens
            )

            throughput_stats.append(metrics['tokens_per_second'])

            results.append({
                "video_path": video_path,
                "video_filename": video_filename,
                "exercise_class": exercise_class,
                "prompt": args.prompt,
                "prediction": prediction,
                "generated_tokens": metrics['generated_tokens'],
                "generation_time": round(metrics['generation_time'], 4),
                "tokens_per_second": round(metrics['tokens_per_second'], 2),
                "status": "success"
            })

        except Exception as e:
            print(f"\n✗ Error processing {video_filename}: {str(e)}")
            results.append({
                "video_path": video_path,
                "video_filename": video_filename,
                "exercise_class": exercise_class,
                "prompt": args.prompt,
                "prediction": "",
                "status": "error",
                "error": str(e)
            })

    # Save JSON results
    with open(json_output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ JSON results saved to: {json_output}")

    # Save Excel results
    try:
        save_results_to_excel(results, excel_output)
    except Exception as e:
        print(f"⚠️ Warning: Failed to create Excel file: {e}")
        print("   (Make sure openpyxl is installed: pip install openpyxl)")

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

    print(f"\nOutput files:")
    print(f"  JSON:  {json_output}")
    print(f"  Excel: {excel_output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
