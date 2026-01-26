#!/usr/bin/env python3
"""
Single Video Inference with Confidence Scoring

This script demonstrates how to run inference on a single video and calculate
comprehensive confidence metrics including:
- Average entropy
- Sequence log probability
- Beam search metrics (top-2 margin, top beam score, score spread)
- Final boolean confidence determination

Usage:
    python utils/confidence_scoring/inference_with_confidence.py --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70 --video_path sample_videos/00000340.mp4
    python utils/confidence_scoring/inference_with_confidence.py --model_path Amshaker/Mobile-VideoGPT-1.5B --video_path dataset/videos/example.mp4 --prompt "Evaluate this exercise"
"""

import sys
import os
import warnings
import logging
import argparse
import time

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from peft import PeftModel
from huggingface_hub import hf_hub_download

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from mobilevideogpt.utils import preprocess_input
from utils.confidence_scoring.calculate_confidence import (
    is_confident,
    ENTROPY_THRESHOLD,
    SEQ_LOGPROB_THRESHOLD,
    BEAM_TOP2_MARGIN_THRESHOLD,
    BEAM_TOP_BEAM_LOGPROB_THRESHOLD,
)


def load_model(pretrained_path: str, device: str = "cuda", base_model: str = "Amshaker/Mobile-VideoGPT-1.5B"):
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
        # Always load tokenizer from base model to avoid custom config issues
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            config=config,
            torch_dtype=torch.float16
        )

    model.to(device)
    return model, tokenizer


def run_inference_with_confidence(
    model,
    tokenizer,
    video_path: str,
    prompt: str,
    device: str = "cuda",
    max_new_tokens: int = 512,
    num_beams: int = 5,
):
    """Runs inference with comprehensive confidence scoring.

    Args:
        model: The model to use for inference
        tokenizer: The tokenizer
        video_path: Path to video file
        prompt: Text prompt for the model
        device: Device to run on
        max_new_tokens: Maximum tokens to generate
        num_beams: Number of beams for beam search

    Returns:
        tuple: (prediction_text, metrics_dict, is_confident_bool)
    """
    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, video_path, prompt
    )

    # Get input token count
    input_token_count = input_ids.shape[1]

    with torch.inference_mode():
        # Time the generation
        start_time = time.time()

        outputs_dict = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().to(device),
            context_images=torch.stack(context_frames, dim=0).half().to(device),
            do_sample=False,  # Use greedy decoding
            num_beams=num_beams,  # Use beam search
            num_return_sequences=num_beams,  # Return all beams for confidence analysis
            max_new_tokens=max_new_tokens,
            use_cache=True,  # KV cache enabled
            return_dict_in_generate=True,  # Required for confidence scoring
            output_scores=True,  # Required for confidence scoring
        )

        # End timing
        end_time = time.time()
        generation_time = end_time - start_time

        # Extract output IDs and scores
        output_ids = outputs_dict.sequences
        scores = outputs_dict.scores  # Tuple of tensors, one per generated token
        sequences_scores = outputs_dict.sequences_scores  # Beam scores

    # Calculate metrics
    # Detect if output_ids includes input tokens or only generated tokens
    total_seq_length = output_ids.shape[1]

    if total_seq_length <= input_token_count:
        # output_ids contains ONLY generated tokens (not including input)
        generated_token_count = total_seq_length
        generated_ids_offset = 0  # No offset needed for confidence calculation
    else:
        # output_ids contains full sequence (input + generated)
        generated_token_count = total_seq_length - input_token_count
        generated_ids_offset = input_token_count

    tokens_per_second = generated_token_count / generation_time if generation_time > 0 else 0

    # Calculate confidence scores and determine overall confidence
    model_is_confident, confidence_metrics = is_confident(
        scores, output_ids, generated_ids_offset, sequences_scores
    )

    # Decode output
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)].strip()

    metrics = {
        "generated_tokens": generated_token_count,
        "generation_time": generation_time,
        "tokens_per_second": tokens_per_second,
        "avg_entropy": confidence_metrics["avg_entropy"],
        "seq_logprob_normalized": confidence_metrics["seq_logprob_normalized"],
        "beam_top2_margin": confidence_metrics["beam_top2_margin"],
        "beam_top_beam_avg_logprob": confidence_metrics["beam_top_beam_avg_logprob"],
        "beam_score_spread": confidence_metrics["beam_score_spread"],
    }

    return outputs, metrics, model_is_confident


def print_confidence_report(metrics: dict, is_confident: bool):
    """Print a detailed confidence report."""
    print("\n" + "=" * 80)
    print("📊 CONFIDENCE SCORING REPORT")
    print("=" * 80)

    print(f"\n⚡ Performance Metrics:")
    print(f"  Generated Tokens:  {metrics['generated_tokens']}")
    print(f"  Generation Time:   {metrics['generation_time']:.4f}s")
    print(f"  Tokens/Second:     {metrics['tokens_per_second']:.2f}")

    print(f"\n🎯 Confidence Metrics:")
    print(f"  Average Entropy:             {metrics['avg_entropy']:.4f}  (threshold: < {ENTROPY_THRESHOLD})")
    entropy_pass = "✓" if metrics['avg_entropy'] < ENTROPY_THRESHOLD else "✗"
    print(f"                               {entropy_pass} {'PASS' if metrics['avg_entropy'] < ENTROPY_THRESHOLD else 'FAIL'}")

    print(f"\n  Seq-LogProb (Normalized):    {metrics['seq_logprob_normalized']:.4f}  (threshold: > {SEQ_LOGPROB_THRESHOLD})")
    logprob_pass = "✓" if metrics['seq_logprob_normalized'] > SEQ_LOGPROB_THRESHOLD else "✗"
    print(f"                               {logprob_pass} {'PASS' if metrics['seq_logprob_normalized'] > SEQ_LOGPROB_THRESHOLD else 'FAIL'}")

    print(f"\n🔍 Beam Search Metrics ({3} beams):")
    print(f"  Top-2 Margin:                {metrics['beam_top2_margin']:.4f}  (threshold: > {BEAM_TOP2_MARGIN_THRESHOLD})")
    margin_pass = "✓" if metrics['beam_top2_margin'] > BEAM_TOP2_MARGIN_THRESHOLD else "✗"
    print(f"                               {margin_pass} {'PASS' if metrics['beam_top2_margin'] > BEAM_TOP2_MARGIN_THRESHOLD else 'FAIL'}")

    print(f"\n  Top Beam Avg LogProb:        {metrics['beam_top_beam_avg_logprob']:.4f}  (threshold: > {BEAM_TOP_BEAM_LOGPROB_THRESHOLD})")
    beam_pass = "✓" if metrics['beam_top_beam_avg_logprob'] > BEAM_TOP_BEAM_LOGPROB_THRESHOLD else "✗"
    print(f"                               {beam_pass} {'PASS' if metrics['beam_top_beam_avg_logprob'] > BEAM_TOP_BEAM_LOGPROB_THRESHOLD else 'FAIL'}")

    print(f"\n  Score Spread (StdDev):       {metrics['beam_score_spread']:.4f}")

    print(f"\n{'=' * 80}")
    if is_confident:
        print("✅ FINAL VERDICT: MODEL IS CONFIDENT")
    else:
        print("❌ FINAL VERDICT: MODEL IS NOT CONFIDENT")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run inference with confidence scoring")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to finetuned model checkpoint or HuggingFace model ID",
    )
    parser.add_argument(
        "--video_path", type=str, required=True, help="Path to input video file"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?",
        help="Prompt for the model",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run inference on (cuda/cpu)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--num_beams", type=int, default=3, help="Number of beams for beam search"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="Amshaker/Mobile-VideoGPT-1.5B",
        help="Base model to use when loading LoRA adapters",
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        sys.exit(1)

    if (
        not args.model_path.startswith("Amshaker/")
        and not args.model_path.startswith("EdgeVLM-Labs/")
        and not os.path.exists(args.model_path)
    ):
        print(f"❌ Error: Model path not found: {args.model_path}")
        sys.exit(1)

    # Load model
    print(f"📦 Loading model from: {args.model_path}")
    model, tokenizer = load_model(args.model_path, device=args.device, base_model=args.base_model)

    print(f"🎥 Processing video: {args.video_path}")
    print(f"💬 Prompt: {args.prompt}")

    # Run inference with confidence scoring
    prediction, metrics, model_is_confident = run_inference_with_confidence(
        model,
        tokenizer,
        args.video_path,
        args.prompt,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
    )

    # Display results
    print("\n" + "=" * 80)
    print("🤖 MODEL PREDICTION:")
    print("=" * 80)
    print(prediction)
    print("=" * 80)

    # Display confidence report
    print_confidence_report(metrics, model_is_confident)


if __name__ == "__main__":
    main()
