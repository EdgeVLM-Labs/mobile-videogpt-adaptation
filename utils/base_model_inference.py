#!/usr/bin/env python3
"""
Base Model Inference Utility

This utility runs inference using the base (non-finetuned) model
to compare against finetuned model predictions.

Usage:
    from utils.base_model_inference import get_base_model_predictions

    predictions = get_base_model_predictions(
        test_data,
        base_model="Amshaker/Mobile-VideoGPT-0.5B",
        data_path="dataset",
        device="cuda"
    )
"""

import json
import torch
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm


def load_base_model(model_path: str, device: str = "cuda"):
    """Load the base Mobile-VideoGPT model."""
    try:
        from mobilevideogpt.model.builder import load_pretrained_model
        from mobilevideogpt.mm_utils import get_model_name_from_path

        print(f"Loading base model from {model_path}...")
        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map=device
        )

        model.eval()
        print("✓ Base model loaded successfully")
        return tokenizer, model, image_processor

    except Exception as e:
        print(f"❌ Error loading base model: {e}")
        raise


def run_base_inference(
    video_path: str,
    prompt: str,
    model,
    tokenizer,
    image_processor,
    device: str = "cuda",
    max_new_tokens: int = 64
) -> str:
    """Run inference on a single video with the base model."""
    try:
        from mobilevideogpt.mm_utils import process_video, tokenizer_image_token
        from mobilevideogpt.constants import IMAGE_TOKEN_INDEX
        from mobilevideogpt.conversation import conv_templates

        # Process video
        video_tensor = process_video(
            video_path,
            image_processor,
            model.config
        ).to(dtype=torch.float16, device=device)

        # Prepare conversation
        conv = conv_templates["qwen_2"].copy()
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_formatted,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(device)

        # Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=video_tensor.unsqueeze(0),
                do_sample=False,
                max_new_tokens=max_new_tokens,
                use_cache=True
            )

        # Decode
        output = tokenizer.batch_decode(
            output_ids,
            skip_special_tokens=True
        )[0].strip()

        return output

    except Exception as e:
        print(f"❌ Error during inference: {e}")
        return f"[ERROR: {str(e)}]"


def get_base_model_predictions(
    test_data: List[Dict],
    base_model: str = "Amshaker/Mobile-VideoGPT-0.5B",
    data_path: str = "dataset",
    device: str = "cuda",
    max_new_tokens: int = 64
) -> List[Dict]:
    """
    Generate predictions for test data using the base model.

    Args:
        test_data: List of test samples with 'video', 'conversations' fields
        base_model: Path or HuggingFace ID of base model
        data_path: Base path for video files
        device: Device to use (cuda/cpu)
        max_new_tokens: Maximum tokens to generate

    Returns:
        List of dictionaries with base model predictions
    """
    print("\n" + "="*60)
    print("Base Model Inference")
    print("="*60)
    print(f"Model: {base_model}")
    print(f"Device: {device}")
    print(f"Samples: {len(test_data)}")
    print("="*60 + "\n")

    # Load model
    tokenizer, model, image_processor = load_base_model(base_model, device)

    results = []

    for idx, sample in enumerate(tqdm(test_data, desc="Running base model inference")):
        try:
            # Get video path
            video_file = sample.get('video', '')
            video_path = Path(data_path) / video_file

            if not video_path.exists():
                results.append({
                    'id': sample.get('id', f'sample_{idx}'),
                    'video': video_file,
                    'base_prediction': '[ERROR: Video not found]',
                    'status': 'error'
                })
                continue

            # Get prompt (first user message)
            conversations = sample.get('conversations', [])
            prompt = ""
            for conv in conversations:
                if conv.get('from') == 'human':
                    prompt = conv.get('value', '').replace('<video>\n', '').replace('<video>', '').strip()
                    break

            if not prompt:
                prompt = "Describe this exercise video."

            # Run inference
            prediction = run_base_inference(
                str(video_path),
                prompt,
                model,
                tokenizer,
                image_processor,
                device,
                max_new_tokens
            )

            results.append({
                'id': sample.get('id', f'sample_{idx}'),
                'video': video_file,
                'base_prediction': prediction,
                'status': 'success'
            })

        except Exception as e:
            results.append({
                'id': sample.get('id', f'sample_{idx}'),
                'video': video_file,
                'base_prediction': f'[ERROR: {str(e)}]',
                'status': 'error'
            })

    print(f"\n✓ Base model inference complete: {len(results)} samples processed")
    return results


def main():
    """CLI interface for base model inference."""
    import argparse

    parser = argparse.ArgumentParser(description="Run base model inference on test set")
    parser.add_argument('--test_json', type=str, required=True, help='Path to test JSON file')
    parser.add_argument('--base_model', type=str, default='Amshaker/Mobile-VideoGPT-0.5B',
                        help='Base model path or HF ID')
    parser.add_argument('--data_path', type=str, default='dataset', help='Base path for videos')
    parser.add_argument('--output', type=str, default='base_model_predictions.json',
                        help='Output JSON file')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda/cpu')
    parser.add_argument('--max_new_tokens', type=int, default=64, help='Max tokens to generate')
    parser.add_argument('--limit', type=int, help='Limit number of samples (for testing)')

    args = parser.parse_args()

    # Load test data
    print(f"Loading test data from {args.test_json}...")
    with open(args.test_json, 'r') as f:
        test_data = json.load(f)

    if args.limit:
        test_data = test_data[:args.limit]
        print(f"Limited to {args.limit} samples")

    # Run inference
    results = get_base_model_predictions(
        test_data,
        base_model=args.base_model,
        data_path=args.data_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens
    )

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
