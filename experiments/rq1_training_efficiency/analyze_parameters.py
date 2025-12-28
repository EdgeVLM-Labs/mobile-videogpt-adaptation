"""
Analyze trainable parameters for LoRA vs Full fine-tuning.
"""
import argparse
import json
import torch
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def count_parameters(model):
    """Count total and trainable parameters."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
        'parameter_reduction': 1 - (trainable_params / total_params) if total_params > 0 else 0
    }


def analyze_lora_parameters(checkpoint_path: str):
    """Analyze LoRA parameters specifically."""
    # Try to load adapter config if exists
    adapter_config_path = Path(checkpoint_path) / "adapter_config.json"

    if adapter_config_path.exists():
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        return {
            'lora_r': config.get('r', 'N/A'),
            'lora_alpha': config.get('lora_alpha', 'N/A'),
            'lora_dropout': config.get('lora_dropout', 'N/A'),
            'target_modules': config.get('target_modules', [])
        }
    return {}


def main():
    parser = argparse.ArgumentParser(description='Analyze model parameters')
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint')
    parser.add_argument('--output', required=True, help='Output JSON file')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)

    # Basic parameter analysis
    # Note: In production, you would load the actual model here
    # For now, we'll create a template output

    results = {
        'checkpoint_path': str(checkpoint_path),
        'analysis_type': 'parameter_count'
    }

    # Try to get LoRA config if it exists
    lora_config = analyze_lora_parameters(checkpoint_path)
    if lora_config:
        results['lora_config'] = lora_config

    # Check for training logs
    parent_dir = checkpoint_path.parent
    training_log = parent_dir / "training.log"

    if training_log.exists():
        # Parse training log for parameter info
        with open(training_log, 'r') as f:
            log_content = f.read()
            # Extract parameter counts if logged
            if 'trainable params' in log_content:
                for line in log_content.split('\n'):
                    if 'trainable params' in line:
                        results['log_info'] = line.strip()
                        break

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Parameter analysis saved to {output_path}")
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
