"""
Compare LoRA and Full fine-tuning results.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.metrics import load_metrics_json


def load_results(base_dir: Path):
    """Load all results from a training run."""
    results = {}

    # Load parameter info
    param_file = base_dir / "parameters.json"
    if param_file.exists():
        results['parameters'] = load_metrics_json(str(param_file))

    # Load inference results
    inference_dir = base_dir / "inference"
    if inference_dir.exists():
        result_file = inference_dir / "test_results.json"
        if result_file.exists():
            results['inference'] = load_metrics_json(str(result_file))

    # Parse training log for loss curves
    log_file = base_dir / "training.log"
    if log_file.exists():
        train_losses = []
        with open(log_file, 'r') as f:
            for line in f:
                if "'loss':" in line:
                    try:
                        loss_val = float(line.split("'loss':")[1].split(',')[0].strip())
                        train_losses.append(loss_val)
                    except:
                        pass
        if train_losses:
            results['training'] = {
                'losses': train_losses,
                'final_loss': train_losses[-1] if train_losses else None,
                'num_steps': len(train_losses)
            }

    return results


def compare_results(lora_results, full_results):
    """Generate comparison metrics."""
    comparison = {
        'parameter_efficiency': {},
        'performance': {},
        'training': {}
    }

    # Parameter comparison
    if 'parameters' in lora_results and 'parameters' in full_results:
        lora_params = lora_results['parameters']
        full_params = full_results['parameters']

        comparison['parameter_efficiency'] = {
            'lora_trainable': lora_params.get('trainable_parameters', 'N/A'),
            'full_trainable': full_params.get('trainable_parameters', 'N/A'),
            'reduction_ratio': lora_params.get('parameter_reduction', 'N/A')
        }

    # Performance comparison
    if 'inference' in lora_results and 'inference' in full_results:
        lora_inf = lora_results['inference']
        full_inf = full_results['inference']

        comparison['performance'] = {
            'lora_accuracy': lora_inf.get('accuracy', 'N/A'),
            'full_accuracy': full_inf.get('accuracy', 'N/A'),
            'accuracy_difference': (full_inf.get('accuracy', 0) - lora_inf.get('accuracy', 0)) if isinstance(lora_inf.get('accuracy'), (int, float)) else 'N/A'
        }

    # Training comparison
    if 'training' in lora_results and 'training' in full_results:
        comparison['training'] = {
            'lora_final_loss': lora_results['training'].get('final_loss'),
            'full_final_loss': full_results['training'].get('final_loss'),
            'lora_steps': lora_results['training'].get('num_steps'),
            'full_steps': full_results['training'].get('num_steps')
        }

    return comparison


def main():
    parser = argparse.ArgumentParser(description='Compare LoRA vs Full fine-tuning')
    parser.add_argument('--lora_dir', required=True, help='LoRA results directory')
    parser.add_argument('--full_dir', required=True, help='Full fine-tuning results directory')
    parser.add_argument('--output', required=True, help='Output comparison JSON')
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    full_dir = Path(args.full_dir)

    print("Loading LoRA results...")
    lora_results = load_results(lora_dir)

    print("Loading Full fine-tuning results...")
    full_results = load_results(full_dir)

    print("Comparing results...")
    comparison = compare_results(lora_results, full_results)

    # Save comparison
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)

    print(f"\nComparison Results:")
    print(json.dumps(comparison, indent=2))
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
