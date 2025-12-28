"""
Analyze convergence speed across different training configurations.
"""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.metrics import calculate_convergence_metrics
from shared.visualization import plot_training_curves


def parse_training_log(log_file):
    """Extract losses from training log."""
    train_losses = []

    with open(log_file, 'r') as f:
        for line in f:
            if "'loss':" in line:
                try:
                    loss_val = float(line.split("'loss':")[1].split(',')[0].strip())
                    train_losses.append(loss_val)
                except:
                    pass

    return train_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_results = {}

    # Find all subdirectories with training logs
    for subdir in results_dir.rglob('*'):
        if subdir.is_dir():
            log_file = subdir / 'training.log'
            if log_file.exists():
                config_name = subdir.name
                losses = parse_training_log(log_file)

                if losses:
                    # Assuming validation losses are similar for this analysis
                    convergence = calculate_convergence_metrics(losses, losses)
                    all_results[config_name] = {
                        'train_losses': losses,
                        'convergence': convergence
                    }

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Plot all training curves
    for config_name, data in all_results.items():
        axes[0].plot(data['train_losses'], label=config_name, alpha=0.7)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss Curves')
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0].grid(True, alpha=0.3)

    # Plot convergence speeds
    config_names = list(all_results.keys())
    convergence_epochs = [data['convergence']['convergence_epoch']
                         for data in all_results.values()]

    axes[1].barh(config_names, convergence_epochs, color='steelblue')
    axes[1].set_xlabel('Convergence Epoch')
    axes[1].set_title('Convergence Speed Comparison')
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(args.output, dpi=300, bbox_inches='tight')
    print(f"Convergence analysis saved to {args.output}")

    # Save numerical results
    summary_file = Path(args.output).with_suffix('.json')
    with open(summary_file, 'w') as f:
        json.dump({k: v['convergence'] for k, v in all_results.items()}, f, indent=2)
    print(f"Numerical results saved to {summary_file}")


if __name__ == '__main__':
    main()
