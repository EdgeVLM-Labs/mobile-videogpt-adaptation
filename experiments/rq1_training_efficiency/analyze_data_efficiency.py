"""
Analyze data efficiency: accuracy vs dataset size.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.visualization import plot_data_efficiency_curve


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Extract results for each data ratio
    ratios = []
    accuracies = []

    for ratio in [0.25, 0.5, 0.75, 1.0]:
        result_file = results_dir / f"data_{ratio}" / "inference" / "test_results.json"

        if result_file.exists():
            with open(result_file, 'r') as f:
                results = json.load(f)
                accuracy = results.get('accuracy', 0)

                ratios.append(ratio)
                accuracies.append(accuracy)
                print(f"Ratio {ratio}: Accuracy {accuracy:.2%}")

    if ratios and accuracies:
        # Plot data efficiency curve
        plot_data_efficiency_curve(
            ratios,
            accuracies,
            args.output,
            title="Data Efficiency: Accuracy vs Dataset Size"
        )

        # Calculate efficiency metrics
        max_accuracy = max(accuracies)
        target_90 = max_accuracy * 0.90
        target_95 = max_accuracy * 0.95

        ratio_90 = next((r for r, a in zip(ratios, accuracies) if a >= target_90), 1.0)
        ratio_95 = next((r for r, a in zip(ratios, accuracies) if a >= target_95), 1.0)

        efficiency_metrics = {
            'max_accuracy': max_accuracy,
            'ratio_for_90_percent': ratio_90,
            'ratio_for_95_percent': ratio_95,
            'full_vs_25_percent': accuracies[-1] - accuracies[0] if len(accuracies) > 1 else 0
        }

        # Save metrics
        metrics_file = Path(args.output).with_suffix('.json')
        with open(metrics_file, 'w') as f:
            json.dump(efficiency_metrics, f, indent=2)

        print(f"\nEfficiency Metrics:")
        print(json.dumps(efficiency_metrics, indent=2))
    else:
        print("No results found!")


if __name__ == '__main__':
    main()
