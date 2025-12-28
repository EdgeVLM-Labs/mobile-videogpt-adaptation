"""
Analyze temporal modeling results.
"""
import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.visualization import plot_temporal_sensitivity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)

    # Parse results by frame count (averaging across FPS)
    frame_results = {}

    for result_dir in results_dir.glob('frames*'):
        # Extract frame count from directory name
        dir_name = result_dir.name
        if 'frames' in dir_name:
            frame_count = int(dir_name.split('frames')[1].split('_')[0])

            # Load accuracy
            results_file = result_dir / 'test_results.json'
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                    accuracy = data.get('accuracy', 0)

                    if frame_count not in frame_results:
                        frame_results[frame_count] = []
                    frame_results[frame_count].append(accuracy)

    # Average accuracies for each frame count
    frame_counts = sorted(frame_results.keys())
    avg_accuracies = [np.mean(frame_results[fc]) for fc in frame_counts]

    # Plot
    if frame_counts and avg_accuracies:
        plot_temporal_sensitivity(
            frame_counts,
            avg_accuracies,
            args.output,
            title="Temporal Sensitivity: Accuracy vs Frame Count"
        )

        # Save numerical results
        metrics = {
            'frame_counts': frame_counts,
            'accuracies': avg_accuracies,
            'optimal_frame_count': frame_counts[np.argmax(avg_accuracies)]
        }

        metrics_file = Path(args.output).with_suffix('.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"Temporal analysis complete!")
        print(f"Optimal frame count: {metrics['optimal_frame_count']}")


if __name__ == '__main__':
    main()
