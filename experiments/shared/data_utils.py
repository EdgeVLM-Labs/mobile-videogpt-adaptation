"""
Dataset manipulation utilities for experiments.
"""
import json
import random
from pathlib import Path
from typing import List, Dict
import shutil


def load_qved_dataset(json_path: str) -> List[Dict]:
    """Load QVED dataset from JSON."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def save_qved_dataset(data: List[Dict], output_path: str):
    """Save QVED dataset to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def create_subset(data: List[Dict], ratio: float, seed: int = 42) -> List[Dict]:
    """Create a subset of the dataset with given ratio."""
    random.seed(seed)
    n_samples = int(len(data) * ratio)
    return random.sample(data, n_samples)


def create_stratified_subset(data: List[Dict], ratio: float,
                            exercise_key: str = 'exercise_id',
                            seed: int = 42) -> List[Dict]:
    """Create stratified subset maintaining class distribution."""
    random.seed(seed)

    # Group by exercise type
    exercise_groups = {}
    for item in data:
        ex_id = item.get(exercise_key, 'unknown')
        if ex_id not in exercise_groups:
            exercise_groups[ex_id] = []
        exercise_groups[ex_id].append(item)

    # Sample from each group
    subset = []
    for ex_id, items in exercise_groups.items():
        n_samples = max(1, int(len(items) * ratio))
        subset.extend(random.sample(items, n_samples))

    random.shuffle(subset)
    return subset


def create_balanced_dataset(data: List[Dict],
                           exercise_key: str = 'exercise_id',
                           seed: int = 42) -> List[Dict]:
    """Create balanced dataset by oversampling minority classes."""
    random.seed(seed)

    # Group by exercise type
    exercise_groups = {}
    for item in data:
        ex_id = item.get(exercise_key, 'unknown')
        if ex_id not in exercise_groups:
            exercise_groups[ex_id] = []
        exercise_groups[ex_id].append(item)

    # Find max class size
    max_size = max(len(items) for items in exercise_groups.values())

    # Oversample to match max size
    balanced = []
    for ex_id, items in exercise_groups.items():
        # Sample with replacement to reach max_size
        oversampled = random.choices(items, k=max_size)
        balanced.extend(oversampled)

    random.shuffle(balanced)
    return balanced


def split_by_exercise(data: List[Dict],
                     exercise_key: str = 'exercise_id') -> Dict[str, List[Dict]]:
    """Split dataset by exercise type."""
    splits = {}
    for item in data:
        ex_id = item.get(exercise_key, 'unknown')
        if ex_id not in splits:
            splits[ex_id] = []
        splits[ex_id].append(item)
    return splits


def get_dataset_statistics(data: List[Dict],
                          exercise_key: str = 'exercise_id') -> Dict:
    """Get dataset statistics."""
    exercise_counts = {}
    for item in data:
        ex_id = item.get(exercise_key, 'unknown')
        exercise_counts[ex_id] = exercise_counts.get(ex_id, 0) + 1

    return {
        'total_samples': len(data),
        'n_exercises': len(exercise_counts),
        'exercise_distribution': exercise_counts,
        'min_samples': min(exercise_counts.values()) if exercise_counts else 0,
        'max_samples': max(exercise_counts.values()) if exercise_counts else 0,
        'mean_samples': sum(exercise_counts.values()) / len(exercise_counts) if exercise_counts else 0
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Dataset utilities')
    parser.add_argument('--action', choices=['create_subsets', 'create_balanced', 'stats'], required=True)
    parser.add_argument('--input', required=True, help='Input JSON file')
    parser.add_argument('--output_dir', help='Output directory')
    parser.add_argument('--ratios', nargs='+', type=float, default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    data = load_qved_dataset(args.input)

    if args.action == 'create_subsets':
        for ratio in args.ratios:
            subset = create_stratified_subset(data, ratio, seed=args.seed)
            output_path = Path(args.output_dir) / f"qved_train_{ratio:.2f}.json"
            save_qved_dataset(subset, str(output_path))
            print(f"Created subset with {len(subset)} samples (ratio={ratio}) -> {output_path}")

    elif args.action == 'create_balanced':
        balanced = create_balanced_dataset(data, seed=args.seed)
        output_path = Path(args.output_dir) / "qved_train_balanced.json"
        save_qved_dataset(balanced, str(output_path))
        print(f"Created balanced dataset with {len(balanced)} samples -> {output_path}")

    elif args.action == 'stats':
        stats = get_dataset_statistics(data)
        print(json.dumps(stats, indent=2))
