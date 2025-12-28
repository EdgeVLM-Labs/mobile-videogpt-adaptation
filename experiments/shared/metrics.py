"""
Common evaluation metrics for QVED experiments.
"""
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import json


def calculate_accuracy_metrics(predictions: List[str], ground_truth: List[str]) -> Dict[str, float]:
    """Calculate accuracy, precision, recall, F1."""
    correct = sum(1 for p, g in zip(predictions, ground_truth) if p == g)
    accuracy = correct / len(predictions) if predictions else 0.0

    return {
        'accuracy': accuracy,
        'total_samples': len(predictions),
        'correct': correct
    }


def calculate_per_exercise_metrics(predictions: List[str],
                                   ground_truth: List[str],
                                   exercise_ids: List[str]) -> Dict[str, Dict[str, float]]:
    """Calculate metrics per exercise type."""
    exercise_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    for pred, gt, ex_id in zip(predictions, ground_truth, exercise_ids):
        exercise_stats[ex_id]['total'] += 1
        if pred == gt:
            exercise_stats[ex_id]['correct'] += 1

    # Calculate accuracy per exercise
    per_exercise = {}
    for ex_id, stats in exercise_stats.items():
        per_exercise[ex_id] = {
            'accuracy': stats['correct'] / stats['total'] if stats['total'] > 0 else 0.0,
            'total_samples': stats['total'],
            'correct': stats['correct']
        }

    return per_exercise


def calculate_confusion_matrix(predictions: List[str],
                               ground_truth: List[str],
                               exercise_classes: List[str]) -> np.ndarray:
    """Calculate confusion matrix for exercise classification."""
    n_classes = len(exercise_classes)
    class_to_idx = {cls: idx for idx, cls in enumerate(exercise_classes)}

    confusion = np.zeros((n_classes, n_classes), dtype=int)

    for pred, gt in zip(predictions, ground_truth):
        if pred in class_to_idx and gt in class_to_idx:
            pred_idx = class_to_idx[pred]
            gt_idx = class_to_idx[gt]
            confusion[gt_idx, pred_idx] += 1

    return confusion


def calculate_convergence_metrics(train_losses: List[float],
                                  val_losses: List[float]) -> Dict[str, float]:
    """Calculate convergence speed metrics."""
    # Find epoch where validation loss stops improving significantly
    min_val_loss = min(val_losses)
    min_val_epoch = val_losses.index(min_val_loss)

    # Calculate convergence speed (epochs to reach 95% of final performance)
    target_loss = min_val_loss * 1.05
    convergence_epoch = next((i for i, loss in enumerate(val_losses) if loss <= target_loss), len(val_losses))

    return {
        'min_val_loss': float(min_val_loss),
        'min_val_epoch': min_val_epoch,
        'convergence_epoch': convergence_epoch,
        'final_train_loss': float(train_losses[-1]) if train_losses else 0.0,
        'total_epochs': len(train_losses)
    }


def save_metrics_json(metrics: Dict, output_path: str):
    """Save metrics to JSON file."""
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


def load_metrics_json(input_path: str) -> Dict:
    """Load metrics from JSON file."""
    with open(input_path, 'r') as f:
        return json.load(f)
