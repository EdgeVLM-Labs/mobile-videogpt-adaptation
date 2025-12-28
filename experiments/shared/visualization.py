"""
Visualization utilities for experiment results.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import json

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_training_curves(train_losses: List[float],
                        val_losses: List[float],
                        output_path: str,
                        title: str = "Training Curves"):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)

    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_accuracy_comparison(model_accuracies: Dict[str, float],
                            output_path: str,
                            title: str = "Model Accuracy Comparison"):
    """Plot bar chart comparing model accuracies."""
    plt.figure(figsize=(10, 6))

    models = list(model_accuracies.keys())
    accuracies = list(model_accuracies.values())

    bars = plt.bar(models, accuracies, color='steelblue', alpha=0.8)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom')

    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy comparison to {output_path}")


def plot_confusion_matrix(confusion: np.ndarray,
                         class_names: List[str],
                         output_path: str,
                         title: str = "Confusion Matrix"):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(12, 10))

    # Normalize by row (ground truth)
    confusion_norm = confusion.astype('float') / confusion.sum(axis=1)[:, np.newaxis]

    sns.heatmap(confusion_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Proportion'})

    plt.xlabel('Predicted')
    plt.ylabel('Ground Truth')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {output_path}")


def plot_data_efficiency_curve(ratios: List[float],
                              accuracies: List[float],
                              output_path: str,
                              title: str = "Data Efficiency Curve"):
    """Plot accuracy vs dataset size."""
    plt.figure(figsize=(10, 6))

    plt.plot(ratios, accuracies, 'o-', linewidth=2, markersize=8, color='steelblue')

    plt.xlabel('Dataset Ratio')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved data efficiency curve to {output_path}")


def plot_temporal_sensitivity(frame_counts: List[int],
                             accuracies: List[float],
                             output_path: str,
                             title: str = "Temporal Sensitivity"):
    """Plot accuracy vs frame count."""
    plt.figure(figsize=(10, 6))

    plt.plot(frame_counts, accuracies, 'o-', linewidth=2, markersize=8, color='coral')

    plt.xlabel('Number of Frames')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved temporal sensitivity plot to {output_path}")


def plot_robustness_curves(degradation_levels: List[float],
                          model_accuracies: Dict[str, List[float]],
                          output_path: str,
                          title: str = "Robustness to Degradation"):
    """Plot accuracy degradation curves for multiple models."""
    plt.figure(figsize=(10, 6))

    for model_name, accuracies in model_accuracies.items():
        plt.plot(degradation_levels, accuracies, 'o-',
                label=model_name, linewidth=2, markersize=6)

    plt.xlabel('Degradation Level')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved robustness curves to {output_path}")


def plot_pareto_frontier(latencies: List[float],
                        accuracies: List[float],
                        model_names: List[str],
                        output_path: str,
                        title: str = "Accuracy vs Latency Pareto Frontier"):
    """Plot Pareto frontier for accuracy-latency tradeoff."""
    plt.figure(figsize=(10, 6))

    plt.scatter(latencies, accuracies, s=100, alpha=0.6, color='steelblue')

    # Add labels for each point
    for i, name in enumerate(model_names):
        plt.annotate(name, (latencies[i], accuracies[i]),
                    xytext=(5, 5), textcoords='offset points')

    plt.xlabel('Latency (ms)')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Pareto frontier to {output_path}")


def plot_per_exercise_accuracy(exercise_accuracies: Dict[str, float],
                               output_path: str,
                               title: str = "Per-Exercise Accuracy"):
    """Plot per-exercise accuracy bar chart."""
    plt.figure(figsize=(12, 6))

    exercises = list(exercise_accuracies.keys())
    accuracies = list(exercise_accuracies.values())

    bars = plt.bar(exercises, accuracies, color='mediumseagreen', alpha=0.8)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2%}',
                ha='center', va='bottom', fontsize=9)

    plt.xlabel('Exercise Type')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.axhline(y=np.mean(accuracies), color='r', linestyle='--',
                label=f'Mean: {np.mean(accuracies):.2%}')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved per-exercise accuracy to {output_path}")
