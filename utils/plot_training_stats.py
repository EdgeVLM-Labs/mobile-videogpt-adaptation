#!/usr/bin/env python3
"""
Training Statistics Plotter for Mobile-VideoGPT Finetuning

This script parses training logs and generates publication-quality plots
for loss, gradient norm, and learning rate over epochs.

Usage:
    python utils/plot_training_stats.py --log_file <path_to_log> --output_dir plots/<model_name>
    python utils/plot_training_stats.py --model_name qved_finetune_mobilevideogpt_0.5B
"""

import os
import re
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple

# Configure matplotlib to use LaTeX for text rendering
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 12,
    "font.size": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.figsize": (8, 6),
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})


def parse_training_log(log_file: str) -> Dict[str, List[float]]:
    """
    Parse training log file and extract metrics.

    Args:
        log_file: Path to the training log file

    Returns:
        Dictionary containing lists of metrics: epoch, loss, grad_norm, learning_rate, eval_loss, eval_epoch
    """
    metrics = {
        'epoch': [],
        'loss': [],
        'grad_norm': [],
        'learning_rate': [],
        'eval_loss': [],
        'eval_epoch': []
    }

    # Pattern to match training log lines with metrics
    # Example: {'loss': 0.2278, 'grad_norm': 0.379, 'learning_rate': 6.929e-08, 'epoch': 9.71}
    train_pattern = r"\{'loss': ([\d.]+), 'grad_norm': ([\d.]+), 'learning_rate': ([\de.\-+]+), 'epoch': ([\d.]+)\}"

    # Pattern to match eval log lines
    # Example: {'eval_loss': 0.5123, 'eval_runtime': 12.34, ..., 'epoch': 2.5}
    eval_pattern = r"\{'eval_loss': ([\d.]+).*?'epoch': ([\d.]+)\}"

    with open(log_file, 'r') as f:
        for line in f:
            # Try training pattern first
            train_match = re.search(train_pattern, line)
            if train_match:
                loss, grad_norm, lr, epoch = train_match.groups()
                metrics['loss'].append(float(loss))
                metrics['grad_norm'].append(float(grad_norm))
                metrics['learning_rate'].append(float(lr))
                metrics['epoch'].append(float(epoch))
                continue

            # Try eval pattern
            eval_match = re.search(eval_pattern, line)
            if eval_match:
                eval_loss, epoch = eval_match.groups()
                metrics['eval_loss'].append(float(eval_loss))
                metrics['eval_epoch'].append(float(epoch))

    return metrics


def parse_training_summary(log_file: str) -> Dict[str, float]:
    """
    Parse final training summary from log.

    Returns:
        Dictionary with train_runtime, train_samples_per_second, train_steps_per_second, train_loss
    """
    summary = {}
    pattern = r"\{'train_runtime': ([\d.]+), 'train_samples_per_second': ([\d.]+), 'train_steps_per_second': ([\d.]+), 'train_loss': ([\d.]+), 'epoch': ([\d.]+)\}"

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                summary['train_runtime'] = float(match.group(1))
                summary['train_samples_per_second'] = float(match.group(2))
                summary['train_steps_per_second'] = float(match.group(3))
                summary['train_loss'] = float(match.group(4))
                summary['final_epoch'] = float(match.group(5))
                break

    return summary


def plot_loss(epochs: List[float], loss: List[float], output_path: str,
              eval_epochs: List[float] = None, eval_loss: List[float] = None):
    """Plot training and validation loss over epochs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot training loss
    ax.plot(epochs, loss, linewidth=2, color='#2E86AB', marker='o', markersize=3,
            alpha=0.8, label='Training Loss')

    # Plot validation loss if available
    if eval_epochs and eval_loss and len(eval_loss) > 0:
        ax.plot(eval_epochs, eval_loss, linewidth=2, color='#E63946', marker='s',
                markersize=4, alpha=0.8, label='Validation Loss')

    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Loss}')
    ax.set_title(r'\textbf{Training and Validation Loss over Epochs}')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add moving average for training loss
    if len(loss) > 10:
        window = min(20, len(loss) // 5)
        ma = np.convolve(loss, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        ax.plot(ma_epochs, ma, linewidth=2, color='#A23B72', linestyle='--',
                label=f'Train MA (w={window})', alpha=0.7)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved loss plot to: {output_path}")


def plot_gradient_norm(epochs: List[float], grad_norm: List[float], output_path: str):
    """Plot gradient norm over epochs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(epochs, grad_norm, linewidth=2, color='#F18F01', marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Gradient Norm}')
    ax.set_title(r'\textbf{Gradient Norm over Epochs}')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add moving average
    if len(grad_norm) > 10:
        window = min(20, len(grad_norm) // 5)
        ma = np.convolve(grad_norm, np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        ax.plot(ma_epochs, ma, linewidth=2, color='#C73E1D', linestyle='--',
                label=f'Moving Avg (window={window})', alpha=0.7)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved gradient norm plot to: {output_path}")


def plot_learning_rate(epochs: List[float], lr: List[float], output_path: str):
    """Plot learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(epochs, lr, linewidth=2, color='#06A77D', marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Learning Rate}')
    ax.set_title(r'\textbf{Learning Rate Schedule}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved learning rate plot to: {output_path}")


def plot_combined(epochs: List[float], metrics: Dict[str, List[float]], output_path: str):
    """Plot all metrics in a single figure with subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Loss
    axes[0].plot(epochs, metrics['loss'], linewidth=2, color='#2E86AB', marker='o', markersize=2, alpha=0.8)
    axes[0].set_xlabel(r'\textbf{Epoch}')
    axes[0].set_ylabel(r'\textbf{Training Loss}')
    axes[0].set_title(r'\textbf{Training Loss}')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    if len(metrics['loss']) > 10:
        window = min(20, len(metrics['loss']) // 5)
        ma = np.convolve(metrics['loss'], np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        axes[0].plot(ma_epochs, ma, linewidth=2, color='#A23B72', linestyle='--',
                    label=f'Moving Avg (w={window})', alpha=0.7)
        axes[0].legend()

    # Gradient Norm
    axes[1].plot(epochs, metrics['grad_norm'], linewidth=2, color='#F18F01', marker='o', markersize=2, alpha=0.8)
    axes[1].set_xlabel(r'\textbf{Epoch}')
    axes[1].set_ylabel(r'\textbf{Gradient Norm}')
    axes[1].set_title(r'\textbf{Gradient Norm}')
    axes[1].grid(True, alpha=0.3, linestyle='--')

    # Learning Rate
    axes[2].plot(epochs, metrics['learning_rate'], linewidth=2, color='#06A77D', marker='o', markersize=2, alpha=0.8)
    axes[2].set_xlabel(r'\textbf{Epoch}')
    axes[2].set_ylabel(r'\textbf{Learning Rate}')
    axes[2].set_title(r'\textbf{Learning Rate Schedule}')
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"✓ Saved combined plot to: {output_path}")


def save_summary_stats(metrics: Dict[str, List[float]], summary: Dict[str, float], output_path: str):
    """Save training statistics summary to a text file."""
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("Training Statistics Summary\n")
        f.write("=" * 60 + "\n\n")

        if summary:
            f.write("Final Training Metrics:\n")
            f.write(f"  Total Runtime: {summary.get('train_runtime', 0):.2f} seconds ({summary.get('train_runtime', 0)/60:.2f} minutes)\n")
            f.write(f"  Samples/Second: {summary.get('train_samples_per_second', 0):.3f}\n")
            f.write(f"  Steps/Second: {summary.get('train_steps_per_second', 0):.3f}\n")
            f.write(f"  Average Train Loss: {summary.get('train_loss', 0):.4f}\n")
            f.write(f"  Final Epoch: {summary.get('final_epoch', 0):.2f}\n\n")

        f.write("Training Progress Statistics:\n")
        f.write(f"  Total Training Steps: {len(metrics['loss'])}\n")
        f.write(f"  Epoch Range: {min(metrics['epoch']):.2f} - {max(metrics['epoch']):.2f}\n\n")

        f.write("Loss Statistics:\n")
        f.write(f"  Initial Loss: {metrics['loss'][0]:.4f}\n")
        f.write(f"  Final Loss: {metrics['loss'][-1]:.4f}\n")
        f.write(f"  Min Loss: {min(metrics['loss']):.4f} (epoch {metrics['epoch'][metrics['loss'].index(min(metrics['loss']))]:.2f})\n")
        f.write(f"  Max Loss: {max(metrics['loss']):.4f} (epoch {metrics['epoch'][metrics['loss'].index(max(metrics['loss']))]:.2f})\n")
        f.write(f"  Mean Loss: {np.mean(metrics['loss']):.4f}\n")
        f.write(f"  Std Loss: {np.std(metrics['loss']):.4f}\n\n")

        f.write("Gradient Norm Statistics:\n")
        f.write(f"  Min Grad Norm: {min(metrics['grad_norm']):.4f}\n")
        f.write(f"  Max Grad Norm: {max(metrics['grad_norm']):.4f}\n")
        f.write(f"  Mean Grad Norm: {np.mean(metrics['grad_norm']):.4f}\n")
        f.write(f"  Std Grad Norm: {np.std(metrics['grad_norm']):.4f}\n\n")

        f.write("Learning Rate Statistics:\n")
        f.write(f"  Initial LR: {metrics['learning_rate'][0]:.2e}\n")
        f.write(f"  Final LR: {metrics['learning_rate'][-1]:.2e}\n")
        f.write(f"  Max LR: {max(metrics['learning_rate']):.2e}\n")

        f.write("\n" + "=" * 60 + "\n")

    print(f"✓ Saved statistics summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot training statistics from Mobile-VideoGPT finetuning logs")
    parser.add_argument("--log_file", type=str, help="Path to training log file")
    parser.add_argument("--model_name", type=str, default="qved_finetune_mobilevideogpt_0.5B",
                        help="Model name for output directory")
    parser.add_argument("--output_dir", type=str, help="Custom output directory (overrides model_name)")

    args = parser.parse_args()

    # Determine log file path
    if args.log_file is None:
        # Try to find the most recent log file in results directory
        results_dir = Path(f"results/{args.model_name}")
        if results_dir.exists():
            log_files = list(results_dir.glob("*.log"))
            if log_files:
                args.log_file = str(max(log_files, key=os.path.getctime))
                print(f"Auto-detected log file: {args.log_file}")
            else:
                print("❌ No log files found in results directory")
                print("Please provide --log_file argument")
                return
        else:
            print(f"❌ Results directory not found: {results_dir}")
            print("Please provide --log_file argument")
            return

    if not os.path.exists(args.log_file):
        print(f"❌ Log file not found: {args.log_file}")
        return

    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("plots") / args.model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Training Statistics Plotter")
    print("=" * 60)
    print(f"Log file: {args.log_file}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

    # Parse metrics
    print("\nParsing training log...")
    metrics = parse_training_log(args.log_file)
    summary = parse_training_summary(args.log_file)

    if not metrics['epoch']:
        print("❌ No training metrics found in log file")
        return

    print(f"✓ Found {len(metrics['epoch'])} training steps")
    print(f"  Epoch range: {min(metrics['epoch']):.2f} - {max(metrics['epoch']):.2f}")

    # Generate plots
    print("\nGenerating plots...")
    epochs = metrics['epoch']
    eval_epochs = metrics.get('eval_epoch', [])
    eval_loss = metrics.get('eval_loss', [])

    if eval_loss:
        print(f"  Found {len(eval_loss)} validation checkpoints")

    plot_loss(epochs, metrics['loss'], str(output_dir / "loss.pdf"), eval_epochs, eval_loss)
    plot_gradient_norm(epochs, metrics['grad_norm'], str(output_dir / "gradient_norm.pdf"))
    plot_learning_rate(epochs, metrics['learning_rate'], str(output_dir / "learning_rate.pdf"))
    plot_combined(epochs, metrics, str(output_dir / "combined_metrics.pdf"))

    # Save PNG versions as well
    plot_loss(epochs, metrics['loss'], str(output_dir / "loss.png"), eval_epochs, eval_loss)
    plot_gradient_norm(epochs, metrics['grad_norm'], str(output_dir / "gradient_norm.png"))
    plot_learning_rate(epochs, metrics['learning_rate'], str(output_dir / "learning_rate.png"))
    plot_combined(epochs, metrics, str(output_dir / "combined_metrics.png"))

    # Save statistics summary
    save_summary_stats(metrics, summary, str(output_dir / "training_summary.txt"))

    print("\n" + "=" * 60)
    print("✓ All plots generated successfully!")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
