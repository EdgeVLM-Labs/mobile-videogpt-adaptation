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
from matplotlib.backends.backend_pdf import PdfPages

# Import dataset configuration constants
try:
    from utils.load_dataset import MAX_PER_CLASS
except ImportError:
    MAX_PER_CLASS = 5  # Default fallback

# Dataset split ratios (matching qved_from_fine_labels.py)
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20

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


def plot_loss(epochs: List[float], loss: List[float], output_path: str = None,
              eval_epochs: List[float] = None, eval_loss: List[float] = None, pdf: PdfPages = None):
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

    if output_path:
        plt.savefig(output_path)
        print(f"✓ Saved loss plot to: {output_path}")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def plot_gradient_norm(epochs: List[float], grad_norm: List[float], output_path: str = None, pdf: PdfPages = None):
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

    if output_path:
        plt.savefig(output_path)
        print(f"✓ Saved gradient norm plot to: {output_path}")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def plot_learning_rate(epochs: List[float], lr: List[float], output_path: str = None, pdf: PdfPages = None):
    """Plot learning rate schedule over epochs."""
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(epochs, lr, linewidth=2, color='#06A77D', marker='o', markersize=3, alpha=0.8)
    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Learning Rate}')
    ax.set_title(r'\textbf{Learning Rate Schedule}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✓ Saved learning rate plot to: {output_path}")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def plot_combined(epochs: List[float], metrics: Dict[str, List[float]], output_path: str = None,
                  eval_epochs: List[float] = None, eval_loss: List[float] = None, pdf: PdfPages = None):
    """Plot all metrics in a single figure with subplots."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))

    # Loss
    axes[0].plot(epochs, metrics['loss'], linewidth=2, color='#2E86AB', marker='o', markersize=2, alpha=0.8, label='Training Loss')

    # Plot validation loss if available
    if eval_epochs and eval_loss and len(eval_loss) > 0:
        axes[0].plot(eval_epochs, eval_loss, linewidth=2, color='#E63946', marker='s',
                markersize=4, alpha=0.8, label='Validation Loss')

    axes[0].set_xlabel(r'\textbf{Epoch}')
    axes[0].set_ylabel(r'\textbf{Loss}')
    axes[0].set_title(r'\textbf{Training and Validation Loss}')
    axes[0].grid(True, alpha=0.3, linestyle='--')

    if len(metrics['loss']) > 10:
        window = min(20, len(metrics['loss']) // 5)
        ma = np.convolve(metrics['loss'], np.ones(window)/window, mode='valid')
        ma_epochs = epochs[window-1:]
        axes[0].plot(ma_epochs, ma, linewidth=2, color='#A23B72', linestyle='--',
                    label=f'Train MA (w={window})', alpha=0.7)

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

    if output_path:
        plt.savefig(output_path)
        print(f"✓ Saved combined plot to: {output_path}")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def plot_eval_metrics(eval_epochs: List[float], eval_loss: List[float], output_path: str = None, pdf: PdfPages = None):
    """Plot validation metrics separately."""
    if not eval_epochs or not eval_loss:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(eval_epochs, eval_loss, linewidth=2, color='#E63946', marker='s', markersize=6, alpha=0.8, label='Validation Loss')

    ax.set_xlabel(r'\textbf{Epoch}')
    ax.set_ylabel(r'\textbf{Loss}')
    ax.set_title(r'\textbf{Validation Loss over Epochs}')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend()

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path)
        print(f"✓ Saved validation metrics plot to: {output_path}")

    if pdf:
        pdf.savefig(fig)

    plt.close()


def get_dataset_info() -> Dict[str, int]:
    """
    Get dataset information from JSON files.

    Returns:
        Dictionary with train_count, val_count, test_count, total_videos, num_classes
    """
    import json
    from pathlib import Path

    dataset_info = {
        'max_per_class': MAX_PER_CLASS,
        'train_count': 0,
        'val_count': 0,
        'test_count': 0,
        'total_videos': 0,
        'num_classes': 0,
        'train_ratio': TRAIN_RATIO,
        'val_ratio': VAL_RATIO,
        'test_ratio': TEST_RATIO,
    }

    dataset_dir = Path("dataset")

    # Try to read train/val/test JSON files
    try:
        train_json = dataset_dir / "qved_train.json"
        if train_json.exists():
            with open(train_json, 'r') as f:
                dataset_info['train_count'] = len(json.load(f))
    except Exception:
        pass

    try:
        val_json = dataset_dir / "qved_val.json"
        if val_json.exists():
            with open(val_json, 'r') as f:
                dataset_info['val_count'] = len(json.load(f))
    except Exception:
        pass

    try:
        test_json = dataset_dir / "qved_test.json"
        if test_json.exists():
            with open(test_json, 'r') as f:
                dataset_info['test_count'] = len(json.load(f))
    except Exception:
        pass

    # Try to get number of exercise classes from manifest
    try:
        manifest_json = dataset_dir / "manifest.json"
        if manifest_json.exists():
            with open(manifest_json, 'r') as f:
                manifest = json.load(f)
                dataset_info['total_videos'] = len(manifest)
                dataset_info['num_classes'] = len(set(manifest.values()))
    except Exception:
        pass

    return dataset_info


def get_summary_text(metrics: Dict[str, List[float]], summary: Dict[str, float]) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("Training Statistics Summary")
    lines.append("=" * 60 + "\n")

    # Dataset Information
    dataset_info = get_dataset_info()
    lines.append("Dataset Configuration:")
    lines.append(f"  Videos per Exercise Class: {dataset_info['max_per_class']}")
    lines.append(f"  Number of Exercise Classes: {dataset_info['num_classes']}")
    lines.append(f"  Total Videos Downloaded: {dataset_info['total_videos']}\n")

    lines.append("Dataset Split:")
    lines.append(f"  Train: {dataset_info['train_count']} samples ({dataset_info['train_ratio']*100:.0f}%)")
    lines.append(f"  Val:   {dataset_info['val_count']} samples ({dataset_info['val_ratio']*100:.0f}%)")
    lines.append(f"  Test:  {dataset_info['test_count']} samples ({dataset_info['test_ratio']*100:.0f}%)\n")

    # Training Summary (Final Metrics)
    if summary:
        lines.append("Training Summary:")
        lines.append(f"  Total Runtime: {summary.get('train_runtime', 0):.2f} seconds ({summary.get('train_runtime', 0)/60:.2f} minutes)")
        lines.append(f"  Samples/Second: {summary.get('train_samples_per_second', 0):.3f}")
        lines.append(f"  Steps/Second: {summary.get('train_steps_per_second', 0):.3f}")
        lines.append(f"  Average Train Loss: {summary.get('train_loss', 0):.4f}")
        lines.append(f"  Final Epoch: {summary.get('final_epoch', 0):.2f}\n")

    lines.append("Training Progress Statistics:")
    lines.append(f"  Total Training Steps: {len(metrics['loss'])}")
    lines.append(f"  Epoch Range: {min(metrics['epoch']):.2f} - {max(metrics['epoch']):.2f}\n")

    # Validation Statistics
    if 'eval_loss' in metrics and metrics['eval_loss']:
        eval_loss = metrics['eval_loss']
        eval_epochs = metrics['eval_epoch']
        lines.append("Validation Statistics:")
        lines.append(f"  Total Validation Checks: {len(eval_loss)}")
        lines.append(f"  Min Validation Loss: {min(eval_loss):.4f} (epoch {eval_epochs[eval_loss.index(min(eval_loss))]:.2f})")
        lines.append(f"  Max Validation Loss: {max(eval_loss):.4f} (epoch {eval_epochs[eval_loss.index(max(eval_loss))]:.2f})")
        lines.append(f"  Mean Validation Loss: {np.mean(eval_loss):.4f}")
        lines.append(f"  Final Validation Loss: {eval_loss[-1]:.4f}\n")

    lines.append("Loss Statistics:")
    lines.append(f"  Initial Loss: {metrics['loss'][0]:.4f}")
    lines.append(f"  Final Loss: {metrics['loss'][-1]:.4f}")
    lines.append(f"  Min Loss: {min(metrics['loss']):.4f} (epoch {metrics['epoch'][metrics['loss'].index(min(metrics['loss']))]:.2f})")
    lines.append(f"  Max Loss: {max(metrics['loss']):.4f} (epoch {metrics['epoch'][metrics['loss'].index(max(metrics['loss']))]:.2f})")
    lines.append(f"  Mean Loss: {np.mean(metrics['loss']):.4f}")
    lines.append(f"  Std Loss: {np.std(metrics['loss']):.4f}\n")

    lines.append("Gradient Norm Statistics:")
    lines.append(f"  Min Grad Norm: {min(metrics['grad_norm']):.4f}")
    lines.append(f"  Max Grad Norm: {max(metrics['grad_norm']):.4f}")
    lines.append(f"  Mean Grad Norm: {np.mean(metrics['grad_norm']):.4f}")
    lines.append(f"  Std Grad Norm: {np.std(metrics['grad_norm']):.4f}\n")

    lines.append("Learning Rate Statistics:")
    lines.append(f"  Initial LR: {metrics['learning_rate'][0]:.2e}")
    lines.append(f"  Final LR: {metrics['learning_rate'][-1]:.2e}")
    lines.append(f"  Max LR: {max(metrics['learning_rate']):.2e}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def create_summary_page(pdf: PdfPages, metrics: Dict[str, List[float]], summary: Dict[str, float]):
    """Create a summary page in the PDF report."""
    text = get_summary_text(metrics, summary)

    fig = plt.figure(figsize=(8.5, 11))
    plt.axis('off')
    plt.text(0.1, 0.95, text, transform=fig.transFigure, fontsize=10, verticalalignment='top', fontfamily='monospace')
    pdf.savefig(fig)
    plt.close()


def save_summary_stats(metrics: Dict[str, List[float]], summary: Dict[str, float], output_path: str):
    """Save training statistics summary to a text file."""
    text = get_summary_text(metrics, summary)
    with open(output_path, 'w') as f:
        f.write(text)
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

    report_path = output_dir / "training_report.pdf"
    with PdfPages(report_path) as pdf:
        # Summary page
        create_summary_page(pdf, metrics, summary)

        # Plots
        plot_loss(epochs, metrics['loss'], str(output_dir / "loss.png"), eval_epochs, eval_loss, pdf=pdf)
        plot_gradient_norm(epochs, metrics['grad_norm'], str(output_dir / "gradient_norm.png"), pdf=pdf)
        plot_learning_rate(epochs, metrics['learning_rate'], str(output_dir / "learning_rate.png"), pdf=pdf)
        plot_combined(epochs, metrics, str(output_dir / "combined_metrics.png"), eval_epochs, eval_loss, pdf=pdf)
        plot_eval_metrics(eval_epochs, eval_loss, str(output_dir / "eval_metrics.png"), pdf=pdf)

    print(f"✓ Saved training report to: {report_path}")

    # Save statistics summary
    save_summary_stats(metrics, summary, str(output_dir / "training_summary.txt"))

    print("\n" + "=" * 60)
    print("✓ All plots generated successfully!")
    print(f"✓ Output directory: {output_dir.absolute()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
