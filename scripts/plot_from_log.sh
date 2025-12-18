#!/bin/bash

# Quick script to generate training plots from existing log files
# Usage: bash scripts/plot_from_log.sh [log_file]

set -e

echo "========================================="
echo "Training Statistics Plotter"
echo "========================================="

if [ -z "$1" ]; then
    # Auto-detect latest log file
    LOG_FILE=$(find results/qved_finetune_mobilevideogpt_0.5B/ -name "training_*.log" -type f 2>/dev/null | sort -r | head -1)

    if [ -z "$LOG_FILE" ]; then
        echo "❌ No log files found in results/qved_finetune_mobilevideogpt_0.5B/"
        echo ""
        echo "Usage: bash scripts/plot_from_log.sh [log_file]"
        echo ""
        echo "Example:"
        echo "  bash scripts/plot_from_log.sh results/qved_finetune_mobilevideogpt_0.5B/training_20251015_141352.log"
        exit 1
    fi

    echo "Auto-detected log file: $LOG_FILE"
else
    LOG_FILE="$1"

    if [ ! -f "$LOG_FILE" ]; then
        echo "❌ Log file not found: $LOG_FILE"
        exit 1
    fi
fi

echo "Log file: $LOG_FILE"
echo "Output directory: plots/qved_finetune_mobilevideogpt_0.5B/"
echo "========================================="

# Activate conda environment
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate mobile_videogpt 2>/dev/null || true
fi

# Generate plots
python utils/plot_training_stats.py \
    --log_file "$LOG_FILE" \
    --model_name "qved_finetune_mobilevideogpt_0.5B"

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ Plots generated successfully!"
    echo ""
    echo "Generated files:"
    echo "  - plots/qved_finetune_mobilevideogpt_0.5B/loss.pdf"
    echo "  - plots/qved_finetune_mobilevideogpt_0.5B/gradient_norm.pdf"
    echo "  - plots/qved_finetune_mobilevideogpt_0.5B/learning_rate.pdf"
    echo "  - plots/qved_finetune_mobilevideogpt_0.5B/combined_metrics.pdf"
    echo "  - plots/qved_finetune_mobilevideogpt_0.5B/training_summary.txt"
    echo ""
    echo "PNG versions also saved for quick preview."
    echo "========================================="
else
    echo "❌ Failed to generate plots"
    exit 1
fi
