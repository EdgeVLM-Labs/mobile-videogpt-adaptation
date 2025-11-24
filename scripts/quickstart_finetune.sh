#!/bin/bash

# QVED Finetuning - Quick Start
# This script runs all necessary steps to start finetuning

set -e  # Exit on error

# Setup logging
mkdir -p results
LOG_FILE="results/finetune_$(date +%Y%m%d_%H%M%S).log"
echo "Logging all output to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "QVED Finetuning - Quick Start"
echo "========================================="

# Step 1: Verify setup
echo -e "\n[Step 1/3] Verifying setup..."
bash scripts/verify_qved_setup.sh

# Step 2: Confirm to proceed
echo -e "\n========================================="
echo -n "Setup verified! Start finetuning? (y/N): "
read -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 0
fi

# Step 3: Start finetuning
echo -e "\n[Step 2/3] Activating conda environment and starting finetuning..."
echo "This will take approximately 30-50 minutes..."
echo "========================================="

# Run finetuning
bash scripts/finetune_qved.sh

# Generate training plots
echo ""
echo "Generating training plots..."
python utils/plot_training_stats.py \
  --log_file "$LOG_FILE" \
  --model_name "qved_finetune_mobilevideogpt_0.5B"

if [ $? -eq 0 ]; then
    echo "✓ Training plots generated successfully!"
    echo "  Location: plots/qved_finetune_mobilevideogpt_0.5B/"
else
    echo "⚠ Warning: Failed to generate plots. You can generate them later with:"
    echo "  python utils/plot_training_stats.py --log_file $LOG_FILE"
fi

echo -e "\n========================================="
echo "[Step 3/3] Finetuning complete!"
echo "========================================="
echo "Model saved to: results/qved_finetune_mobilevideogpt_0.5B/"
echo ""
echo "Finding latest checkpoint..."
LATEST_CKPT=$(ls -d results/qved_finetune_mobilevideogpt_0.5B/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint: $LATEST_CKPT"
    echo ""
    echo "To use the finetuned model:"
    echo "  python scripts/infer_qved.py \\"
    echo "    --model_path $LATEST_CKPT \\"
    echo "    --video_path sample_videos/00000340.mp4"
else
    echo "Note: LoRA adapters saved in results/qved_finetune_mobilevideogpt_0.5B/"
    echo ""
    echo "To use the finetuned model:"
    echo "  python scripts/infer_qved.py \\"
    echo "    --model_path results/qved_finetune_mobilevideogpt_0.5B \\"
    echo "    --video_path sample_videos/00000340.mp4"
fi
echo ""
echo "Adjustable parameters in scripts/infer_qved.py:"
echo "  --model_path       Path to model checkpoint (default: Amshaker/Mobile-VideoGPT-0.5B)"
echo "  --video_path       Path to video file (default: sample_videos/00000340.mp4)"
echo "  --prompt           Custom prompt (default: physiotherapy evaluation prompt)"
echo "  --device           Device to use (default: cuda, options: cuda/cpu)"
echo "  --max_new_tokens   Max tokens to generate (default: 512)"
echo ""
echo "========================================="
