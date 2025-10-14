#!/bin/bash

# QVED Finetuning - Quick Start
# This script runs all necessary steps to start finetuning

set -e  # Exit on error

echo "========================================="
echo "QVED Finetuning - Quick Start"
echo "========================================="

# Step 1: Verify setup
echo -e "\n[Step 1/3] Verifying setup..."
bash scripts/verify_qved_setup.sh

# Step 2: Confirm to proceed
echo -e "\n========================================="
read -p "Setup verified! Start finetuning? (y/N): " -n 1 -r
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

echo -e "\n========================================="
echo "[Step 3/3] Finetuning complete!"
echo "========================================="
echo "Model saved to: results/qved_finetune_mobilevideogpt_0.5B/"
echo ""
echo "To use the finetuned model:"
echo "  python inference.py \\"
echo "    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-final \\"
echo "    --video_path dataset/squats/00093026.mp4 \\"
echo "    --prompt 'Analyze this squats video and provide corrective feedback.'"
echo "========================================="
