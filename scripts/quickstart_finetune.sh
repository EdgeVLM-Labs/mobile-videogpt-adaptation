#!/bin/bash

# QVED Finetuning - Quick Start
# This script runs all necessary steps to start finetuning

set -e  # Exit on error

# Setup logging
mkdir -p results
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="results/finetune_${TIMESTAMP}.log"
echo "Logging all output to: $LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "QVED Finetuning - Quick Start"
echo "========================================="

# Step 1: Verify setup
echo -e "\n[Step 1/4] Verifying setup..."
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
echo -e "\n[Step 2/4] Activating conda environment and starting finetuning..."
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
echo "[Step 3/4] Finetuning complete!"
echo "========================================="
echo "Model saved to: results/qved_finetune_mobilevideogpt_0.5B/"
echo ""
echo "Finding latest checkpoint..."
LATEST_CKPT=$(ls -d results/qved_finetune_mobilevideogpt_0.5B/checkpoint-* 2>/dev/null | sort -V | tail -1)
if [ -n "$LATEST_CKPT" ]; then
    echo "Latest checkpoint: $LATEST_CKPT"
    MODEL_PATH="$LATEST_CKPT"
else
    echo "Note: LoRA adapters saved in results/qved_finetune_mobilevideogpt_0.5B/"
    MODEL_PATH="results/qved_finetune_mobilevideogpt_0.5B"
fi

# Step 4: Upload to HuggingFace
echo -e "\n========================================="
echo "[Step 4/4] Upload to HuggingFace"
echo "========================================="
echo -n "Upload finetuned model to HuggingFace? (y/N): "
read -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    HF_REPO_NAME="mobile-videogpt-finetune-${TIMESTAMP}"
    echo "Uploading to EdgeVLM-Labs/${HF_REPO_NAME}..."
    python utils/hf_upload.py \
        --model_path "$MODEL_PATH" \
        --repo_name "$HF_REPO_NAME" \
        --org "EdgeVLM-Labs"

    if [ $? -eq 0 ]; then
        echo "✓ Model uploaded successfully to HuggingFace!"
        echo "  URL: https://huggingface.co/EdgeVLM-Labs/${HF_REPO_NAME}"
    else
        echo "⚠ Warning: Failed to upload model to HuggingFace"
        echo "  You can upload manually later with:"
        echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
    fi
else
    echo "Skipping HuggingFace upload."
    echo "You can upload later with:"
    echo "  python utils/hf_upload.py --model_path $MODEL_PATH"
fi

echo -e "\n========================================="
echo "All steps complete!"
echo "========================================="
echo ""
echo "To use the finetuned model:"
echo "  python utils/infer_qved.py \\"
echo "    --model_path $MODEL_PATH \\"
echo "    --video_path sample_videos/00000340.mp4"
echo ""
echo "Adjustable parameters in utils/infer_qved.py:"
echo "  --model_path       Path to model checkpoint (default: Amshaker/Mobile-VideoGPT-0.5B)"
echo "  --video_path       Path to video file (default: sample_videos/00000340.mp4)"
echo "  --prompt           Custom prompt (default: physiotherapy evaluation prompt)"
echo "  --device           Device to use (default: cuda, options: cuda/cpu)"
echo "  --max_new_tokens   Max tokens to generate (default: 512)"
echo ""
echo "To run the inference script:"
echo ""
echo "Using local checkpoint:"
echo "  bash scripts/run_inference.sh --model_path $MODEL_PATH"
echo ""
echo "Using HuggingFace model:"
echo "  bash scripts/run_inference.sh --hf_repo EdgeVLM-Labs/${HF_REPO_NAME}"
echo ""
echo "========================================="
