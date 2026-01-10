#!/bin/bash

# QVED Finetuning Script for Mobile-VideoGPT-0.5B
# This script performs Stage 3 (finetuning) only, using the pre-trained Mobile-VideoGPT-0.5B checkpoint
# The checkpoint already includes pre-trained video and image projectors from Stages 1 and 2

# Environment setup
export PYTHONPATH="./:$PYTHONPATH"
export DATASET_DIR="$(pwd)/playground/data"

# Suppress DeepSpeed hostfile warning for single-GPU training
export PDSH_RCMD_TYPE=ssh

# WandB Configuration
export WANDB_PROJECT="mobile-videogpt"
export WANDB_ENTITY="fyp-21"
export WANDB_NAME="qved-finetune-$(date +%Y%m%d_%H%M%S)"

# Model paths - using pre-trained Mobile-VideoGPT-1.5B checkpoint
BASE_LLM_PATH="Amshaker/Mobile-VideoGPT-1.5B"
VISION_TOWER="OpenGVLab/VideoMamba"
IMAGE_VISION_TOWER="openai/clip-vit-base-patch16"
PROJECTOR_TYPE="etp"

# Output directory for finetuned model
OUTPUT_DIR_PATH="results/qved_finetune_mobilevideogpt_1.5B"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR_PATH"

# =========================================
# HYPERPARAMETERS - Configure all values here
# =========================================

# Video processing parameters
FPS=1                        # Frame sampling rate
MAX_FRAMES=16                # Maximum frames per video
INPUT_RESOLUTION=224         # Input resolution to encoder (224 x 224)

# Training epochs and learning rates
EPOCHS=3                     # Number of training epochs
LR=2e-4                      # Learning rate
MM_PROJ_LR=1e-4              # Projector learning rate

# LoRA configuration
LORA_R=64                    # LoRA rank
LORA_ALPHA=128               # LoRA alpha
LORA_DROPOUT=0.05            # LoRA dropout

# Batch size configuration
BATCH=8                      # Per device train batch size
GACC=4                       # Gradient accumulation steps
EVAL_BATCH=8                 # Per device eval batch size

# Sequence length
MAXLEN=2048                  # Max sequence length

# Evaluation and saving
EVAL_STRATEGY="steps"        # Evaluation strategy
EVAL_STEPS=30                # Evaluation steps
SAVE_STRATEGY="steps"        # Save strategy
SAVE_STEPS=30                # Save steps
SAVE_TOTAL_LIMIT=3           # Maximum checkpoints to keep

# Training configuration
WARMUP_RATIO=0.05            # Warmup ratio
LOGGING_STEPS=1              # Logging steps
DATALOADER_WORKERS=2         # Dataloader num workers
GRADIENT_CHECKPOINTING=True  # Enable gradient checkpointing

# Precision settings (True/False)
BF16=True                    # Use bfloat16
TF32=True                    # Use TensorFloat-32
FP16=False                   # Use float16

# DeepSpeed configuration
DEEPSPEED_CONFIG="scripts/zero2.json"  # ZeRO optimization config

echo "========================================="
echo "QVED Dataset Finetuning Configuration"
echo "========================================="
echo "Base Model: $BASE_LLM_PATH"
echo "Output Dir: $OUTPUT_DIR_PATH"
echo "Epochs: $EPOCHS"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH x $GACC accumulation steps = effective batch of $((BATCH * GACC))"
echo "FPS: $FPS"
echo "Max Frames: $MAX_FRAMES"
echo "========================================="

# Save hyperparameters to a config file
CONFIG_FILE="$OUTPUT_DIR_PATH/hyperparameters.json"
cat <<EOF > "$CONFIG_FILE"
{
  "base_model": "$BASE_LLM_PATH",
  "dataset": "QVED",
  "epochs": $EPOCHS,
  "learning_rate": $LR,
  "mm_projector_lr": $MM_PROJ_LR,
  "lora_r": $LORA_R,
  "lora_alpha": $LORA_ALPHA,
  "batch_size": $BATCH,
  "gradient_accumulation_steps": $GACC,
  "max_length": $MAXLEN,
  "wandb_project": "$WANDB_PROJECT",
  "wandb_entity": "$WANDB_ENTITY",
  "wandb_run_name": "$WANDB_NAME"
}
EOF
echo "Hyperparameters saved to $CONFIG_FILE"

# Stage 3: Fine-tuning on QVED dataset
# The Mobile-VideoGPT-0.5B checkpoint already includes trained projectors,
# so we don't need to specify pretrain_mm_mlp_adapter or pretrain_image_mm_mlp_adapter
#
# Note: Using ZeRO-2 instead of ZeRO-3 due to Mamba SSM compatibility issues
# ZeRO-3 causes tensor initialization errors with mamba_ssm modules

deepspeed mobilevideogpt/train/train.py \
  --deepspeed $DEEPSPEED_CONFIG \
  --lora_enable True \
  --lora_r $LORA_R \
  --lora_alpha $LORA_ALPHA \
  --lora_dropout $LORA_DROPOUT \
  --lora_bias none \
  --mm_projector_lr $MM_PROJ_LR \
  --model_name_or_path "$BASE_LLM_PATH" \
  --version qwen2_instruct \
  --dataset_use QVED_TRAIN \
  --dataset_val QVED_VAL \
  --vision_tower "$VISION_TOWER" \
  --image_vision_tower "$IMAGE_VISION_TOWER" \
  --mm_projector_type "$PROJECTOR_TYPE" \
  --image_mm_projector_type "$PROJECTOR_TYPE" \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --image_aspect_ratio pad \
  --group_by_modality_length True \
  --bf16 $BF16 \
  --tf32 $TF32 \
  --fp16 $FP16 \
  --gradient_checkpointing $GRADIENT_CHECKPOINTING \
  --output_dir "$OUTPUT_DIR_PATH" \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH \
  --per_device_eval_batch_size $EVAL_BATCH \
  --gradient_accumulation_steps $GACC \
  --eval_strategy $EVAL_STRATEGY \
  --eval_steps $EVAL_STEPS \
  --save_strategy $SAVE_STRATEGY \
  --save_steps $SAVE_STEPS \
  --save_total_limit $SAVE_TOTAL_LIMIT \
  --learning_rate $LR \
  --weight_decay 0. \
  --warmup_ratio $WARMUP_RATIO \
  --lr_scheduler_type "cosine" \
  --logging_steps $LOGGING_STEPS \
  --model_max_length $MAXLEN \
  --dataloader_num_workers $DATALOADER_WORKERS \
  --lazy_preprocess True \
  --report_to wandb \
  --run_name $WANDB_NAME \
  --num_select_k_frames_in_chunk 4 \
  --topk True \
  --fps $FPS \
  --max_frames $MAX_FRAMES

echo "========================================="
echo "Finetuning completed!"
echo "Model saved to: $OUTPUT_DIR_PATH"
echo "========================================="
