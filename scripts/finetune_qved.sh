#!/bin/bash

# env and paths
export PYTHONPATH="./:$PYTHONPATH"
export DATASET_DIR="$(pwd)/playground/data"
BASE_LLM_PATH="Amshaker/Mobile-VideoGPT-0.5B"     # released checkpoint
VISION_TOWER="OpenGVLab/VideoMamba"
IMAGE_VISION_TOWER="openai/clip-vit-base-patch16"
PROJECTOR_TYPE="etp"
OUTPUT_DIR_PATH="results/qved_finetune_mobilevideogpt"

# hyperparams adapted for small data (PDF):
EPOCHS=8                    # 5–10 recommended; pick 8
LR=1e-4                     # lower than 2e-4 default for LoRA
MM_PROJ_LR=2e-5             # keep small
LORA_R=64                   # reduce from 128
LORA_ALPHA=128
BATCH=8
GACC=2
MAXLEN=2048

deepspeed mobilevideogpt/train/train.py \
  --deepspeed scripts/zero3.json \
  --model_name_or_path "$BASE_LLM_PATH" \
  --version qwen2_instruct \
  --dataset_use MobileGPT \
  --vision_tower "$VISION_TOWER" \
  --image_vision_tower "$IMAGE_VISION_TOWER" \
  --mm_projector_type "$PROJECTOR_TYPE" \
  --image_mm_projector_type "$PROJECTOR_TYPE" \
  --lora_enable True --lora_r $LORA_R --lora_alpha $LORA_ALPHA \
  --mm_projector_lr $MM_PROJ_LR \
  --bf16 True --tf32 True --gradient_checkpointing True \
  --output_dir $OUTPUT_DIR_PATH \
  --num_train_epochs $EPOCHS \
  --per_device_train_batch_size $BATCH --gradient_accumulation_steps $GACC \
  --per_device_eval_batch_size 4 \
  --learning_rate $LR --warmup_ratio 0.03 --lr_scheduler_type "cosine" \
  --model_max_length $MAXLEN \
  --dataloader_num_workers 4 --lazy_preprocess True \
  --num_select_k_frames_in_chunk 4 --topk True \
  --report_to none
