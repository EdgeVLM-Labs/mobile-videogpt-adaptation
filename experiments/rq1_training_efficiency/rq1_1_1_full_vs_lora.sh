#!/bin/bash
# RQ1.1.1: Full Model Fine-tuning vs LoRA
# Compare parameter efficiency and performance

set -e

# Configuration
MODEL=${1:-"Mobile-VideoGPT-0.5B"}
OUTPUT_BASE="experiments/results/rq1_training_efficiency/rq1_1_1"
TRAIN_DATA="dataset/qved_train.json"
VAL_DATA="dataset/qved_train.json"  # Use subset for validation

echo "=== RQ1.1.1: Full vs LoRA Fine-tuning ==="
echo "Model: $MODEL"

# Create output directories
mkdir -p "$OUTPUT_BASE/lora" "$OUTPUT_BASE/full"

# 1. LoRA Fine-tuning (default)
echo "Training with LoRA..."
bash scripts/finetune_qved.sh \
    --model_name_or_path "$MODEL" \
    --data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_BASE/lora/checkpoint" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-4 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --bf16 True \
    --use_lora True \
    --lora_r 16 \
    --lora_alpha 32 \
    2>&1 | tee "$OUTPUT_BASE/lora/training.log"

# Count trainable parameters
echo "Counting LoRA trainable parameters..."
python experiments/rq1_training_efficiency/analyze_parameters.py \
    --checkpoint "$OUTPUT_BASE/lora/checkpoint" \
    --output "$OUTPUT_BASE/lora/parameters.json"

# 2. Full Fine-tuning (disable LoRA)
echo "Training with Full Fine-tuning..."
bash scripts/finetune_qved.sh \
    --model_name_or_path "$MODEL" \
    --data_path "$TRAIN_DATA" \
    --output_dir "$OUTPUT_BASE/full/checkpoint" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --warmup_ratio 0.05 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --save_strategy "epoch" \
    --bf16 True \
    --use_lora False \
    2>&1 | tee "$OUTPUT_BASE/full/training.log"

# Count trainable parameters
echo "Counting full fine-tuning parameters..."
python experiments/rq1_training_efficiency/analyze_parameters.py \
    --checkpoint "$OUTPUT_BASE/full/checkpoint" \
    --output "$OUTPUT_BASE/full/parameters.json"

# 3. Run inference on test set for both
echo "Running inference with LoRA model..."
python utils/test_inference.py \
    --model_path "$OUTPUT_BASE/lora/checkpoint" \
    --test_json "dataset/ground_truth.json" \
    --output_dir "$OUTPUT_BASE/lora/inference"

echo "Running inference with full model..."
python utils/test_inference.py \
    --model_path "$OUTPUT_BASE/full/checkpoint" \
    --test_json "dataset/ground_truth.json" \
    --output_dir "$OUTPUT_BASE/full/inference"

# 4. Generate comparison report
echo "Generating comparison report..."
python experiments/rq1_training_efficiency/compare_lora_full.py \
    --lora_dir "$OUTPUT_BASE/lora" \
    --full_dir "$OUTPUT_BASE/full" \
    --output "$OUTPUT_BASE/comparison_report.json"

echo "=== RQ1.1.1 Complete ==="
echo "Results saved to: $OUTPUT_BASE"
