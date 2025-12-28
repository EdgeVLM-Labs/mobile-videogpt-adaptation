#!/bin/bash
# RQ1.1.2: Learning Rate Schedules Comparison

set -e

MODEL=${1:-"Mobile-VideoGPT-0.5B"}
OUTPUT_BASE="experiments/results/rq1_training_efficiency/rq1_1_2"
TRAIN_DATA="dataset/qved_train.json"

echo "=== RQ1.1.2: Learning Rate Schedules ==="

# Test different schedulers
SCHEDULERS=("cosine" "linear" "constant")
WARMUP_RATIOS=(0.0 0.03 0.05 0.1)

for SCHEDULE in "${SCHEDULERS[@]}"; do
    for WARMUP in "${WARMUP_RATIOS[@]}"; do
        OUTPUT_DIR="$OUTPUT_BASE/${SCHEDULE}_warmup${WARMUP}"
        echo "Training with scheduler=$SCHEDULE, warmup=$WARMUP..."

        mkdir -p "$OUTPUT_DIR"

        bash scripts/finetune_qved.sh \
            --model_name_or_path "$MODEL" \
            --data_path "$TRAIN_DATA" \
            --output_dir "$OUTPUT_DIR/checkpoint" \
            --num_train_epochs 3 \
            --learning_rate 2e-4 \
            --warmup_ratio "$WARMUP" \
            --lr_scheduler_type "$SCHEDULE" \
            --use_lora True \
            2>&1 | tee "$OUTPUT_DIR/training.log"

        # Run inference
        python utils/test_inference.py \
            --model_path "$OUTPUT_DIR/checkpoint" \
            --test_json "dataset/ground_truth.json" \
            --output_dir "$OUTPUT_DIR/inference"
    done
done

# Analyze convergence
python experiments/rq1_training_efficiency/analyze_convergence.py \
    --results_dir "$OUTPUT_BASE" \
    --output "$OUTPUT_BASE/convergence_analysis.pdf"

echo "=== RQ1.1.2 Complete ==="
echo "Results: $OUTPUT_BASE"
