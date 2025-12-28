#!/bin/bash
# RQ1.1.3: Data Efficiency Analysis

set -e

MODEL=${1:-"Mobile-VideoGPT-0.5B"}
OUTPUT_BASE="experiments/results/rq1_training_efficiency/rq1_1_3"
TRAIN_DATA="dataset/qved_train.json"
SUBSET_DIR="experiments/data/subsets"

echo "=== RQ1.1.3: Data Efficiency ==="

# Generate stratified subsets
echo "Generating dataset subsets..."
python experiments/shared/data_utils.py \
    --action create_subsets \
    --input "$TRAIN_DATA" \
    --output_dir "$SUBSET_DIR" \
    --ratios 0.25 0.5 0.75 1.0

# Train on each subset
for RATIO in 0.25 0.5 0.75 1.0; do
    SUBSET_FILE="$SUBSET_DIR/qved_train_${RATIO}.json"
    OUTPUT_DIR="$OUTPUT_BASE/data_${RATIO}"

    echo "Training on ${RATIO} of data..."
    mkdir -p "$OUTPUT_DIR"

    bash scripts/finetune_qved.sh \
        --model_name_or_path "$MODEL" \
        --data_path "$SUBSET_FILE" \
        --output_dir "$OUTPUT_DIR/checkpoint" \
        --num_train_epochs 3 \
        --use_lora True \
        2>&1 | tee "$OUTPUT_DIR/training.log"

    # Evaluate
    python utils/test_inference.py \
        --model_path "$OUTPUT_DIR/checkpoint" \
        --test_json "dataset/ground_truth.json" \
        --output_dir "$OUTPUT_DIR/inference"
done

# Analyze data efficiency curves
python experiments/rq1_training_efficiency/analyze_data_efficiency.py \
    --results_dir "$OUTPUT_BASE" \
    --output "$OUTPUT_BASE/data_efficiency_curves.pdf"

echo "=== RQ1.1.3 Complete ==="
