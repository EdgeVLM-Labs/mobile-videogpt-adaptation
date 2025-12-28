#!/bin/bash
# RQ1.3.3: Visual Degradation Robustness

set -e

OUTPUT_BASE="experiments/results/rq1_robustness"
CHECKPOINT=${1:-"results/checkpoint-70"}
TEST_DATA="dataset/ground_truth.json"
DEGRADED_DIR="experiments/data/degraded_videos"

echo "=== RQ1.3.3: Visual Degradation Robustness ==="

mkdir -p "$OUTPUT_BASE" "$DEGRADED_DIR"

# Generate degraded videos
echo "Generating degraded test videos..."
python experiments/rq1_robustness/generate_degradations.py \
    --input_json "$TEST_DATA" \
    --output_dir "$DEGRADED_DIR" \
    --degradation_types blur brightness noise compression

# Test on each degradation type
DEGRADATIONS=("original" "blur_1.5" "blur_2.5" "brightness_-30" "brightness_30" "noise_100" "compression_35")

for DEG in "${DEGRADATIONS[@]}"; do
    OUTPUT_DIR="$OUTPUT_BASE/$DEG"
    TEST_FILE="$DEGRADED_DIR/${DEG}_test.json"

    if [ -f "$TEST_FILE" ]; then
        echo "Testing with degradation: $DEG..."
        python utils/test_inference.py \
            --model_path "$CHECKPOINT" \
            --test_json "$TEST_FILE" \
            --output "$OUTPUT_DIR/predictions.json"
    fi
done

# Analyze robustness
python experiments/rq1_robustness/analyze_robustness.py \
    --results_dir "$OUTPUT_BASE" \
    --output "$OUTPUT_BASE/robustness_curves.pdf"

echo "=== RQ1.3.3 Complete ==="
