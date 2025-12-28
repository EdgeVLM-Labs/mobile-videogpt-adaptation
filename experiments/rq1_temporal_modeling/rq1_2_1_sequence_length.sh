#!/bin/bash
# RQ1.2: Temporal Modeling - Frame Sampling Analysis

set -e

OUTPUT_BASE="experiments/results/rq1_temporal_modeling"
CHECKPOINT=${1:-"results/checkpoint-70"}  # Use existing finetuned model
TEST_DATA="dataset/ground_truth.json"
TEMP_DIR="experiments/data/temporal_variants"

echo "=== RQ1.2: Temporal Modeling ==="

# Test different frame counts
FRAME_COUNTS=(2 4 8 16 32)
FPS_RATIOS=(1.0 0.5 0.25)

mkdir -p "$OUTPUT_BASE" "$TEMP_DIR"

# Generate temporal variants
echo "Generating temporal variants..."
python experiments/rq1_temporal_modeling/prepare_frame_variants.py \
    --input "$TEST_DATA" \
    --output_dir "$TEMP_DIR" \
    --frame_counts "${FRAME_COUNTS[@]}" \
    --fps_ratios "${FPS_RATIOS[@]}"

# Test each configuration (inference only, no retraining)
for FRAMES in "${FRAME_COUNTS[@]}"; do
    for FPS_RATIO in "${FPS_RATIOS[@]}"; do
        OUTPUT_DIR="$OUTPUT_BASE/frames${FRAMES}_fps${FPS_RATIO}"
        TEST_VARIANT="$TEMP_DIR/fps${FPS_RATIO}_frames${FRAMES}.json"

        echo "Testing frames=$FRAMES, fps_ratio=$FPS_RATIO..."

        export NUM_FRAMES=$FRAMES
        python utils/test_inference.py \
            --model_path "$CHECKPOINT" \
            --test_json "$TEST_VARIANT" \
            --output "$OUTPUT_DIR/predictions.json"
    done
done

# Analyze results
python experiments/rq1_temporal_modeling/analyze_temporal.py \
    --results_dir "$OUTPUT_BASE" \
    --output "$OUTPUT_BASE/temporal_sensitivity.pdf"

echo "=== RQ1.2 Complete ==="
