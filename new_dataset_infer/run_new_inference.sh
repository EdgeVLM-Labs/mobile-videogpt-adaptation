#!/bin/bash

# New Dataset Inference Script
# This script runs inference on the new dataset videos using a finetuned model

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================${NC}"
echo -e "${BLUE}  New Dataset Inference${NC}"
echo -e "${BLUE}=========================================${NC}"

# Default values
MODEL_PATH="EdgeVLM-Labs/mobile-videogpt-finetune-v2-mixed"
DATA_PATH="dataset"
OUTPUT_DIR="new_dataset_infer/results"
DEVICE="cuda"
MAX_NEW_TOKENS=64
BASE_MODEL="Amshaker/Mobile-VideoGPT-0.5B"
LIMIT=""
PROMPT="Watch the exercise being performed and provide short corrective feedback to help improve the form."

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data_path)
            DATA_PATH="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --max_new_tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --base_model)
            BASE_MODEL="$2"
            shift 2
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        --prompt)
            PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: bash new_dataset_infer/run_new_inference.sh [options]"
            echo ""
            echo "Model: EdgeVLM-Labs/mobile-videogpt-finetune-v2-mixed (hardcoded)"
            echo ""
            echo "Optional:"
            echo "  --data_path       Base path for video files (default: dataset)"
            echo "  --output_dir      Output directory for results (default: new_dataset_infer/results)"
            echo "  --device          Device to use: cuda/cpu (default: cuda)"
            echo "  --max_new_tokens  Max tokens to generate (default: 64)"
            echo "  --base_model      Base model for LoRA adapters (default: Amshaker/Mobile-VideoGPT-0.5B)"
            echo "  --limit           Limit number of videos (for testing)"
            echo "  --prompt          Custom prompt (default: 'Watch the exercise being performed and provide short corrective feedback to help improve the form.')"
            echo ""
            echo "Examples:"
            echo "  # Run inference on all videos:"
            echo "  bash new_dataset_infer/run_new_inference.sh"
            echo ""
            echo "  # With limit for testing:"
            echo "  bash new_dataset_infer/run_new_inference.sh --limit 10"
            echo ""
            echo "  # Custom output directory:"
            echo "  bash new_dataset_infer/run_new_inference.sh --output_dir my_results"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if data path exists
if [ ! -d "$DATA_PATH" ]; then
    echo -e "${RED}❌ Error: Data path not found: $DATA_PATH${NC}"
    exit 1
fi

echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model path:      $MODEL_PATH"
echo "  Data path:       $DATA_PATH"
echo "  Output dir:      $OUTPUT_DIR"
echo "  Device:          $DEVICE"
echo "  Max new tokens:  $MAX_NEW_TOKENS"
echo "  Base model:      $BASE_MODEL"
echo "  Prompt:          \"$PROMPT\""
if [ -n "$LIMIT" ]; then
    echo "  Video limit:     $LIMIT"
fi
echo -e "${BLUE}=========================================${NC}"
echo ""

# Build command
CMD="python new_dataset_infer/run_new_inference.py \
    --data_path \"$DATA_PATH\" \
    --output_dir \"$OUTPUT_DIR\" \
    --device \"$DEVICE\" \
    --max_new_tokens \"$MAX_NEW_TOKENS\" \
    --base_model \"$BASE_MODEL\" \
    --prompt \"$PROMPT\""

if [ -n "$LIMIT" ]; then
    CMD="$CMD --limit $LIMIT"
fi

# Run inference
echo -e "${GREEN}Running inference...${NC}"
echo ""

eval $CMD

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Inference failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}=========================================${NC}"
echo -e "${GREEN}  ✅ Inference Complete!${NC}"
echo -e "${GREEN}=========================================${NC}"
echo ""
echo "Output files saved in: $OUTPUT_DIR"
echo "  - inference_results.json"
echo "  - inference_results.xlsx"
echo ""
