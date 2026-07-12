#!/bin/bash

# Script to download and prepare new exercise video dataset for inference
# This downloads all videos from a tar file (no sampling or filtering)

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  New Dataset Initialization (Inference)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

echo -e "${YELLOW}ℹ  This script will:${NC}"
echo "   1. Download exercise_videos.tar from HuggingFace"
echo "   2. Extract all videos to dataset/ folder"
echo "   3. Create manifest.json for video organization"
echo ""
echo -e "${YELLOW}ℹ  Dataset location: dataset/${NC}"
echo ""

# Ask for confirmation
echo -n "Do you want to proceed with the download? (y/N): "
read -r CONFIRM

CONFIRM=$(echo "$CONFIRM" | tr '[:upper:]' '[:lower:]')

if [[ "$CONFIRM" != "y" && "$CONFIRM" != "yes" ]]; then
    echo -e "${RED}⊘ Download cancelled${NC}"
    exit 0
fi

echo ""

# Ask about keeping tar file
echo -n "Keep the tar file after extraction? (y/N): "
read -r KEEP_TAR

KEEP_TAR=$(echo "$KEEP_TAR" | tr '[:upper:]' '[:lower:]')

TAR_FLAG=""
if [[ "$KEEP_TAR" == "y" || "$KEEP_TAR" == "yes" ]]; then
    TAR_FLAG="--keep-tar"
    echo -e "${BLUE}ℹ  Tar file will be kept after extraction${NC}"
else
    echo -e "${BLUE}ℹ  Tar file will be removed after extraction (to save space)${NC}"
fi

echo ""

# Run the download script
echo -e "${RED}Downloading and Extracting Dataset${NC}"
echo -e "${BLUE}Running: python new_dataset_infer/load_new_dataset.py ${TAR_FLAG}${NC}"
echo ""

python new_dataset_infer/load_new_dataset.py $TAR_FLAG

if [ $? -ne 0 ]; then
    echo ""
    echo -e "${RED}❌ Dataset download/extraction failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Dataset Initialization Complete! ✅${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary of generated files:"
echo "  - dataset/                    (extracted videos organized by exercise class)"
echo "  - dataset/manifest.json       (video path to exercise class mapping)"
echo ""
echo "Next steps:"
echo "  1. Verify the downloaded videos in the dataset/ folder"
echo "  2. Run inference using your model on these videos"
echo ""
echo "Example inference command:"
echo "  python inference.py --video-path dataset/exercise_class/video.mov"
echo ""
