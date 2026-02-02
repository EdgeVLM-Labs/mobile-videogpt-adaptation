#!/bin/bash

# Script to initialize QVED dataset with optional cleaning
# This script orchestrates the complete dataset preparation pipeline

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  QVED Dataset Initialization Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Ask for number of videos per exercise
echo -e "${RED}Step 1: Dataset Download Configuration${NC}"
echo -n "Enter number of videos to download per exercise class: "
read -r VIDEO_COUNT

# Validate input
if ! [[ "$VIDEO_COUNT" =~ ^[0-9]+$ ]] || [ "$VIDEO_COUNT" -lt 1 ]; then
    echo -e "${RED}Error: Please enter a valid positive number${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Will download ${VIDEO_COUNT} videos per exercise class${NC}"
echo ""

# Ask about parallel downloads
echo -n "Use parallel downloads for faster processing? (y/N): "
read -r PARALLEL_RESPONSE

PARALLEL_RESPONSE=$(echo "$PARALLEL_RESPONSE" | tr '[:upper:]' '[:lower:]')

PARALLEL_FLAG=""
if [[ "$PARALLEL_RESPONSE" == "y" || "$PARALLEL_RESPONSE" == "yes" ]]; then
    PARALLEL_FLAG="--parallel"
    echo -e "${GREEN}✓ Parallel downloads enabled${NC}"
else
    echo -e "${BLUE}ℹ Using sequential downloads (default)${NC}"
fi

echo ""

# Step 2: Download dataset
echo -e "${RED}Step 2: Downloading Dataset from HuggingFace${NC}"
echo -e "${BLUE}Running: python utils/dataset/load_dataset.py ${VIDEO_COUNT} ${PARALLEL_FLAG}${NC}"
python utils/dataset/load_dataset.py "$VIDEO_COUNT" $PARALLEL_FLAG

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset download failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset download completed${NC}"
echo ""

# Step 3: Filter ground truth (Optional)
echo -e "${RED}Step 3: Filtering Ground Truth Labels (Optional)${NC}"
echo "Ground truth filtering will process and filter the downloaded labels."
echo ""
echo -n "Do you want to filter ground truth labels? (y/N): "
read -r FILTER_RESPONSE

FILTER_RESPONSE=$(echo "$FILTER_RESPONSE" | tr '[:upper:]' '[:lower:]')

if [[ "$FILTER_RESPONSE" == "y" || "$FILTER_RESPONSE" == "yes" ]]; then
    echo ""
    echo -e "${BLUE}Running: python utils/dataset/filter_ground_truth.py${NC}"
    python utils/dataset/filter_ground_truth.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Ground truth filtering failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Ground truth filtering completed${NC}"
else
    echo -e "${RED}⊘ Skipping ground truth filtering${NC}"
fi

echo ""

# Step 4: Ask about dataset cleaning (BEFORE generating splits)
echo -e "${RED}Step 4: Dataset Cleaning (Optional)${NC}"
echo "Dataset cleaning will analyze video quality (resolution, brightness, sharpness, motion)"
echo "and filter out low-quality videos."
echo ""
echo "⚠️  Important: Cleaning happens BEFORE generating train/val/test splits"
echo "   so splits will only include videos that pass quality checks."
echo ""
echo -n "Do you want to clean the dataset? (y/N): "
read -r CLEAN_RESPONSE

CLEAN_RESPONSE=$(echo "$CLEAN_RESPONSE" | tr '[:upper:]' '[:lower:]')

if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo ""
    echo -e "${BLUE}Running: python utils/dataset/clean_dataset.py${NC}"
    python utils/dataset/clean_dataset.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Dataset cleaning failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Dataset cleaning completed${NC}"
else
    echo -e "${RED}⊘ Skipping dataset cleaning${NC}"
fi

echo ""

# Step 5: Ask about dataset augmentation (BEFORE generating splits)
echo -e "${RED}Step 5: Dataset Augmentation (Optional)${NC}"
echo "Dataset augmentation will create additional training samples by applying"
echo "transformations like flips, rotations, blur, brightness changes, etc."
echo ""
echo "⚠️  Important: Augmentation happens BEFORE generating train/val/test splits"
echo "   so augmented videos will be included in the dataset splits."
echo ""
echo -n "Do you want to augment the dataset? (y/N): "
read -r AUGMENT_RESPONSE

AUGMENT_RESPONSE=$(echo "$AUGMENT_RESPONSE" | tr '[:upper:]' '[:lower:]')

if [[ "$AUGMENT_RESPONSE" == "y" || "$AUGMENT_RESPONSE" == "yes" ]]; then
    echo ""
    echo -e "${BLUE}Running: python utils/dataset/augment_videos.py${NC}"
    python utils/dataset/augment_videos.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Dataset augmentation failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Dataset augmentation completed${NC}"
else
    echo -e "${RED}⊘ Skipping dataset augmentation${NC}"
fi

echo ""

# Step 6: Generate QVED splits (AFTER cleaning and augmentation)
echo -e "${RED}Step 6: Generating QVED Train/Val/Test Splits${NC}"
echo -e "${BLUE}Running: python utils/dataset/qved_from_fine_labels.py${NC}"
python utils/dataset/qved_from_fine_labels.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: QVED split generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ QVED splits generated${NC}"

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Dataset Initialization Complete! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary of generated files:"
echo "  - dataset/manifest.json               (downloaded video manifest)"

if [[ "$FILTER_RESPONSE" == "y" || "$FILTER_RESPONSE" == "yes" ]]; then
    echo "  - dataset/ground_truth.json           (filtered ground truth labels)"
fi

echo "  - dataset/qved_train.json             (training split - fine-grained labels)"
echo "  - dataset/qved_val.json               (validation split - fine-grained labels)"
echo "  - dataset/qved_test.json              (test split - fine-grained labels)"
echo "  - dataset/qved_feedbacks_train.json   (training split - feedbacks)"
echo "  - dataset/qved_feedbacks_val.json     (validation split - feedbacks)"
echo "  - dataset/qved_feedbacks_test.json    (test split - feedbacks)"

if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo "  - cleaned_dataset/                    (quality-filtered videos)"
    echo "  - cleaned_dataset/cleaning_report.csv"
fi

if [[ "$AUGMENT_RESPONSE" == "y" || "$AUGMENT_RESPONSE" == "yes" ]]; then
    echo "  - Augmented videos added to exercise folders"
    echo "  - JSON files updated with augmented video paths"
fi

echo ""
echo "Note: Same videos are in the same splits (train/val/test) for both fine-grained labels and feedbacks datasets."
echo ""
echo "You can now proceed with model training!"
