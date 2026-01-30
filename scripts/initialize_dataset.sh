#!/bin/bash

# Script to initialize QVED dataset with optional cleaning and augmentation
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

# Step 2: Download dataset
echo -e "${RED}Step 2: Downloading Dataset from HuggingFace${NC}"
echo -e "${BLUE}Running: python utils/dataset/load_dataset.py ${VIDEO_COUNT}${NC}"
python utils/dataset/load_dataset.py "$VIDEO_COUNT"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset download failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset download completed${NC}"
echo ""

# Step 3: Filter ground truth and feedback labels
echo -e "${RED}Step 3: Filtering Ground Truth and Feedback Labels${NC}"
echo -e "${BLUE}Running: python utils/dataset/filter_ground_truth.py${NC}"
python utils/dataset/filter_ground_truth.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Label filtering failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ground truth and feedback filtering completed${NC}"
echo ""

# Step 4: Ask about optional data processing
echo -e "${RED}Step 4: Optional Data Processing${NC}"
echo ""
echo "Available optional processing steps:"
echo "  1. Dataset Cleaning - Analyze and filter low-quality videos"
echo "  2. Dataset Augmentation - Create additional training samples"
echo ""
echo "⚠️  Important: These processes happen BEFORE generating train/val/test splits"
echo ""

# Ask about cleaning
echo -n "Do you want to clean the dataset? (y/N) [default: N]: "
read -r CLEAN_RESPONSE
CLEAN_RESPONSE=$(echo "$CLEAN_RESPONSE" | tr '[:upper:]' '[:lower:]')

# Default to "no" if empty
if [[ -z "$CLEAN_RESPONSE" ]]; then
    CLEAN_RESPONSE="n"
fi

# Ask about augmentation
echo -n "Do you want to augment the dataset? (y/N) [default: N]: "
read -r AUGMENT_RESPONSE
AUGMENT_RESPONSE=$(echo "$AUGMENT_RESPONSE" | tr '[:upper:]' '[:lower:]')

# Default to "no" if empty
if [[ -z "$AUGMENT_RESPONSE" ]]; then
    AUGMENT_RESPONSE="n"
fi

echo ""

# Execute cleaning if requested
if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo -e "${BLUE}Running dataset cleaning...${NC}"
    python utils/dataset/clean_dataset.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Dataset cleaning failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Dataset cleaning completed${NC}"
    echo ""
else
    echo -e "${RED}⊘ Skipping dataset cleaning${NC}"
    echo ""
fi

# Execute augmentation if requested
if [[ "$AUGMENT_RESPONSE" == "y" || "$AUGMENT_RESPONSE" == "yes" ]]; then
    echo -e "${BLUE}Running dataset augmentation...${NC}"
    python utils/dataset/augment_videos.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Dataset augmentation failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Dataset augmentation completed${NC}"
    echo ""
else
    echo -e "${RED}⊘ Skipping dataset augmentation${NC}"
    echo ""
fi

# Step 5: Generate QVED splits (AFTER cleaning and augmentation)
echo -e "${RED}Step 5: Generating QVED Train/Val/Test Splits${NC}"
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
echo "  - dataset/manifest.json          (downloaded video manifest)"
echo "  - dataset/ground_truth.json      (filtered ground truth labels)"
echo "  - dataset/feedbacks.json         (filtered feedback labels)"
echo "  - dataset/qved_train.json        (ground truth training split)"
echo "  - dataset/qved_val.json          (ground truth validation split)"
echo "  - dataset/qved_test.json         (ground truth test split)"
echo "  - dataset/feedback_train.json    (feedback training split)"
echo "  - dataset/feedback_val.json      (feedback validation split)"
echo "  - dataset/feedback_test.json     (feedback test split)"

if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo "  - cleaned_dataset/               (quality-filtered videos)"
    echo "  - cleaned_dataset/cleaning_report.csv"
fi

if [[ "$AUGMENT_RESPONSE" == "y" || "$AUGMENT_RESPONSE" == "yes" ]]; then
    echo "  - Augmented videos added to exercise folders"
    echo "  - JSON files updated with augmented video paths"
fi

echo ""
echo "You can now proceed with model training!"
