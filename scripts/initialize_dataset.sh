#!/bin/bash

# Script to initialize QVED dataset with optional cleaning
# This script orchestrates the complete dataset preparation pipeline

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  QVED Dataset Initialization Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Step 1: Ask for number of videos per exercise
echo -e "${YELLOW}Step 1: Dataset Download Configuration${NC}"
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
echo -e "${YELLOW}Step 2: Downloading Dataset from HuggingFace${NC}"
echo -e "${BLUE}Running: python utils/load_dataset.py ${VIDEO_COUNT}${NC}"
python utils/load_dataset.py "$VIDEO_COUNT"

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Dataset download failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dataset download completed${NC}"
echo ""

# Step 3: Filter ground truth
echo -e "${YELLOW}Step 3: Filtering Ground Truth Labels${NC}"
echo -e "${BLUE}Running: python utils/filter_ground_truth.py${NC}"
python utils/filter_ground_truth.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Ground truth filtering failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Ground truth filtering completed${NC}"
echo ""

# Step 4: Generate QVED splits
echo -e "${YELLOW}Step 4: Generating QVED Train/Val/Test Splits${NC}"
echo -e "${BLUE}Running: python utils/qved_from_fine_labels.py${NC}"
python utils/qved_from_fine_labels.py

if [ $? -ne 0 ]; then
    echo -e "${RED}Error: QVED split generation failed${NC}"
    exit 1
fi

echo -e "${GREEN}✓ QVED splits generated${NC}"
echo ""

# Step 5: Ask about dataset cleaning
echo -e "${YELLOW}Step 5: Dataset Cleaning (Optional)${NC}"
echo "Dataset cleaning will analyze video quality (resolution, brightness, sharpness, motion)"
echo "and filter out low-quality videos."
echo ""
echo -n "Do you want to clean the dataset? (y/N): "
read -r CLEAN_RESPONSE

CLEAN_RESPONSE=$(echo "$CLEAN_RESPONSE" | tr '[:upper:]' '[:lower:]')

if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo ""
    echo -e "${BLUE}Running: python utils/clean_dataset.py${NC}"
    python utils/clean_dataset.py

    if [ $? -ne 0 ]; then
        echo -e "${RED}Error: Dataset cleaning failed${NC}"
        exit 1
    fi

    echo -e "${GREEN}✓ Dataset cleaning completed${NC}"
else
    echo -e "${YELLOW}⊘ Skipping dataset cleaning${NC}"
fi

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Dataset Initialization Complete! ✓${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Summary of generated files:"
echo "  - dataset/manifest.json          (downloaded video manifest)"
echo "  - dataset/ground_truth.json      (filtered ground truth labels)"
echo "  - dataset/qved_train.json        (training split)"
echo "  - dataset/qved_val.json          (validation split)"
echo "  - dataset/qved_test.json         (test split)"

if [[ "$CLEAN_RESPONSE" == "y" || "$CLEAN_RESPONSE" == "yes" ]]; then
    echo "  - cleaned_dataset/               (quality-filtered videos)"
    echo "  - cleaned_dataset/cleaning_report.csv"
fi

echo ""
echo "You can now proceed with model training!"
