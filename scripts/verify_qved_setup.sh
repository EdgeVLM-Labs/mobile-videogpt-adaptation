#!/bin/bash

echo "========================================="
echo "QVED Finetuning Setup Verification"
echo "========================================="

# Check dataset files
echo -e "\n[1] Checking dataset files..."
if [ -f "dataset/qved_train.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_train.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_train.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_train.json NOT found"
fi

if [ -f "dataset/manifest.json" ]; then
    echo "✓ dataset/manifest.json found"
else
    echo "✗ dataset/manifest.json NOT found"
fi

if [ -f "dataset/ground_truth.json" ]; then
    echo "✓ dataset/ground_truth.json found"
else
    echo "✗ dataset/ground_truth.json NOT found"
fi

# Check video files
echo -e "\n[2] Checking video files..."

exercise_dirs=(
  "alternating_single_leg_glutes_bridge"
  "cat-cow_pose"
  "elbow_plank"
  "glute_hamstring_walkout"
  "glutes_bridge"
  "heel_lift"
  "high_plank"
  "lunges_leg_out_in_front"
  "opposite_arm_and_leg_lifts_on_knees"
  "pushups"
  "side_plank"
  "squats"
  "toe_touch"
  "tricep_stretch"
)

for dir in "${exercise_dirs[@]}"; do
    if [ -d "dataset/$dir" ]; then
        num_videos=$(ls -1 dataset/$dir/*.mp4 2>/dev/null | wc -l)
        echo "✓ dataset/$dir/ found ($num_videos videos)"
    else
        echo "✗ dataset/$dir/ NOT found"
    fi
done

# Check if paths in qved_train.json match actual files
echo -e "\n[3] Verifying video paths in qved_train.json..."
missing_count=0
total_count=0

# Use Python to extract video paths (fallback if jq not available)
python -c "
import json
with open('dataset/qved_train.json') as f:
    data = json.load(f)
    for item in data:
        print(item['video'])
" 2>/dev/null | while IFS= read -r video_path; do
    total_count=$((total_count + 1))
    if [ ! -f "dataset/$video_path" ]; then
        echo "✗ Missing: dataset/$video_path"
        missing_count=$((missing_count + 1))
    fi
done

# Get actual counts using Python
result=$(python -c "
import json
import os
with open('dataset/qved_train.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['video'])))
    print(f'{total},{missing}')
" 2>/dev/null)

total_count=$(echo $result | cut -d',' -f1)
missing_count=$(echo $result | cut -d',' -f2)

if [ "$missing_count" -eq 0 ]; then
    echo "✓ All $total_count video paths are valid"
else
    echo "✗ $missing_count out of $total_count videos are missing"
fi

# Check required scripts
echo -e "\n[4] Checking required scripts..."
if [ -f "scripts/finetune_qved.sh" ]; then
    echo "✓ scripts/finetune_qved.sh found"
else
    echo "✗ scripts/finetune_qved.sh NOT found"
fi

if [ -f "scripts/zero3.json" ]; then
    echo "✓ scripts/zero3.json found"
else
    echo "✗ scripts/zero3.json NOT found"
fi

# Check conda environment
echo -e "\n[5] Checking conda environment..."
if conda env list | grep -q "mobile_videogpt"; then
    echo "✓ Conda environment 'mobile_videogpt' exists"
else
    echo "✗ Conda environment 'mobile_videogpt' NOT found"
fi

# Check GPU availability
echo -e "\n[6] Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    gpu_count=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ $gpu_count GPU(s) detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "✗ nvidia-smi not found - GPU may not be available"
fi

echo -e "\n========================================="
echo "Setup verification complete!"
echo "========================================="
echo -e "\nTo start finetuning, run:"
echo "  conda activate mobile_videogpt"
echo "  bash scripts/finetune_qved.sh"
echo "========================================="
