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

if [ -f "dataset/qved_val.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_val.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_val.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_val.json NOT found"
fi

if [ -f "dataset/qved_test.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_test.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_test.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_test.json NOT found"
fi

if [ -f "dataset/qved_feedbacks_train.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_feedbacks_train.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_feedbacks_train.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_feedbacks_train.json NOT found"
fi

if [ -f "dataset/qved_feedbacks_val.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_feedbacks_val.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_feedbacks_val.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_feedbacks_val.json NOT found"
fi

if [ -f "dataset/qved_feedbacks_test.json" ]; then
    num_samples=$(python -c "import json; print(len(json.load(open('dataset/qved_feedbacks_test.json'))))" 2>/dev/null || echo "?")
    echo "✓ dataset/qved_feedbacks_test.json found ($num_samples samples)"
else
    echo "✗ dataset/qved_feedbacks_test.json NOT found"
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
  "alternating_forward_lunges"
  "floor_touches"
  "high_knees"
  "jumping_jacks"
  "mountain-climbers"
  "plank_taps"
  "pushups"
  "shoulder_gators"
  "squats"
  "toe_touch"
)

for dir in "${exercise_dirs[@]}"; do
    if [ -d "dataset/$dir" ]; then
        num_videos=$(ls -1 dataset/$dir/*.mp4 2>/dev/null | wc -l)
        echo "✓ dataset/$dir/ found ($num_videos videos)"
    else
        echo "✗ dataset/$dir/ NOT found"
    fi
done

# Check if paths in dataset split files match actual files
echo -e "\n[3] Verifying video paths in dataset splits..."

# Check training set
if [ -f "dataset/qved_train.json" ]; then
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
        echo "✓ Training set: All $total_count video paths are valid"
    else
        echo "✗ Training set: $missing_count out of $total_count videos are missing"
    fi
fi

# Check validation set
if [ -f "dataset/qved_val.json" ]; then
    result=$(python -c "
import json
import os
with open('dataset/qved_val.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['video'])))
    print(f'{total},{missing}')
" 2>/dev/null)

    total_count=$(echo $result | cut -d',' -f1)
    missing_count=$(echo $result | cut -d',' -f2)

    if [ "$missing_count" -eq 0 ]; then
        echo "✓ Validation set: All $total_count video paths are valid"
    else
        echo "✗ Validation set: $missing_count out of $total_count videos are missing"
    fi
fi

# Check test set
if [ -f "dataset/qved_test.json" ]; then
    result=$(python -c "
import json
import os
with open('dataset/qved_test.json') as f:
    data = json.load(f)
    total = len(data)
    missing = sum(1 for item in data if not os.path.exists(os.path.join('dataset', item['video'])))
    print(f'{total},{missing}')
" 2>/dev/null)

    total_count=$(echo $result | cut -d',' -f1)
    missing_count=$(echo $result | cut -d',' -f2)

    if [ "$missing_count" -eq 0 ]; then
        echo "✓ Test set: All $total_count video paths are valid"
    else
        echo "✗ Test set: $missing_count out of $total_count videos are missing"
    fi
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
if command -v conda &> /dev/null; then
    if conda env list | grep -q "mobile_videogpt"; then
        echo "✓ Conda environment 'mobile_videogpt' exists"
    else
        echo "✗ Conda environment 'mobile_videogpt' NOT found"
    fi
elif command -v python &> /dev/null; then
    python_path=$(which python)
    if [[ $python_path == *"mobile_videogpt"* ]]; then
        echo "✓ Python environment 'mobile_videogpt' is active"
    else
        echo "⚠ Conda not found, but Python is available at: $python_path"
    fi
else
    echo "✗ Neither conda nor python found"
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
