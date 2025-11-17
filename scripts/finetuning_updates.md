# QVED Finetuning Guide for Mobile-VideoGPT-0.5B

This guide explains how to finetune Mobile-VideoGPT-0.5B on the QVED (exercise feedback) dataset.

## Overview

The Mobile-VideoGPT training pipeline has 3 stages:

1. **Stage 1**: Video projection pre-training (VideoMamba encoder)
2. **Stage 2**: Image projection pre-training (CLIP encoder)
3. **Stage 3**: Mobile-VideoGPT finetuning on instruction data

Since we're using the pre-trained `Amshaker/Mobile-VideoGPT-0.5B` checkpoint, **we only need to run Stage 3** (finetuning on QVED dataset).

## Dataset Structure

```
dataset/
├── qved_train.json              # Training annotations (30 samples)
├── ground_truth.json            # Original labels
├── manifest.json                # Video path mapping
├── knee_circles/                # 5 videos
├── opposite_arm_and_leg_lifts_on_knees/  # 5 videos
├── pushups_on_knees/            # 5 videos
├── squat_jump/                  # 5 videos
├── squats/                      # 5 videos
└── tricep_stretch/              # 5 videos
```

## Dataset Format

The `qved_train.json` file follows the Mobile-VideoGPT format:

```json
[
  {
    "video": "dataset/tricep_stretch/00002012.mp4",
    "conversations": [
      {
        "from": "human",
        "value": "Analyze this tricep stretch video and provide corrective feedback."
      },
      {
        "from": "gpt",
        "value": "tricep stretch (right arm) - User is holding the stretch"
      }
    ],
    "split": "train"
  }
]
```

## Setup Verification

Before finetuning, verify your setup:

```bash
bash scripts/verify_qved_setup.sh
```

This checks:

- Dataset files exist
- All 30 videos are present
- Video paths in annotations are valid
- Required scripts exist
- Conda environment is configured
- GPU is available

## Regenerating Dataset (Optional)

If you need to regenerate `qved_train.json` from `ground_truth.json`:

```bash
conda activate mobile_videogpt
python utils/qved_from_fine_labels.py
```

This script:

- Reads `dataset/ground_truth.json` (original labels)
- Uses `dataset/manifest.json` to map filenames to full paths
- Extracts exercise names and corrective feedback
- Generates `dataset/qved_train.json` in Mobile-VideoGPT format

## Finetuning Configuration

The finetuning script uses the following hyperparameters (optimized for small dataset):

| Parameter             | Value                           | Description                   |
| --------------------- | ------------------------------- | ----------------------------- |
| Base Model            | `Amshaker/Mobile-VideoGPT-0.5B` | Pre-trained checkpoint        |
| Epochs                | 10                              | More epochs for small dataset |
| Learning Rate         | 1e-4                            | Conservative for stability    |
| MM Projector LR       | 2e-5                            | Lower for projection layers   |
| LoRA Rank             | 64                              | Parameter-efficient tuning    |
| LoRA Alpha            | 128                             | LoRA scaling factor           |
| Batch Size            | 4                               | Per-device batch size         |
| Gradient Accumulation | 4                               | Effective batch = 16          |
| Max Length            | 2048                            | Max sequence length           |
| DeepSpeed             | ZeRO-3                          | Memory optimization           |

## Running Finetuning

### Start Training

```bash
conda activate mobile_videogpt
bash scripts/finetune_qved.sh
```

### Monitor Training

The script will:

1. Load the pre-trained Mobile-VideoGPT-0.5B model
2. Load QVED dataset (30 samples, auto-detected via `dataset/qved_train.json`)
3. Finetune with LoRA for 10 epochs
4. Save checkpoints every 50 steps to `results/qved_finetune_mobilevideogpt_0.5B/`

Training progress will be logged to the console.

### Expected Training Time

- **Per epoch**: ~2-5 minutes (depending on GPU)
- **Total time**: ~30-50 minutes for 10 epochs
- **GPU**: NVIDIA RTX 4070 Laptop (8GB VRAM)

### Output

The finetuned model will be saved to:

```
results/qved_finetune_mobilevideogpt_0.5B/
├── checkpoint-50/
├── checkpoint-100/
├── ...
└── checkpoint-final/
```

## Key Differences from Official Training Script

The `finetune_qved.sh` script differs from `Mobile-VideoGPT-0.5B_training.sh`:

1. **Skips Stage 1 & 2**: Uses pre-trained projectors from checkpoint
2. **No pretrain_mm_mlp_adapter**: Not needed (already in checkpoint)
3. **Smaller batch size**: 4 instead of 8 (fits 8GB VRAM)
4. **More epochs**: 10 instead of 2 (small dataset needs more training)
5. **Auto-dataset detection**: QVED is auto-included via config
6. **Adjusted save frequency**: Every 50 steps (more frequent for small dataset)

## Dataset Auto-Detection

The QVED dataset is automatically included in training if `dataset/qved_train.json` exists. This is configured in `mobilevideogpt/config/__init__.py`:

```python
# Auto-include QVED if present
if os.path.exists(QVED_TRAIN["annotation_path"]):
    _mobilegpt_datasets.append(QVED_TRAIN)
```

No manual dataset registration is needed!

## Troubleshooting

### Out of Memory Error

If you get OOM errors:

1. Reduce `BATCH` to 2 or 1 in `finetune_qved.sh`
2. Increase `GACC` to maintain effective batch size
3. Reduce `MAXLEN` to 1024

### Dataset Not Found

If training can't find videos:

1. Run verification: `bash scripts/verify_qved_setup.sh`
2. Check video paths in `qved_train.json` match actual files
3. Regenerate dataset: `python utils/qved_from_fine_labels.py`

### Model Not Downloading

If `Amshaker/Mobile-VideoGPT-0.5B` fails to download:

1. Check internet connection
2. Check Hugging Face access
3. Pre-download: `huggingface-cli download Amshaker/Mobile-VideoGPT-0.5B`

## Inference After Finetuning

After training, use the finetuned model for inference:

```bash
conda activate mobile_videogpt
python inference.py \
  --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-final \
  --video_path dataset/squats/00093026.mp4 \
  --prompt "Analyze this squats video and provide corrective feedback."
```

## References

- Original Mobile-VideoGPT: https://github.com/Amshaker/Mobile-VideoGPT
- Pre-trained checkpoint: https://huggingface.co/Amshaker/Mobile-VideoGPT-0.5B
- VideoMamba encoder: https://github.com/OpenGVLab/VideoMamba
- Base LLM: Qwen2.5-0.5B-Instruct
