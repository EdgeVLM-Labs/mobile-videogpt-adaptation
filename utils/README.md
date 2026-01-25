# Utils - Mobile-VideoGPT Utilities

This folder contains utility scripts for dataset preparation, model training, inference, evaluation, deployment, and feedback naturalization.

## Directory Structure

```
utils/
├── dataset/           # Dataset preparation and augmentation
├── inference/         # Model inference and evaluation
├── naturalizer/       # Feedback naturalizer for varied responses
└── credentials.json   # Google Drive API credentials (optional)
```

## Table of Contents

- [Utils - Mobile-VideoGPT Utilities](#utils---mobile-videogpt-utilities)
  - [Directory Structure](#directory-structure)
  - [Table of Contents](#table-of-contents)
  - [Inference \& Evaluation](#inference--evaluation)
    - [infer\_qved.py](#infer_qvedpy)
    - [base\_model\_inference.py](#base_model_inferencepy)
    - [test\_inference.py](#test_inferencepy)
    - [generate\_test\_report.py](#generate_test_reportpy)
  - [Model Deployment](#model-deployment)
    - [hf\_upload.py](#hf_uploadpy)
  - [Training Visualization](#training-visualization)
    - [plot\_training\_stats.py](#plot_training_statspy)
  - [Dataset Preparation](#dataset-preparation)
    - [load\_dataset.py](#load_datasetpy)
    - [load\_drive\_folder.py](#load_drive_folderpy)
    - [qved\_from\_fine\_labels.py](#qved_from_fine_labelspy)
    - [filter\_ground\_truth.py](#filter_ground_truthpy)
    - [clean\_dataset.py](#clean_datasetpy)
    - [motion\_classifier.py](#motion_classifierpy)
    - [augment\_videos.py](#augment_videospy)
  - [Feedback Naturalizer](#feedback-naturalizer)
    - [feedback\_naturalizer.py](#feedback_naturalizerpy)
    - [inference\_with\_naturalizer.py](#inference_with_naturalizerpy)
  - [Linked Scripts](#linked-scripts)
    - [`scripts/initialize_dataset.sh`](#scriptsinitialize_datasetsh)
    - [`scripts/quickstart_finetune.sh`](#scriptsquickstart_finetunesh)
    - [`scripts/run_inference.sh`](#scriptsrun_inferencesh)
    - [`scripts/plot_from_log.sh`](#scriptsplot_from_logsh)
    - [`polling/gradio_app.py`](#pollinggradio_apppy)
    - [`polling/run_polling_with_naturalizer.py`](#pollingrun_polling_with_naturalizerpy)
  - [Quick Reference](#quick-reference)

## Inference & Evaluation

All inference scripts are located in `utils/inference/`

### infer_qved.py

**Location:** `utils/inference/infer_qved.py`

**Purpose:** Run single video inference using a finetuned Mobile-VideoGPT model.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | `Amshaker/Mobile-VideoGPT-0.5B` | Path to model (HuggingFace ID or local checkpoint) |
| `--video_path` | str | `sample_videos/00000340.mp4` | Path to input video file |
| `--prompt` | str | Physiotherapy evaluation prompt | Custom prompt for the model |
| `--device` | str | `cuda` | Device to run inference (`cuda`/`cpu`) |
| `--max_new_tokens` | int | `512` | Maximum new tokens to generate |
| `--base_model` | str | `Amshaker/Mobile-VideoGPT-0.5B` | Base model for LoRA adapters |

**Sample Commands:**

```bash
# Using base model (no finetuning)
python utils/inference/infer_qved.py \
    --video_path sample_videos/00000340.mp4

# Using local finetuned checkpoint
python utils/inference/infer_qved.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70 \
    --video_path sample_videos/00000340.mp4

# Using HuggingFace model with custom prompt
python utils/inference/infer_qved.py \
    --model_path EdgeVLM-Labs/qved-finetune-20241128 \
    --video_path sample_videos/00000340.mp4 \
    --prompt "Describe this exercise video"
```

### base_model_inference.py

**Location:** `utils/inference/base_model_inference.py`

**Purpose:** Run inference using the base Mobile-VideoGPT model without any finetuning or LoRA adapters.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | `Amshaker/Mobile-VideoGPT-0.5B` | Base model path (HuggingFace ID) |
| `--video_path` | str | _required_ | Path to input video file |
| `--prompt` | str | Default exercise prompt | Custom prompt for the model |
| `--device` | str | `cuda` | Device to run inference (`cuda`/`cpu`) |
| `--max_new_tokens` | int | `512` | Maximum new tokens to generate |

**Sample Commands:**

```bash
# Basic inference with 0.5B model
python utils/inference/base_model_inference.py \
    --video_path sample_videos/00000340.mp4

# Using 1.5B model
python utils/inference/base_model_inference.py \
    --model_path Amshaker/Mobile-VideoGPT-1.5B \
    --video_path sample_videos/exercise.mp4

# Custom prompt
python utils/inference/base_model_inference.py \
    --video_path sample_videos/00000340.mp4 \
    --prompt "What exercise is being performed?"
```

### test_inference.py

**Location:** `utils/inference/test_inference.py`

**Purpose:** Run batch inference on the QVED test set and save predictions to JSON.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | _required_ | Path to finetuned model checkpoint |
| `--test_json` | str | `dataset/qved_test.json` | Path to test set JSON |
| `--data_path` | str | `dataset` | Base path for video files |
| `--output` | str | Model directory | Output file for predictions |
| `--device` | str | `cuda` | Device to use (`cuda`/`cpu`) |
| `--max_new_tokens` | int | `64` | Maximum new tokens to generate |
| `--base_model` | str | `Amshaker/Mobile-VideoGPT-0.5B` | Base model for LoRA adapters |
| `--limit` | int | `None` | Limit samples to process (for testing) |

**Sample Commands:**

```bash
# Run on full test set
python utils/inference/test_inference.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70

# Run with sample limit (for quick testing)
python utils/inference/test_inference.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B \
    --limit 10 \
    --output test_predictions_sample.json
```

**Output:** JSON file containing predictions, ground truth, and status for each video.

### generate_test_report.py

**Location:** `utils/inference/generate_test_report.py`

**Purpose:** Generate an Excel evaluation report with similarity scores (BERT cosine similarity and METEOR score) comparing predictions to ground truth.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--predictions` | str | _required_ | Path to predictions JSON from `test_inference.py` |
| `--output` | str | Same directory as predictions | Output Excel file path |
| `--no-bert` | flag | `False` | Skip BERT similarity (faster evaluation) |

**Sample Commands:**

```bash
# Generate full report with BERT similarity
python utils/inference/generate_test_report.py \
    --predictions results/qved_finetune_mobilevideogpt_0.5B/test_predictions.json

# Generate report without BERT (faster)
python utils/inference/generate_test_report.py \
    --predictions test_predictions.json \
    --output evaluation_report.xlsx \
    --no-bert
```

**Output:** Excel file with:

- Color-coded similarity scores (green ≥0.7, yellow ≥0.4, red <0.4 for BERT)
- Summary statistics (mean, median, std dev, min, max)
- BERT cosine similarity and METEOR scores

**Dependencies:** `sentence-transformers`, `evaluate`, `openpyxl`, `sklearn`

## Model Deployment

### hf_upload.py

**Location:** `utils/inference/hf_upload.py`

**Purpose:** Upload finetuned Mobile-VideoGPT models to HuggingFace Hub.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model_path` | str | _required_ | Path to finetuned model directory |
| `--repo_name` | str | `qved-finetune-TIMESTAMP` | Name for HuggingFace repository |
| `--org` | str | `EdgeVLM-Labs` | HuggingFace organization name |
| `--private` | flag | `False` | Create a private repository |
| `--commit_message` | str | Auto-generated | Custom commit message |

**Sample Commands:**

```bash
# Upload with auto-generated repo name
python utils/inference/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B

# Upload with custom repo name
python utils/inference/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70 \
    --repo_name qved-finetune-v1.0

# Upload as private repository
python utils/inference/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B \
    --repo_name qved-finetune-private \
    --private
```

**Prerequisites:**

- Login to HuggingFace: `huggingface-cli login`
- Or set `HF_TOKEN` environment variable

## Training Visualization

### plot_training_stats.py

**Location:** `utils/inference/plot_training_stats.py`

**Purpose:** Generate LaTeX-quality training plots from log files (loss, gradient norm, learning rate).

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--log_file` | str | _required_ | Path to training log file |
| `--model_name` | str | `model` | Name for output files |
| `--output_dir` | str | `plots/{model_name}` | Output directory for plots |

**Sample Commands:**

```bash
# Generate plots from training log
python utils/inference/plot_training_stats.py \
    --log_file results/finetune_20241128_143022.log \
    --model_name qved_finetune_mobilevideogpt_0.5B

# Custom output directory
python utils/inference/plot_training_stats.py \
    --log_file training.log \
    --model_name my_model \
    --output_dir my_plots/
```

**Output:**

- `loss.png` - Training loss over steps
- `gradient_norm.png` - Gradient norm over steps
- `learning_rate.png` - Learning rate schedule
- `combined_metrics.png` - All metrics in one figure
- `training_report.pdf` - PDF report with all plots

## Dataset Preparation

All dataset scripts are located in `utils/dataset/`

### load_dataset.py

**Location:** `utils/dataset/load_dataset.py`

**Purpose:** Download videos from the HuggingFace QVED dataset with automatic rate limit handling.

**Configuration (edit in file):**

```python
REPO_ID = "EdgeVLM-Labs/QVED-Test-Dataset"
MAX_PER_CLASS = 5  # Videos per exercise class
```

**Sample Command:**

```bash
python utils/dataset/load_dataset.py
```

**Output:** Downloads videos to `dataset/` folder organized by exercise class.

### load_drive_folder.py

**Location:** `utils/dataset/load_drive_folder.py`

**Purpose:** Download files from a public Google Drive folder.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `folder_url` | str | _required_ | Google Drive folder URL or ID |
| `--output` | str | `downloads` | Output directory |

**Sample Commands:**

```bash
# Download folder contents
python utils/dataset/load_drive_folder.py \
    "https://drive.google.com/drive/folders/FOLDER_ID" \
    --output dataset/videos

# Using folder ID directly
python utils/dataset/load_drive_folder.py \
    "FOLDER_ID" \
    --output downloads
```

**Prerequisites:**

- `credentials.json` from Google Cloud Console (OAuth 2.0 Client ID)
- First run will open browser for authentication

### qved_from_fine_labels.py

**Location:** `utils/dataset/qved_from_fine_labels.py`

**Purpose:** Convert `fine_grained_labels.json` to QVED train/val/test splits.

**Configuration (edit in file):**

```python
RANDOM_SEED = 42
# Split ratios: 60% train, 20% val, 20% test
```

**Sample Command:**

```bash
python utils/dataset/qved_from_fine_labels.py
```

**Input:** `dataset/fine_grained_labels.json`

**Output:**

- `dataset/qved_train.json` (60%)
- `dataset/qved_val.json` (20%)
- `dataset/qved_test.json` (20%)

### filter_ground_truth.py

**Location:** `utils/dataset/filter_ground_truth.py`

**Purpose:** Filter ground truth labels to only include downloaded videos (based on manifest).

**Sample Command:**

```bash
python utils/dataset/filter_ground_truth.py
```

**Input:**

- `dataset/fine_grained_labels.json`
- `dataset/manifest.json`

**Output:** `dataset/ground_truth.json`

### clean_dataset.py

**Location:** `utils/dataset/clean_dataset.py`

**Purpose:** Filter low-quality videos from dataset based on quality metrics.

**Quality Criteria:**

- Resolution thresholds
- Brightness levels
- Sharpness (blur detection)
- Motion detection

**Sample Command:**

```bash
python utils/dataset/clean_dataset.py
```

### motion_classifier.py

**Location:** `utils/dataset/motion_classifier.py`

**Purpose:** Detect motion in exercise videos using frame differencing.

**Configuration (edit in file):**

```python
N = 30  # Number of frames to sample
```

**Sample Command:**

```bash
python utils/dataset/motion_classifier.py
```

**Output:**

- CSV report with motion detection per video
- JSON report for programmatic access

### augment_videos.py

**Location:** `utils/dataset/augment_videos.py`

**Purpose:** Apply video augmentation techniques to increase dataset diversity.

**Augmentation Techniques:**

- Temporal augmentation (speed changes, frame sampling)
- Spatial augmentation (crops, flips, rotations)
- Color jitter and brightness adjustments

**Sample Command:**

```bash
python utils/dataset/augment_videos.py
```

## Feedback Naturalizer

### feedback_naturalizer.py

**Location:** `utils/naturalizer/feedback_naturalizer.py`

**Purpose:** Convert structured model predictions into natural language feedback using LLM (Gemini).

**Key Features:**

- Converts JSON predictions to human-readable feedback
- Uses Google Gemini API for natural language generation
- Supports batch processing and streaming responses

**Usage:**

```python
from utils.naturalizer.feedback_naturalizer import FeedbackNaturalizer

naturalizer = FeedbackNaturalizer(
    model_name="gemini-1.5-flash",
    api_key="your-api-key"
)

# Convert prediction to natural feedback
feedback = naturalizer.naturalize(
    prediction="correct",
    exercise="squats",
    context={"frame": 45, "confidence": 0.95}
)
```

### inference_with_naturalizer.py

**Location:** `utils/naturalizer/inference_with_naturalizer.py`

**Purpose:** End-to-end inference pipeline with natural language feedback generation.

**Sample Command:**

```bash
python utils/naturalizer/inference_with_naturalizer.py \
    --model_path checkpoints/mobile-videogpt-qved \
    --video sample_videos/squat_001.mp4 \
    --api_key YOUR_GEMINI_API_KEY
```

**Output:**

- Raw model prediction (JSON)
- Natural language feedback
- Confidence scores

## Linked Scripts

These utilities are called by the main pipeline scripts:

### `scripts/initialize_dataset.sh`

Dataset initialization pipeline that calls:

1. `utils/dataset/load_dataset.py` - Download videos from dataset source
2. `utils/dataset/filter_ground_truth.py` - Filter labels based on downloaded videos
3. `utils/dataset/clean_dataset.py` - Apply quality filtering
4. `utils/dataset/augment_videos.py` - Generate augmented variations
5. `utils/dataset/qved_from_fine_labels.py` - Create train/val/test splits

### `scripts/quickstart_finetune.sh`

Finetuning pipeline that calls:

1. `utils/inference/plot_training_stats.py` - Generate training plots after finetuning
2. `utils/inference/hf_upload.py` - Upload model to HuggingFace (optional, prompted)
3. `utils/inference/infer_qved.py` - Referenced in final instructions

### `scripts/run_inference.sh`

Inference pipeline that combines:

1. `utils/inference/test_inference.py` - Run batch inference on test set
2. `utils/inference/generate_test_report.py` - Generate evaluation report

### `scripts/plot_from_log.sh`

Training visualization utility that calls:

1. `utils/inference/plot_training_stats.py` - Parse log files and generate plots

### `polling/gradio_app.py`

Interactive polling inference app that imports:

1. `utils.naturalizer.feedback_naturalizer.FeedbackNaturalizer` - Natural language feedback generation

### `polling/run_polling_with_naturalizer.py`

Polling inference with naturalizer that imports:

1. `utils.naturalizer.feedback_naturalizer.FeedbackNaturalizer` - Batch feedback processing

---

## Quick Reference

**Dataset Preparation:**

```bash
# Initialize complete dataset
bash scripts/initialize_dataset.sh

# Individual steps
python utils/dataset/load_dataset.py
python utils/dataset/filter_ground_truth.py
python utils/dataset/clean_dataset.py
python utils/dataset/augment_videos.py
python utils/dataset/qved_from_fine_labels.py
```

**Training:**

```bash
# Quick finetuning (includes plotting)
bash scripts/quickstart_finetune.sh

# Plot existing logs
bash scripts/plot_from_log.sh path/to/train.log
```

**Inference:**

```bash
# Batch inference on test set
bash scripts/run_inference.sh

# Individual inference
python utils/inference/infer_qved.py --video path/to/video.mp4

# Base model inference (no finetuning)
python utils/inference/base_model_inference.py --video path/to/video.mp4

# With natural feedback
python utils/naturalizer/inference_with_naturalizer.py \
    --model_path checkpoints/model \
    --video path/to/video.mp4 \
    --api_key YOUR_API_KEY
```

**Model Deployment:**

```bash
# Upload to HuggingFace
python utils/inference/hf_upload.py \
    --model_path checkpoints/mobile-videogpt-qved \
    --repo_name your-username/model-name \
    --token YOUR_HF_TOKEN
```
