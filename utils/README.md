# Utils - Mobile-VideoGPT Utilities

This folder contains utility scripts for dataset preparation, model training, inference, evaluation, and deployment.

## Table of Contents

- [Utils - Mobile-VideoGPT Utilities](#utils---mobile-videogpt-utilities)
  - [Table of Contents](#table-of-contents)
  - [Inference \& Evaluation](#inference--evaluation)
    - [infer\_qved.py](#infer_qvedpy)
    - [test\_inference.py](#test_inferencepy)
    - [generate\_test\_report.py](#generate_test_reportpy)
  - [Model Deployment](#model-deployment)
    - [hf\_upload.py](#hf_uploadpy)
  - [Training Visualization](#training-visualization)
    - [plot\_training\_stats.py](#plot_training_statspy)
  - [Dataset Preparation](#dataset-preparation)
    - [load\_dataset.py](#load_datasetpy)
    - [qved\_from\_fine\_labels.py](#qved_from_fine_labelspy)
    - [filter\_ground\_truth.py](#filter_ground_truthpy)
    - [clean\_dataset.py](#clean_datasetpy)
    - [motion\_classifier.py](#motion_classifierpy)
  - [Linked Scripts](#linked-scripts)
    - [`scripts/quickstart_finetune.sh`](#scriptsquickstart_finetunesh)
    - [`scripts/run_inference.sh`](#scriptsrun_inferencesh)

## Inference & Evaluation

### infer_qved.py

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
python utils/infer_qved.py \
    --video_path sample_videos/00000340.mp4

# Using local finetuned checkpoint
python utils/infer_qved.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70 \
    --video_path sample_videos/00000340.mp4

# Using HuggingFace model with custom prompt
python utils/infer_qved.py \
    --model_path EdgeVLM-Labs/qved-finetune-20241128 \
    --video_path sample_videos/00000340.mp4 \
    --prompt "Describe this exercise video"
```

### test_inference.py

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
python utils/test_inference.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70

# Run with sample limit (for quick testing)
python utils/test_inference.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B \
    --limit 10 \
    --output test_predictions_sample.json
```

**Output:** JSON file containing predictions, ground truth, and status for each video.

### generate_test_report.py

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
python utils/generate_test_report.py \
    --predictions results/qved_finetune_mobilevideogpt_0.5B/test_predictions.json

# Generate report without BERT (faster)
python utils/generate_test_report.py \
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
python utils/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B

# Upload with custom repo name
python utils/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B/checkpoint-70 \
    --repo_name qved-finetune-v1.0

# Upload as private repository
python utils/hf_upload.py \
    --model_path results/qved_finetune_mobilevideogpt_0.5B \
    --repo_name qved-finetune-private \
    --private
```

**Prerequisites:**

- Login to HuggingFace: `huggingface-cli login`
- Or set `HF_TOKEN` environment variable

## Training Visualization

### plot_training_stats.py

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
python utils/plot_training_stats.py \
    --log_file results/finetune_20241128_143022.log \
    --model_name qved_finetune_mobilevideogpt_0.5B

# Custom output directory
python utils/plot_training_stats.py \
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

### load_dataset.py

**Purpose:** Download videos from the HuggingFace QVED dataset with automatic rate limit handling.

**Configuration (edit in file):**

```python
REPO_ID = "EdgeVLM-Labs/QVED-Test-Dataset"
MAX_PER_CLASS = 5  # Videos per exercise class
```

**Sample Command:**

```bash
python utils/load_dataset.py
```

**Output:** Downloads videos to `dataset/` folder organized by exercise class.

<!-- ### load_drive_folder.py

**Purpose:** Download files from a public Google Drive folder.

**Arguments:**
| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `folder_url` | str | _required_ | Google Drive folder URL |
| `--output` | str | `downloads` | Output directory |

**Sample Commands:**

```bash
# Download folder contents
python utils/load_drive_folder.py \
    "https://drive.google.com/drive/folders/FOLDER_ID" \
    --output dataset/videos

# Using folder ID directly
python utils/load_drive_folder.py \
    "FOLDER_ID" \
    --output downloads
```

**Prerequisites:**

- `credentials.json` from Google Cloud Console (OAuth 2.0 Client ID)
- First run will open browser for authentication -->

### qved_from_fine_labels.py

**Purpose:** Convert `fine_grained_labels.json` to QVED train/val/test splits.

**Configuration (edit in file):**

```python
RANDOM_SEED = 42
# Split ratios: 60% train, 20% val, 20% test
```

**Sample Command:**

```bash
python utils/qved_from_fine_labels.py
```

**Input:** `dataset/fine_grained_labels.json`

**Output:**

- `dataset/qved_train.json` (60%)
- `dataset/qved_val.json` (20%)
- `dataset/qved_test.json` (20%)

### filter_ground_truth.py

**Purpose:** Filter ground truth labels to only include downloaded videos (based on manifest).

**Sample Command:**

```bash
python utils/filter_ground_truth.py
```

**Input:**

- `dataset/fine_grained_labels.json`
- `dataset/manifest.json`

**Output:** `dataset/ground_truth.json`

### clean_dataset.py

**Purpose:** Filter low-quality videos from dataset based on quality metrics.

**Quality Criteria:**

- Resolution thresholds
- Brightness levels
- Sharpness (blur detection)
- Motion detection

**Sample Command:**

```bash
python utils/clean_dataset.py
```

### motion_classifier.py

**Purpose:** Detect motion in exercise videos using frame differencing.

**Configuration (edit in file):**

```python
N = 30  # Number of frames to sample
```

**Sample Command:**

```bash
python utils/motion_classifier.py
```

**Output:**

- CSV report with motion detection per video
- JSON report for programmatic access

## Linked Scripts

These utilities are called by the main pipeline scripts:

### `scripts/quickstart_finetune.sh`

Calls the following utilities:

1. `utils/plot_training_stats.py` - Generate training plots after finetuning
2. `utils/hf_upload.py` - Upload model to HuggingFace (optional, prompted)
3. `utils/infer_qved.py` - Referenced in final instructions

### `scripts/run_inference.sh`

Wrapper script that combines:

1. `utils/test_inference.py` - Run batch inference on test set
2. `utils/generate_test_report.py` - Generate evaluation report
