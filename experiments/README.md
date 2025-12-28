# QVED Experiments Guide

This directory contains all experimental scripts and analysis tools for evaluating vision-language models on the QVED physiotherapy dataset.

## Quick Start

```bash
# 1. Run a single experiment
bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh Mobile-VideoGPT-0.5B

# 2. Run all RQ1 experiments
bash experiments/run_all_rq1.sh

# 3. Analyze results
python experiments/shared/visualization.py --results_dir experiments/results/rq1_training_efficiency
```

## Directory Structure

```
experiments/
├── shared/                 # Common utilities
│   ├── metrics.py         # Evaluation metrics
│   ├── data_utils.py      # Dataset manipulation
│   ├── visualization.py   # Plotting utilities
│   └── config.py          # Shared configuration
├── rq1_training_efficiency/
├── rq1_temporal_modeling/
├── rq1_robustness/
├── rq2_error_localization/
├── rq3_efficiency/
├── rq3_compression/
├── rq4_failure_analysis/
├── rq5_clinical_guidance/
├── data/                  # Generated experimental data
└── results/               # Experimental results
```

## Research Questions

### RQ1: Training Efficiency

#### RQ1.1.1: Full vs LoRA Fine-tuning

**Script:** `rq1_training_efficiency/rq1_1_1_full_vs_lora.sh`

Compares parameter-efficient LoRA fine-tuning with full model fine-tuning.

**Usage:**

```bash
bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh Mobile-VideoGPT-0.5B
```

**Outputs:**

- `results/rq1_1_1/lora/` - LoRA training results
- `results/rq1_1_1/full/` - Full fine-tuning results
- `results/rq1_1_1/comparison_report.json` - Side-by-side comparison

#### RQ1.1.2: Learning Rate Schedules

**Script:** `rq1_training_efficiency/rq1_1_2_lr_schedules.sh`

Tests different learning rate schedules and warmup ratios.

**Usage:**

```bash
bash experiments/rq1_training_efficiency/rq1_1_2_lr_schedules.sh Mobile-VideoGPT-0.5B
```

**Outputs:**

- Convergence speed analysis
- Training curves comparison
- Optimal scheduler recommendations

#### RQ1.1.3: Data Efficiency

**Script:** `rq1_training_efficiency/rq1_1_3_data_efficiency.sh`

Analyzes how much training data is needed for acceptable accuracy.

**Usage:**

```bash
bash experiments/rq1_training_efficiency/rq1_1_3_data_efficiency.sh Mobile-VideoGPT-0.5B
```

**Outputs:**

- Data efficiency curves
- Minimum dataset size recommendations
- Per-exercise data requirements

### RQ1.2: Temporal Modeling

#### RQ1.2.1 + RQ1.2.3: Frame Sampling Analysis

**Script:** `rq1_temporal_modeling/rq1_2_1_sequence_length.sh`

Tests model sensitivity to frame count and sampling rate.

**Usage:**

```bash
bash experiments/rq1_temporal_modeling/rq1_2_1_sequence_length.sh
```

**Outputs:**

- Accuracy vs frame count curves
- Accuracy vs FPS degradation
- Temporal resolution recommendations

### RQ1.3: Robustness

#### RQ1.3.3: Visual Degradation

**Script:** `rq1_robustness/rq1_3_3_visual_degradation.sh`

Tests model robustness to video quality degradation.

**Usage:**

```bash
bash experiments/rq1_robustness/rq1_3_3_visual_degradation.sh
```

**Degradation types:**

- Motion blur (vidaug)
- Brightness variations (vidaug)
- Noise (vidaug)
- Compression artifacts (ffmpeg)

### RQ2: Error Detection & Localization

#### RQ2.1.1: Form Error Detection

**Script:** `rq2_error_localization/rq2_1_1_error_detection.sh`

Evaluates model's ability to detect and localize form errors.

**Usage:**

```bash
bash experiments/rq2_error_localization/rq2_1_1_error_detection.sh
```

### RQ3: Computational Efficiency

#### RQ3.1: Latency & Memory Profiling

**Scripts:**

- `rq3_efficiency/rq3_1_1_baseline_profile.sh` - Baseline metrics
- `rq3_efficiency/rq3_1_2_batch_scaling.sh` - Batch size effects
- `rq3_efficiency/rq3_1_3_resolution_sweep.sh` - Resolution/sequence tradeoffs

**Usage:**

```bash
bash experiments/rq3_efficiency/rq3_1_1_baseline_profile.sh
```

#### RQ3.2: Quantization & Compression

**Scripts:**

- `rq3_compression/rq3_2_1_accuracy_headroom.sh` - Pre-quantization margin
- `rq3_compression/rq3_2_2_precision_sweep.sh` - Precision reduction impact

### RQ4: Failure Analysis

#### RQ4.1.2 + RQ4.1.3: Class Imbalance

**Scripts:**

- `rq4_failure_analysis/rq4_1_2_confusion_matrix.sh` - Exercise-specific failures
- `rq4_failure_analysis/rq4_1_3_class_imbalance.sh` - Data distribution effects

### RQ5: Clinical Guidance Quality

#### RQ5.2.1: Feedback Quality

**Script:** `rq5_clinical_guidance/rq5_2_1_feedback_quality.sh`

Evaluates corrective guidance quality using LLM-as-judge.

**Usage:**

```bash
bash experiments/rq5_clinical_guidance/rq5_2_1_feedback_quality.sh
```

## Common Utilities

### Metrics (`shared/metrics.py`)

```python
from experiments.shared.metrics import calculate_accuracy_metrics

predictions = ["correct", "incorrect", "correct"]
ground_truth = ["correct", "correct", "correct"]
metrics = calculate_accuracy_metrics(predictions, ground_truth)
print(metrics)  # {'accuracy': 0.667, 'total_samples': 3, 'correct': 2}
```

### Data Utils (`shared/data_utils.py`)

```python
from experiments.shared.data_utils import create_stratified_subset

data = load_qved_dataset("dataset/qved_train.json")
subset = create_stratified_subset(data, ratio=0.25)
save_qved_dataset(subset, "experiments/data/subsets/qved_train_0.25.json")
```

### Visualization (`shared/visualization.py`)

```python
from experiments.shared.visualization import plot_training_curves

plot_training_curves(
    train_losses=[0.5, 0.4, 0.3],
    val_losses=[0.6, 0.5, 0.4],
    output_path="training_curves.pdf"
)
```

## Configuration (`shared/config.py`)

Global configuration for experiments:

- Model definitions
- Training hyperparameters
- Frame sampling configurations
- Degradation parameters
- Exercise class names

## Running Batch Experiments

### Run all RQ1 experiments:

```bash
for model in Mobile-VideoGPT-0.5B Mobile-VideoGPT-1.5B; do
    bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh $model
    bash experiments/rq1_training_efficiency/rq1_1_2_lr_schedules.sh $model
    bash experiments/rq1_training_efficiency/rq1_1_3_data_efficiency.sh $model
done
```

### Parallel execution:

```bash
# Run multiple experiments in parallel
bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh Mobile-VideoGPT-0.5B &
bash experiments/rq1_training_efficiency/rq1_1_2_lr_schedules.sh Mobile-VideoGPT-0.5B &
wait
```

## Results Organization

Results are automatically saved to:

```
experiments/results/
├── rq1_training_efficiency/
│   ├── rq1_1_1/
│   │   ├── lora/
│   │   │   ├── checkpoint/
│   │   │   ├── inference/
│   │   │   ├── parameters.json
│   │   │   └── training.log
│   │   ├── full/
│   │   └── comparison_report.json
│   ├── rq1_1_2/
│   └── rq1_1_3/
└── ...
```

## Best Practices

1. **Always run from project root:**

   ```bash
   cd /path/to/mobile-videogpt-adaptation
   bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh
   ```

2. **Check GPU availability:**

   ```bash
   nvidia-smi
   ```

3. **Monitor training:**

   ```bash
   tail -f experiments/results/rq1_1_1/lora/training.log
   ```

4. **Cleanup:**
   ```bash
   # Remove intermediate checkpoints (keep only final)
   find experiments/results -name "checkpoint-*" -type d -exec rm -rf {} +
   ```

## Troubleshooting

### Out of Memory

- Reduce batch size in training scripts
- Enable gradient checkpointing
- Use smaller sequence length

### Slow Training

- Check GPU utilization: `nvidia-smi`
- Increase gradient accumulation steps
- Use mixed precision (bf16/fp16)

### Missing Dependencies

```bash
pip install matplotlib seaborn numpy pandas torch transformers
```

## Citation

If you use these experiments, please cite:

```bibtex
@article{mobile-videogpt-qved,
  title={Efficient Vision-Language Models for Physiotherapy Assessment},
  author={Your Name},
  year={2025}
}
```

## Contact

For questions or issues, please open a GitHub issue or contact the maintainers.
