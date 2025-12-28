#!/bin/bash
# Master script to run all RQ1 (Training Efficiency) experiments

set -e

MODEL=${1:-"Mobile-VideoGPT-0.5B"}

echo "========================================"
echo " Running All RQ1 Experiments"
echo " Model: $MODEL"
echo "========================================"

# RQ1.1.1: Full vs LoRA
echo -e "\n[1/3] Running RQ1.1.1: Full vs LoRA Fine-tuning"
bash experiments/rq1_training_efficiency/rq1_1_1_full_vs_lora.sh "$MODEL"

# RQ1.1.2: Learning Rate Schedules
echo -e "\n[2/3] Running RQ1.1.2: Learning Rate Schedules"
bash experiments/rq1_training_efficiency/rq1_1_2_lr_schedules.sh "$MODEL"

# RQ1.1.3: Data Efficiency
echo -e "\n[3/3] Running RQ1.1.3: Data Efficiency"
bash experiments/rq1_training_efficiency/rq1_1_3_data_efficiency.sh "$MODEL"

echo -e "\n========================================"
echo "✅ All RQ1 experiments complete!"
echo "========================================"
echo "Results directory: experiments/results/rq1_training_efficiency/"
