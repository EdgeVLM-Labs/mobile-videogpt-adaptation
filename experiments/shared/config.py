"""
Shared configuration and constants for experiments.
"""
from pathlib import Path

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
DATASET_ROOT = PROJECT_ROOT / "dataset"
RESULTS_ROOT = EXPERIMENTS_ROOT / "results"
DATA_ROOT = EXPERIMENTS_ROOT / "data"

# Model configurations
MODELS = {
    'Mobile-VideoGPT-0.5B': {
        'model_name': 'Amshaker/Mobile-VideoGPT-0.5B',
        'size': '0.5B'
    },
    'Mobile-VideoGPT-1.5B': {
        'model_name': 'Amshaker/Mobile-VideoGPT-1.5B',
        'size': '1.5B'
    },
    'VideoLLaMA3-2B': {
        'model_name': 'VideoLLaMA3-2B',
        'size': '2B',
        'requires_adapter': True
    },
    'NVILA-Lite-2B': {
        'model_name': 'NVILA-Lite-2B',
        'size': '2B',
        'requires_adapter': True
    }
}

# Training configurations
TRAINING_CONFIG = {
    'lora': {
        'r': 16,
        'alpha': 32,
        'dropout': 0.05
    },
    'learning_rate': 2e-4,
    'warmup_ratio': 0.05,
    'num_epochs': 3,
    'batch_size': 1,
    'gradient_accumulation_steps': 8
}

# Frame sampling configurations
FRAME_CONFIGS = {
    'default': 16,
    'experiments': [2, 4, 8, 16, 32]
}

# FPS sampling ratios
FPS_RATIOS = [1.0, 0.5, 0.25]

# Data efficiency ratios
DATA_RATIOS = [0.25, 0.5, 0.75, 1.0]

# Degradation types
DEGRADATION_TYPES = {
    'blur': {
        'levels': [1.0, 1.5, 2.5, 3.5],
        'param': 'sigma'
    },
    'brightness': {
        'levels': [-60, -30, 30, 60, 90],
        'param': 'value'
    },
    'noise': {
        'levels': [50, 100, 150, 200],
        'param': 'ratio'
    },
    'compression': {
        'levels': [28, 35, 40, 45],
        'param': 'crf'
    }
}

# Exercise classes
EXERCISE_CLASSES = [
    'knee_circles',
    'opposite_arm_and_leg_lifts_on_knees',
    'pushups_on_knees',
    'squat_jump',
    'squats',
    'tricep_stretch'
]

# Evaluation metrics
EVALUATION_METRICS = [
    'accuracy',
    'per_exercise_accuracy',
    'latency_ms',
    'throughput_samples_per_sec',
    'memory_mb'
]

# LLM Judge configuration
LLM_JUDGE_CONFIG = {
    'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
    'temperature': 0.7,
    'max_tokens': 1024
}
