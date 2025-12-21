#!/usr/bin/env python3
"""
HuggingFace Model Upload Utility

Uploads finetuned Mobile-VideoGPT models to HuggingFace Hub.

Usage:
    python utils/hf_upload.py --model_path results/qved_finetune_mobilevideogpt_0.5B
    python utils/hf_upload.py --model_path results/qved_finetune_mobilevideogpt_0.5B --repo_name qved-finetune-20241128
    python utils/hf_upload.py --model_path results/qved_finetune_mobilevideogpt_0.5B --private
"""

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_folder, login
import json


# Default organization name
DEFAULT_ORG = "EdgeVLM-Labs"


def get_default_repo_name() -> str:
    """Generate a default repository name with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"qved-finetune-{timestamp}"


def check_hf_login() -> bool:
    """Check if user is logged into HuggingFace."""
    try:
        api = HfApi()
        user_info = api.whoami()
        print(f"✓ Logged in as: {user_info['name']}")
        return True
    except Exception:
        return False


def create_model_card(model_path: Path, repo_id: str, has_adapter: bool) -> str:
    """Create a comprehensive model card with hyperparameters and dataset info."""
    
    # Try to load hyperparameters from saved config
    hyperparams = {}
    config_file = model_path / "hyperparameters.json"
    if not config_file.exists():
        # Check parent directory
        config_file = model_path.parent / "hyperparameters.json"
    
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                hyperparams = json.load(f)
        except Exception:
            pass
    
    # Try to load dataset info
    dataset_info = {}
    try:
        import sys
        sys.path.insert(0, str(Path.cwd()))
        train_file = Path("dataset/qved_train.json")
        val_file = Path("dataset/qved_val.json")
        test_file = Path("dataset/qved_test.json")
        
        if train_file.exists() and val_file.exists() and test_file.exists():
            with open(train_file, 'r') as f:
                train_data = json.load(f)
            with open(val_file, 'r') as f:
                val_data = json.load(f)
            with open(test_file, 'r') as f:
                test_data = json.load(f)
            
            dataset_info = {
                "train": len(train_data),
                "val": len(val_data),
                "test": len(test_data),
                "total": len(train_data) + len(val_data) + len(test_data)
            }
    except Exception:
        pass
    
    # Build model card
    model_card = f"""---
tags:
- video-text-to-text
- mobile-videogpt
- qved
- physiotherapy
- exercise-assessment
library_name: transformers
license: apache-2.0
---

# Mobile-VideoGPT QVED Finetuned Model

This model is a finetuned version of [Amshaker/Mobile-VideoGPT-0.5B](https://huggingface.co/Amshaker/Mobile-VideoGPT-0.5B) on the QVED (Qualitative Video-based Exercise Dataset) for physiotherapy exercise assessment.

## Model Description

- **Base Model:** {hyperparams.get('base_model', 'Amshaker/Mobile-VideoGPT-0.5B')}
- **Architecture:** Mobile-VideoGPT with LoRA adapters
- **Vision Encoder:** VideoMamba + CLIP
- **Task:** Video-based exercise quality assessment and feedback generation
- **Dataset:** QVED (Physiotherapy Exercise Videos)

## Training Details

### Hyperparameters

"""
    
    if hyperparams:
        model_card += f"""- **Epochs:** {hyperparams.get('epochs', 'N/A')}
- **Learning Rate:** {hyperparams.get('learning_rate', 'N/A')}
- **MM Projector LR:** {hyperparams.get('mm_projector_lr', 'N/A')}
- **LoRA Rank:** {hyperparams.get('lora_r', 'N/A')}
- **LoRA Alpha:** {hyperparams.get('lora_alpha', 'N/A')}
- **Batch Size:** {hyperparams.get('batch_size', 'N/A')}
- **Gradient Accumulation Steps:** {hyperparams.get('gradient_accumulation_steps', 'N/A')}
- **Effective Batch Size:** {hyperparams.get('batch_size', 0) * hyperparams.get('gradient_accumulation_steps', 0)}
- **Max Sequence Length:** {hyperparams.get('max_length', 'N/A')}
- **Weight Decay:** 0.0
- **Warmup Ratio:** 0.05
- **LR Scheduler:** Cosine
- **Precision:** bfloat16 + TF32
- **Gradient Checkpointing:** Enabled

"""
    else:
        model_card += "See training script for details.\n\n"
    
    model_card += """### Training Infrastructure

- **Framework:** DeepSpeed with ZeRO-2
- **Mixed Precision:** bfloat16 + TF32
- **Optimization:** LoRA (Low-Rank Adaptation)

"""
    
    if dataset_info:
        model_card += f"""### Dataset Splits

- **Train:** {dataset_info['train']} samples
- **Validation:** {dataset_info['val']} samples
- **Test:** {dataset_info['test']} samples
- **Total:** {dataset_info['total']} samples

"""
    
    model_card += f"""### Training Configuration

- **Vision Tower:** OpenGVLab/VideoMamba
- **Image Vision Tower:** openai/clip-vit-base-patch16
- **Projector Type:** ETP (Efficient Token Projection)
- **Frames per Video:** 4 frames selected via TopK
- **Image Aspect Ratio:** Pad
- **Group by Modality Length:** Enabled

## Usage

### Loading the Model

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    "Amshaker/Mobile-VideoGPT-0.5B",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")
```

### Running Inference

```python
# Prepare video input
video_path = "path/to/exercise_video.mp4"
prompt = "Analyze this physiotherapy exercise video and provide feedback."

# Generate response
response = model.generate(
    video_path=video_path,
    prompt=prompt,
    max_new_tokens=512
)

print(response)
```

### Using the Inference Script

```bash
python utils/test_inference.py \\
    --model_path {repo_id} \\
    --video_path sample_videos/exercise.mp4 \\
    --prompt "Evaluate this exercise" \\
    --max_new_tokens 512
```

## Evaluation Metrics

The model is evaluated on:

- **BERT Similarity:** Semantic similarity between generated and ground truth descriptions
- **METEOR Score:** Translation quality metric for generated text
- **ROUGE-L Score:** Longest common subsequence based similarity
- **Exercise Identification:** Accuracy in identifying the correct exercise type

## Intended Use

This model is designed for:

- Automated physiotherapy exercise assessment
- Generating feedback on exercise form and technique
- Identifying exercise types from video
- Educational and research purposes in healthcare AI

## Limitations

- Trained on a limited dataset ({dataset_info.get('total', 'specific')} samples)
- Performance may vary on exercises not seen during training
- Should not replace professional medical advice
- Video quality and angle significantly affect performance

## Training Procedure

This model was finetuned using:

1. **Dataset Preparation:** QVED videos with quality filtering and optional augmentation
2. **LoRA Finetuning:** Efficient parameter-efficient finetuning of Mobile-VideoGPT-0.5B
3. **Validation:** Continuous evaluation on validation set during training
4. **Metrics Tracking:** WandB integration for experiment tracking

## Citation

If you use this model, please cite:

```bibtex
@misc{{mobile-videogpt-qved-finetune,
  author = {{EdgeVLM Labs}},
  title = {{Mobile-VideoGPT QVED Finetuned Model}},
  year = {{2025}},
  publisher = {{HuggingFace}},
  url = {{https://huggingface.co/{repo_id}}}
}}
```

## Model Card Authors

EdgeVLM Labs

## Model Card Contact

For questions or feedback, please open an issue in the model repository.
"""
    
    return model_card


def upload_model_to_hf(
    model_path: str,
    repo_name: str = None,
    org_name: str = DEFAULT_ORG,
    private: bool = False,
    commit_message: str = None,
) -> str:
    """
    Upload a finetuned model to HuggingFace Hub.

    Args:
        model_path: Path to the model directory (can be checkpoint or base finetuning dir)
        repo_name: Name for the HuggingFace repository
        org_name: HuggingFace organization name
        private: Whether to create a private repository
        commit_message: Custom commit message

    Returns:
        URL of the uploaded model on HuggingFace
    """
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model path not found: {model_path}")

    # Check for adapter files or model files
    has_adapter = (model_path / "adapter_config.json").exists()
    has_model = (model_path / "config.json").exists() or (model_path / "pytorch_model.bin").exists()

    if not has_adapter and not has_model:
        # Maybe it's a checkpoint directory
        checkpoints = list(model_path.glob("checkpoint-*"))
        if checkpoints:
            # Use the latest checkpoint
            latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.name.split("-")[1]))[-1]
            print(f"Using latest checkpoint: {latest_checkpoint}")
            model_path = latest_checkpoint
            has_adapter = (model_path / "adapter_config.json").exists()
            has_model = (model_path / "config.json").exists()

    if not has_adapter and not has_model:
        raise ValueError(
            f"No model or adapter files found in {model_path}. "
            "Expected adapter_config.json or config.json"
        )

    # Generate repo name if not provided
    if repo_name is None:
        repo_name = get_default_repo_name()

    # Full repository ID
    repo_id = f"{org_name}/{repo_name}"

    print(f"\n{'='*60}")
    print("HuggingFace Model Upload")
    print(f"{'='*60}")
    print(f"Model path: {model_path}")
    print(f"Repository: {repo_id}")
    print(f"Private: {private}")
    print(f"Type: {'LoRA Adapter' if has_adapter else 'Full Model'}")
    print(f"{'='*60}\n")

    # Check login status
    if not check_hf_login():
        print("⚠ Not logged into HuggingFace. Please login first:")
        print("  huggingface-cli login")
        print("  or set HF_TOKEN environment variable")

        # Try to login with token from environment
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if hf_token:
            print("\nFound HF_TOKEN in environment, attempting login...")
            login(token=hf_token)
        else:
            sys.exit(1)

    api = HfApi()

    # Create repository
    print(f"📦 Creating repository: {repo_id}")
    try:
        create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=private,
            exist_ok=True,
        )
        print(f"✓ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"⚠ Warning: Could not create repository: {e}")
        print("  Will try to upload anyway...")

    # Generate and save model card
    print(f"\n📝 Generating model card...")
    model_card_content = create_model_card(model_path, repo_id, has_adapter)
    readme_path = model_path / "README.md"
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(model_card_content)
    print(f"✓ Model card saved to {readme_path}")
    
    # Prepare commit message
    if commit_message is None:
        if has_adapter:
            commit_message = f"Upload LoRA adapters from {model_path.name}"
        else:
            commit_message = f"Upload finetuned model from {model_path.name}"

    # Upload model
    print(f"\n🚀 Uploading model to {repo_id}...")
    print("  This may take a few minutes depending on model size...")

    try:
        upload_folder(
            folder_path=str(model_path),
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            ignore_patterns=["*.py", "__pycache__", "*.pyc", "runs/*", "wandb/*"],
        )
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        raise

    # Get repository URL
    repo_url = f"https://huggingface.co/{repo_id}"

    print(f"\n{'='*60}")
    print("✅ Upload Complete!")
    print(f"{'='*60}")
    print(f"Repository URL: {repo_url}")
    print(f"\nTo use this model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{repo_id}')")
    if has_adapter:
        print(f"\n  # For LoRA adapters:")
        print(f"  from peft import PeftModel")
        print(f"  base_model = AutoModelForCausalLM.from_pretrained('Amshaker/Mobile-VideoGPT-0.5B')")
        print(f"  model = PeftModel.from_pretrained(base_model, '{repo_id}')")
    print(f"{'='*60}")

    return repo_url


def main():
    parser = argparse.ArgumentParser(
        description="Upload finetuned Mobile-VideoGPT model to HuggingFace Hub"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the finetuned model directory",
    )
    parser.add_argument(
        "--repo_name",
        type=str,
        default=None,
        help=f"Name for the HuggingFace repository (default: qved-finetune-TIMESTAMP)",
    )
    parser.add_argument(
        "--org",
        type=str,
        default=DEFAULT_ORG,
        help=f"HuggingFace organization name (default: {DEFAULT_ORG})",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create a private repository",
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for the upload",
    )

    args = parser.parse_args()

    try:
        repo_url = upload_model_to_hf(
            model_path=args.model_path,
            repo_name=args.repo_name,
            org_name=args.org,
            private=args.private,
            commit_message=args.commit_message,
        )
        print(f"\n🎉 Success! Model available at: {repo_url}")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
