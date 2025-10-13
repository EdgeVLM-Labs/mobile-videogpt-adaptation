# QVED Fine-Tuning Setup - Implementation Summary

## Files Created

### 1. `tools/qved_from_fine_labels.py`

Converts QVED fine_grained_labels.json to Mobile-VideoGPT format.

- **Config variables at top** (no argparse)
- Extracts exercise name from video path or metadata
- Uses `labels_descriptive` or falls back to `labels[0]`
- Samples up to 100 videos per class (deterministic seed=1337)
- Outputs: `playground/data/MobileGPT/qved_train.json`
- Format: system/user/assistant conversation style

### 2. `scripts/finetune_qved.sh`

Stage-3 only training script.

- Uses `Amshaker/Mobile-VideoGPT-0.5B` (released checkpoint)
- No pretrain adapter paths (uses adapters from base model)
- Hyperparams: epochs=8, lr=1e-4, mm_proj_lr=2e-5, lora_r=64, lora_alpha=128
- Batch=8, grad_accum=2, max_len=2048
- Keeps top-k frame selection, bf16, gradient_checkpointing
- Dataset: `--dataset_use MobileGPT` (auto-includes qved_train.json)

### 3. `scripts/infer_qved.py`

Simple inference script (no argparse).

- **Config variables at top**
- Loads from fine-tuned checkpoint or falls back to base model
- Uses `mobilevideogpt.utils.preprocess_input`
- Prints response for configured video

## Files Modified

### 4. `mobilevideogpt/config/dataset_config.py`

Added QVED_TRAIN dataset definition:

```python
QVED_TRAIN = {
    "annotation_path": f"{DATASET_DIR}/MobileGPT/qved_train.json",
    "data_path": f"{DATASET_DIR}/MobileGPT",
}
```

### 5. `mobilevideogpt/config/__init__.py`

Auto-includes QVED when present:

```python
if os.path.exists(QVED_TRAIN["annotation_path"]):
    _mobilegpt_datasets.append(QVED_TRAIN)
```

### 6. `mobilevideogpt/model/arch.py`

Made adapter loading tolerant:

- If `pretrain_mm_mlp_adapter` is None, prints "Using video projector from base model checkpoint"
- If `pretrain_image_mm_mlp_adapter` is None, prints "Using image projector from base model checkpoint"
- Allows Stage-3 to run directly from released weights without Stage-1/2 artifacts

### 7. `README.md`

Added under Training section:

```markdown
### QVED Fine-Tuning (Stage-3 Only)

To fine-tune Stage-3 on QVED dataset:

1. `python tools/qved_from_fine_labels.py`
2. `bash scripts/finetune_qved.sh`
3. `python scripts/infer_qved.py`
```

## Acceptance Checklist ✓

- [x] `tools/qved_from_fine_labels.py` converts fine_grained_labels.json → qved_train.json
- [x] User-set SYSTEM_PROMPT, user prompt template, assistant from descriptive label
- [x] `scripts/finetune_qved.sh` runs Stage-3 only
- [x] Hyperparams: epochs=8, lr=1e-4, projector_lr=2e-5, lora_r=64, alpha=128, batch=8, grad_accum=2
- [x] Top-k frames, bf16, gradient_checkpointing enabled
- [x] Loader auto-adds qved_train.json when present
- [x] Adapter flags tolerant (None skips loading, uses base model)
- [x] `scripts/infer_qved.py` prints response for configured video
- [x] No argparse/CLI parsers used
- [x] Short inline comments, no block comments
- [x] Minimal changes preserving repo behavior

## Usage Instructions

### Before Running

1. **Update paths in `tools/qved_from_fine_labels.py`:**

   ```python
   FINE_LABELS_JSON = "/path/to/your/fine_grained_labels.json"
   VIDEO_ROOT = "/path/to/your/QVED_root"
   ```

2. **Ensure video paths in fine_grained_labels.json are:**
   - Absolute paths, OR
   - Relative to repo root, OR
   - The script will convert them appropriately

### Running the Pipeline

```bash
# 1. Convert QVED labels to Mobile-VideoGPT format
python tools/qved_from_fine_labels.py

# 2. Fine-tune Stage-3 (requires GPU with DeepSpeed)
bash scripts/finetune_qved.sh

# 3. Test inference
python scripts/infer_qved.py
```

### Expected Behavior

**Data Conversion:**

- Reads fine_grained_labels.json
- Groups by exercise class
- Samples up to 100 videos per class
- Outputs conversational JSON with system/user/assistant turns

**Training:**

- Loads Amshaker/Mobile-VideoGPT-0.5B directly
- Prints "Using video/image projector from base model checkpoint"
- If qved_train.json exists, automatically includes it in training
- Trains for 8 epochs with LoRA on projectors + LLM

**Inference:**

- Loads from fine-tuned checkpoint or base model
- Processes video with top-k frame selection
- Generates corrective feedback response

## Key Design Decisions

1. **No Stage-1/2 dependencies**: Script works with released checkpoint directly
2. **Auto-include dataset**: No code changes needed if qved_train.json doesn't exist
3. **Configurable via file editing**: No CLI means simpler, more visible configuration
4. **Minimal invasive changes**: Only added features, didn't refactor existing code
5. **Preserves repo patterns**: Follows existing dataset/training conventions

## Troubleshooting

**If training fails with adapter errors:**

- Ensure `pretrain_mm_mlp_adapter` and `pretrain_image_mm_mlp_adapter` are NOT set in the script
- The released model includes adapters, so these should be omitted

**If dataset not found:**

- Check that `playground/data/MobileGPT/qved_train.json` exists
- Verify DATASET_DIR environment variable: `echo $DATASET_DIR`

**If videos not loading:**

- Ensure video paths in qved_train.json are relative to repo root or accessible
- Check data_path in QVED_TRAIN config points to correct directory
