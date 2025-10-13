import sys
import os
import warnings
import torch
from pathlib import Path

# Suppress warnings
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
sys.path.append(".")
from mobilevideogpt.utils import preprocess_input

# CONFIG
MODEL = "Amshaker/Mobile-VideoGPT-0.5B"
CKPT_DIR = "results/qved_finetune_mobilevideogpt"  # if you want to load from output dir
VIDEO = "sample_videos/v_JspVuT6rsLA.mp4"
PROMPT = "What corrective feedback applies to this exercise video?"

def main():
    # Load from fine-tuned checkpoint if available, else base model
    model_path = CKPT_DIR if os.path.exists(CKPT_DIR) else MODEL

    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16
    )
    model.to("cuda")

    # Preprocess video and prompt
    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, VIDEO, PROMPT
    )

    # Generate response
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().cuda(),
            context_images=torch.stack(context_frames, dim=0).half().cuda(),
            do_sample=False,
            num_beams=1,
            max_new_tokens=512,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    print(f"Model: {model_path}")
    print(f"Video: {VIDEO}")
    print(f"Prompt: {PROMPT}")
    print(f"\nResponse:\n{outputs}")

if __name__ == "__main__":
    main()
