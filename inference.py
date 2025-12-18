import sys
import os
import warnings
import logging

os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

logging.getLogger('mmengine').setLevel(logging.CRITICAL)
logging.getLogger('transformers').setLevel(logging.CRITICAL)
logging.getLogger('transformers.modeling_utils').setLevel(logging.CRITICAL)

import torch
from pathlib import Path
from PIL import Image
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

sys.path.append("Mobile-VideoGPT")

from mobilevideogpt.utils import preprocess_input

def load_model(pretrained_path: str, device: str = "cuda"):
    """Loads the pre-trained model and tokenizer."""
    config = AutoConfig.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_path,
        config=config,
        torch_dtype=torch.float16
    )
    model.to(device)
    return model, tokenizer


def run_inference(model, tokenizer, video_path: str, prompt: str):
    """Runs inference on the given video file."""
    input_ids, video_frames, context_frames, stop_str = preprocess_input(
        model, tokenizer, video_path, prompt
    )

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames, dim=0).half().cuda(),
            context_images=torch.stack(context_frames, dim=0).half().cuda(),
            do_sample=False,  # Use greedy decoding
            # temperature=0,
            # top_p=1,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )

    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)].strip()

    return outputs


def main():
    pretrained_path = "Amshaker/Mobile-VideoGPT-0.5B"
    video_path = "sample_videos/00000340.mp4"
    prompt = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"
    model, tokenizer = load_model(pretrained_path)
    output = run_inference(model, tokenizer, video_path, prompt)
    print("🤖 Mobile-VideoGPT Output: ", output)


if __name__ == "__main__":
    main()

