"""
Polling-based inference engine for Mobile-VideoGPT.
Loads LoRA adapters and performs inference at configurable intervals.
"""

import os
import sys
import time
import logging
import warnings
from typing import Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass

# Suppress warnings before imports
os.environ['PYTHONWARNINGS'] = 'ignore'
warnings.filterwarnings("ignore")

# Reduce CUDA allocator fragmentation on memory-constrained devices (Jetson 8GB)
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

import torch
from transformers import AutoTokenizer, AutoConfig

# Patch Triton's autotuner cache-flush buffer — default 256MB fails on fragmented
# Jetson memory. A tiny buffer keeps autotuning functional; the only cost is
# slightly noisier benchmark timings on the first call (no accuracy impact).
def _install_triton_patch():
    try:
        import triton
        # Patch the active driver instance
        try:
            driver = triton.runtime.driver.active
            def _tiny(*args, **kwargs):
                return torch.empty(1024, dtype=torch.int, device='cuda')
            driver.get_empty_cache_for_benchmark = _tiny
        except Exception:
            pass

        # Also patch the class so any future instances get the tiny version
        try:
            from triton.backends.nvidia import driver as _trtdrv
            for name in dir(_trtdrv):
                cls = getattr(_trtdrv, name)
                if isinstance(cls, type) and hasattr(cls, "get_empty_cache_for_benchmark"):
                    cls.get_empty_cache_for_benchmark = lambda self: torch.empty(
                        1024, dtype=torch.int, device='cuda'
                    )
        except Exception:
            pass
    except Exception:
        pass  # Triton not installed — safe to skip

_install_triton_patch()

# PyTorch optimizations for faster inference
torch.set_float32_matmul_precision('high')  # Use TF32 on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune kernels for input shapes
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Monkey-patch mamba_ssm to work with causal_conv1d >= 1.5.0
# In newer versions, causal_conv1d_cuda.causal_conv1d_fwd requires a pre-allocated
# output tensor (8 args) instead of returning it (7 args).
# This patch is applied in our repo so fresh clones don't need to edit site-packages.
# ---------------------------------------------------------------------------
def _patch_mamba_causal_conv1d():
    try:
        import causal_conv1d_cuda
        import mamba_ssm.ops.selective_scan_interface as ssi
        import inspect

        source = inspect.getsource(ssi.MambaInnerFn.forward)
        # Check if already using 8-arg API (has torch.empty_like before the call)
        if 'torch.empty_like' not in source and 'causal_conv1d_cuda.causal_conv1d_fwd' in source:
            _orig_fwd = causal_conv1d_cuda.causal_conv1d_fwd

            def _patched_fwd(x, weight, bias, seq_idx=None, initial_states=None,
                             final_states_out=None, activation=True):
                out = torch.empty_like(x)
                _orig_fwd(x, weight, bias, seq_idx, initial_states, out, final_states_out, activation)
                return out

            causal_conv1d_cuda.causal_conv1d_fwd = _patched_fwd
            logging.getLogger(__name__).info("Patched mamba_ssm for causal_conv1d >= 1.5 API")
    except (ImportError, Exception):
        pass  # mamba_ssm or causal_conv1d not installed, skip

_patch_mamba_causal_conv1d()

from mobilevideogpt.model import MobileVideoGPTQwenForCausalLM
from mobilevideogpt.mm_utils import tokenizer_image_token
from mobilevideogpt.conversation import conv_templates
from mobilevideogpt.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    NUM_FRAMES,
    NUM_CONTEXT_IMAGES,
)

from polling.config import PollingConfig
from polling.metrics import MetricsTracker, InferenceMetrics
from polling.stream_handler import VideoStreamHandler
from utils.confidence_scoring.calculate_confidence import is_confident


class FirstTokenStreamer:
    """Helper to capture time to first token during generation."""

    def __init__(self):
        self.first_token_time: Optional[float] = None
        self.start_time: float = 0.0

    def reset(self):
        self.first_token_time = None
        self.start_time = time.time()

    def on_token(self):
        if self.first_token_time is None:
            self.first_token_time = time.time()

    @property
    def time_to_first_token(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return self.first_token_time - self.start_time


class PollingInferenceEngine:
    """
    Main engine for polling-based streaming inference.

    Loads the model with LoRA adapters and performs inference
    at configurable polling intervals on video streams.
    """

    def __init__(self, config: PollingConfig):
        self.config = config
        self.logger = self._setup_logging()

        # Model components (loaded lazily)
        self.model = None
        self.tokenizer = None
        self.image_processor = None
        self.video_processor = None

        # Metrics tracking
        self.metrics = MetricsTracker(
            log_dir=config.log_dir,
            save_metrics=config.save_metrics,
        )

        # Stream handler
        self.stream_handler = VideoStreamHandler(
            buffer_size=config.frame_buffer_size,
            num_frames=config.num_frames,
            fps=config.fps,
            image_resolution=config.image_resolution,
            inference_window_seconds=getattr(config, "inference_window_seconds", 0.0),
        )

        # First token streamer
        self._first_token_streamer = FirstTokenStreamer()

        # State
        self._is_loaded = False

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("PollingInference")
        logger.setLevel(getattr(logging, self.config.log_level))

        # Clear existing handlers
        logger.handlers = []

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_format = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)

        # Note: File handler is created by MetricsTracker.start_session()
        # to ensure consistent session_id across all files

        return logger

    def load_model(self) -> bool:
        """
        Load the base model with LoRA adapters from HuggingFace.

        Returns:
            True if model loaded successfully
        """
        # Check if model is already loaded
        if self._is_loaded and self.model is not None:
            self.logger.info("Model already loaded, skipping reload")
            return True

        self.logger.info("=" * 60)
        self.logger.info("LOADING MODEL WITH LORA ADAPTERS")
        self.logger.info("=" * 60)
        self.logger.info(f"Base model: {self.config.base_model_path}")
        self.logger.info(f"LoRA weights: {self.config.lora_weights_path}")

        load_start = time.time()

        try:
            # Import peft for LoRA
            from peft import PeftModel

            # Setup kwargs for model loading
            kwargs = {}
            if self.config.load_8bit:
                kwargs['load_in_8bit'] = True
            elif self.config.load_4bit:
                from transformers import BitsAndBytesConfig
                kwargs['quantization_config'] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4',
                    llm_int8_skip_modules=['lm_head'],
                )
                # Note: Flash Attention incompatible with 4-bit quantization
            elif os.environ.get("USE_QUANTO", "0") == "1":
                # optimum-quanto INT8 weight quantization (Jetson-friendly).
                # Saves ~50% memory vs FP16, accuracy drop <0.5% for structured outputs.
                # We DON'T use transformers' QuantoConfig (requires old 'quanto' pkg);
                # instead we load the model normally then apply quanto post-load below.
                kwargs['torch_dtype'] = torch.float16
                kwargs['attn_implementation'] = 'sdpa'
                quanto_weight = os.environ.get("QUANTO_WEIGHTS", "int8")
                self.logger.info(f"Post-load INT8 quantization via optimum-quanto (weights={quanto_weight})")
            else:
                kwargs['torch_dtype'] = torch.float16  # Use float16 (matches model config)
                try:
                    import flash_attn  # noqa: F401
                    kwargs['attn_implementation'] = 'flash_attention_2'
                except ImportError:
                    kwargs['attn_implementation'] = 'sdpa'  # PyTorch native SDPA fallback
                    self.logger.warning("Flash Attention 2 not available, using SDPA")

            # Load config from base model (LoRA adapters don't have config.json)
            self.logger.info("Loading model configuration from base model...")
            model_cfg = AutoConfig.from_pretrained(self.config.base_model_path)

            # Load tokenizer from base model
            self.logger.info("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.base_model_path,
                use_fast=False
            )
            self.tokenizer.add_tokens(["<image>"], special_tokens=True)

            # Optionally preload TensorRT CLIP engine BEFORE the main model.
            # Enable with: USE_TRT_CLIP=1 in environment.
            # Disabled by default because TRT CLIP + Qwen2 don't reliably fit on 8GB
            # Jetson Orin Nano — the lm_head during generation OOMs.
            # Works fine on boards with more memory (Orin NX 8GB+, etc.)
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            trt_clip = None
            if os.environ.get("USE_TRT_CLIP", "0") == "1":
                from mobilevideogpt.model.multimodal_encoder.clip_trt import preload_trt_clip
                trt_clip = preload_trt_clip()
                if trt_clip is not None:
                    self.logger.info("TensorRT CLIP preloaded on GPU (USE_TRT_CLIP=1)")
                else:
                    self.logger.info("TensorRT CLIP preload failed — using PyTorch CPU")

            # Load base model
            # On Jetson (8GB unified), use device_map="auto" with max_memory
            # so accelerate splits the model between CUDA and CPU as needed.
            use_quanto = os.environ.get("USE_QUANTO", "0") == "1"
            use_full_gpu = os.environ.get("USE_FULL_GPU", "0") == "1"
            if torch.cuda.is_available():
                free_mem = torch.cuda.mem_get_info()[0]
                # CUDA budget % depends on whether TRT CLIP is holding memory and
                # whether we're using quanto (INT8 weights = ~50% smaller, so can fit more).
                if use_full_gpu:
                    # Aggressive: fit all Qwen2 layers on GPU to avoid CPU<->GPU
                    # transfer bounce during autoregressive generation.
                    # Risk: lm_head output tensor (~140MB) might OOM during generate().
                    # If it does, we fall back to moving lm_head explicitly to CPU.
                    budget_pct = 0.75
                elif use_quanto:
                    # With INT8 Qwen2 (~500MB instead of 1GB), bigger budget to keep
                    # the whole model on GPU and avoid CPU offload.
                    budget_pct = 0.55 if trt_clip is not None else 0.65
                elif trt_clip is not None:
                    budget_pct = 0.30  # With TRT CLIP holding ~500MB
                else:
                    budget_pct = 0.40  # Baseline FP16, no TRT CLIP
                cuda_budget = max(int(free_mem * budget_pct), 512 * 1024 * 1024)
                max_memory = {0: cuda_budget, "cpu": "2GiB"}
                if use_full_gpu:
                    mode_tag = "FULL-GPU" + ("+TRT-CLIP" if trt_clip else "")
                elif use_quanto:
                    mode_tag = "QUANTO-INT8"
                else:
                    mode_tag = "FP16+TRT-CLIP" if trt_clip else "FP16"
                self.logger.info(
                    f"CUDA budget: {cuda_budget / 1e9:.2f}GB ({int(budget_pct*100)}% of "
                    f"{free_mem / 1e9:.2f}GB free, mode={mode_tag})"
                )
            else:
                max_memory = None
                use_full_gpu = False

            self.logger.info("Loading base model...")
            self.model = MobileVideoGPTQwenForCausalLM.from_pretrained(
                self.config.base_model_path,
                low_cpu_mem_usage=True,
                device_map="auto",
                max_memory=max_memory,
                offload_folder="offload",
                config=model_cfg,
                num_select_k_frames_in_chunk=self.config.num_select_k_frames_in_chunk,
                topk=self.config.topk,
                **kwargs
            )

            # Diagnostic: report where Qwen2 layers actually ended up after
            # accelerate's device_map placement. Helps verify USE_FULL_GPU works.
            try:
                layer_devices = {}
                for i, layer in enumerate(self.model.model.layers):
                    dev = str(next(layer.parameters()).device)
                    layer_devices[dev] = layer_devices.get(dev, 0) + 1
                self.logger.info(
                    f"Qwen2 layer placement: {layer_devices}  "
                    f"(lm_head: {next(self.model.lm_head.parameters()).device})"
                )
            except Exception:
                pass

            # Post-load INT8 quantization of Qwen2 LLM layers only.
            # We intentionally skip the vision tower, image tower, projectors, and
            # lm_head — those use custom kernels (mamba_ssm, VideoMamba) or are the
            # final output layer where precision matters most.
            if use_quanto:
                from optimum.quanto import quantize, freeze, qint8, qint4
                qtype = qint4 if quanto_weight == "int4" else qint8
                # Only quantize the Qwen2 transformer backbone (self.model.model.layers)
                # Leaves embeddings, lm_head, vision_tower, image_vision_tower, projectors in FP16.
                qwen_backbone = self.model.model
                t0 = time.time()
                self.logger.info(f"Quantizing Qwen2 backbone ({quanto_weight})...")
                quantize(qwen_backbone.layers, weights=qtype)
                freeze(qwen_backbone.layers)
                self.logger.info(f"Quantization done in {time.time() - t0:.1f}s")
                gc.collect()
                torch.cuda.empty_cache()
                if torch.cuda.is_available():
                    free_after = torch.cuda.mem_get_info()[0] / 1e9
                    self.logger.info(f"CUDA free after quantization: {free_after:.2f}GB")

            # Resize token embeddings
            token_num, token_dim = self.model.lm_head.out_features, self.model.lm_head.in_features
            if self.model.lm_head.weight.shape[0] != token_num:
                self.model.lm_head.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype)
                )
                self.model.model.embed_tokens.weight = torch.nn.Parameter(
                    torch.empty(token_num, token_dim, device=self.model.device, dtype=self.model.dtype)
                )

            # Load non-LoRA trainables
            self.logger.info("Loading non-LoRA trainables...")
            non_lora_path = os.path.join(self.config.lora_weights_path, 'non_lora_trainables.bin')

            # Try to load from HuggingFace hub
            from huggingface_hub import hf_hub_download
            try:
                non_lora_local = hf_hub_download(
                    repo_id=self.config.lora_weights_path,
                    filename="non_lora_trainables.bin"
                )
                non_lora_trainables = torch.load(non_lora_local, map_location='cpu')
            except Exception as e:
                self.logger.warning(f"Could not load non_lora_trainables from hub: {e}")
                # Try local path
                if os.path.exists(non_lora_path):
                    non_lora_trainables = torch.load(non_lora_path, map_location='cpu')
                else:
                    self.logger.warning("No non_lora_trainables found, proceeding without")
                    non_lora_trainables = {}

            # Clean up keys
            non_lora_trainables = {
                (k[11:] if k.startswith('base_model.') else k): v
                for k, v in non_lora_trainables.items()
            }
            if any(k.startswith('model.model.') for k in non_lora_trainables):
                non_lora_trainables = {
                    (k[6:] if k.startswith('model.') else k): v
                    for k, v in non_lora_trainables.items()
                }

            self.model.load_state_dict(non_lora_trainables, strict=False)

            # Load and merge LoRA weights
            self.logger.info("Loading LoRA adapter weights...")
            try:
                self.model = PeftModel.from_pretrained(self.model, self.config.lora_weights_path)

                self.logger.info("Merging LoRA weights...")
                self.model = self.model.merge_and_unload()
                self.logger.info("LoRA adapters loaded and merged successfully")
            except Exception as e:
                self.logger.warning(f"Could not load LoRA adapters: {e}")
                self.logger.info("Proceeding with base model only")

            # Setup special tokens
            mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
            mm_use_im_patch_token = getattr(self.model.config, "mm_use_im_patch_token", True)

            from mobilevideogpt.constants import (
                DEFAULT_IMAGE_PATCH_TOKEN,
                DEFAULT_IM_START_TOKEN,
                DEFAULT_IM_END_TOKEN
            )

            if mm_use_im_patch_token:
                self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            if mm_use_im_start_end:
                self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

            self.model.resize_token_embeddings(len(self.tokenizer))

            # Model placed by device_map="auto" — clean up and set eval mode
            gc.collect()
            torch.cuda.empty_cache()
            self.model.eval()

            # Setup vision processors
            self.logger.info("Setting up vision processors...")
            vision_tower = self.model.get_vision_tower()
            vision_tower.load_model(self.model.config.mm_vision_tower)
            self.video_processor = vision_tower.image_processor

            image_vision_tower = self.model.get_image_vision_tower()
            image_vision_tower.load_model()
            self.image_processor = image_vision_tower.image_processor

            load_time = time.time() - load_start
            self.logger.info(f"Model loaded successfully in {load_time:.2f}s")
            self.logger.info("=" * 60)

            self._is_loaded = True
            return True

        except Exception as e:
            self.logger.error(f"Failed to load model: {e}", exc_info=True)
            return False

    def prepare_prompt(self, prompt: str, slice_len: int) -> torch.Tensor:
        """Prepare the prompt with image tokens."""
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)

        if mm_use_im_start_end:
            from mobilevideogpt.constants import DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + prompt
        else:
            qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt

        # Use Qwen2 instruct template
        conv = conv_templates["qwen2_instruct"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        formatted_prompt = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            formatted_prompt,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.config.device)

        return input_ids, conv.sep

    def warmup(self, num_runs: int = 1):
        """
        Perform warmup runs to load model into memory and optimize caching.

        Args:
            num_runs: Number of warmup inference runs
        """
        self.logger.info("="*60)
        self.logger.info(f"Starting warmup with {num_runs} run(s)...")
        self.logger.info("="*60)

        # Create dummy data
        dummy_frames = [
            torch.zeros(
                (3, self.config.image_resolution, self.config.image_resolution),
                dtype=torch.float16,
                device=self.config.device
            )
            for _ in range(self.config.num_frames)
        ]

        dummy_context = [
            torch.zeros(
                (3, self.config.image_resolution, self.config.image_resolution),
                dtype=torch.float16,
                device=self.config.device
            )
            for _ in range(self.config.num_context_images)
        ]

        warmup_prompt = "Analyze this exercise."

        for i in range(num_runs):
            start_time = time.time()
            self.logger.info(f"Warmup run {i+1}/{num_runs}...")

            try:
                _, ttft, _, _ = self.run_single_inference(
                    dummy_frames,
                    dummy_context,
                    warmup_prompt,
                    self.config.num_frames
                )

                warmup_time = time.time() - start_time
                self.logger.info(f"  Completed in {warmup_time:.2f}s (TTFT: {ttft*1000:.1f}ms)")

            except Exception as e:
                self.logger.warning(f"  Warmup run {i+1} failed: {e}")

        self.logger.info("Warmup complete!")
        self.logger.info("="*60)

    def run_single_inference(
        self,
        video_frames: torch.Tensor,
        context_frames: torch.Tensor,
        prompt: str,
        slice_len: int,
    ) -> Tuple[str, float, int, int]:
        """
        Run a single inference on the provided frames.

        Returns:
            Tuple of (response, time_to_first_token, input_tokens, output_tokens)
        """
        # Prepare input
        input_ids, stop_str = self.prepare_prompt(prompt, slice_len)

        # Prepare frames with bfloat16 to match model dtype
        video_tensor = torch.stack(video_frames, dim=0).to(dtype=torch.float16, device=self.config.device)
        context_tensor = torch.stack(context_frames, dim=0).to(dtype=torch.float16, device=self.config.device)

        input_token_count = input_ids.shape[1]

        # Reset first token timer
        self._first_token_streamer.reset()

        # Generate with inference_mode (more efficient than no_grad)
        # Return scores for confidence calculation if enabled
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=video_tensor,
                context_images=context_tensor,
                do_sample=self.config.do_sample,
                num_beams=self.config.num_beams,
                max_new_tokens=self.config.max_new_tokens,
                use_cache=True,  # Always use KV cache for faster generation
                return_dict_in_generate=self.config.enable_confidence_scoring,
                output_scores=self.config.enable_confidence_scoring,
            )

        # Record first token time (approximate since we can't hook into generate)
        generation_end = time.time()

        # Extract output_ids from the generation result
        if self.config.enable_confidence_scoring and hasattr(output_ids, 'sequences'):
            # output_ids is a GenerateOutput object
            sequences = output_ids.sequences
            scores = output_ids.scores if hasattr(output_ids, 'scores') else None
            sequences_scores = output_ids.sequences_scores if hasattr(output_ids, 'sequences_scores') else None
            beam_indices = output_ids.beam_indices if hasattr(output_ids, 'beam_indices') else None
            output_ids = sequences
        else:
            scores = None
            sequences_scores = None
            beam_indices = None

        # Decode output
        output_tokens = output_ids.shape[1] - input_token_count
        response = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        if response.endswith(stop_str):
            response = response[:-len(stop_str)].strip()

        # Check confidence if enabled
        if self.config.enable_confidence_scoring and scores is not None:
            try:
                confident, confidence_metrics = is_confident(
                    scores=scores,
                    output_ids=output_ids,
                    input_token_count=input_token_count,
                    sequences_scores=sequences_scores,
                    beam_indices=beam_indices,
                )
                
                # Log confidence metrics
                self.logger.debug(f"Confidence metrics: {confidence_metrics}")
                
                # Append "(NOT CONFIDENT)" if model is not confident
                if not confident:
                    response = response + " (NOT CONFIDENT)"
                    self.logger.info("Low confidence detected, appended (NOT CONFIDENT) to response")
            except Exception as e:
                self.logger.warning(f"Failed to calculate confidence: {e}")

        # Estimate TTFT (first token is roughly 1/output_tokens of total time)
        # This is approximate since transformers doesn't expose per-token timing
        ttft = self._first_token_streamer.time_to_first_token
        if ttft == 0:
            # Estimate: TTFT is typically the encoding time + first decoding step
            ttft = (generation_end - self._first_token_streamer.start_time) / max(output_tokens, 1) * 2

        return response, ttft, input_token_count, output_tokens

    def run_single_inference_streaming(
        self,
        video_frames,
        context_frames,
        prompt: str,
        slice_len: int,
    ):
        """
        Streaming variant of run_single_inference.

        Yields tuples of (partial_response, is_final, elapsed_seconds, metrics).
        The final yield has is_final=True with final metrics populated:
            metrics = dict(ttft=float, input_tokens=int, output_tokens=int)

        Usage:
            for partial, done, elapsed, metrics in engine.run_single_inference_streaming(...):
                ui.update(partial)
                if done: break
        """
        import threading
        from transformers import TextIteratorStreamer

        input_ids, stop_str = self.prepare_prompt(prompt, slice_len)
        video_tensor = torch.stack(video_frames, dim=0).to(
            dtype=torch.float16, device=self.config.device
        )
        context_tensor = torch.stack(context_frames, dim=0).to(
            dtype=torch.float16, device=self.config.device
        )
        input_token_count = input_ids.shape[1]

        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
            timeout=60.0,  # safety net
        )

        gen_kwargs = dict(
            images=video_tensor,
            context_images=context_tensor,
            do_sample=self.config.do_sample,
            num_beams=self.config.num_beams,
            max_new_tokens=self.config.max_new_tokens,
            use_cache=True,
            streamer=streamer,
        )

        thread_error = {"exc": None}

        def _generate():
            try:
                with torch.inference_mode():
                    self.model.generate(input_ids, **gen_kwargs)
            except Exception as exc:  # pass error out to main thread
                thread_error["exc"] = exc
            finally:
                streamer.end()

        t_start = time.time()
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # Poll the streamer queue with short timeout so the caller can yield
        # progress updates while prefill is still running (no tokens yet).
        # Returns None every `poll_interval` seconds during quiet periods.
        import queue as _queue
        poll_interval = 0.8  # seconds between progress pings
        accumulated = ""
        ttft = 0.0
        try:
            while True:
                try:
                    token_text = streamer.text_queue.get(timeout=poll_interval)
                except _queue.Empty:
                    # No token yet — yield a "no-progress" heartbeat so caller
                    # can refresh UI with a progress indicator.
                    yield None, False, time.time() - t_start, None
                    continue

                # `streamer.end()` sends the stop_signal sentinel
                if token_text is streamer.stop_signal:
                    break

                if ttft == 0.0 and token_text:
                    ttft = time.time() - t_start
                accumulated += token_text
                yield accumulated, False, time.time() - t_start, None
        finally:
            thread.join()

        if thread_error["exc"] is not None:
            raise thread_error["exc"]

        # Strip end-of-turn marker if present
        final_text = accumulated
        if final_text.endswith(stop_str):
            final_text = final_text[: -len(stop_str)].strip()

        # Estimate output_tokens by re-tokenizing final text (fast enough)
        try:
            output_tokens = len(self.tokenizer(final_text, add_special_tokens=False).input_ids)
        except Exception:
            output_tokens = 0

        yield final_text, True, time.time() - t_start, {
            "ttft": ttft,
            "input_tokens": input_token_count,
            "output_tokens": output_tokens,
        }

    def run_polling_loop(
        self,
        video_source: str,
        prompt: Optional[str] = None,
        on_response: Optional[Callable[[int, str, InferenceMetrics], None]] = None,
        max_polls: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the main polling loop on a video source.

        Args:
            video_source: Path to video file or stream URL
            prompt: Inference prompt (uses config default if None)
            on_response: Callback called after each inference (poll_index, response, metrics)
            max_polls: Maximum number of polls (None = run until video ends or max_duration)

        Returns:
            Session summary dictionary
        """
        if not self._is_loaded:
            self.logger.error("Model not loaded. Call load_model() first.")
            return {"error": "Model not loaded"}

        prompt = prompt or self.config.prompt

        self.logger.info("=" * 60)
        self.logger.info("STARTING POLLING LOOP")
        self.logger.info("=" * 60)
        self.logger.info(f"Video source: {video_source}")
        self.logger.info(f"Polling interval: {self.config.polling_interval}s")
        self.logger.info(f"Prompt: {prompt[:100]}...")
        self.logger.info("=" * 60)

        # Open video source
        if os.path.isfile(video_source):
            if not self.stream_handler.open_video_file(video_source):
                return {"error": f"Failed to open video file: {video_source}"}
        else:
            self.stream_handler.start_stream_capture(video_source)

        # Start metrics session
        self.metrics.start_session(
            video_source=video_source,
            prompt=prompt,
            polling_interval=self.config.polling_interval,
        )

        poll_index = 0
        start_time = time.time()

        try:
            while True:
                # Check termination conditions
                elapsed = time.time() - start_time
                if elapsed >= self.config.max_polling_duration:
                    self.logger.info(f"Max polling duration ({self.config.max_polling_duration}s) reached")
                    break

                if max_polls is not None and poll_index >= max_polls:
                    self.logger.info(f"Max polls ({max_polls}) reached")
                    break

                if self.stream_handler.is_exhausted:
                    self.logger.info("Video exhausted")
                    break

                self.logger.info(f"\n{'='*40}")
                self.logger.info(f"POLL #{poll_index + 1}")
                self.logger.info(f"Video position: {self.stream_handler.current_position:.2f}s / {self.stream_handler.total_duration:.2f}s")
                self.logger.info(f"{'='*40}")

                # Start metrics for this inference
                self.metrics.start_inference(poll_index)

                try:
                    # Extract frames from current position
                    frame_start = time.time()
                    video_frames, context_frames, slice_len = self.stream_handler.get_frames_for_inference(
                        self.image_processor,
                        self.video_processor,
                        num_video_frames=self.config.num_frames,
                        num_context_images=self.config.num_context_images,
                        polling_interval=self.config.polling_interval,
                    )
                    frame_time = time.time() - frame_start
                    self.metrics.record_timing("frame_extraction_time", frame_time)

                    if slice_len == 0:
                        self.logger.warning("No frames extracted, end of video reached")
                        break

                    self.logger.info(f"Extracted {slice_len} frames in {frame_time*1000:.1f}ms")

                    # Run inference with one OOM-retry for reliability on
                    # memory-constrained devices. First CUDA OOM we clear cache
                    # and try again — usually succeeds after defragmentation.
                    inference_start = time.time()
                    try:
                        response, ttft, input_tokens, output_tokens = self.run_single_inference(
                            video_frames, context_frames, prompt, slice_len
                        )
                    except RuntimeError as oom_err:
                        if 'out of memory' in str(oom_err).lower() or 'NVML_SUCCESS' in str(oom_err):
                            self.logger.warning(f"Inference OOM — clearing cache and retrying once")
                            import gc
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                torch.cuda.synchronize()
                            response, ttft, input_tokens, output_tokens = self.run_single_inference(
                                video_frames, context_frames, prompt, slice_len
                            )
                        else:
                            raise
                    inference_time = time.time() - inference_start
                    self.metrics.record_timing("generation_time", inference_time)

                    # Record metrics
                    metrics = self.metrics.end_inference(
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        frames_processed=slice_len,
                        buffer_size=len(self.stream_handler.frame_buffer),
                        response=response,
                        time_to_first_token=ttft,
                    )

                    # Log response
                    self.logger.info(f"\n📝 Response:\n{response}\n")

                    # Callback
                    if on_response:
                        on_response(poll_index, response, metrics)

                except Exception as e:
                    import traceback
                    self.metrics.record_error(poll_index, str(e), traceback.format_exc())
                    self.logger.error(f"Inference failed: {e}", exc_info=True)

                poll_index += 1

                # Free CUDA tensor cache between polls — prevents fragmentation
                # that causes OOM on the lm_head projection during generate().
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Wait for next poll
                if not self.stream_handler.is_exhausted:
                    self.logger.info(f"Waiting {self.config.polling_interval}s until next poll...")
                    time.sleep(self.config.polling_interval)

        except KeyboardInterrupt:
            self.logger.info("\nPolling interrupted by user")

        finally:
            self.stream_handler.close()

        # End session and get summary
        summary = self.metrics.end_session()

        return summary

    def cleanup(self):
        """Clean up resources."""
        self.stream_handler.close()

        # Unload model and processors
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if self.image_processor is not None:
            del self.image_processor
            self.image_processor = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        self._is_loaded = False
        self.logger.info("Resources cleaned up")
