"""
TensorRT runtime wrapper for CLIP ViT-Base.

Loads a pre-built TensorRT engine (clip_vit_base_fp16.engine) and provides
a PyTorch-compatible interface so it can be swapped in for CLIPVisionModel
in clip_encoder.py with minimal changes.

Memory-aware: falls back gracefully if the engine file is missing or CUDA
doesn't have enough memory to load the engine.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


# Paths relative to project root
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
_ENGINES_DIR = os.path.join(_PROJECT_ROOT, "models", "tensorrt")

# Try engines in this order:
#   1. FP32 engine (current default) — accurate, 180ms, 206x faster than CPU
#   2. FP16 engine — faster (84ms) but has overflow issue on TRT 10.3/Jetson; disabled
#      until a working FP16 path is found. Build it with build_clip_tensorrt_safe.py
#      and rename to .BROKEN to skip it, or fix the overflow first.
ENGINE_CANDIDATES = [
    os.path.join(_ENGINES_DIR, "clip_vit_base_fp32.engine"),
    os.path.join(_ENGINES_DIR, "clip_vit_base_fp16.engine"),
]
DEFAULT_ENGINE_PATH = ENGINE_CANDIDATES[0]


def find_best_engine() -> Optional[str]:
    """Return path to first available engine in preference order."""
    for path in ENGINE_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

# Minimum free CUDA memory required (MB) to load the engine on GPU.
# FP32 engine: 304MB weights + ~150MB activations + fragmentation headroom = ~900MB.
# If less free, fall back to CPU PyTorch (41s vs 0.18s — big regression, but avoids OOM).
MIN_FREE_CUDA_MB = 900


class _HiddenStatesProxy:
    """
    Mock object mimicking CLIPVisionModelOutput.hidden_states so that
    pipeline code like `out.hidden_states[-2]` works unchanged.

    Since we baked layer=-2 into the ONNX export, we always return the
    same tensor regardless of the index requested.
    """

    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor

    def __getitem__(self, idx):  # noqa: ARG002
        return self._tensor

    def __len__(self):
        # CLIP ViT-Base has 13 hidden states (embeddings + 12 layers)
        return 13


class _ModelOutputProxy:
    """Mimics huggingface transformers model output with .hidden_states."""

    def __init__(self, hidden_states_tensor: torch.Tensor):
        self.hidden_states = _HiddenStatesProxy(hidden_states_tensor)
        self.last_hidden_state = hidden_states_tensor


class _ConfigProxy:
    """Mimics CLIPVisionConfig — exposes only fields the pipeline uses."""
    # CLIP ViT-Base-Patch16 constants
    hidden_size = 768
    image_size = 224
    patch_size = 16
    num_attention_heads = 12
    num_hidden_layers = 12


class TensorRTCLIPVisionModel:
    """
    Drop-in replacement for CLIPVisionModel using a TensorRT engine.

    Matches the interface used by CLIPVisionTower in clip_encoder.py:
        out = vision_tower(pixel_values, output_hidden_states=True)
        features = out.hidden_states[self.select_layer]
    """

    def __init__(self, engine_path: str = DEFAULT_ENGINE_PATH):
        import tensorrt as trt  # Imported here so fallback works if unavailable

        self.engine_path = engine_path
        self._trt = trt
        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime: Optional[trt.Runtime] = None
        self._engine = None
        self._context = None

        # Determine dtype from engine filename
        if "fp32" in os.path.basename(engine_path):
            self._dtype = torch.float32
        else:
            self._dtype = torch.float16
        self._device = torch.device("cuda:0")

        # Config proxy so CLIPVisionTower.config / .hidden_size work
        self.config = _ConfigProxy()
        self.hidden_size = self.config.hidden_size  # 768
        self.num_patches_per_dim = self.config.image_size // self.config.patch_size  # 14
        self.num_patches = self.num_patches_per_dim ** 2  # 196

        self._load_engine()

    def _load_engine(self):
        """Load the TensorRT engine from disk into GPU memory."""
        if not os.path.exists(self.engine_path):
            raise FileNotFoundError(
                f"TensorRT engine not found at {self.engine_path}. "
                f"Run: bash scripts/build_clip_tensorrt.sh"
            )

        free_bytes, _ = torch.cuda.mem_get_info()
        logger.info(
            f"Loading TensorRT CLIP engine: {self.engine_path} "
            f"(CUDA free: {free_bytes / 1e9:.2f}GB)"
        )
        self._runtime = self._trt.Runtime(self._logger)

        with open(self.engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")

        # Creating execution context allocates activation memory — can OOM
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(
                "Failed to create execution context (likely CUDA OOM during "
                "activation allocation). Falling back to CPU CLIP."
            )
        logger.info(
            f"  Engine loaded. Input: pixel_values, Output: hidden_states (layer=-2)"
        )

    @torch.no_grad()
    def __call__(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = True,  # noqa: ARG002
    ) -> _ModelOutputProxy:
        """
        Run CLIP forward pass via TensorRT.

        Args:
            pixel_values: (B, 3, 224, 224) tensor on CUDA, FP16

        Returns:
            Object with .hidden_states indexable like transformers output
        """
        # Ensure correct device/dtype
        if pixel_values.device != self._device:
            pixel_values = pixel_values.to(self._device)
        if pixel_values.dtype != self._dtype:
            pixel_values = pixel_values.to(self._dtype)
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        batch_size = pixel_values.shape[0]

        # Set dynamic input shape
        input_name = "pixel_values"
        output_name = "hidden_states"
        self._context.set_input_shape(input_name, pixel_values.shape)

        # Allocate output tensor — shape is (batch, 197, 768) for CLIP ViT-Base
        # 197 = 1 CLS token + 196 patches
        # Use zeros (not empty) so we can detect if TRT isn't writing at all
        output = torch.zeros(
            (batch_size, 197, self.hidden_size),
            dtype=self._dtype,
            device=self._device,
        )

        # Bind tensors to engine via data pointers
        self._context.set_tensor_address(input_name, pixel_values.data_ptr())
        self._context.set_tensor_address(output_name, output.data_ptr())

        # Execute on current CUDA stream
        stream = torch.cuda.current_stream().cuda_stream
        self._context.execute_async_v3(stream_handle=stream)
        torch.cuda.current_stream().synchronize()

        return _ModelOutputProxy(output)

    # Mimic nn.Module interface
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def eval(self):
        return self

    def requires_grad_(self, mode: bool = True):  # noqa: ARG002
        # TensorRT engines don't track gradients — this is a no-op.
        return self

    def to(self, *args, **kwargs):  # noqa: ARG002
        # TensorRT engine is already on GPU; moving not supported.
        return self

    def parameters(self):
        return iter([])

    def __del__(self):
        # Explicit cleanup order matters for TensorRT
        self._context = None
        self._engine = None
        self._runtime = None


def is_engine_available(engine_path: str = DEFAULT_ENGINE_PATH) -> bool:
    """Return True if the TensorRT engine file exists."""
    return os.path.exists(engine_path)


def has_enough_cuda_memory(min_free_mb: int = MIN_FREE_CUDA_MB) -> bool:
    """Return True if CUDA has enough free memory to load the engine."""
    if not torch.cuda.is_available():
        return False
    free_bytes, _ = torch.cuda.mem_get_info()
    free_mb = free_bytes / (1024 * 1024)
    return free_mb >= min_free_mb


# Module-level cache — once loaded, subsequent calls return the same engine.
# This lets us "preload" the engine early (before the main model) to grab
# contiguous CUDA memory before fragmentation sets in.
_CACHED_TRT_CLIP: Optional["TensorRTCLIPVisionModel"] = None
_PRELOAD_ATTEMPTED: bool = False


def preload_trt_clip(
    engine_path: Optional[str] = None,
    min_free_mb: int = MIN_FREE_CUDA_MB,
) -> Optional["TensorRTCLIPVisionModel"]:
    """
    Try to load the TensorRT engine *now* and cache it for later use.

    Call this before loading the main Qwen model — while CUDA memory is
    still fresh and contiguous allocations are more likely to succeed.
    """
    global _CACHED_TRT_CLIP, _PRELOAD_ATTEMPTED
    _PRELOAD_ATTEMPTED = True
    _CACHED_TRT_CLIP = _try_load_trt_clip_uncached(engine_path, min_free_mb)
    return _CACHED_TRT_CLIP


def try_load_trt_clip(
    engine_path: Optional[str] = None,
    min_free_mb: int = MIN_FREE_CUDA_MB,
) -> Optional["TensorRTCLIPVisionModel"]:
    """
    Return TensorRT CLIP, using cached engine if preloaded.
    Falls back to fresh load attempt if not preloaded.
    """
    if _CACHED_TRT_CLIP is not None:
        logger.info("Using preloaded TensorRT CLIP engine")
        return _CACHED_TRT_CLIP
    if _PRELOAD_ATTEMPTED:
        # Preload was tried and failed — don't retry (memory only gets worse)
        logger.info("TensorRT preload already failed earlier — using PyTorch CLIP")
        return None
    return _try_load_trt_clip_uncached(engine_path, min_free_mb)


def _try_load_trt_clip_uncached(
    engine_path: Optional[str],
    min_free_mb: int,
) -> Optional["TensorRTCLIPVisionModel"]:
    """Actual load logic (no caching)."""
    if engine_path is None:
        engine_path = find_best_engine()
        if engine_path is None:
            logger.info("No TensorRT engine found — using PyTorch CLIP")
            return None

    if not is_engine_available(engine_path):
        logger.info(f"TensorRT engine not found at {engine_path} — using PyTorch CLIP")
        return None

    if not has_enough_cuda_memory(min_free_mb):
        free_bytes, _ = torch.cuda.mem_get_info() if torch.cuda.is_available() else (0, 0)
        logger.warning(
            f"Not enough CUDA memory for TensorRT CLIP "
            f"(need {min_free_mb}MB, have {free_bytes / 1024 / 1024:.0f}MB) "
            f"— using PyTorch CLIP on CPU"
        )
        return None

    try:
        return TensorRTCLIPVisionModel(engine_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load TensorRT CLIP ({e}) — using PyTorch CLIP")
        return None
