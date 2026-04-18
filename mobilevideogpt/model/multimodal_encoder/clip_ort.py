"""
ONNX Runtime (CPU) wrapper for CLIP ViT-Base.

Runs the same ONNX file that TensorRT uses, but on CPU via ONNX Runtime.
Key advantage over PyTorch CPU: ORT has optimized FP32 CPU kernels with
graph fusion, AVX/NEON vectorization, and multi-threading — typically
3-5x faster than PyTorch CPU for transformer inference.

Uses no CUDA memory at all, so it's safe alongside the main GPU model.
"""
from __future__ import annotations

import os
import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# Same ONNX file as TensorRT pipeline uses
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)
DEFAULT_ONNX_PATH = os.path.join(
    _PROJECT_ROOT, "models", "tensorrt", "clip_vit_base.onnx"
)


class _HiddenStatesProxy:
    """Mock so `out.hidden_states[-2]` works (ONNX already returns layer -2)."""
    def __init__(self, tensor: torch.Tensor):
        self._tensor = tensor
    def __getitem__(self, idx):  # noqa: ARG002
        return self._tensor
    def __len__(self):
        return 13


class _ModelOutputProxy:
    def __init__(self, tensor: torch.Tensor):
        self.hidden_states = _HiddenStatesProxy(tensor)
        self.last_hidden_state = tensor


class _ConfigProxy:
    """CLIP ViT-Base-Patch16 constants."""
    hidden_size = 768
    image_size = 224
    patch_size = 16
    num_attention_heads = 12
    num_hidden_layers = 12


class ONNXRuntimeCLIPVisionModel:
    """
    CLIP vision encoder running via ONNX Runtime on CPU.
    Drop-in replacement for CLIPVisionModel — same interface as the
    TensorRT wrapper, so clip_encoder.py can use either transparently.
    """

    def __init__(self, onnx_path: str = DEFAULT_ONNX_PATH, num_threads: Optional[int] = None):
        import onnxruntime as ort

        self.onnx_path = onnx_path
        self._device = torch.device("cpu")
        self._dtype = torch.float32  # ONNX was exported in FP32

        # Config proxy (for CLIPVisionTower.config.hidden_size etc.)
        self.config = _ConfigProxy()
        self.hidden_size = self.config.hidden_size
        self.num_patches = (self.config.image_size // self.config.patch_size) ** 2

        # Session options for best CPU performance on Jetson's ARM cores
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Default to all 6 cores on Jetson Orin Nano
        if num_threads is None:
            num_threads = os.cpu_count() or 6
        opts.intra_op_num_threads = num_threads
        opts.inter_op_num_threads = 1

        logger.info(
            f"Loading ONNX Runtime CLIP: {onnx_path} (threads={num_threads})"
        )
        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        logger.info(
            f"  ORT session ready. Input: {self._input_name}, Output: {self._output_name}"
        )

    @torch.no_grad()
    def __call__(
        self,
        pixel_values: torch.Tensor,
        output_hidden_states: bool = True,  # noqa: ARG002
    ) -> _ModelOutputProxy:
        """
        Run CLIP forward via ONNX Runtime.

        Args:
            pixel_values: (B, 3, 224, 224) — any device, any float dtype.
                           We move to CPU + FP32 for the ONNX session.
        Returns:
            ModelOutput-like with .hidden_states[-2]
        """
        # ORT needs CPU FP32 numpy array
        if pixel_values.device.type != "cpu":
            pixel_values = pixel_values.cpu()
        if pixel_values.dtype != torch.float32:
            pixel_values = pixel_values.float()
        if not pixel_values.is_contiguous():
            pixel_values = pixel_values.contiguous()

        np_input = pixel_values.numpy()
        np_output = self._session.run(
            [self._output_name], {self._input_name: np_input}
        )[0]

        output = torch.from_numpy(np_output)
        return _ModelOutputProxy(output)

    # nn.Module compatibility
    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def eval(self):
        return self

    def requires_grad_(self, mode: bool = True):  # noqa: ARG002
        return self

    def to(self, *args, **kwargs):  # noqa: ARG002
        return self

    def parameters(self):
        return iter([])


def try_load_ort_clip(onnx_path: str = DEFAULT_ONNX_PATH) -> Optional[ONNXRuntimeCLIPVisionModel]:
    """
    Load ONNX Runtime CLIP. Returns None if ORT not installed or ONNX missing.
    """
    if not os.path.exists(onnx_path):
        logger.info(f"ONNX file not found at {onnx_path} — using PyTorch CLIP")
        return None
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        logger.info("onnxruntime not installed — using PyTorch CLIP")
        return None
    try:
        return ONNXRuntimeCLIPVisionModel(onnx_path)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Failed to load ONNX Runtime CLIP: {e}")
        return None
