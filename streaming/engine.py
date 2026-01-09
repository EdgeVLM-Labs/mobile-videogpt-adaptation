"""
Main streaming inference engine for Mobile-VideoGPT.

This module provides the StreamingMobileVideoGPT class that coordinates
all components to enable real-time video processing and feedback generation.
"""

from typing import Optional, Dict, Any, Tuple
import time
import torch
import numpy as np
import logging
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from .buffer import VideoFrameBuffer
from .context import TemporalContextManager, KVCacheManager
from .predictor import ActionTokenPredictor
from .utils import (
    load_config,
    setup_logging,
    preprocess_frame,
    frames_to_tensor,
    add_special_tokens,
    format_feedback,
    PerformanceMonitor,
)

# Import Mobile-VideoGPT utilities
import sys
sys.path.append(".")
from mobilevideogpt.utils import preprocess_input
from mobilevideogpt.mm_utils import tokenizer_image_token
from mobilevideogpt.conversation import conv_templates
from mobilevideogpt.constants import IMAGE_TOKEN_INDEX
from einops import rearrange

logger = logging.getLogger(__name__)


class StreamingMobileVideoGPT:
    """
    Streaming inference wrapper for Mobile-VideoGPT.

    Enables real-time video processing with autonomous action prediction
    for exercise form feedback. Maintains temporal context across chunks
    and generates feedback only when needed.

    Attributes:
        model: Mobile-VideoGPT model
        tokenizer: HuggingFace tokenizer
        config: Configuration dictionary
        frame_buffer: Video frame buffer
        context_manager: Temporal context manager
        kv_cache_manager: KV cache manager
        action_predictor: Action token predictor
        performance_monitor: Performance metrics tracker
    """

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        config_dict: Optional[Dict] = None,
        device: str = "cuda",
    ):
        """
        Initialize streaming inference engine.

        Args:
            model_path: Path to pretrained Mobile-VideoGPT model
            config_path: Path to YAML config file
            config_dict: Config dictionary (overrides config_path)
            device: Device to run on ("cuda" or "cpu")
        """
        # Load configuration
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            # Default configuration
            self.config = self._get_default_config()

        # Setup logging
        log_config = self.config.get("logging", {})
        setup_logging(
            level=log_config.get("level", "INFO"),
            log_file=log_config.get("log_file"),
            console=log_config.get("console", True),
        )

        self.device = device
        self.model_path = model_path

        # Load model and tokenizer
        logger.info(f"Loading model from {model_path}")
        self._load_model()

        # Add special action tokens
        self._add_action_tokens()

        # Initialize components
        self._init_components()

        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        self.start_time = time.time()

        logger.info("StreamingMobileVideoGPT initialized successfully")

    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            "video": {
                "chunk_size": 8,
                "overlap": 4,
                "num_context_images": 16,
            },
            "temporal": {
                "max_history": 3,
                "context_aggregation": "concatenate",
            },
            "action_prediction": {
                "strategy": "rule_based",
                "confidence_threshold": 0.75,
                "rules": {
                    "time_based_interval": 5,
                    "motion_threshold": 0.7,
                    "min_feedback_interval": 3.0,
                },
            },
            "special_tokens": {
                "next": "<next>",
                "feedback": "<feedback>",
                "correct": "<correct>",
            },
            "generation": {
                "max_new_tokens": 256,
                "temperature": 0.7,
                "do_sample": True,
                "num_beams": 1,
            },
        }

    def _load_model(self):
        """Load Mobile-VideoGPT model and tokenizer."""
        model_config = AutoConfig.from_pretrained(self.model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)

        # Load model with FP16 for efficiency
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=model_config,
            torch_dtype=dtype,
        )
        self.model.to(self.device)
        self.model.eval()

        # Load vision towers
        vision_tower = self.model.get_vision_tower()
        vision_tower.load_model(model_config.mm_vision_tower)
        self.video_processor = vision_tower.image_processor

        image_vision_tower = self.model.get_image_vision_tower()
        image_vision_tower.load_model()
        self.image_processor = image_vision_tower.image_processor

        logger.info(f"Model loaded: {self.model.__class__.__name__}")

    def _add_action_tokens(self):
        """Add special action tokens to vocabulary."""
        tokens = self.config.get("special_tokens", {})
        self.token_ids = add_special_tokens(self.tokenizer, self.model, tokens)
        self.action_tokens = {v: k for k, v in tokens.items()}  # Reverse mapping
        logger.info(f"Action tokens: {self.token_ids}")

    def _init_components(self):
        """Initialize streaming components."""
        video_config = self.config.get("video", {})
        temporal_config = self.config.get("temporal", {})
        action_config = self.config.get("action_prediction", {})

        # Frame buffer
        self.frame_buffer = VideoFrameBuffer(
            chunk_size=video_config.get("chunk_size", 8),
            overlap=video_config.get("overlap", 4),
        )

        # Temporal context manager
        self.context_manager = TemporalContextManager(
            max_history=temporal_config.get("max_history", 3),
            aggregation=temporal_config.get("context_aggregation", "concatenate"),
            device=self.device,
        )

        # KV cache manager (currently not used, kept for future optimization)
        self.kv_cache_manager = KVCacheManager(device=self.device)

        # Action predictor
        self.action_predictor = ActionTokenPredictor(
            strategy=action_config.get("strategy", "rule_based"),
            config=action_config.get("rules", {}),
            special_tokens=self.config.get("special_tokens", {}),
        )

        logger.info("Components initialized")

    def process_frame(self, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process a single video frame.

        This is the main entry point for streaming inference. Add frames
        one at a time, and this method will handle chunk extraction,
        encoding, action prediction, and feedback generation.

        Args:
            frame: Video frame as numpy array (H, W, C) in RGB format

        Returns:
            Dictionary with feedback result if action is <feedback>:
                {
                    "action": str,  # Action token
                    "confidence": float,  # Prediction confidence
                    "feedback_text": str,  # Generated text (if feedback)
                    "timestamp": float,  # Time since start
                    "chunk_id": int,  # Chunk identifier
                }
            Returns None if action is <next> (continue observing)
        """
        frame_start = time.time()

        # Preprocess frame
        processed_frame = preprocess_frame(
            frame,
            target_size=(224, 224),
            normalize=False,  # Processor handles normalization
        )

        # Add to buffer
        chunk_ready = self.frame_buffer.add_frame(processed_frame)

        self.performance_monitor.record_frame_time(time.time() - frame_start)

        if not chunk_ready:
            return None  # Need more frames

        # Extract chunk
        chunk_start = time.time()
        result = self.frame_buffer.get_chunk()

        if result is None:
            return None

        video_chunk, context_frames = result

        # Encode video chunk
        chunk_embeddings, context_embeddings = self._encode_video_chunk(
            video_chunk, context_frames
        )

        # Update temporal context
        self.context_manager.update(chunk_embeddings)
        full_context = self.context_manager.get_context()

        self.performance_monitor.record_chunk_time(time.time() - chunk_start)

        # Predict action
        inference_start = time.time()
        current_time = time.time()
        action, confidence = self.action_predictor.predict(
            chunk_embeddings,
            full_context,
            current_time,
        )

        # Check confidence threshold
        threshold = self.config.get("action_prediction", {}).get("confidence_threshold", 0.75)

        if action == self.config["special_tokens"]["feedback"] and confidence >= threshold:
            # Check minimum interval
            time_since_feedback = current_time - self.action_predictor.last_feedback_time
            min_interval = self.config.get("action_prediction", {}).get("rules", {}).get("min_feedback_interval", 3.0)

            if time_since_feedback >= min_interval:
                # Generate feedback
                feedback_text = self._generate_feedback(full_context)

                # Update feedback time
                self.action_predictor.update_feedback_time(current_time)

                self.performance_monitor.record_inference_time(time.time() - inference_start)

                result = {
                    "action": action,
                    "confidence": confidence,
                    "feedback_text": feedback_text,
                    "timestamp": current_time - self.start_time,
                    "chunk_id": self.frame_buffer.chunk_count,
                }

                logger.info(f"Feedback generated: {feedback_text[:50]}...")
                return result

        self.performance_monitor.record_inference_time(time.time() - inference_start)

        # Continue observing
        return None

    def _encode_video_chunk(
        self,
        video_frames: list,
        context_frames: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video chunk using Mobile-VideoGPT encoders.

        Args:
            video_frames: List of 8 frames for VideoMamba
            context_frames: List of context frames for CLIP

        Returns:
            Tuple of (video_embeddings, context_embeddings)
        """
        # Preprocess frames using Mobile-VideoGPT processors
        video_tensors = self.video_processor.preprocess(video_frames)['pixel_values']
        context_tensors = [
            self.image_processor.preprocess(f, return_tensors='pt')['pixel_values'][0]
            for f in context_frames
        ]

        # Pad context frames if needed
        num_context = self.config.get("video", {}).get("num_context_images", 16)
        while len(context_tensors) < num_context:
            context_tensors.append(
                torch.zeros((3, 224, 224))
            )

        # Convert to tensors
        video_tensor = torch.stack(video_tensors).half().to(self.device)
        context_tensor = torch.stack(context_tensors).half().to(self.device)

        # Reshape for model
        batch_size = 1
        video_tensor = rearrange(video_tensor, 't c h w -> (b t) c h w', b=batch_size)
        context_tensor = rearrange(context_tensor, 't c h w -> (b t) c h w', b=batch_size)

        # Encode using model's encoders
        with torch.no_grad():
            # Encode video and context
            video_features, context_features = self.model.encode_videos_by_seletive_frames(
                video_tensor, context_tensor, batch_size=batch_size
            )

            # Project to LLM embedding space
            merged_features = self.model.project(
                video_features, context_features, input_type="video"
            )

        # merged_features shape: (1, total_tokens, embed_dim)
        return merged_features.squeeze(0), context_features

    def _generate_feedback(self, context_embeddings: torch.Tensor) -> str:
        """
        Generate feedback text using the language model.

        Args:
            context_embeddings: Temporal context embeddings

        Returns:
            Generated feedback text
        """
        # Prepare prompt
        prompt = self.config.get("prompts", {}).get(
            "feedback_prefix",
            "Please evaluate the exercise form shown. What mistakes, if any, are present?"
        )

        # Use conversation template
        conv_mode = "qwen2_instruct"
        conv = conv_templates[conv_mode].copy()

        # Add image tokens (already encoded in context_embeddings)
        num_tokens = context_embeddings.shape[0]
        qs = "<image> " * (num_tokens // 49) + "\n" + prompt  # Approximate token count

        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt_formatted = conv.get_prompt()

        # Tokenize
        input_ids = tokenizer_image_token(
            prompt_formatted,
            self.tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        ).unsqueeze(0).to(self.device)

        # Prepare embeddings (combine context with text)
        with torch.no_grad():
            text_embeds = self.model.get_model().embed_tokens(input_ids)

            # Replace image tokens with video context
            # (Simplified: just use context_embeddings directly)
            # In practice, need to handle token replacement properly
            inputs_embeds = context_embeddings.unsqueeze(0)

        # Generate
        gen_config = self.config.get("generation", {})
        with torch.no_grad():
            try:
                output_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    max_new_tokens=gen_config.get("max_new_tokens", 256),
                    temperature=gen_config.get("temperature", 0.7),
                    do_sample=gen_config.get("do_sample", True),
                    num_beams=gen_config.get("num_beams", 1),
                    use_cache=True,
                )

                # Decode
                output_text = self.tokenizer.batch_decode(
                    output_ids,
                    skip_special_tokens=True
                )[0].strip()

                # Clean up (remove prompt if present)
                if prompt in output_text:
                    output_text = output_text.replace(prompt, "").strip()

                return output_text

            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return "Form analysis in progress..."

    def reset(self):
        """Reset all streaming components."""
        self.frame_buffer.reset()
        self.context_manager.clear()
        self.kv_cache_manager.clear()
        self.action_predictor.reset()
        self.start_time = time.time()
        logger.info("Streaming engine reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        return {
            "buffer": self.frame_buffer.stats,
            "context": self.context_manager.stats,
            "kv_cache": self.kv_cache_manager.stats,
            "predictor": self.action_predictor.stats,
            "performance": self.performance_monitor.get_stats(),
        }

    def print_stats(self):
        """Print statistics."""
        stats = self.get_stats()

        print(f"\n{'='*60}")
        print(f"Streaming Inference Statistics")
        print(f"{'='*60}")

        print(f"\n📹 Buffer:")
        for k, v in stats["buffer"].items():
            print(f"  {k}: {v}")

        print(f"\n🧠 Context:")
        for k, v in stats["context"].items():
            print(f"  {k}: {v}")

        print(f"\n🎯 Predictor:")
        for k, v in stats["predictor"].items():
            print(f"  {k}: {v}")

        print(f"\n⚡ Performance:")
        for k, v in stats["performance"].items():
            print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

        print(f"{'='*60}\n")
