"""
Action token prediction for determining when to provide feedback.

This module implements strategies for predicting action tokens (<next>, <feedback>, <correct>)
based on video chunk analysis. Supports both rule-based and model-based prediction.
"""

from typing import Tuple, Optional, Dict
import time
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)


class ActionTokenPredictor:
    """
    Predicts action tokens to determine model behavior.

    Action tokens:
        - <next>: Continue observing, no speech output
        - <feedback>: Generate form correction feedback
        - <correct>: Provide positive reinforcement (optional)

    Supports two strategies:
        1. Rule-based: Simple heuristics for testing infrastructure
        2. Model-based: Trained classifier (for future use)

    Attributes:
        strategy: Prediction strategy ("rule_based" or "model_based")
        config: Configuration dictionary with strategy-specific settings
        chunk_count: Number of chunks processed
        last_feedback_time: Timestamp of last feedback action
    """

    def __init__(
        self,
        strategy: str = "rule_based",
        config: Optional[Dict] = None,
        special_tokens: Optional[Dict[str, str]] = None,
    ):
        """
        Initialize action token predictor.

        Args:
            strategy: "rule_based" or "model_based"
            config: Configuration dict with strategy settings
            special_tokens: Dict mapping action names to token strings

        Raises:
            ValueError: If strategy is not supported
        """
        if strategy not in ["rule_based", "model_based"]:
            raise ValueError(f"Unsupported strategy: {strategy}. "
                           f"Must be 'rule_based' or 'model_based'")

        self.strategy = strategy
        self.config = config or {}

        # Default special tokens
        self.tokens = special_tokens or {
            "next": "<next>",
            "feedback": "<feedback>",
            "correct": "<correct>",
        }

        # State tracking
        self.chunk_count = 0
        self.last_feedback_time = 0
        self.motion_history = []

        # Strategy-specific initialization
        if strategy == "rule_based":
            self._init_rule_based()
        elif strategy == "model_based":
            self._init_model_based()

        logger.info(f"Initialized ActionTokenPredictor: strategy={strategy}")

    def _init_rule_based(self):
        """Initialize rule-based strategy parameters."""
        self.time_interval = self.config.get("time_based_interval", 5)
        self.motion_threshold = self.config.get("motion_threshold", 0.7)
        self.min_feedback_interval = self.config.get("min_feedback_interval", 3.0)

        logger.info(f"Rule-based config: interval={self.time_interval}, "
                   f"motion_threshold={self.motion_threshold}, "
                   f"min_interval={self.min_feedback_interval}s")

    def _init_model_based(self):
        """Initialize model-based strategy (load trained model)."""
        checkpoint_path = self.config.get("checkpoint_path")

        if checkpoint_path is None:
            logger.warning("Model-based strategy selected but no checkpoint provided. "
                          "Falling back to rule-based.")
            self.strategy = "rule_based"
            self._init_rule_based()
            return

        # TODO: Load trained action prediction model
        # self.model = load_action_model(checkpoint_path)
        logger.info(f"Model-based prediction: loaded from {checkpoint_path}")

    def predict(
        self,
        chunk_embeddings: torch.Tensor,
        full_context: torch.Tensor,
        current_time: Optional[float] = None,
    ) -> Tuple[str, float]:
        """
        Predict action token for current video context.

        Args:
            chunk_embeddings: Embeddings for current chunk (N, D)
            full_context: Aggregated temporal context (M, D)
            current_time: Current timestamp (None = use time.time())

        Returns:
            Tuple of (action_token, confidence):
                - action_token: One of the special tokens (<next>, <feedback>, <correct>)
                - confidence: Prediction confidence score [0, 1]
        """
        self.chunk_count += 1
        current_time = current_time or time.time()

        if self.strategy == "rule_based":
            return self._predict_rule_based(
                chunk_embeddings, full_context, current_time
            )
        elif self.strategy == "model_based":
            return self._predict_model_based(
                chunk_embeddings, full_context, current_time
            )

    def _predict_rule_based(
        self,
        chunk_embeddings: torch.Tensor,
        full_context: torch.Tensor,
        current_time: float,
    ) -> Tuple[str, float]:
        """
        Rule-based prediction using simple heuristics.

        Rules:
            1. Time-based: Provide feedback every N chunks
            2. Motion-based: Provide feedback when motion exceeds threshold
            3. Cooldown: Enforce minimum interval between feedback

        Returns:
            (action_token, confidence)
        """
        time_since_feedback = current_time - self.last_feedback_time

        # Rule 1: Enforce minimum cooldown period
        if time_since_feedback < self.min_feedback_interval:
            return self.tokens["next"], 1.0

        # Rule 2: Time-based interval
        if self.chunk_count % self.time_interval == 0:
            logger.debug(f"Time-based feedback trigger: chunk {self.chunk_count}")
            return self.tokens["feedback"], 0.85

        # Rule 3: Motion-based detection
        motion_score = self._compute_motion_score(chunk_embeddings)
        self.motion_history.append(motion_score)

        # Keep only recent motion history
        if len(self.motion_history) > 10:
            self.motion_history.pop(0)

        if motion_score > self.motion_threshold:
            # Check if motion is sustained or sudden spike
            recent_avg = np.mean(self.motion_history[-3:]) if len(self.motion_history) >= 3 else motion_score

            if recent_avg > self.motion_threshold * 0.8:
                logger.debug(f"Motion-based feedback trigger: score={motion_score:.3f}")
                return self.tokens["feedback"], motion_score

        # Default: continue observing
        return self.tokens["next"], 1.0

    def _predict_model_based(
        self,
        chunk_embeddings: torch.Tensor,
        full_context: torch.Tensor,
        current_time: float,
    ) -> Tuple[str, float]:
        """
        Model-based prediction using trained classifier.

        TODO: Implement after training action prediction model.

        Returns:
            (action_token, confidence)
        """
        # Placeholder: fallback to rule-based
        logger.warning("Model-based prediction not yet implemented, using rules")
        return self._predict_rule_based(chunk_embeddings, full_context, current_time)

    def _compute_motion_score(self, embeddings: torch.Tensor) -> float:
        """
        Compute motion score from chunk embeddings.

        Uses embedding variance as a proxy for motion intensity.
        High variance indicates significant visual changes (motion).

        Args:
            embeddings: Chunk embeddings tensor (N, D)

        Returns:
            Motion score in [0, 1] range
        """
        if embeddings.numel() == 0:
            return 0.0

        # Compute variance across tokens
        variance = embeddings.var(dim=0).mean().item()

        # Normalize to [0, 1] range (heuristic scaling)
        # Typical variance ranges from 0.01 to 0.5 for video embeddings
        normalized = min(variance / 0.3, 1.0)

        return normalized

    def update_feedback_time(self, timestamp: Optional[float] = None):
        """
        Update last feedback timestamp.

        Should be called when feedback is actually provided.

        Args:
            timestamp: Feedback time (None = use current time)
        """
        self.last_feedback_time = timestamp or time.time()
        logger.debug(f"Updated last feedback time: {self.last_feedback_time}")

    def reset(self):
        """Reset predictor state."""
        self.chunk_count = 0
        self.last_feedback_time = 0
        self.motion_history.clear()
        logger.info("Predictor state reset")

    @property
    def stats(self) -> dict:
        """Get predictor statistics."""
        return {
            "strategy": self.strategy,
            "chunks_processed": self.chunk_count,
            "last_feedback_time": self.last_feedback_time,
            "motion_history_len": len(self.motion_history),
            "avg_motion": np.mean(self.motion_history) if self.motion_history else 0.0,
        }


class MotionDetector:
    """
    Simple optical flow-based motion detector.

    Can be used as an auxiliary signal for action prediction.
    Currently not used in the main predictor but available for future enhancements.
    """

    def __init__(self):
        """Initialize motion detector."""
        self.prev_frame = None
        logger.info("Initialized MotionDetector")

    def compute_motion(self, frame: np.ndarray) -> float:
        """
        Compute motion score between consecutive frames.

        Args:
            frame: Current frame as numpy array (H, W, C)

        Returns:
            Motion score [0, 1]
        """
        if self.prev_frame is None:
            self.prev_frame = frame
            return 0.0

        # Simple pixel difference
        diff = np.abs(frame.astype(float) - self.prev_frame.astype(float))
        motion_score = diff.mean() / 255.0  # Normalize to [0, 1]

        self.prev_frame = frame
        return motion_score

    def reset(self):
        """Reset detector state."""
        self.prev_frame = None
