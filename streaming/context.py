"""
Temporal context management for maintaining history across chunks.

This module implements managers for storing and retrieving temporal context
(chunk embeddings and KV cache) to enable the model to "remember" recent
video content across multiple inference calls.
"""

from collections import deque
from typing import Optional, List, Tuple, Dict
import torch
import logging

logger = logging.getLogger(__name__)


class TemporalContextManager:
    """
    Manages temporal context by storing recent chunk embeddings.

    This allows the model to maintain awareness of recent video content
    when processing new chunks. Context can be aggregated via concatenation,
    averaging, or attention mechanisms.

    Attributes:
        max_history: Maximum number of chunks to remember
        aggregation: Method to combine chunks ("concatenate", "average", "attention")
        history: Deque storing chunk embedding tensors
    """

    def __init__(
        self,
        max_history: int = 3,
        aggregation: str = "concatenate",
        device: str = "cuda",
    ):
        """
        Initialize temporal context manager.

        Args:
            max_history: Number of previous chunks to store
            aggregation: How to combine chunks - "concatenate", "average", "attention"
            device: Device to store tensors on

        Raises:
            ValueError: If aggregation method is not supported
        """
        if aggregation not in ["concatenate", "average", "attention"]:
            raise ValueError(f"Unsupported aggregation: {aggregation}. "
                           f"Must be 'concatenate', 'average', or 'attention'")

        self.max_history = max_history
        self.aggregation = aggregation
        self.device = device

        # History storage: each element is (chunk_id, embeddings_tensor)
        self.history: deque = deque(maxlen=max_history)
        self.chunk_counter = 0

        logger.info(f"Initialized TemporalContextManager: max_history={max_history}, "
                   f"aggregation={aggregation}")

    def update(self, chunk_embeddings: torch.Tensor, chunk_id: Optional[int] = None):
        """
        Add new chunk embeddings to history.

        Args:
            chunk_embeddings: Embeddings tensor for current chunk (N, D)
                where N is number of tokens, D is embedding dimension
            chunk_id: Optional chunk identifier (auto-increments if None)
        """
        if chunk_id is None:
            chunk_id = self.chunk_counter
            self.chunk_counter += 1

        # Move to device and detach from computation graph
        chunk_embeddings = chunk_embeddings.detach().to(self.device)

        # Store in history
        self.history.append((chunk_id, chunk_embeddings))

        logger.debug(f"Updated context with chunk {chunk_id}: "
                    f"shape={chunk_embeddings.shape}, history_size={len(self.history)}")

    def get_context(
        self,
        include_current: bool = True,
        return_separate: bool = False,
    ) -> torch.Tensor:
        """
        Retrieve aggregated temporal context.

        Args:
            include_current: Whether to include most recent chunk
            return_separate: If True, return list of chunks instead of aggregated

        Returns:
            Aggregated context tensor (M, D) where M depends on aggregation method:
                - concatenate: M = sum of all chunk tokens
                - average: M = tokens from single chunk (averaged)
                - attention: M = tokens from single chunk (attention-weighted)
            Or list of tensors if return_separate=True
        """
        if len(self.history) == 0:
            # Return empty tensor if no history
            logger.warning("No temporal context available")
            return torch.empty(0, 0, device=self.device)

        # Extract embeddings from history
        chunks = [emb for _, emb in self.history]

        if not include_current and len(chunks) > 0:
            chunks = chunks[:-1]

        if len(chunks) == 0:
            return torch.empty(0, 0, device=self.device)

        if return_separate:
            return chunks

        # Aggregate based on method
        if self.aggregation == "concatenate":
            # Concatenate all chunks along token dimension
            context = torch.cat(chunks, dim=0)  # (M, D)
            logger.debug(f"Concatenated context: {len(chunks)} chunks -> {context.shape}")

        elif self.aggregation == "average":
            # Average across chunks (token-wise if same size, else mean pool)
            if all(c.shape[0] == chunks[0].shape[0] for c in chunks):
                # Same token count, average directly
                stacked = torch.stack(chunks, dim=0)  # (num_chunks, N, D)
                context = stacked.mean(dim=0)  # (N, D)
            else:
                # Different token counts, concatenate then mean pool
                concatenated = torch.cat(chunks, dim=0)
                context = concatenated.mean(dim=0, keepdim=True)  # (1, D)
            logger.debug(f"Averaged context: {len(chunks)} chunks -> {context.shape}")

        elif self.aggregation == "attention":
            # Use attention mechanism (simplified: most recent gets higher weight)
            weights = torch.softmax(
                torch.linspace(0, 1, len(chunks), device=self.device), dim=0
            )

            # Weighted average
            if all(c.shape[0] == chunks[0].shape[0] for c in chunks):
                stacked = torch.stack(chunks, dim=0)  # (num_chunks, N, D)
                weighted = stacked * weights.view(-1, 1, 1)
                context = weighted.sum(dim=0)  # (N, D)
            else:
                # Fallback to concatenation for variable token counts
                context = torch.cat(chunks, dim=0)
            logger.debug(f"Attention-weighted context: {len(chunks)} chunks -> {context.shape}")

        return context

    def clear(self):
        """Clear all temporal context."""
        self.history.clear()
        self.chunk_counter = 0
        logger.info("Temporal context cleared")

    def get_latest(self) -> Optional[torch.Tensor]:
        """Get most recent chunk embeddings."""
        if len(self.history) == 0:
            return None
        return self.history[-1][1]

    def __len__(self) -> int:
        """Return number of chunks in history."""
        return len(self.history)

    @property
    def stats(self) -> dict:
        """Get context manager statistics."""
        total_tokens = sum(emb.shape[0] for _, emb in self.history)
        memory_mb = sum(emb.element_size() * emb.nelement() for _, emb in self.history) / (1024 ** 2)

        return {
            "num_chunks": len(self.history),
            "total_tokens": total_tokens,
            "memory_mb": memory_mb,
            "aggregation": self.aggregation,
            "max_history": self.max_history,
        }


class KVCacheManager:
    """
    Manages KV cache for the language model across chunks.

    This enables efficient reuse of computed key-value pairs from previous
    tokens, avoiding redundant computation for the LLM.

    Attributes:
        past_key_values: Cached key-value pairs from previous forward passes
        max_length: Maximum cache length before pruning
    """

    def __init__(
        self,
        max_length: int = 2048,
        device: str = "cuda",
    ):
        """
        Initialize KV cache manager.

        Args:
            max_length: Maximum sequence length to cache
            device: Device to store cache on
        """
        self.max_length = max_length
        self.device = device
        self.past_key_values: Optional[Tuple] = None
        self.sequence_length = 0

        logger.info(f"Initialized KVCacheManager: max_length={max_length}")

    def update(self, new_past_key_values: Tuple):
        """
        Update cache with new key-value pairs.

        Args:
            new_past_key_values: Tuple of (key, value) tensors from model output
                Format: tuple of tuples, one per layer: ((k1, v1), (k2, v2), ...)
                where k shape: (batch, num_heads, seq_len, head_dim)
        """
        if new_past_key_values is None:
            return

        if self.past_key_values is None:
            # First cache
            self.past_key_values = new_past_key_values
            self.sequence_length = new_past_key_values[0][0].shape[2]
        else:
            # Concatenate with existing cache
            concatenated = []
            for (k_old, v_old), (k_new, v_new) in zip(self.past_key_values, new_past_key_values):
                k_concat = torch.cat([k_old, k_new], dim=2)  # Concat along seq_len
                v_concat = torch.cat([v_old, v_new], dim=2)
                concatenated.append((k_concat, v_concat))

            self.past_key_values = tuple(concatenated)
            self.sequence_length = self.past_key_values[0][0].shape[2]

        # Prune if exceeding max length
        if self.sequence_length > self.max_length:
            self._prune_cache()

        logger.debug(f"Updated KV cache: sequence_length={self.sequence_length}")

    def _prune_cache(self):
        """Prune cache to stay within max_length by removing oldest tokens."""
        if self.past_key_values is None:
            return

        # Keep only the most recent max_length tokens
        keep_length = self.max_length // 2  # Prune aggressively
        pruned = []

        for k, v in self.past_key_values:
            k_pruned = k[:, :, -keep_length:, :]
            v_pruned = v[:, :, -keep_length:, :]
            pruned.append((k_pruned, v_pruned))

        self.past_key_values = tuple(pruned)
        self.sequence_length = keep_length

        logger.info(f"Pruned KV cache to {keep_length} tokens")

    def get(self) -> Optional[Tuple]:
        """Retrieve current KV cache."""
        return self.past_key_values

    def clear(self):
        """Clear all cached key-value pairs."""
        self.past_key_values = None
        self.sequence_length = 0
        logger.info("KV cache cleared")

    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        if self.past_key_values is None:
            return {
                "cached": False,
                "sequence_length": 0,
                "num_layers": 0,
                "memory_mb": 0,
            }

        # Calculate memory usage
        memory_bytes = 0
        for k, v in self.past_key_values:
            memory_bytes += k.element_size() * k.nelement()
            memory_bytes += v.element_size() * v.nelement()

        return {
            "cached": True,
            "sequence_length": self.sequence_length,
            "num_layers": len(self.past_key_values),
            "memory_mb": memory_bytes / (1024 ** 2),
            "max_length": self.max_length,
        }
