"""
Unit tests for streaming inference components.

Run with: python -m pytest tests/test_streaming.py -v
"""

import pytest
import numpy as np
import torch
from collections import deque

import sys
sys.path.append(".")

from streaming.buffer import VideoFrameBuffer
from streaming.context import TemporalContextManager, KVCacheManager
from streaming.predictor import ActionTokenPredictor


class TestVideoFrameBuffer:
    """Test cases for VideoFrameBuffer."""

    def test_initialization(self):
        """Test buffer initialization."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)
        assert buffer.chunk_size == 8
        assert buffer.overlap == 4
        assert buffer.stride == 4
        assert len(buffer) == 0

    def test_invalid_params(self):
        """Test invalid parameter handling."""
        with pytest.raises(ValueError):
            VideoFrameBuffer(chunk_size=8, overlap=8)  # overlap >= chunk_size

        with pytest.raises(ValueError):
            VideoFrameBuffer(chunk_size=8, overlap=-1)  # negative overlap

    def test_add_frame(self):
        """Test adding frames to buffer."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)
        frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        # First frame
        chunk_ready = buffer.add_frame(frame)
        assert not chunk_ready  # Need more frames
        assert len(buffer) == 1

        # Add 7 more frames
        for i in range(7):
            chunk_ready = buffer.add_frame(frame)
            if i < 6:
                assert not chunk_ready

        assert chunk_ready  # 8th frame triggers chunk
        assert len(buffer) == 8

    def test_get_chunk(self):
        """Test chunk extraction."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)

        # Add 8 frames
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        for frame in frames:
            buffer.add_frame(frame)

        # Extract chunk
        result = buffer.get_chunk()
        assert result is not None

        video_chunk, context_frames = result
        assert len(video_chunk) == 8
        assert len(context_frames) <= 16  # May be less if buffer is small

        # Check overlap preserved
        assert len(buffer) == 4  # Should have overlap frames remaining

    def test_sliding_window(self):
        """Test sliding window behavior."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)

        # Add 12 frames (should produce 2 chunks)
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(12)]
        chunks_extracted = 0

        for i, frame in enumerate(frames):
            chunk_ready = buffer.add_frame(frame)

            if chunk_ready and i >= 7:  # First chunk at frame 8
                result = buffer.get_chunk()
                if result:
                    chunks_extracted += 1

        assert chunks_extracted >= 1  # At least one chunk
        assert buffer.chunk_count >= 1

    def test_reset(self):
        """Test buffer reset."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)

        # Add some frames
        for _ in range(5):
            frame = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            buffer.add_frame(frame)

        assert len(buffer) == 5

        # Reset
        buffer.reset()
        assert len(buffer) == 0
        assert buffer.frame_count == 0
        assert buffer.chunk_count == 0


class TestTemporalContextManager:
    """Test cases for TemporalContextManager."""

    def test_initialization(self):
        """Test context manager initialization."""
        manager = TemporalContextManager(max_history=3, aggregation="concatenate")
        assert manager.max_history == 3
        assert manager.aggregation == "concatenate"
        assert len(manager) == 0

    def test_invalid_aggregation(self):
        """Test invalid aggregation method."""
        with pytest.raises(ValueError):
            TemporalContextManager(aggregation="invalid")

    def test_update(self):
        """Test adding chunk embeddings."""
        manager = TemporalContextManager(max_history=3, device="cpu")

        # Add embeddings
        emb1 = torch.randn(100, 896)
        manager.update(emb1)

        assert len(manager) == 1

        # Add more
        emb2 = torch.randn(100, 896)
        manager.update(emb2)
        assert len(manager) == 2

    def test_max_history(self):
        """Test maximum history limit."""
        manager = TemporalContextManager(max_history=2, device="cpu")

        # Add 3 embeddings
        for i in range(3):
            emb = torch.randn(100, 896)
            manager.update(emb)

        # Should only keep last 2
        assert len(manager) == 2

    def test_concatenate_aggregation(self):
        """Test concatenate aggregation."""
        manager = TemporalContextManager(max_history=3, aggregation="concatenate", device="cpu")

        # Add embeddings
        emb1 = torch.randn(100, 896)
        emb2 = torch.randn(100, 896)
        manager.update(emb1)
        manager.update(emb2)

        # Get context
        context = manager.get_context()
        assert context.shape[0] == 200  # 100 + 100
        assert context.shape[1] == 896

    def test_average_aggregation(self):
        """Test average aggregation."""
        manager = TemporalContextManager(max_history=3, aggregation="average", device="cpu")

        # Add same-size embeddings
        emb1 = torch.randn(100, 896)
        emb2 = torch.randn(100, 896)
        manager.update(emb1)
        manager.update(emb2)

        # Get context
        context = manager.get_context()
        assert context.shape[0] == 100  # Averaged across chunks
        assert context.shape[1] == 896

    def test_clear(self):
        """Test clearing context."""
        manager = TemporalContextManager(device="cpu")

        emb = torch.randn(100, 896)
        manager.update(emb)
        assert len(manager) == 1

        manager.clear()
        assert len(manager) == 0


class TestKVCacheManager:
    """Test cases for KVCacheManager."""

    def test_initialization(self):
        """Test cache manager initialization."""
        manager = KVCacheManager(max_length=2048, device="cpu")
        assert manager.max_length == 2048
        assert manager.past_key_values is None
        assert manager.sequence_length == 0

    def test_update(self):
        """Test cache update."""
        manager = KVCacheManager(device="cpu")

        # Create fake KV cache (2 layers)
        batch, heads, seq_len, head_dim = 1, 8, 10, 64
        fake_cache = tuple([
            (
                torch.randn(batch, heads, seq_len, head_dim),
                torch.randn(batch, heads, seq_len, head_dim)
            )
            for _ in range(2)
        ])

        # Update
        manager.update(fake_cache)
        assert manager.sequence_length == seq_len
        assert len(manager.past_key_values) == 2

    def test_concatenate(self):
        """Test cache concatenation."""
        manager = KVCacheManager(device="cpu")

        # First update
        batch, heads, seq_len1, head_dim = 1, 8, 10, 64
        cache1 = tuple([
            (
                torch.randn(batch, heads, seq_len1, head_dim),
                torch.randn(batch, heads, seq_len1, head_dim)
            )
            for _ in range(2)
        ])
        manager.update(cache1)

        # Second update
        seq_len2 = 5
        cache2 = tuple([
            (
                torch.randn(batch, heads, seq_len2, head_dim),
                torch.randn(batch, heads, seq_len2, head_dim)
            )
            for _ in range(2)
        ])
        manager.update(cache2)

        # Check concatenation
        assert manager.sequence_length == seq_len1 + seq_len2
        assert manager.past_key_values[0][0].shape[2] == seq_len1 + seq_len2

    def test_clear(self):
        """Test cache clearing."""
        manager = KVCacheManager(device="cpu")

        # Add cache
        cache = tuple([
            (torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))
            for _ in range(2)
        ])
        manager.update(cache)
        assert manager.sequence_length > 0

        # Clear
        manager.clear()
        assert manager.past_key_values is None
        assert manager.sequence_length == 0


class TestActionTokenPredictor:
    """Test cases for ActionTokenPredictor."""

    def test_initialization(self):
        """Test predictor initialization."""
        predictor = ActionTokenPredictor(strategy="rule_based")
        assert predictor.strategy == "rule_based"
        assert predictor.chunk_count == 0

    def test_invalid_strategy(self):
        """Test invalid strategy."""
        with pytest.raises(ValueError):
            ActionTokenPredictor(strategy="invalid")

    def test_rule_based_prediction(self):
        """Test rule-based prediction."""
        config = {
            "time_based_interval": 5,
            "motion_threshold": 0.7,
            "min_feedback_interval": 3.0,
        }
        predictor = ActionTokenPredictor(strategy="rule_based", config=config)

        # Create fake embeddings
        chunk_emb = torch.randn(100, 896)
        full_context = torch.randn(300, 896)

        # First prediction (should be <next>)
        action, confidence = predictor.predict(chunk_emb, full_context, time.time())
        assert action == "<next>"
        assert 0 <= confidence <= 1

    def test_time_based_trigger(self):
        """Test time-based feedback trigger."""
        import time

        config = {
            "time_based_interval": 2,  # Every 2 chunks
            "motion_threshold": 0.7,
            "min_feedback_interval": 0.1,  # Short interval for testing
        }
        predictor = ActionTokenPredictor(strategy="rule_based", config=config)

        chunk_emb = torch.randn(100, 896)
        full_context = torch.randn(300, 896)

        # Process chunks
        actions = []
        for i in range(5):
            time.sleep(0.15)  # Ensure cooldown passes
            action, conf = predictor.predict(chunk_emb, full_context, time.time())
            actions.append(action)

            if action == "<feedback>":
                predictor.update_feedback_time()

        # Should have at least one feedback
        assert "<feedback>" in actions

    def test_reset(self):
        """Test predictor reset."""
        predictor = ActionTokenPredictor(strategy="rule_based")

        # Process some chunks
        chunk_emb = torch.randn(100, 896)
        full_context = torch.randn(300, 896)
        predictor.predict(chunk_emb, full_context)
        predictor.predict(chunk_emb, full_context)

        assert predictor.chunk_count == 2

        # Reset
        predictor.reset()
        assert predictor.chunk_count == 0
        assert predictor.last_feedback_time == 0


class TestIntegration:
    """Integration tests for streaming components."""

    def test_buffer_context_flow(self):
        """Test data flow from buffer to context manager."""
        buffer = VideoFrameBuffer(chunk_size=8, overlap=4)
        context_manager = TemporalContextManager(device="cpu")

        # Add frames
        frames = [np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(8)]
        for frame in frames:
            buffer.add_frame(frame)

        # Extract chunk
        result = buffer.get_chunk()
        assert result is not None

        # Simulate encoding
        fake_embeddings = torch.randn(100, 896)
        context_manager.update(fake_embeddings)

        assert len(context_manager) == 1
        context = context_manager.get_context()
        assert context.shape[0] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
