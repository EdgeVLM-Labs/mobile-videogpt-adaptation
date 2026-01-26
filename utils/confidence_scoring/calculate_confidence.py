#!/usr/bin/env python3
"""
Confidence score utilities

Provides `calculate_confidence_scores` for computing:
- Average entropy across generated tokens
- Length-normalized sequence log probability

These metrics can be used to gauge model confidence in generated sequences.
"""

from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn.functional as F


# Confidence thresholds for determining if model is confident
# Lower entropy = more confident (typical range: 0-10+)
ENTROPY_THRESHOLD = 2.0

# Higher (less negative) log probability = more confident
SEQ_LOGPROB_THRESHOLD = -16.0

# Higher margin between top 2 beams = more confident
BEAM_TOP2_MARGIN_THRESHOLD = 0.001

# Higher (less negative) top beam log prob = more confident
BEAM_TOP_BEAM_LOGPROB_THRESHOLD = -0.2

# Lower score spread might indicate consistency (optional metric)
BEAM_SCORE_SPREAD_THRESHOLD = 1.0


def calculate_confidence_scores(
    scores: Tuple[torch.Tensor, ...],
    output_ids: torch.Tensor,
    input_token_count: int,
) -> Dict[str, float]:
    """Calculate confidence scores from generation scores.

    Args:
        scores: Tuple of score tensors (logits), one per generated token. Each
            tensor is shaped [batch_size, vocab_size].
        output_ids: Generated token IDs. Can be either:
            - Full sequence including input: [batch_size, input_len + gen_len]
            - Only generated tokens: [batch_size, gen_len]
        input_token_count: Offset into output_ids where generated tokens start.
            Set to 0 if output_ids contains only generated tokens.

    Returns:
        dict: Contains 'avg_entropy' and 'seq_logprob_normalized'.
    """
    if not scores or len(scores) == 0:
        return {"avg_entropy": 0.0, "seq_logprob_normalized": 0.0}

    entropies: List[float] = []
    log_probs: List[float] = []

    # Get the actual sequence length to avoid out of bounds errors
    seq_len = output_ids.shape[1]
    num_generated_tokens = seq_len - input_token_count

    # Process each generated token's logits
    # Note: len(scores) might be 1 more than num_generated_tokens if EOS triggered stop
    num_tokens_to_process = min(len(scores), num_generated_tokens)
    
    for idx in range(num_tokens_to_process):
        score = scores[idx]
        # Check if we're within bounds
        token_position = input_token_count + idx
        if token_position >= seq_len:
            # Stop if we've gone beyond the actual generated sequence
            break

        # score: [batch_size, vocab_size] - use first beam (best sequence)
        probs = F.softmax(score[0], dim=-1)  # [vocab_size]
        log_prob_dist = F.log_softmax(score[0], dim=-1)  # [vocab_size]

        # Entropy: -sum(p * log(p))
        entropy = -(probs * log_prob_dist).sum().item()
        entropies.append(entropy)

        # Log-prob of the actually selected token
        selected_token_id = output_ids[0, token_position].item()
        token_log_prob = log_prob_dist[selected_token_id].item()
        log_probs.append(token_log_prob)

    avg_entropy = float(np.mean(entropies)) if entropies else 0.0
    seq_logprob = float(sum(log_probs))
    seq_logprob_normalized = seq_logprob / len(log_probs) if log_probs else 0.0

    return {
        "avg_entropy": avg_entropy,
        "seq_logprob_normalized": seq_logprob_normalized,
    }


def calculate_beam_scores(
    sequences_scores: torch.Tensor,
    beam_indices: torch.Tensor = None,
) -> Dict[str, float]:
    """Calculate beam search confidence scores.

    Args:
        sequences_scores: Beam scores for each sequence [num_beams]
        beam_indices: Optional beam indices (not used currently)

    Returns:
        dict: Contains 'top2_margin', 'top_beam_avg_logprob', and 'score_spread'
    """
    if sequences_scores is None or len(sequences_scores) == 0:
        return {
            "top2_margin": 0.0,
            "top_beam_avg_logprob": 0.0,
            "score_spread": 0.0,
        }

    # Convert to numpy for easier manipulation
    scores = sequences_scores.cpu().numpy() if torch.is_tensor(sequences_scores) else np.array(sequences_scores)
    
    # Sort scores in descending order
    sorted_scores = np.sort(scores)[::-1]
    
    # 1. Top-2 Score Margin: difference between best and 2nd best
    if len(sorted_scores) >= 2:
        top2_margin = float(sorted_scores[0] - sorted_scores[1])
    else:
        top2_margin = 0.0
    
    # 2. Average Token Log-Probability of Top Beam (the best score)
    top_beam_avg_logprob = float(sorted_scores[0]) if len(sorted_scores) > 0 else 0.0
    
    # 3. Score Spread: std deviation across all beams
    score_spread = float(np.std(scores)) if len(scores) > 1 else 0.0
    
    return {
        "top2_margin": top2_margin,
        "top_beam_avg_logprob": top_beam_avg_logprob,
        "score_spread": score_spread,
    }


def is_confident(
    scores: Tuple[torch.Tensor, ...],
    output_ids: torch.Tensor,
    input_token_count: int,
    sequences_scores: torch.Tensor = None,
    beam_indices: torch.Tensor = None,
) -> Tuple[bool, Dict[str, float]]:
    """Determine if the model is confident in its generation.

    Combines both greedy and beam search confidence metrics using a scoring system.
    The model is considered confident if it meets at least 2 out of 4 criteria:
    - Low average entropy (< ENTROPY_THRESHOLD)
    - High sequence log probability (> SEQ_LOGPROB_THRESHOLD)
    - High top-2 beam margin (> BEAM_TOP2_MARGIN_THRESHOLD)
    - Low beam score spread (< BEAM_SCORE_SPREAD_THRESHOLD)

    Args:
        scores: Tuple of score tensors (logits), one per generated token.
        output_ids: Generated token IDs.
        input_token_count: Offset into output_ids where generated tokens start.
        sequences_scores: Beam scores for each sequence [num_beams]
        beam_indices: Optional beam indices (not used currently)

    Returns:
        tuple: (is_confident: bool, metrics: dict) where metrics contains:
            - avg_entropy: float
            - seq_logprob_normalized: float
            - beam_top2_margin: float
            - beam_top_beam_avg_logprob: float
            - beam_score_spread: float
    """
    # Calculate base confidence scores
    confidence_scores = calculate_confidence_scores(scores, output_ids, input_token_count)
    
    # Calculate beam scores if available
    beam_scores = calculate_beam_scores(sequences_scores, beam_indices)
    
    # Combine all metrics into a single dictionary
    all_metrics = {
        "avg_entropy": confidence_scores["avg_entropy"],
        "seq_logprob_normalized": confidence_scores["seq_logprob_normalized"],
        "beam_top2_margin": beam_scores["top2_margin"],
        "beam_top_beam_avg_logprob": beam_scores["top_beam_avg_logprob"],
        "beam_score_spread": beam_scores["score_spread"],
    }
    
    # Use scoring system instead of hard-AND gating
    score = 0
    
    if confidence_scores["avg_entropy"] < ENTROPY_THRESHOLD:
        score += 1
    
    if confidence_scores["seq_logprob_normalized"] > SEQ_LOGPROB_THRESHOLD:
        score += 1
    
    if beam_scores["top2_margin"] > BEAM_TOP2_MARGIN_THRESHOLD:
        score += 1
    
    if beam_scores["top_beam_avg_logprob"] > BEAM_TOP_BEAM_LOGPROB_THRESHOLD:
        score += 1
    
    if beam_scores["score_spread"] < BEAM_SCORE_SPREAD_THRESHOLD:
        score += 1
    
    # Model is confident if at least 3 out of 5 criteria are met
    confident = score >= 3
    
    return confident, all_metrics