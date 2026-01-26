# Confidence Metrics Documentation

## 1️⃣ Average Entropy (avg_entropy)

### What it is:
Entropy measures the uncertainty of the model when predicting each token. High entropy = model is "unsure," low entropy = model is "confident" in its token choices.

### How it's calculated:

For each predicted token, the model outputs a probability distribution over the vocabulary (p(token)).

Entropy per token is computed as:

$$H = -\sum_i p_i \log(p_i)$$

where $p_i$ is the probability of the i-th token.

Average entropy is just the mean of these token-wise entropies across the generated sequence.

### What it represents:

A measure of token-level uncertainty.

Captures how "sure" the model is about each token it generates.

### High vs Low values:

- **Low entropy**: Model is confident → good.
- **High entropy**: Model is uncertain → bad.

**Threshold**: `< ENTROPY_THRESHOLD` → considered confident.

---

## 2️⃣ Sequence Log Probability Normalized (seq_logprob_normalized)

### What it is:
This is the overall likelihood of the generated sequence, normalized by the number of tokens. Essentially, it tells you how probable the whole sentence is according to the model.

### How it's calculated:

The model provides a log probability for each token: log p(token_i | previous_tokens).

Sum all log probabilities for the generated sequence:

$$\text{SeqLogProb} = \sum_i \log p(token_i)$$

Normalize by sequence length:

$$\text{SeqLogProbNormalized} = \frac{\text{SeqLogProb}}{\text{\#tokens}}$$

### What it represents:

Measures the overall confidence in the sequence.

High normalized log probability = the model thinks the whole output is likely.

Low normalized log probability = model thinks the output is unusual or improbable.

### High vs Low values:

- **High value**: Model is confident → good.
- **Low value**: Model is less confident → bad.

**Threshold**: `> SEQ_LOGPROB_THRESHOLD` → considered confident.

---

## 3️⃣ Top-2 Beam Margin (beam_top2_margin)

### What it is:
This comes from beam search, where the model keeps the top N sequences at each step. The Top-2 margin is the difference in score between the best beam and the second-best beam.

### How it's calculated:

$$\text{Top2Margin} = \text{score(top beam)} - \text{score(second beam)}$$

### What it represents:

Shows how clearly the model prefers its top prediction over alternatives.

A big margin → top prediction is clearly better.

A small margin → top predictions are ambiguous.

### High vs Low values:

- **High margin**: Model strongly favors one sequence → confident.
- **Low margin**: Model is torn between options → not confident.

**Threshold**: `> BEAM_TOP2_MARGIN_THRESHOLD` → considered confident.

---

## 4️⃣ Top Beam Average Log Probability (beam_top_beam_avg_logprob)

### What it is:
Average log probability of tokens in the best beam sequence.

### How it's calculated:

Take the top beam sequence (from beam search).

Sum all token log probabilities.

Divide by the number of tokens → average log probability per token.

### What it represents:

Token-level confidence for the most likely sequence.

Similar to seq_logprob_normalized, but specifically for beam search's top sequence.

### High vs Low values:

- **High**: Each token is individually likely → confident prediction.
- **Low**: Some tokens are improbable → model is less confident.

**Threshold**: `> BEAM_TOP_BEAM_LOGPROB_THRESHOLD` → considered confident.

---

## 5️⃣ Beam Score Spread (beam_score_spread)

### What it is:
Measures the standard deviation of log probabilities across all beams. Captures consistency of beam scores.

### How it's calculated:

$$\text{BeamScoreSpread} = \text{std\_dev}([\text{beam1 score}, \text{beam2 score}, \ldots])$$

### What it represents:

High spread → big differences among beam sequences → top sequence is clearly better.

Low spread → beams are similar → the model is uncertain because it cannot differentiate well.

### High vs Low values:

- **Low spread**: Beams are similar → more ambiguous → can indicate low confidence.
- **High spread**: Clear winner → high confidence.

**Threshold**: `< BEAM_SCORE_SPREAD_THRESHOLD` → considered confident (here the logic is flipped: too much spread can also indicate inconsistent scoring).

---

## ✅ Summary of All Metrics

| Metric | High Value → | Low Value → | Confidence Threshold |
|--------|-------------|-------------|---------------------|
| Average Entropy | Uncertain / unsure | Confident | < ENTROPY_THRESHOLD |
| Sequence LogProb Norm | Confident | Unlikely | > SEQ_LOGPROB_THRESHOLD |
| Top-2 Beam Margin | Confident | Ambiguous | > BEAM_TOP2_MARGIN_THRESHOLD |
| Top Beam Avg LogProb | Confident | Unlikely | > BEAM_TOP_BEAM_LOGPROB_THRESHOLD |
| Beam Score Spread | Stable / clear winner | Ambiguous / inconsistent | < BEAM_SCORE_SPREAD_THRESHOLD |
