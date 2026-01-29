# Training Guide

This document covers expected training behavior, loss values, and how to interpret them.

## Configuration

Default training configuration (`config.toml`):

```toml
[training]
dropout = 0.2              # Probability of dropping neurons during training (regularization)
eps = 1e-5                 # Small constant for numerical stability in layer normalization
learning_rate = 3e-4       # Step size for weight updates (AdamW optimizer)
warmup_steps = 200         # Gradually increase LR from 0 to learning_rate over this many steps
max_grad_norm = 1.0        # Clip gradients to this maximum norm (prevents exploding gradients)
weight_decay = 0.01        # L2 regularization strength (prevents overfitting)
checkpoint_interval = 500  # Save model checkpoint every N steps

[hyperparams]
batch_size = 16            # Number of sequences processed in parallel per step
block_size = 256           # Context window size (max sequence length in tokens)
num_embed = 384            # Embedding dimension (size of token/position vectors)
num_heads = 6              # Number of attention heads in each transformer block
num_layers = 6             # Number of transformer blocks stacked
```

## Training Parameters Explained

### dropout
Controls how many neurons are randomly "turned off" during each training step. This prevents the model from relying too heavily on any single neuron and improves generalization.

| Value | Effect |
|-------|--------|
| 0.0 | No dropout (may overfit) |
| 0.1-0.2 | Light regularization (recommended for most cases) |
| 0.3-0.5 | Heavy regularization (use for small datasets or overfitting) |

### learning_rate
How much to adjust weights based on the gradient. Too high causes instability, too low causes slow training.

| Value | Use Case |
|-------|----------|
| 1e-3 | Aggressive, may be unstable |
| 3e-4 | Standard for AdamW (recommended) |
| 1e-4 | Conservative, slower but stable |
| 1e-5 | Very slow, use for fine-tuning |

### warmup_steps
Gradually increases learning rate from 0 to avoid early training instability when the model weights are random.

- **Rule of thumb**: 5-10% of total training steps
- **Small datasets**: 100-200 steps
- **Large datasets**: 500-2000 steps

### max_grad_norm
Clips gradients when their total norm exceeds this value. Prevents "exploding gradients" that can destabilize training.

| Value | Effect |
|-------|--------|
| 0.5 | Aggressive clipping (very stable, slower learning) |
| 1.0 | Standard (recommended) |
| 5.0 | Light clipping |
| inf | No clipping (risky) |

### weight_decay
L2 regularization that penalizes large weights, helping prevent overfitting.

| Value | Effect |
|-------|--------|
| 0.0 | No regularization |
| 0.01 | Light regularization (recommended) |
| 0.1 | Strong regularization |

## Hyperparameters Explained

### batch_size
Number of independent sequences processed per training step. Larger batches give more stable gradients but use more memory.

| Value | Trade-off |
|-------|-----------|
| 4-8 | Low memory, noisy gradients |
| 16-32 | Balanced (recommended) |
| 64-128 | Stable gradients, high memory |

**Memory scaling**: Doubling batch_size roughly doubles memory usage.

### block_size (Context Window)
Maximum number of tokens the model can "see" when making predictions. Longer context captures more dependencies but uses quadratically more memory (due to attention).

| Value | Context | Memory |
|-------|---------|--------|
| 64 | ~10-15 words | Low |
| 256 | ~50 words | Medium (recommended) |
| 512 | ~100 words | High |
| 1024+ | ~200+ words | Very high |

**Memory scaling**: Doubling block_size roughly quadruples attention memory.

### num_embed (Embedding Dimension)
Size of the vector representing each token. Larger dimensions can capture more nuance but require more parameters.

| Value | Parameters | Capacity |
|-------|------------|----------|
| 128 | Small | Limited |
| 256 | Medium | Moderate |
| 384 | Medium-large | Good (recommended) |
| 512+ | Large | High |

**Constraint**: Must be divisible by `num_heads` (determines head_size = num_embed / num_heads).

### num_heads
Number of parallel attention heads. Each head can learn different patterns (e.g., one for syntax, one for semantics).

| Value | head_size (with num_embed=384) | Notes |
|-------|-------------------------------|-------|
| 4 | 96 | Fewer, larger heads |
| 6 | 64 | Balanced (recommended, GPT-2 style) |
| 8 | 48 | More, smaller heads |
| 12 | 32 | Many small heads |

**Common head_size**: 64 is standard (GPT-2/3 style). Ensure `num_embed % num_heads == 0`.

### num_layers
Number of stacked transformer blocks. Deeper models can learn more complex patterns but are harder to train.

| Value | Capacity | Training Difficulty |
|-------|----------|---------------------|
| 2-4 | Low | Easy |
| 6 | Medium | Moderate (recommended for toy models) |
| 12 | High | Harder |
| 24+ | Very high | Requires careful tuning |

## Model Size Estimation

Approximate parameter count:

```
params ≈ vocab_size × num_embed                    # Token embeddings
       + block_size × num_embed                    # Position embeddings
       + num_layers × (4 × num_embed² + 8 × num_embed²)  # Attention + FFN per layer
       + vocab_size × num_embed                    # Output projection
```

For default config (vocab=65, embed=384, layers=6):
- ~10M parameters

## Scaling Guidelines

### Small dataset (<1MB)
```toml
batch_size = 8
block_size = 128
num_embed = 256
num_heads = 4
num_layers = 4
dropout = 0.2
```

### Medium dataset (1-10MB)
```toml
batch_size = 16
block_size = 256
num_embed = 384
num_heads = 6
num_layers = 6
dropout = 0.2
```

### Large dataset (10MB+)
```toml
batch_size = 32
block_size = 512
num_embed = 512
num_heads = 8
num_layers = 8
dropout = 0.1
```

## Expected Loss Progression

For character-level training on Tiny Shakespeare (~1.1MB, 65 character vocabulary):

| Steps | Train Loss | Val Loss | Notes |
|-------|------------|----------|-------|
| 0 | ~4.2 | ~4.2 | Random (ln(65)) |
| 200 | ~3.0-3.5 | ~3.0-3.5 | End of warmup |
| 500 | ~2.3-2.7 | ~2.4-2.8 | First checkpoint |
| 1000 | ~1.8-2.2 | ~1.9-2.3 | Learning well |
| 2500 | ~1.4-1.7 | ~1.5-1.8 | Good progress |
| 5000 | ~1.2-1.5 | ~1.3-1.6 | Good stopping point |
| 10000 | ~1.1-1.3 | ~1.2-1.5 | Diminishing returns |

### Target Loss Values

- **Good**: Train loss < 1.5, Val loss < 1.7
- **Very good**: Train loss < 1.3, Val loss < 1.5
- **Overfitting warning**: Val loss > Train loss + 0.3

## Understanding Perplexity

Perplexity is the exponential of the cross-entropy loss:

```
Perplexity = e^(loss)
```

### Loss to Perplexity Conversion

| Loss | Perplexity | Interpretation |
|------|------------|----------------|
| 4.17 | 65 | Random guessing among 65 characters |
| 2.5 | 12.2 | Choosing between ~12 likely characters |
| 1.5 | 4.5 | Choosing between ~4-5 likely characters |
| 1.0 | 2.7 | Choosing between ~3 likely characters |

### Intuition

Perplexity represents **how many choices the model is "confused" between** on average when predicting the next token.

- **Perplexity of 1** = Perfect prediction (always 100% confident and correct)
- **Perplexity of 65** = Random guessing among all 65 characters
- **Perplexity of 4** = Model narrows prediction down to ~4 equally likely options

For Tiny Shakespeare, perplexity ~4-5 (loss ~1.4-1.6) produces readable text because English has predictable patterns. For example, after "th", the model confidently predicts "e" or "a".

## Training Commands

```bash
# Train with GPU (default)
just train 5000

# Train on CPU
just gpu=false train 5000

# Resume from checkpoint
just train-resume 5000 ./models/latest.safetensors
```

## Troubleshooting

### Loss stuck at ~2.5+

If training loss plateaus above 2.5, check:

1. **Learning rate**: Try 1e-4 or 1e-3
2. **Warmup steps**: Ensure warmup is completing (default 200 steps)
3. **Gradient clipping**: Verify `max_grad_norm` is set (default 1.0)
4. **Weight decay**: Should be ~0.01

### Loss explodes (NaN or very high)

1. Reduce learning rate
2. Increase warmup steps
3. Check for numerical instability in data

### Overfitting (val loss >> train loss)

1. Increase dropout (try 0.3)
2. Reduce model size (fewer layers/heads)
3. Add more training data
4. Stop training earlier
