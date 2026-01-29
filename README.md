# nanogpt

A Rust workspace containing toy implementations of ML models using [candle](https://github.com/huggingface/candle) for tensor operations.

## Crates

### nanogpt

GPT-style transformer language model implementation featuring:

- Token and position embeddings
- Transformer blocks with multi-head self-attention
- Language model head for text generation
- Configurable hyperparameters via `config.toml`

### nanotok

BPE tokenizer implementations:

- **BasicTokenizer** - Basic BPE tokenizer
- **RegexTokenizer** - Regex-based tokenizer (GPT-2/4 style patterns)

Includes a CLI tool (`nanotok-cli`) for training and testing tokenizers.

## Requirements

- Rust (stable)
- [just](https://github.com/casey/just) command runner

## Quick Start

```bash
# Build all crates
just build

# Run training (uses config.toml for hyperparameters)
just train

# Format and lint
just clippy
```

## Configuration

Edit `config.toml` to adjust training parameters:

```toml
[training]
dropout = 0.2
learning_rate = 3e-4

[hyperparams]
batch_size = 8
block_size = 256
num_embed = 384
num_heads = 4
num_layers = 4
```

## Hardware Acceleration

Candle automatically uses hardware acceleration based on platform:

- **macOS**: Metal backend
- **Windows/Linux**: CUDA backend
- **Fallback**: CPU

## Data

Training data goes in `./data/llm/`. The default dataset is `input.txt` (Tiny Shakespeare).

## License

MIT
