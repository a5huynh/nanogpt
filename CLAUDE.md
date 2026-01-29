# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Development Commands

```bash
# Build all crates
just build   # or: just b

# Format and lint
just clippy  # or: just c, just check

# Run nanogpt training
just train   # Runs with RUST_LOG=info

# Run tests
cargo test                           # All tests
cargo test -p nanogpt               # Tests for specific crate
cargo test test_name                # Single test by name

# Download MNIST data (required for nanodiffuse)
just setup-mnist
```

## Architecture

This is a Rust workspace containing toy implementations of ML models using [candle](https://github.com/huggingface/candle) for tensor operations.

### Crates

- **nanogpt** - GPT-style transformer language model
  - `model/mod.rs` - `BigramModel` with token/position embeddings, transformer blocks, and language model head
  - `model/block.rs` - Transformer blocks with multi-head self-attention
  - `dataset.rs` - Training/validation data batching
  - Uses `nanotok` for tokenization

- **nanodiffuse** - Toy diffusion model (WIP)
  - `model.rs` - Basic U-Net with convolutional down/up paths
  - `lib.rs` - MNIST loading, image corruption for diffusion training
  - `util.rs` - Tensor-to-BMP image output for visualization

- **nanotok** - BPE tokenizer implementations
  - `tokenizers/basic.rs` - Basic BPE tokenizer
  - `tokenizers/regex.rs` - Regex-based tokenizer (GPT-2/4 style patterns)
  - `Tokenizer` trait: `train`, `encode`, `decode`, `save`, `load`

### Hardware Acceleration

Candle is configured per-platform in each crate's Cargo.toml:
- macOS: Metal backend
- Windows/Linux: CUDA backend
- CPU fallback via `--gpu` flag (default is CPU)

### Configuration

nanogpt reads `config.toml` for training config and hyperparameters. See `TrainingConfig` and `Hyperparams` structs in `model/mod.rs` for available options.

### Data Paths

- LLM training data: `./data/llm/` (default: `input.txt` - Shakespeare)
  - `input.txt` - Tiny Shakespeare (~1.1 MB)
  - `simple.txt` - Simple text (~361 KB)
  - `taylorswift.txt` - Taylor Swift lyrics (~181 KB)
  - `wikitext2.txt` - Wikipedia articles (~10 MB, not in git)
  - `tinystories.txt` - Simple stories (~2.1 GB, not in git)
- MNIST images: `./data/images/mnist/` (download via `just setup-mnist`)
- Model checkpoints: `./models/`
