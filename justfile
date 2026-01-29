dataset := "./data/llm/input.txt"
gpu := "true"

alias b := build
alias c := clippy
alias check := clippy

# build
build:
    cargo build

clippy:
    cargo fmt
    cargo clippy

# train the model
train steps="5000":
    RUST_LOG=info cargo run --release --bin nanogpt -- {{ if gpu == "true" { "--gpu" } else { "" } }} train --dataset-path {{dataset}} --num-steps {{steps}}

# resume training
train-resume steps="5000" checkpoint="./models/latest.safetensors":
    RUST_LOG=info cargo run --release --bin nanogpt -- {{ if gpu == "true" { "--gpu" } else { "" } }} train --dataset-path {{dataset}} --num-steps {{steps}} --checkpoint {{checkpoint}}

# generate text from trained model
generate tokens="256":
    RUST_LOG=info cargo run --release --bin nanogpt -- {{ if gpu == "true" { "--gpu" } else { "" } }} generate --num-tokens {{tokens}}

# train a BPE tokenizer (models: BasicTokenizer, Gpt2, Gpt4)
train-tokenizer vocab_size="512" model="Gpt4":
    mkdir -p models/tokenizers
    RUST_LOG=info cargo run --release --bin nanotok-cli -- --model {{model}} train {{vocab_size}} {{dataset}}

# download mnist training/testing images
setup-mnist:
    mkdir -p data/images/mnist
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    gunzip -k data/images/mnist/*.gz
