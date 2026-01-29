alias b := build
alias c := clippy
alias check := clippy

# build
build:
    cargo build

clippy:
    cargo fmt
    cargo clippy

# train the model (default: shakespeare, 5000 steps)
train dataset="./data/llm/input.txt" steps="5000":
    RUST_LOG=info cargo run --release --bin nanogpt -- train --dataset-path {{dataset}} --num-steps {{steps}}

# generate text from trained model
generate tokens="256":
    RUST_LOG=info cargo run --release --bin nanogpt -- generate --num-tokens {{tokens}}

# download mnist training/testing images
setup-mnist:
    mkdir -p data/images/mnist
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz"
    curl -L --remote-name --output-dir data/images/mnist "https://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
    gunzip -k data/images/mnist/*.gz
