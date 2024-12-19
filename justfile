alias b := build
alias c := clippy
alias check := clippy

# build
build:
    cargo build

clippy:
    cargo fmt
    cargo clippy

train:
    RUST_LOG=info cargo run -- train