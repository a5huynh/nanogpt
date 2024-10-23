.PHONY: clippy run

clippy:
	cargo fmt
	cargo clippy

train:
	RUST_LOG=info cargo run -- train

train-debug:
	RUST_LOG=debug cargo run -- train