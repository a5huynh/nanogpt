.PHONY: clippy run

clippy:
	cargo fmt
	cargo clippy

run:
	RUST_LOG=debug cargo run -- train
