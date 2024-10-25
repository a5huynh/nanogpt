use candle_core::{Device, Result, Tensor};
use candle_nn::{linear_no_bias, ops, seq, Activation, LayerNorm, Module, Sequential, VarBuilder};

use crate::{EPS, NUM_EMBED, NUM_HEADS};

use super::head::MultiHeadAttention;

pub const FEED_FORWARD_OUT_SCALE: usize = 4;

/// Transformer block: communication followed by computation.
pub struct Block {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl Block {
    pub fn new(dropout: f32, device: &Device, var_builder: VarBuilder) -> Self {
        let head_size = NUM_EMBED / NUM_HEADS;

        Self {
            attention: MultiHeadAttention::new(
                head_size,
                dropout,
                device,
                var_builder.push_prefix("attention"),
            ),
            feed_forward: FeedForward::new(dropout, var_builder.push_prefix("ffwd")),
            layer_norm1: LayerNorm::new_no_bias(
                Tensor::ones(NUM_EMBED, candle_core::DType::F32, device).unwrap(),
                EPS,
            ),
            layer_norm2: LayerNorm::new_no_bias(
                Tensor::ones(NUM_EMBED, candle_core::DType::F32, device).unwrap(),
                EPS,
            ),
        }
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = (xs + self.attention.forward(&self.layer_norm1.forward(xs)?)?)?;
        xs.clone() + self.feed_forward.forward(&self.layer_norm2.forward(&xs)?)
    }
}

/// Simple multi-layer perceptron
/// Implementation of the position-wise feed-forward network in the transformer paper.
pub struct FeedForward {
    net: Sequential,
}

impl FeedForward {
    pub fn new(dropout: f32, var_builder: VarBuilder) -> Self {
        let net = seq()
            .add(
                linear_no_bias(
                    NUM_EMBED,
                    FEED_FORWARD_OUT_SCALE * NUM_EMBED,
                    var_builder.push_prefix("linear1"),
                )
                .expect("Unable to create linear layer"),
            )
            .add(Activation::Relu)
            .add(
                linear_no_bias(
                    FEED_FORWARD_OUT_SCALE * NUM_EMBED,
                    NUM_EMBED,
                    var_builder.push_prefix("projection"),
                )
                .expect("Unable to create linear layer"),
            )
            .add_fn(move |xs| ops::dropout(xs, dropout));

        Self { net }
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.net.forward(xs)
    }
}
