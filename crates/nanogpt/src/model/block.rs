use candle_core::{Device, Result, Tensor};
use candle_nn::{linear_no_bias, ops, Activation, LayerNorm, Linear, Module, VarBuilder};

use crate::Config;

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
    pub fn new(config: &Config, dropout: f32, device: &Device, var_builder: VarBuilder) -> Self {
        Self {
            attention: MultiHeadAttention::new(
                &config.hyperparams,
                dropout,
                device,
                var_builder.push_prefix("attention"),
            ),
            feed_forward: FeedForward::new(
                config.hyperparams.num_embed,
                dropout,
                var_builder.push_prefix("ffwd"),
            ),
            layer_norm1: LayerNorm::new_no_bias(
                Tensor::ones(
                    config.hyperparams.num_embed,
                    candle_core::DType::F32,
                    device,
                )
                .unwrap(),
                config.training.eps,
            ),
            layer_norm2: LayerNorm::new_no_bias(
                Tensor::ones(
                    config.hyperparams.num_embed,
                    candle_core::DType::F32,
                    device,
                )
                .unwrap(),
                config.training.eps,
            ),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.attention.set_training(training);
        self.feed_forward.set_training(training);
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
    linear1: Linear,
    projection: Linear,
    dropout: f32,
    training: bool,
}

impl FeedForward {
    pub fn new(num_embed: usize, dropout: f32, var_builder: VarBuilder) -> Self {
        let linear1 = linear_no_bias(
            num_embed,
            FEED_FORWARD_OUT_SCALE * num_embed,
            var_builder.push_prefix("linear1"),
        )
        .expect("Unable to create linear layer");

        let projection = linear_no_bias(
            FEED_FORWARD_OUT_SCALE * num_embed,
            num_embed,
            var_builder.push_prefix("projection"),
        )
        .expect("Unable to create linear layer");

        Self {
            linear1,
            projection,
            dropout,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let xs = self.linear1.forward(xs)?;
        let xs = Activation::Relu.forward(&xs)?;
        let xs = self.projection.forward(&xs)?;
        let dropout = if self.training { self.dropout } else { 0.0 };
        ops::dropout(&xs, dropout)
    }
}
