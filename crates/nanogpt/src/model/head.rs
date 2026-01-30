use candle_core::{Device, IndexOp, Result, Tensor, D};
use candle_nn::{
    linear_no_bias,
    ops::{self, softmax_last_dim},
    Linear, Module, VarBuilder,
};

use crate::utils;

use super::Hyperparams;

pub struct Head {
    key: Linear,
    query: Linear,
    value: Linear,
    mask: Tensor,
    device: Device,
    dropout: f32,
    training: bool,
    hyperparams: Hyperparams,
}

impl Head {
    pub fn new(
        hparams: &Hyperparams,
        dropout: f32,
        device: &Device,
        var_builder: VarBuilder,
    ) -> Self {
        // what do i contain?
        let key = candle_nn::linear_no_bias(
            hparams.num_embed,
            hparams.head_size(),
            var_builder.push_prefix("key"),
        )
        .expect("Unable to create key layer");
        // what am i looking for?
        let query = candle_nn::linear_no_bias(
            hparams.num_embed,
            hparams.head_size(),
            var_builder.push_prefix("query"),
        )
        .expect("Unable to create key layer");
        let value = candle_nn::linear_no_bias(
            hparams.num_embed,
            hparams.head_size(),
            var_builder.push_prefix("value"),
        )
        .expect("Unable to create key layer");

        let mask = Tensor::tril2(hparams.block_size, candle_core::DType::U32, device)
            .expect("Unable to create mask");

        Self {
            key,
            query,
            value,
            mask,
            device: device.clone(),
            dropout,
            training: true,
            hyperparams: hparams.clone(),
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
    }
}

impl Module for Head {
    fn forward(&self, input: &candle_core::Tensor) -> Result<Tensor> {
        let (_, block_size, _) = input.shape().dims3()?;

        let k = self.key.forward(input)?; // (B, T, head_size)
        let q = self.query.forward(input)?;
        // Calculate the attention scores, aka the "affinities"
        // This is a scaled dot-product attention.  Intuitively, if the key & query
        // values align because what it contains matches what it's looking for, then
        // the weights (or affinity in that particular spot) will be high.
        let scores = q.matmul(&k.transpose(D::Minus2, D::Minus1)?)?;
        // Scales by 1 / sqrt(head_size))
        // The weights are normalized so that variance is keep around 1.
        let (_, _, hs) = k.shape().dims3()?;
        // (B, T, hs) @ (B, hs, T) -> (B, T, T)
        let scores = (scores * (hs as f64).powf(-0.5))?;
        // Ignore future positions by applying causal mask.
        // Slice the pre-computed mask to match current sequence length.
        let mask = if block_size < self.hyperparams.block_size {
            self.mask.i((..block_size, ..block_size))?
        } else {
            self.mask.clone()
        };
        let mask = mask.broadcast_as(scores.shape())?;
        let scores = utils::masked_fill(&scores, &mask, f32::NEG_INFINITY, &self.device)?;
        let scores = softmax_last_dim(&scores)?;
        // Adding dropout to prevent overfitting by randomly shutting off neurons
        let dropout = if self.training { self.dropout } else { 0.0 };
        let scores = ops::dropout(&scores, dropout)?;
        // Weighted aggregation of the values.
        let v = self.value.forward(input)?;
        scores.matmul(&v)
    }
}

/// Multiple heads of attention applied in parallel
pub struct MultiHeadAttention {
    heads: Vec<Head>,
    projection: Linear,
    dropout: f32,
    training: bool,
}

impl MultiHeadAttention {
    pub fn new(
        hparams: &Hyperparams,
        dropout: f32,
        device: &Device,
        var_builder: VarBuilder,
    ) -> Self {
        let heads = (0..hparams.num_heads)
            .map(|idx| {
                Head::new(
                    hparams,
                    dropout,
                    device,
                    var_builder.push_prefix(format!("head_{idx}")),
                )
            })
            .collect::<Vec<_>>();

        let projection = linear_no_bias(
            hparams.num_embed,
            hparams.num_embed,
            var_builder.push_prefix("projection"),
        )
        .expect("Unable to create projection layer");

        Self {
            heads,
            projection,
            dropout,
            training: true,
        }
    }

    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for head in &mut self.heads {
            head.set_training(training);
        }
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let head_outputs: Result<Vec<_>> = self
            .heads
            .iter()
            .map(|head| head.forward(xs))
            .collect();
        // Concat on the channel dimension
        let out = Tensor::cat(&head_outputs?, D::Minus1)?;

        let projected = self.projection.forward(&out)?;
        let dropout = if self.training { self.dropout } else { 0.0 };
        ops::dropout(&projected, dropout)
    }
}
