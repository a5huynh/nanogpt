use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{
    embedding, linear_no_bias, loss, ops::softmax_last_dim, seq, Activation, Embedding, Linear,
    Module, Sequential, VarBuilder, VarMap,
};
use head::MultiHeadAttention;
use rand::prelude::Distribution;
use rand_pcg::Lcg64Xsh32;

use crate::{BLOCK_SIZE, NUM_EMBED};

pub mod head;

/// Transformer block: communication followed by computation.
pub struct Block {
    attention: MultiHeadAttention,
    feed_forward: FeedForward,
}

impl Block {
    pub fn new(
        num_embeddings: usize,
        num_heads: usize,
        device: &Device,
        var_builder: VarBuilder,
    ) -> Self {
        let head_size = num_embeddings / num_heads;

        Self {
            attention: MultiHeadAttention::new(
                head_size,
                num_heads,
                device,
                var_builder.push_prefix("attention"),
            ),
            feed_forward: FeedForward::new(num_embeddings, var_builder.push_prefix("ffwd")),
        }
    }
}

impl Module for Block {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x = self.attention.forward(xs)?;
        self.feed_forward.forward(&x)
    }
}

// Simple multi-layer perceptron
pub struct FeedForward {
    net: Sequential,
}

impl FeedForward {
    pub fn new(num_embed: usize, var_builder: VarBuilder) -> Self {
        let net = seq()
            .add(
                linear_no_bias(num_embed, num_embed, var_builder.push_prefix("linear1"))
                    .expect("Unable to create linear layer"),
            )
            .add(Activation::Relu);

        Self { net }
    }
}

impl Module for FeedForward {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.net.forward(xs)
    }
}

pub struct BigramModel {
    token_embedding_table: Embedding,
    position_embedding_table: Embedding,
    lm_head: Linear,
    blocks: Sequential,
    device: Device,
    rng: Lcg64Xsh32,
    pub parameters: VarMap,
}

impl BigramModel {
    pub fn new(device: &candle_core::Device, rng: &Lcg64Xsh32, vocab_size: usize) -> Self {
        // Similar to nn.Parameter in pytorch.
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Each token directly reads off the logits for the next token from a lookup table.
        let token_embedding_table =
            embedding(vocab_size, NUM_EMBED, var_builder.push_prefix("token"))
                .expect("Unable to create token_embedding_table");

        let position_embedding_table =
            embedding(BLOCK_SIZE, NUM_EMBED, var_builder.push_prefix("position"))
                .expect("Unable to create position_embedding_table");

        let blocks = seq()
            .add(Block::new(
                NUM_EMBED,
                4,
                device,
                var_builder.push_prefix("block_0"),
            ))
            .add(Block::new(
                NUM_EMBED,
                4,
                device,
                var_builder.push_prefix("block_1"),
            ))
            .add(Block::new(
                NUM_EMBED,
                4,
                device,
                var_builder.push_prefix("block_2"),
            ))
            .add(Block::new(
                NUM_EMBED,
                4,
                device,
                var_builder.push_prefix("block_3"),
            ));

        let lm_head = linear_no_bias(NUM_EMBED, vocab_size, var_builder.push_prefix("lm"))
            .expect("Unable to create lm_head layer");

        Self {
            token_embedding_table,
            position_embedding_table,
            blocks,
            device: device.clone(),
            rng: rng.clone(),
            parameters: var_map,
            lm_head,
        }
    }

    pub fn train(&self, inputs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.forward(inputs)?;
        // dbg!(logits.shape()); // shape = [B, T, vocab_size]

        log::debug!("reshaping logits");
        let (batch_size, time_size, channels_size) = logits.shape().dims3()?;
        let logits = logits.reshape(Shape::from((batch_size * time_size, channels_size)))?;

        // dbg!(targets.shape()); = [32]
        let targets = targets.reshape(Shape::from((batch_size * time_size,)))?;

        log::debug!("applying cross entropy");
        let loss = loss::cross_entropy(&logits, &targets)?;
        Ok((logits, loss))
    }

    /// ctxt: The current context of characters as a (B, T) array of indices.
    /// Extends ctxt by <max_new_tokens>
    pub fn generate(&mut self, ctxt: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        log::info!("Generating {max_new_tokens} token(s)");
        log::info!("Starting shape: {:?}", ctxt.shape());

        let mut ctxt = ctxt.clone();

        // get predictions
        for _ in 0..max_new_tokens {
            // Crop to the last BLOCK_SIZE tokens.
            let (_, block) = ctxt.shape().dims2()?;
            let cropped = if block > BLOCK_SIZE {
                ctxt.i((.., block - BLOCK_SIZE..))?
            } else {
                ctxt.clone()
            };

            let logits = self.forward(&cropped)?;
            // focus only on the last time step
            let (_, last, _) = logits.shape().dims3()?;
            let logits = logits.i((.., last - 1, ..))?; // Becomes [B, C]

            // Apply softmax to get probabilities
            // This gives us a tensor of [B, C] with the probabilities for each character
            // for each batch. e.g., a single batch will give us [1, C]
            let probs = softmax_last_dim(&logits)?;

            // Sample from the distribution for each batch
            // Build a tensor where each row contains something sampled from the probability distribution
            // given by the softmax, This essentially simulates torch.multinomal(probs, num_samples=1)
            // which is not implemented in candle.
            let mut samples: Vec<i64> = Vec::new();
            let num_batches = probs.dim(0)?;
            for idx in 0..num_batches {
                let batch_probs = probs.i((idx, ..))?;
                // Each element in this vec is the probability of that particular character
                // in the vocab occuring next.
                let batch_probs = batch_probs.to_vec1::<f32>()?;
                // We put this into a weighted index & sample for the next token.
                let dist = rand::distributions::WeightedIndex::new(&batch_probs).unwrap();
                let next_token = dist.sample(&mut self.rng) as i64;
                samples.push(next_token);
            }
            // Append the sampled index to the running sequence.
            let samples = Tensor::new(samples, &self.device)?;
            let samples = samples.reshape((num_batches, 1))?;
            ctxt = Tensor::cat(&[&ctxt, &samples], 1)?;
        }

        Ok(ctxt)
    }
}

impl Module for BigramModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (_, time_size) = input.shape().dims2()?;
        // each token directly reads off the logits for the next token from a lookup table.
        log::debug!("encoding embeddings");
        let tok_embed = self.token_embedding_table.forward(input)?; // shape = [B, T, C]

        // encode the positions of each token
        log::debug!("encoding positions");
        let positions = Tensor::arange::<u32>(0, time_size as u32, &self.device)?;
        let pos_embed = self.position_embedding_table.forward(&positions)?; // shape = [T, C]

        // Vector with encoded tokens & positions
        let x = tok_embed.broadcast_add(&pos_embed)?;
        // Appply a single head of self-attention.
        log::debug!("applying transformer blocks");
        let x = self.blocks.forward(&x)?; // shape = [B, T, C];

        log::debug!("applying lm_head");
        self.lm_head.forward(&x)
    }
}
