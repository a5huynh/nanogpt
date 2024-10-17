use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{loss, ops::{softmax, softmax_last_dim}, Embedding, Module};
use rand::prelude::Distribution;
use rand_pcg::Lcg64Xsh32;

pub struct BigramModel {
    token_embedding_table: Embedding,
    device: Device,
    rng: Lcg64Xsh32,
}

impl BigramModel {
    pub fn new(device: &candle_core::Device, rng: &Lcg64Xsh32, vocab_size: usize) -> Self {
        let embedding = Embedding::new(
            Tensor::zeros((vocab_size, vocab_size), DType::F32, device).unwrap(),
            vocab_size,
        );

        Self {
            token_embedding_table: embedding,
            device: device.clone(),
            rng: rng.clone(),
        }
    }

    pub fn train(&self, inputs: &Tensor, targets: &Tensor) -> Result<(Tensor, f32)> {
        let logits = self.forward(inputs)?;
        // logits.shape() = [4, 8, 65] = [B, T, C]

        let (batch_size, time_size, channel_size) = logits.shape().dims3()?;

        // dbg!(logits.shape()); = [32, 65]
        let logits = logits.reshape(Shape::from((batch_size * time_size, channel_size)))?;
        // dbg!(targets.shape()); = [32]
        let targets = targets.reshape(Shape::from((batch_size * time_size,)))?;

        let loss = loss::cross_entropy(&logits, &targets)?;
        Ok((logits, loss.to_vec0()?))
    }

    /// ctxt: The current context of characters as a (B, T) array of indices.
    /// Extends ctxt by <max_new_tokens>
    pub fn generate(&mut self, ctxt: &Tensor, max_new_tokens: usize) -> Result<Tensor> {
        let mut ctxt = ctxt.clone();
        // get predictions
        for _ in 0..max_new_tokens {
            dbg!(ctxt.shape());
            let logits = self.forward(&ctxt)?;
            // focus only on the last time step
            let (_, last, _) = logits.shape().dims3()?;
            let logits = logits.i((.., last - 1, ..))?; // Becomes [B, C]
            dbg!(logits.shape());

            // apply softmax to get probabilities
            let probs = softmax_last_dim(&logits)?;

            // sample from the distribution for each batch
            // let dist = rand::distributions::WeightedIndex::new(&probs).unwrap();
            // let next_token = dist.sample(&mut self.rng) as f32;
            // append the sampled index to the running sequence.
            // ctxt = Tensor::cat(&[&ctxt, &Tensor::new(&[next_token], &self.device)?], 1)?;
        }

        Ok(ctxt)
    }
}

impl Module for BigramModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.token_embedding_table.forward(input)
    }
}
