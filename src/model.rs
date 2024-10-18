use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor};
use candle_nn::{loss, ops::softmax_last_dim, Embedding, Module, VarBuilder, VarMap};
use rand::prelude::Distribution;
use rand_pcg::Lcg64Xsh32;

pub struct BigramModel {
    token_embedding_table: Embedding,
    device: Device,
    rng: Lcg64Xsh32,
    pub parameters: VarMap,
}

impl BigramModel {
    pub fn new(device: &candle_core::Device, rng: &Lcg64Xsh32, vocab_size: usize) -> Self {
        // Similar to nn.Parameter in pytorch.
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);
        let embedding = var_builder.get((vocab_size, vocab_size), "embeddings")
                .unwrap();

        let embedding = Embedding::new(
            embedding,
            vocab_size,
        );

        Self {
            token_embedding_table: embedding,
            device: device.clone(),
            rng: rng.clone(),
            parameters: var_map,
        }
    }

    pub fn train(&self, inputs: &Tensor, targets: &Tensor) -> Result<(Tensor, Tensor)> {
        let logits = self.forward(inputs)?;
        // logits.shape() = [4, 8, 65] = [B, T, C]

        let (batch_size, time_size, channel_size) = logits.shape().dims3()?;

        // dbg!(logits.shape()); = [32, 65]
        let logits = logits.reshape(Shape::from((batch_size * time_size, channel_size)))?;
        // dbg!(targets.shape()); = [32]
        let targets = targets.reshape(Shape::from((batch_size * time_size,)))?;

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
            let logits = self.forward(&ctxt)?;
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
        self.token_embedding_table.forward(input)
    }
}
