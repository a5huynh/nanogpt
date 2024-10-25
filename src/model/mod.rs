use candle_core::{DType, Device, Error, IndexOp, Result, Shape, Tensor};
use candle_nn::{
    embedding, linear_no_bias, loss, ops::softmax_last_dim, seq, AdamW, Embedding, LayerNorm, Linear, Module, Optimizer, Sequential, VarBuilder, VarMap
};

use rand::prelude::Distribution;
use rand_pcg::Lcg64Xsh32;

use crate::{
    dataset::Dataset, utils::print_probs, vocab::Vocab, BATCH_SIZE, BLOCK_SIZE, EPS, LEARNING_RATE,
    NUM_EMBED, NUM_HEADS,
};

pub mod block;
pub mod head;
pub mod norm;

pub fn estimate_loss(logits: &Tensor, targets: &Tensor) -> Result<Tensor> {
    log::debug!("reshaping logits");
    let (batch_size, time_size, channels_size) = logits.shape().dims3()?;
    let logits = logits.reshape(Shape::from((batch_size * time_size, channels_size)))?;
    let targets = targets.reshape(Shape::from((batch_size * time_size,)))?;

    log::debug!("applying cross entropy");
    loss::cross_entropy(&logits, &targets)
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
    pub fn new(
        num_layers: usize,
        dropout: f32,
        device: &candle_core::Device,
        rng: &Lcg64Xsh32,
        vocab_size: usize,
    ) -> Self {
        // Similar to nn.Parameter in pytorch.
        let var_map = VarMap::new();
        let var_builder = VarBuilder::from_varmap(&var_map, DType::F32, device);

        // Each token directly reads off the logits for the next token from a lookup table.
        let token_embedding_table = embedding(
            vocab_size,
            NUM_EMBED,
            var_builder.push_prefix("token_embedding"),
        )
        .expect("Unable to create token_embedding_table");

        let position_embedding_table = embedding(
            BLOCK_SIZE,
            NUM_EMBED,
            var_builder.push_prefix("position_embedding"),
        )
        .expect("Unable to create position_embedding_table");

        let mut blocks = seq();
        for block_idx in 0..num_layers {
            blocks = blocks.add(block::Block::new(
                NUM_EMBED,
                NUM_HEADS,
                dropout,
                device,
                var_builder.push_prefix(format!("block_{}", block_idx)),
            ))
        }

        blocks = blocks.add(LayerNorm::new_no_bias(Tensor::ones(NUM_EMBED, DType::F32, device).unwrap(), EPS));

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

    pub fn train(&self, dataset: &mut Dataset, num_steps: usize) -> Result<()> {
        // setup some timers for see how efficient we are.
        let train_start = std::time::Instant::now();
        let mut timer = std::time::Instant::now();

        let mut optimizer = AdamW::new_lr(self.parameters.all_vars(), LEARNING_RATE)?;
        for step in 0..num_steps {
            // sample a batch of data
            let (input, target) = dataset.get_batch(BATCH_SIZE, BLOCK_SIZE);
            // evaluate the loss
            let logits = self.forward(&input)?;
            let loss = estimate_loss(&logits, &target)?;
            // Combines loss.backward() & optimizer.step() from pytorch.
            optimizer.backward_step(&loss)?;
            if step % 100 == 0 || step == num_steps - 1 {
                let train_loss = loss.to_scalar::<f32>()?;
                let (val_input, val_target) = dataset.get_validation_batch(BATCH_SIZE, BLOCK_SIZE);
                let val_logits = self.forward(&val_input)?;
                let val_loss = estimate_loss(&val_logits, &val_target)?.to_scalar::<f32>()?;

                let tps = (timer.elapsed().as_secs_f32() / 100.0) * 1000.0;
                timer = std::time::Instant::now();
                log::info!(
                    "step {step} - train loss = {train_loss:0.3}, val loss = {val_loss:0.3}, per step: {tps:0.3}ms",
                );
            }
        }

        log::info!(
            "total training time: {:0.3}s",
            train_start.elapsed().as_secs_f32()
        );
        Ok(())
    }

    /// ctxt: The current context of characters as a (B, T) array of indices.
    /// Extends ctxt by <max_new_tokens>
    pub fn generate(
        &mut self,
        vocab: &Vocab,
        ctxt: &Tensor,
        max_new_tokens: usize,
    ) -> Result<Tensor> {
        log::info!("Generating {max_new_tokens} token(s)");
        log::info!("Starting shape: {:?}", ctxt.shape());

        let mut ctxt = ctxt.clone();
        for _ in 0..max_new_tokens {
            // crop the ctxt to the last BLOCK_SIZE tokens
            let (_, block) = ctxt.shape().dims2()?;
            let cropped = if block > BLOCK_SIZE {
                ctxt.i((.., block - BLOCK_SIZE..))?
            } else {
                ctxt.clone()
            };

            let logits = self.forward(&cropped)?;
            // focus only on the last time step
            let (_, last, _) = logits.shape().dims3()?;
            // Becomes [B, C]
            let logits = logits.i((.., last - 1, ..))?;
            // Apply softmax to get probabilities
            // This gives us a tensor of [B, C] with the probabilities for each character
            // for each batch. e.g., a single batch will give us [1, C]
            let probs = softmax_last_dim(&logits)?;

            // Sample from the distribution for each batch
            // Build a tensor where each row contains something sampled from the probability distribution
            // given by the softmax, This essentially simulates torch.multinomal(probs, num_samples=1)
            // which is not implemented in candle.
            let mut samples: Vec<u32> = Vec::new();
            let num_batches = probs.dim(0)?;
            for idx in 0..num_batches {
                // Each element in this vec is the probability of that particular character
                // in the vocab occuring next.
                let batch_probs = probs.i((idx, ..))?.to_vec1::<f32>()?;
                if max_new_tokens == 1 {
                    print_probs(vocab, &batch_probs);
                }
                // We put this into a weighted index & sample for the next token.
                let dist =
                    rand::distributions::WeightedIndex::new(&batch_probs).map_err(Error::wrap)?;
                let next_token = dist.sample(&mut self.rng) as u32;
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
        // shape = [B, T, C]
        let tok_embed = self.token_embedding_table.forward(input)?;
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
