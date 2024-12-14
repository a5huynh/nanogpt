use candle_core::{DType, Device, Error, IndexOp, Result, Tensor};
use candle_nn::{
    embedding, linear_no_bias, ops::softmax_last_dim, seq, AdamW, Embedding, LayerNorm, Linear,
    Module, Optimizer, Sequential, VarBuilder, VarMap,
};

use rand::prelude::Distribution;
use rand_pcg::Lcg64Xsh32;
use serde::Deserialize;
use tokio::sync::mpsc::Sender;

use super::utils;
use crate::{dataset::Dataset, stream::TokenSample, EPS, LEARNING_RATE};

pub mod block;
pub mod head;
pub mod norm;

#[derive(Clone, Deserialize)]
pub struct Hyperparams {
    pub batch_size: usize,
    pub block_size: usize,
    pub num_embed: usize,
    pub num_heads: usize,
    pub num_layers: usize,
}

impl std::fmt::Display for Hyperparams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Hyperparams: batch_size={}, block_size={}, num_embed={}, num_heads={}, num_layers = {}",
            self.batch_size, self.block_size, self.num_embed, self.num_heads, self.num_layers
        )
    }
}

impl Hyperparams {
    pub fn head_size(&self) -> usize {
        self.num_embed / self.num_heads
    }
}

impl Default for Hyperparams {
    fn default() -> Self {
        Hyperparams {
            batch_size: 8,
            block_size: 256,
            // Gives us a head size of 64 (num_embed / num_heads)
            num_embed: 384,
            num_heads: 6,
            num_layers: 6,
        }
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
    hyperparams: Hyperparams,
}

impl BigramModel {
    pub fn new(
        hyperparams: &Hyperparams,
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
            hyperparams.num_embed,
            var_builder.push_prefix("token_embedding"),
        )
        .expect("Unable to create token_embedding_table");

        let position_embedding_table = embedding(
            hyperparams.block_size,
            hyperparams.num_embed,
            var_builder.push_prefix("position_embedding"),
        )
        .expect("Unable to create position_embedding_table");

        let mut blocks = seq();
        for block_idx in 0..hyperparams.num_layers {
            blocks = blocks.add(block::Block::new(
                hyperparams,
                dropout,
                device,
                var_builder.push_prefix(format!("block_{}", block_idx)),
            ))
        }

        blocks = blocks.add(LayerNorm::new_no_bias(
            Tensor::ones(hyperparams.num_embed, DType::F32, device).unwrap(),
            EPS,
        ));

        let lm_head = linear_no_bias(
            hyperparams.num_embed,
            vocab_size,
            var_builder.push_prefix("lm"),
        )
        .expect("Unable to create lm_head layer");

        Self {
            hyperparams: hyperparams.clone(),
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
            let (input, target) =
                dataset.get_batch(self.hyperparams.batch_size, self.hyperparams.block_size);
            // evaluate the loss
            let logits = self.forward(&input)?;
            let loss = utils::estimate_loss(&logits, &target)?;
            // Combines loss.backward() & optimizer.step() from pytorch.
            optimizer.backward_step(&loss)?;
            // Go at least one step before printing out any stats.
            if step > 0 && (step == 1 || step % 100 == 0 || step == num_steps - 1) {
                let train_loss = loss.to_scalar::<f32>()?;
                let (val_input, val_target) = dataset
                    .get_validation_batch(self.hyperparams.batch_size, self.hyperparams.block_size);
                let val_logits = self.forward(&val_input)?;
                let val_loss =
                    utils::estimate_loss(&val_logits, &val_target)?.to_scalar::<f32>()?;

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
    pub async fn generate(
        &mut self,
        ctxt: &Tensor,
        max_new_tokens: usize,
        stream: Option<Sender<TokenSample>>,
    ) -> Result<(Tensor, Vec<Vec<f32>>)> {
        log::info!("Generating {max_new_tokens} token(s)");
        log::info!("Starting shape: {:?}", ctxt.shape());

        if let Some(ref stream) = stream {
            stream.send(TokenSample::Start).await.unwrap();
        }

        let mut ctxt = ctxt.clone();
        let mut saved_probs = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            // crop the ctxt to the last BLOCK_SIZE tokens
            let (_, block) = ctxt.shape().dims2()?;
            let cropped = if block > self.hyperparams.block_size {
                ctxt.i((.., block - self.hyperparams.block_size..))?
            } else {
                ctxt.clone()
            };

            let logits = self.forward(&cropped)?;
            // focus only on the last time step
            let (_, last_size, _) = logits.shape().dims3()?;
            // Becomes [B, C]
            let logits = logits.i((.., last_size - 1, ..))?;
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
                let dist =
                    rand::distributions::WeightedIndex::new(&batch_probs).map_err(Error::wrap)?;
                let next_token = dist.sample(&mut self.rng) as u32;
                saved_probs.push(batch_probs);
                if let Some(ref stream) = stream {
                    stream
                        .send(TokenSample::NewSample(next_token))
                        .await
                        .unwrap();
                }

                samples.push(next_token);
            }

            // Append the sampled index to the running sequence.
            let samples = Tensor::new(samples, &self.device)?;
            let samples = samples.reshape((num_batches, 1))?;
            ctxt = Tensor::cat(&[&ctxt, &samples], 1)?;
        }

        if let Some(ref stream) = stream {
            stream.send(TokenSample::End).await.unwrap();
        }

        Ok((ctxt, saved_probs))
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
