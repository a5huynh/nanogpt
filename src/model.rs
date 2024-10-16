use candle_core::{DType, Result, Tensor};
use candle_nn::{Embedding, Module};

pub struct BigramModel {
    token_embedding_table: Embedding,
}

impl BigramModel {
    pub fn new(device: &candle_core::Device, vocab_size: usize) -> Self {
        let embedding = Embedding::new(
            Tensor::zeros((vocab_size, vocab_size), DType::I64, device).unwrap(),
            vocab_size,
        );

        Self {
            token_embedding_table: embedding,
        }
    }
}

impl Module for BigramModel {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let logits = self.token_embedding_table.forward(xs);
        logits
    }
}
