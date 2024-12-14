use candle_core::{Device, Shape, Tensor};
use candle_nn::loss;
use nanotok::tokenizers::Tokenizer;

pub fn masked_fill(
    input: &Tensor,
    mask: &Tensor,
    value: f32,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    mask.where_cond(
        // when value != 0
        input,
        // when value = 0
        &Tensor::new(&[value], device)?.broadcast_as(mask.shape().dims())?,
    )
}

pub fn print_probs(tokenizer: &dyn Tokenizer, probs: &[f32]) {
    for (idx, prob) in probs.iter().enumerate() {
        println!("'{}': {prob:0.3}", tokenizer.decode(&[idx as u32]))
    }
}

pub fn estimate_loss(logits: &Tensor, targets: &Tensor) -> candle_core::Result<Tensor> {
    log::debug!("reshaping logits");
    let (batch_size, time_size, channels_size) = logits.shape().dims3()?;
    let logits = logits.reshape(Shape::from((batch_size * time_size, channels_size)))?;
    let targets = targets.reshape(Shape::from((batch_size * time_size,)))?;

    log::debug!("applying cross entropy");
    loss::cross_entropy(&logits, &targets)
}
