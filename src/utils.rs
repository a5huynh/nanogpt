use candle_core::{Device, Tensor};

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
