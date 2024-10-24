use candle_core::{Device, Result, Tensor};
use candle_nn::Module;

pub struct LayerNorm {
    gamma: Tensor,
    beta: Tensor,
    eps: f64,
}

impl LayerNorm {
    pub fn new(dim: usize, eps: f64, device: &Device) -> Self {
        Self {
            gamma: Tensor::ones((1, dim), candle_core::DType::F32, device).unwrap(),
            beta: Tensor::zeros((1, dim), candle_core::DType::F32, device).unwrap(),
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        // calculate the forward pass
        let xmean = xs.mean_keepdim(1)?; // batch mean
        let xvar = xs.var_keepdim(2)?; // batch variance

        // normalize to unit variance
        let xhat = xs
            .broadcast_sub(&xmean)?
            .broadcast_div(&(xvar + self.eps)?)?;
        self.gamma.broadcast_mul(&xhat)?.broadcast_add(&self.beta)
    }
}
