use std::path::PathBuf;

use candle_core::{Device, Tensor};
use candle_nn::Module;
use mnist::*;
use model::BasicUnet;
use thiserror::Error;
use util::tensor_as_image;

mod model;
mod util;

const NUM_IMAGES: usize = 100;

#[derive(Error, Debug)]
pub enum DiffuseError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error("Other error: {0}")]
    Other(String),
}

/// Adds a random amount of noise to a tensor
/// Assumes input tensor has DType::f32
fn corrupt(input: &Tensor, amount: f32, device: &Device) -> Result<Tensor, candle_core::Error> {
    // Covert input to floating point
    let noise = input.rand_like(0.0, 1.0)?;
    let amount = Tensor::full(amount, (1, 28), device)?;

    // input * ( 1 - amount ) + noise * amount
    let percent = (1.0 - amount.clone())?;
    let rhs = noise.broadcast_mul(&amount)?;
    input.broadcast_mul(&percent)?.broadcast_add(&rhs)
}

fn read_mnist_data() -> Mnist {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("../../data/images/mnist");

    MnistBuilder::new()
        .base_path(path.as_os_str().to_str().unwrap())
        .test_set_length(NUM_IMAGES as u32)
        .training_set_length(NUM_IMAGES as u32)
        .finalize()
}

pub fn run_diffusion(device: &candle_core::Device) -> Result<(), DiffuseError> {
    let mnist = read_mnist_data();

    // Convert data to values between 0 and 1.0
    let data = Tensor::from_vec(mnist.trn_img, &[NUM_IMAGES, 28, 28], device)?;
    let data = data.to_dtype(candle_core::DType::F32)?;
    let data = (data / 255.0)?;

    let single = data.get(0)?;
    // let _labels = Tensor::from_vec(mnist.trn_lbl, &[NUM_IMAGES], device)?;
    tensor_as_image(&single, &"img_test.bmp".into())
        .map_err(|err| DiffuseError::Other(err.to_string()))?;

    // Corrupt image
    for idx in 0..10 {
        let corruped = corrupt(&single, idx as f32 / 10.0, device)?;
        tensor_as_image(&corruped, &format!("img_corrupted_{idx}.bmp").into())
            .map_err(|err| DiffuseError::Other(err.to_string()))?;
    }

    let _model = BasicUnet::new(1, 1, device)?;
    _model.forward(&single)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_loader() {
        read_mnist_data();
    }
}
