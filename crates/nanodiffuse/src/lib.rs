use std::path::PathBuf;

use candle_core::Tensor;
use mnist::*;
use thiserror::Error;
use util::tensor_as_image;

mod util;

const NUM_IMAGES: usize = 100;

#[derive(Error, Debug)]
pub enum DiffuseError {
    #[error(transparent)]
    Candle(#[from] candle_core::Error),
    #[error("Other error: {0}")]
    Other(String),
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

    let data = Tensor::from_vec(mnist.trn_img, &[NUM_IMAGES, 28, 28], device)?;
    let single = data.get(0)?;
    dbg!(single.shape());

    let labels = Tensor::from_vec(mnist.trn_lbl, &[NUM_IMAGES, 1], device)?;
    let single = labels.get(0)?;
    dbg!(single.shape());
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
