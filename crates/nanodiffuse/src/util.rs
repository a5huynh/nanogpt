use std::{fs::File, path::PathBuf};
use candle_core::Tensor;
use image::codecs::bmp::BmpEncoder;


/// Saves a tensor as an image.
/// Assumes the tensor is a 2D tensor of (width, height).
pub fn tensor_as_image(tensor: &Tensor, path: &PathBuf) -> anyhow::Result<()> {
    let shape = tensor.shape();
    let (width, height) = shape.dims2()?;

    let mut file = File::create(path)?;
    let mut encoder = BmpEncoder::new(&mut file);

    // Convert the tensor values into grayscale.
    let raw: Vec<u8> = tensor.to_vec2::<u8>()?
        .into_iter()
        .flatten()
        // reverse the values so that the number appears as black on white.
        .map(|x| 255 - x)
        .collect();
    encoder.encode(&raw, width as u32, height as u32,  image::ExtendedColorType::L8)?;
    Ok(())
}
