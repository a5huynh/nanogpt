use candle_core::{backend::BackendDevice, Device};
use nanodiffuse::{run_diffusion, DiffuseError};

#[tokio::main]
async fn main() -> anyhow::Result<(), DiffuseError> {
    let device = if cfg!(target_os = "macos") {
        Device::Metal(candle_core::MetalDevice::new(0)?)
    } else if cfg!(target_os = "windows") || cfg!(target_os = "linux") {
        Device::Cuda(candle_core::CudaDevice::new(0)?)
    } else {
        return Err(DiffuseError::Other("OS not supported for GPU".into()));
    };

    let _ = run_diffusion(&device)?;

    Ok(())
}